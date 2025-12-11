"""
Generate Training Dataset with Simulated Motion Correction

Test data will be provided upon request

If working on the h4h server:
salloc -p gpu --account=uludag_gpu --gres=gpu:v100:1 -C gpu32g -t 1:00:00 -c 4 --mem 100G

"""

import os
import pathlib as plib
from time import time
from functools import partial
import itertools
import numpy as np

# CPU_FLAG = 0
CPU_FLAG = 1 #TEMPORARY force to use CPU
if CPU_FLAG:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 
    os.environ['JAX_PLATFORMS'] = 'cpu'
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['JAX_PLATFORMS'] = 'cuda'
    #
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="0" #turn off GPU pre-allocation
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import jax
import jax.numpy as xp

import encode.encode_op as eop
import recon.recon_op as rec
import cnn.run_unet as cnn
import utils.metrics as mtc
import utils.visualize as vis
import motion.motion_sim as msi


#-------------------------------------------------------------------------------
#---------------------------------LOADING DATA----------------------------------
#-------------------------------------------------------------------------------
#Load data
mpath = r'/cluster/projects/uludag/Brian/PyMoCo_v2'
case = 1 #4, 5, 6, 7
test_case = 'Test{}'.format(case)
dpath = mpath + r'/data/Simulations/{}'.format(test_case)

res = xp.array([1,1,1])
m_GT = xp.load(dpath + r'/m_complex/img_CG.npy') #SI, LR, AP
C = xp.load(dpath + r'/sens/sens.npy')

mask = rec.getMask(C)
cerebrum_mask = xp.ones(m_GT.shape)

#Defining path to NN weights
# NB. UNet takes in data as [LR, AP, SI]
cnn_path = r'/home/nghiemb/PyMoCo/cnn/3DUNet_SAP'
wpath = cnn_path + r'/weights/PE1_AP/Complex/combo/train_n240_interleaved_P1EF_2025-04-02/slices'
pads = [11,3]

#---------------------------------------------------------------------------
#--------------------------MOTION SIMULATION BLOCK--------------------------
#---------------------------------------------------------------------------

#Defining motion simulation parameters
mild_specs = {'Tx':[0.1,0.1],'Ty':[0.2,0.15],'Tz':[0.2,0.15],\
            'Rx':[0.2,0.15],'Ry':[0.1,0.1],'Rz':[0.1,0.1]} #[max_rate, prob]
moderate_specs = {'Tx':[0.2,0.1],'Ty':[0.4,0.2],'Tz':[0.4,0.2],\
            'Rx':[0.5,0.2],'Ry':[0.2,0.1],'Rz':[0.2,0.1]} #[max_rate, prob]
severe_specs1 = {'Tx':[0.4,0.15],'Ty':[0.9,0.3],'Tz':[0.9,0.3],\
            'Rx':[1,0.3],'Ry':[0.5,0.15],'Rz':[0.5,0.15]} #[max_rate, prob]
severe_specs2 = {'Tx':[0.8,0.3],'Ty':[1.8,0.6],'Tz':[1.8,0.6],\
            'Rx':[2,0.6],'Ry':[1.0,0.3],'Rz':[1.0,0.3]} #Double the max_rate and probability
severe_specs3 = {'Tx':[1.6,0.6],'Ty':[3.6,1.0],'Tz':[3.6,1.0],\
            'Rx':[4,1.0],'Ry':[2.0,0.6],'Rz':[2.0,0.6]} #Quadruple the probability
motion_specs = {'moderate':moderate_specs,'severe1':severe_specs1,\
                'severe2':severe_specs2, 'severe3':severe_specs3}

#Setting up nominal motion trajectory
TR = 1.6 #T1w MPRAGE acquisition parameter
Rs = 1 #SENSE acceleration factor
TR_shot = 16
print("Simulated motion temporal resolution: {} sec".format(TR * TR_shot))

U = msi.make_samp(m_GT, Rs, TR_shot, order='interleaved', mode = 'list')

specs_scale = [1, 1] # [r_scale, p_scale]
motion_lv = 'severe1'
j = 1; k = 1 #legacy parameters, from training dataset script
rand_keys = msi._gen_key(60+case, j, k)
Mtraj_GT = msi._gen_traj(rand_keys, len(U), motion_specs.get(motion_lv), specs_scale)

# vis.plot_Mtraj(Mtraj_GT, Mtraj_GT, m_GT.shape, rescale = 0)

#----------------------------------------
#SIMULATING INTRASHOT MOTION --> SMOOTH [LINEAR] INTERPOLATION (NO OTHER CHANGES)

dscale = 4
TR_shot_effective = TR_shot // dscale
U_dscale = TR_shot//TR_shot_effective
U_effective = msi._U_subdivide(U, U_dscale)

Mtraj_GT_effective = msi.Mtraj_interp(Mtraj_GT, U_dscale)
# vis.plot_Mtraj(Mtraj_GT_effective, Mtraj_GT_effective, m_GT.shape, rescale = 0)

#Apply motion simulation with interpolated trajectories
R_pad = (10, 10, 10)
batch = 1
t1 = time()
s_corrupted = eop.Encode(m_GT, C, U_effective, Mtraj_GT_effective, res, batch=batch) #on h4h server CPU, 1 shot takes 20 seconds! 
t2 = time()
print("Elapsed time for effective temporal res of {} sec: {} sec".format(TR * TR_shot_effective, t2 - t1))

#----------------------------------------
#INITIAL IMAGE RECON, ASSUMING ZERO MOTION'
U_zero = msi.make_samp(m_GT, 1, m_GT.shape[1], order='interleaved', mode = 'list')
Mtraj_zero = xp.zeros((len(U_zero), 6))
m_corrupted = eop.Encode_Adj(s_corrupted, C, U_zero, Mtraj_zero, res, batch=batch) #E.H*s
m_est_rmse = mtc.evalPE(m_corrupted, m_GT, mask)
m_est_ssim = mtc.evalSSIM(m_corrupted, m_GT, mask=mask)

m_est = m_corrupted
DC_init_alt = rec._f(Mtraj_zero, m_est=m_corrupted, C=C, res=res, \
                     U=U_zero, R_pad=R_pad, s_corrupted=s_corrupted)

#---------------------------------------------------------------------------
#------------------JOINT IMAGE RECON AND MOTION ESTIMATION------------------
#---------------------------------------------------------------------------

#Defining algorithm parameters
CG_maxiter = 3 #limit CG_iter to 3 iters for fully-sampled data to prevent artifacts
ME_maxiter = 1 #motion estimation maxiter
LS_maxiter = 20 #line search maxiter for BFGS algorithm
CG_tol = 1e-7 #relative tolerance
CG_atol = 1e-4 #absolute tolerance
CG_lamda = 0
CG_mask = 0 #turn on for in-vivo dataset, turn off for simulated dataset

rmse_tol = 0.0 #impossible
ssim_tol = 2.0 #impossible
trans_axes = (0,1,2,0) 
cnn_flag = 0 #binary, flag for using UNet
JE_flag = 1 #binary, flag for using JE
thresh = {'severe': 500, 'moderate': 0.1}

dscale = 1
continuity = 0
grad_tol = 1e-4 #threshold for finite dif of shotwise DC loss

#Initialize stores
m_loss_store = []
m_loss_store.append([m_est_rmse, m_est_ssim])
DC_store = []

m_cnn_store = []
Mtraj_store_lv1 = []


if JE_flag and cnn_flag: #UNet + JE
    spath = dpath + r'/Intrashot/Upres_{}x/Hierarchical/M3_2025-04-15'.format(dscale)
    # max_loops = 200
elif JE_flag and not cnn_flag: #only JE
    spath = dpath + r'/Intrashot/Upres_{}x/Hierarchical/M2_2025-04-15'.format(dscale)
    # max_loops = 200
elif not JE_flag and cnn_flag: #only UNet
    spath = dpath + r'/Intrashot/Upres_{}x/Hierarchical/M1_2025-04-15'.format(dscale)
    # max_loops = 1

plib.Path(spath).mkdir(parents=True, exist_ok=True)

xp.save(spath + r'/Mtraj_GT.npy', Mtraj_GT_effective)
xp.save(spath + r'/m_corrupted.npy', m_corrupted)
xp.save(spath + r'/s_corrupted.npy', s_corrupted)
xp.save(spath + r'/U_GT.npy', U_effective)
xp.save(spath + r"/DC_init_alt.npy", DC_init_alt)

#----------------------------------------
#---------------------------LEVEL 1 --> 2x UPRES----------------------------

max_loops_lv1 = 100

dscale = 2
TR_shot_effective = TR_shot // dscale
U_dscale = TR_shot//TR_shot_effective
U_effective = msi._U_subdivide(U, U_dscale) #REDEFINE THE SAMPLING PATTERN
Mtraj_init = xp.zeros((len(U_effective), 6))
Mtraj_est = Mtraj_init

#Initializing update vars
JE_params = [m_est_rmse, rmse_tol, m_est_ssim, ssim_tol, max_loops_lv1, ME_maxiter, LS_maxiter, \
                CG_maxiter, CG_tol, CG_atol, CG_mask, batch, mask, continuity, grad_tol]
CNN_params = [cnn_flag, JE_flag, trans_axes, pads, wpath, wpath, wpath, thresh]
init_est = [m_est, Mtraj_est]
fixed_vars = [m_corrupted, s_corrupted, C, U, dscale, res, spath, m_GT, R_pad, cerebrum_mask]

DC_store.append(rec.eval_TotalDC(Mtraj_est, fixed_vars, JE_params))
xp.save(spath + r"/DC_store.npy", DC_store)

#
stores = [m_cnn_store, Mtraj_store_lv1, m_loss_store, DC_store]
m_est, m_loss_store, Mtraj_store_lv1, m_cnn_store = rec.JointEst(init_est, fixed_vars, \
                                                                stores, cnn, \
                                                                CNN_params, JE_params)

Mtraj_est_lv1 = Mtraj_store_lv1[-1][0]

xp.save(spath + r'/m_est_lv1.npy', m_est)
xp.save(spath + r'/Mtraj_store_lv1.npy', Mtraj_store_lv1)
xp.save(spath + r'/Mtraj_final_lv1.npy', Mtraj_est_lv1)
xp.save(spath + r'/m_loss_store_lv1.npy', m_loss_store)
xp.save(spath + r'/DC_store_lv1.npy', DC_store)




#---------------------------LEVEL 2 --> 4x UPRES----------------------------

max_loops_lv2 = 50

lv_scale = 2
dscale *= lv_scale
TR_shot_effective = TR_shot // dscale
U_dscale = TR_shot//TR_shot_effective
U_effective = msi._U_subdivide(U, U_dscale) #REDEFINE THE SAMPLING PATTERN

Mtraj_est_lv2 = msi.Mtraj_interp(Mtraj_est_lv1, lv_scale)
Mtraj_store_lv2 = []

#Initializing update vars
JE_params = [m_est_rmse, rmse_tol, m_est_ssim, ssim_tol, max_loops_lv2, ME_maxiter, LS_maxiter, \
                CG_maxiter, CG_tol, CG_atol, CG_mask, batch, mask, continuity, grad_tol]
CNN_params = [cnn_flag, JE_flag, trans_axes, pads, wpath, wpath, wpath, thresh]
init_est = [m_est, Mtraj_est_lv2]
fixed_vars = [m_corrupted, s_corrupted, C, U, dscale, res, spath, m_GT, R_pad, cerebrum_mask]

#
stores = [m_cnn_store, Mtraj_store_lv2, m_loss_store, DC_store]
m_est, m_loss_store, Mtraj_store_lv2, m_cnn_store = rec.JointEst(init_est, fixed_vars, \
                                                                stores, cnn, \
                                                                CNN_params, JE_params)

Mtraj_est_lv2 = Mtraj_store_lv2[-1][0]

xp.save(spath + r'/m_est_lv2.npy', m_est)
xp.save(spath + r'/Mtraj_store_lv2.npy', Mtraj_store_lv2)
xp.save(spath + r'/Mtraj_final_lv2.npy', Mtraj_est_lv2)
xp.save(spath + r'/m_loss_store_lv2.npy', m_loss_store)
xp.save(spath + r'/DC_store_lv2.npy', DC_store)




'''

# import jax.numpy as xp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False


def plot_views(img, vmax = 1.0):
    if vmax == "auto": #if auto, set as max val of volume
        vmax = abs(img.flatten().detach().cpu()).max()
    #
    fig, axes = plt.subplots(1,3)
    for i, ax in enumerate(axes):
        if i==0:
            ax.imshow(img[img.shape[0]//2,:,:], cmap = "gray", vmax = vmax)
        if i==1:
            ax.imshow(img[:,img.shape[1]//2,:], cmap = "gray", vmax = vmax)
        if i==2:
            ax.imshow(img[:,:,img.shape[2]//2], cmap = "gray", vmax = vmax)
        #
    plt.show()

def plot_Mtraj(Mtraj_GT, Mtraj, img_dims, rescale = 0):
    Nx, Ny, Nz = img_dims
    if rescale:
        Tx_scale = (Nx/2)
        Ty_scale = (Ny/2)
        Tz_scale = (Nz/2)
        R_scale = 1/(np.pi/180)
    else:
        Tx_scale = 1; Ty_scale = 1; Tz_scale = 1
        R_scale = 1
    #
    T_GT = Mtraj_GT[:,:3]
    R_GT = Mtraj_GT[:,3:]
    T = Mtraj[:,:3]
    R = Mtraj[:,3:]
    plt.figure()
    plt.plot(T_GT[:,0]*Tx_scale, '--r', alpha = 0.75, label="Tx - GT")
    plt.plot(T_GT[:,1]*Ty_scale, '--b', alpha = 0.75, label="Ty - GT")
    plt.plot(T_GT[:,2]*Tz_scale, '--g', alpha = 0.75, label="Tz - GT")
    plt.plot(T[:,0]*Tx_scale, 'r', label="Tx")
    plt.plot(T[:,1]*Ty_scale, 'b', label="Ty")
    plt.plot(T[:,2]*Tz_scale, 'g', label="Tz")
    # plt.legend(loc="lower left")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=5)
    plt.ylabel("Translations (mm)")
    plt.xlabel("Shot Index")
    plt.title("Estimated Motion Trajectories: Translations")
    plt.show()
    #
    plt.figure()
    plt.plot(R_GT[:,0]*R_scale, '--r', alpha = 0.75, label="Rx - GT")
    plt.plot(R_GT[:,1]*R_scale, '--b', alpha = 0.75, label="Ry - GT")
    plt.plot(R_GT[:,2]*R_scale, '--g', alpha = 0.75, label="Rz - GT")
    plt.plot(R[:,0]*R_scale, 'r', label="Rx")
    plt.plot(R[:,1]*R_scale, 'b', label="Ry")
    plt.plot(R[:,2]*R_scale, 'g', label="Rz")
    # plt.legend(loc="upper left")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=5)
    plt.ylabel("Rotations (deg)")
    plt.xlabel("Shot Index")
    plt.title("Estimated Motion Trajectories: Rotations")
    plt.show()


DC_init = np.load("DC_init_alt.npy")
DC_store = np.load("DC_store.npy"); DC_store[0] = DC_init

plt.figure()
plt.plot(DC_store, label = "JE")
plt.xlabel("JE iteration")
plt.ylabel("DC Loss")
plt.title("Data Consistency Loss Trajectory for Outlier Test Case")
plt.legend(loc = "upper right")
plt.show()



U_effective = np.load("U_effective.npy", allow_pickle=1)


Mtraj_store = np.load("Mtraj_store.npy", allow_pickle=1)
Mtraj_GT = np.load("Mtraj_GT_effective.npy")
Mtraj_final = Mtraj_store[-1][0]

m_corrupted = np.load("m_corrupted.npy")
m_final = np.load("m_intmd.npy")

plot_Mtraj(Mtraj_GT, Mtraj_final, m_final.shape)

plot_views(abs(m_corrupted))
plot_views(abs(m_final))



'''