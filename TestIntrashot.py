"""
Generate Training Dataset with Simulated Motion Correction

Test data will be provided upon request
"""

import os
import pathlib as plib
from time import time
from functools import partial
import itertools
import numpy as np

import jax
import jax.numpy as xp
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="0" #turn off GPU pre-allocation
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

os.environ['CUDA_VISIBLE_DEVICES'] = '' #TEMPORARY force to use CPU
print(jax.numpy.ones(3).device()) # TFRT_CPU_0

import encode.encode_op as eop
import recon.recon_op as rec
import cnn.run_unet as cnn
import utils.metrics as mtc
import utils.visualize as vis
import motion.motion_sim as msi


#-------------------------------------------------------------------------------
#-------------------------Image Acquisition Simulation--------------------------
#-------------------------------------------------------------------------------
#Load data
mpath = r'/home/nghiemb/PyMoCo/data/cc/test/PE1_AP/Complex/R1/Paradigm_1E'
case = 1
test_case = 'Test{}'.format(case)
dpath = mpath + r'/{}'.format(test_case)
spath = r'/home/nghiemb/PyMoCo/data/cc/test/PE1_AP/Complex/R1/Intrashot/Paradigm_1D'

res = xp.array([1,1,1])
m_GT_init = xp.load(dpath + r'/current_test_GT.npy') #SI, LR, AP
m_GT = xp.pad(m_GT_init[:,:,:,0,0] + 1j*m_GT_init[:,:,:,1,0], ((1,1), (0,0), (0,0)))
del m_GT_init
C = xp.load(dpath + r'/sens.npy')

mask = rec.getMask(C); xp.save(dpath + r'/m_GT_brain_mask.npy', mask)
cerebrum_mask = xp.ones(m_GT.shape)

#Transpose to reorient as LR, AP, SI
m_GT = xp.transpose(m_GT, (1,2,0))
m_GT = xp.abs(m_GT[6:-6, 3:-3, :])

#---------------------------------------
TR = 1.6 #T1w MPRAGE acquisition parameter
Rs = 1 #SENSE acceleration factor
TR_shot = 16
print("Simulated motion temporal resolution: {} sec".format(TR * TR_shot))

U = msi.make_samp(m_GT, Rs, TR_shot, order='interleaved', mode = 'list')

#---------------------------------------
#Generating discrete random motion trajectory

# TR_shot_effective = 2
# U_len_effective = m_GT.shape[1]//TR_shot_effective
# r_scale = (TR_shot_effective / 16) * 4 #NEED TO ADJUST AS DESIRED
# p_scale = (TR_shot_effective / 16) * 2 #NEED TO ADJUST AS DESIRED
# specs_scale = [r_scale, p_scale]
specs_scale = [1, 1]

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

motion_lv = 'severe1'
j = 1; k = 1 #legacy parameters, from training dataset script
rand_keys = msi._gen_key(60+case, j, k)
Mtraj_GT = msi._gen_traj(rand_keys, len(U), motion_specs.get(motion_lv), specs_scale)

# vis.plot_Mtraj(Mtraj_GT, Mtraj_GT, m_GT.shape, rescale = 0)


#---------------------------------------
#SIMULATING INTRASHOT MOTION --> SMOOTH [LINEAR] INTERPOLATION (NO OTHER CHANGES)

TR_shot_effective = 2
U_dscale = TR_shot//TR_shot_effective
U_effective = msi._U_subdivide(U, U_dscale)

Mtraj_GT_effective = msi.Mtraj_interp(Mtraj_GT, U_dscale)
# vis.plot_Mtraj(Mtraj_GT_effective, Mtraj_GT_effective, m_GT.shape, rescale = 0)

#Apply motion simulation
R_pad = (10, 10, 10)
batch = 1
t1 = time()
s_corrupted = eop.Encode(m_GT, C, U_effective, Mtraj_GT, res, batch=batch)
t2 = time()
print("Elapsed time for effective temporal res of {} sec: {} sec".format(TR * TR_shot_effective, t2 - t1))



'''


#---------------------------------------------------------------------------
#------------------JOINT IMAGE RECON AND MOTION ESTIMATION------------------
#---------------------------------------------------------------------------
#Initializing update vars
Mtraj_init = xp.zeros((len(U), 6))
Mtraj_est = Mtraj_init
CG_maxiter = 3 #limit CG_iter to 3 iters for fully-sampled data to prevent artifacts
ME_maxiter = 1 #motion estimation maxiter
LS_maxiter = 20 #line search maxiter for BFGS algorithm
CG_tol = 1e-7 #relative tolerance
CG_atol = 1e-4 #absolute tolerance
CG_lamda = 0
CG_mask = 0 #turn on for in-vivo dataset, turn off for simulated dataset
#Initialize stores
m_loss_store = []
m_cnn_store = []
Mtraj_store = []
DC_store = []
#---------------------------------------------------------------------------
#Reconstruct image via EH, since data is fully-sampled
m_init = eop.Encode_Adj(s_corrupted, C, U, Mtraj_init, res, batch=batch) #E.H*s
#Motion-corrupted reconstruction
A = partial(eop._EH_E, C=C, U=U, Mtraj=Mtraj_est, res=res, \
            lamda = CG_lamda, batch=batch)
b = eop.Encode_Adj(s_corrupted, C, U, Mtraj_est, res, batch=batch)
#
m_corrupted = m_init
#----------------------------------------
m_est_rmse = mtc.evalPE(m_corrupted, m_GT, mask)
m_est_ssim = mtc.evalSSIM(m_corrupted, m_GT, mask=mask)
m_loss_store.append([m_est_rmse, m_est_ssim])
print("RMSE of Corrupted Image: {:.2f} %".format(m_est_rmse))
print("SSIM of Corrupted Image: {}".format(m_est_ssim))
m_est = m_corrupted
#---------------------------------------------------------------------------
#Loading trained CNN model
# NB. UNet takes in data as [LR, AP, SI]
# For my Data (SI, AP, LR), need to transpose --> (2,1,0)
cnn_path = r'/home/nghiemb/PyMoCo/cnn/3DUNet_SAP'
# wpath_severe = cnn_path + r'/weights/PE1_AP/Complex/{}/train_n360'.format('combo')
# wpath_moderate = cnn_path + r'/weights/PE1_AP/Complex/{}/train_n360'.format('combo')
# wpath_mild = cnn_path + r'/weights/PE1_AP/Complex/{}/train_n360'.format('combo')
wpath = r'/home/nghiemb/PyMoCo/cnn/3DUNet_SAP/weights/PE1_AP/Complex/combo/train_n240_sequential'
wpath_severe = wpath; wpath_moderate = wpath; wpath_mild = wpath
pads = [11,3]
#---------------------------------------------------------------------------
#Alternating image & motion estimation (coordinate descent)
rmse_tol = 0.0 #impossible
ssim_tol = 2.0 #impossible
trans_axes = (0,1,2,0) 
cnn_flag = test_flag[0] #turn on / off CNN
JE_flag = test_flag[1] #turn JE algorithm on / off
thresh = {'severe': 500, 'moderate': 0.1}
if JE_flag and cnn_flag: #UNet + JE
    spath = spath_root + r'/w_cnn_SEQUENTIAL_RETRAINEDUNET2_2025-02-22'
    max_loops = 200
elif JE_flag and not cnn_flag: #only JE
    spath = spath_root + r'/wo_cnn_SEQUENTIAL_RETRAINEDUNET2_2025-02-22'
    max_loops = 200
elif not JE_flag and cnn_flag: #only UNet
    spath = spath_root + r'/w_only_cnn_SEQUENTIAL_RETRAINEDUNET2_2025-02-22'
    max_loops = 1
plib.Path(spath).mkdir(parents=True, exist_ok=True)
xp.save(spath + r'/m_corrupted.npy', m_corrupted)
xp.save(spath + r'/Mtraj.npy', Mtraj_GT)
xp.save(spath + r'/s_corrupted_mag.npy', s_corrupted)
#
#---------------------------------------------------------------------------
dscale = 1
continuity = 0
grad_tol = 1e-4 #
JE_params = [m_est_rmse, rmse_tol, m_est_ssim, ssim_tol, max_loops, ME_maxiter, LS_maxiter, \
                CG_maxiter, CG_tol, CG_atol, CG_mask, batch, mask, continuity, grad_tol]
CNN_params = [cnn_flag, JE_flag, trans_axes, pads, wpath_severe, wpath_moderate, wpath_mild, thresh]
init_est = [m_est, Mtraj_est]
fixed_vars = [m_init, s_corrupted, C, U, dscale, res, spath, m_GT, R_pad, cerebrum_mask]
#
DC_store.append(rec.eval_TotalDC(Mtraj_est, fixed_vars, JE_params))
xp.save(spath + r"/DC_store.npy", DC_store)
DC_init_alt = rec._f(Mtraj_init, m_est=m_corrupted, C=C, res=res, U=U, R_pad=R_pad, s_corrupted=s_corrupted)
xp.save(spath + r"/DC_init_alt.npy", DC_init_alt)
#
stores = [m_cnn_store, Mtraj_store, m_loss_store, DC_store]
m_est, m_loss_store, Mtraj_store, m_cnn_store = rec.JointEst(init_est, fixed_vars, \
                                                                stores, cnn, \
                                                                CNN_params, JE_params)
#




#---------------------------------------------------------------------------
#Generate sliding window
#---------------------------------------

PE1_combined_init = [list(U[i][1]) for i in range(len(U))]
PE1_combined = list(itertools.chain.from_iterable(PE1_combined_init))

strides = [0,8] #takes value between [0, 15]
# strides = [0,4,8,12] #takes value between [0, 15]
# strides = [0,2,4,6,8,10,12,14] #takes value between [0, 15]
PE1_sliding_window = []
for shot_nominal_iter in range(len(U)):
    for stride in strides:
        window_temp = PE1_combined[TR_shot*shot_nominal_iter+stride:TR_shot*(shot_nominal_iter+1)+stride]
        PE1_sliding_window.append(window_temp)

U_sliding_window = []
U_RO_vals = U[0][0]
U_PE2_vals = U[0][2]
for i in range(len(PE1_sliding_window)):
    U_PE1_vals = np.asarray(PE1_sliding_window[i])
    U_vals_temp = np.asarray([U_RO_vals, U_PE1_vals, U_PE2_vals])
    U_sliding_window.append(U_vals_temp)


# U_temp = msi._gen_U_n(U_sliding_window[9], m_GT.shape)


'''




'''
#---------------------------------------------------------------------------
#-------------------------IN VIVO CONTINUOUS MOTION-------------------------
#---------------------------------------------------------------------------
#Loading in vivo data with continuous head motion
mpath_sub1 = r'/home/nghiemb/Data/TWH/MPRAGE_PE1Reordered/Scan20231204/Sub1/h5data'
dpath_sub1 = mpath_sub1 + r'/scan1-ContinuousMotion-16shots/npy'
gt_name_sub1 = r'scan1-ContinuousMotion-16shots/npy'
cerebrum_slice_sub1 = 190
paths_sub1 = [mpath_sub1, dpath_sub1, gt_name_sub1, cerebrum_slice_sub1]



s_corrupted = xp.load(dpath + r'/kdat_trunc.npy') #NC, SI, AP, LR
if sub == 4:
    C_init = xp.load(dpath + r'/sens.npy')
else:
    C_init = xp.load(mpath + r'/{}/sens.npy'.format(gt_name))
C = xp.transpose(C_init, (3,0,1,2))
mask = rec.getMask(C); xp.save(dpath + r'/m_GT_brain_mask.npy', mask)
del C_init

U = np.load(dpath + r'/samp_order.npy', allow_pickle=1) #LR, AP, SI
res = xp.array([1,1,1])
#---------------------------------------
# 
# maxval = abs(m_GT.flatten()).max()
# m_GT /= maxval
# s_corrupted /= maxval
try:
    m_GT = xp.load(mpath + r'/{}/img_CG.npy'.format(gt_name)) ###I THINK I'VE OVERWRITTEN IMG_CG, SO MAXVAL NOW = 1
except:
    m_GT = xp.ones(s_corrupted.shape[1:]) #BYPASSING LOADING M_GT
maxval = abs(m_GT.flatten()).max()
m_GT /= maxval
s_corrupted /= maxval

#---------------------------------------
#Loading the skull-stripping mask, generated from FreeSurfer SynthStrip tool
cerebrum_mask = xp.ones(m_GT.shape)
cerebrum_mask = cerebrum_mask.at[cerebrum_slice:,...].set(0)

#---------------------------------------
#Motion trajectory
R_pad = (10, 10, 10)
batch = 1


'''
