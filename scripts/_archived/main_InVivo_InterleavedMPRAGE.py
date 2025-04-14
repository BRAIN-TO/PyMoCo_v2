"""
MAIN SCRIPT
Running Joint Motion and Image Estimation
on Real Motion-Corrupted Data, with Interleaved PE1 Reordering (acquired July 2023)
"""
import os
import pathlib as plib
from time import time
from functools import partial

import jax.numpy as xp
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="0" #turn off GPU pre-allocation

import encode.encode_op as eop
import recon.recon_op as rec
import cnn.run_unet as cnn
import utils.metrics as mtc
import motion.motion_sim as msi

#-------------------------------------------------------------------------------
def main(dpath, spath_root, mpath, gt_name):
    #---------------------------------------------------------------------------
    #-----------------------Image Acquisition Simulation------------------------
    #---------------------------------------------------------------------------
    #Load data
    s_corrupted = xp.load(dpath + r'/kdat_trunc.npy') #NC, SI, AP, LR
    C_init = xp.load(dpath + r'/sens.npy')
    C = xp.transpose(C_init, (3,0,1,2))
    mask = rec.getMask(C); xp.save(dpath + r'/m_GT_brain_mask.npy', mask)
    res = xp.array([1,1,1])
    #Loading sampling pattern
    Rs = 1
    TR_shot = 15
    order = 'interleaved'
    U = msi.make_samp(s_corrupted[0,...], Rs, TR_shot, order)
    U = xp.transpose(U, axes = (0,2,1,3)) #PE1 was along AP, need to match
    xp.save(dpath + r'/U.npy', U)
    #
    m_GT = xp.load(mpath + r'/{}/img_rss_trunc.npy'.format(gt_name))
    maxval = abs(m_GT.flatten()).max()
    m_GT /= maxval
    s_corrupted /= maxval
    #Motion trajectory
    R_pad = (10, 10, 10)
    batch = 1
    #Removing neck coil elements:
    s_corrupted_new = xp.concatenate((s_corrupted[:8,...], s_corrupted[10:-2,...]), axis = 0)
    C_new = xp.concatenate((C[:8,...], C[10:-2,...]), axis = 0)
    s_corrupted = s_corrupted_new
    C = C_new
    #
    #---------------------------------------------------------------------------
    #------------------JOINT IMAGE RECON AND MOTION ESTIMATION------------------
    #---------------------------------------------------------------------------
    #Initializing update vars
    Mtraj_init = xp.zeros((U.shape[0], 6))
    Mtraj_est = Mtraj_init
    CG_maxiter = 3 #limit CG_iter to 3 iters for fully-sampled data to prevent artifacts
    ME_maxiter = 1 #motion estimation maxiter
    LS_maxiter = 20 #line search maxiter for BFGS algorithm
    CG_tol = 1e-7 #relative tolerance
    CG_atol = 1e-4 #absolute tolerance
    CG_lamda = 0
    CG_mask = 1 #turn on for in-vivo dataset, turn off for CC dataset
    #Initialize stores
    m_loss_store = []
    m_cnn_store = []
    Mtraj_store = []
    #---------------------------------------------------------------------------
    #Reconstruct image using CG SENSE algorithm
    m_init = eop.Encode_Adj(s_corrupted, C, U, Mtraj_init, res, batch=batch) #E.H*s
    #Motion-corrupted reconstruction
    A = partial(eop._EH_E, C=C, U=U, Mtraj=Mtraj_est, res=res, \
                lamda = CG_lamda, batch=batch)
    b = eop.Encode_Adj(s_corrupted, C, U, Mtraj_est, res, batch=batch)
    #
    m_corrupted = m_init
    m_est_rmse = mtc.evalPE(m_corrupted, m_GT, mask)
    m_est_ssim = mtc.evalSSIM(m_corrupted, m_GT, mask=mask)
    m_loss_store.append([m_est_rmse, m_est_ssim])
    print("RMSE of Corrupted Image: {:.2f} %".format(m_est_rmse))
    print("SSIM of Corrupted Image: {}".format(m_est_ssim))
    m_est = m_corrupted
    #---------------------------------------------------------------------------
    #Loading trained CNN model
    # NB. UNet takes in data as [LR, AP, SI]
    # For my Ref Data (SI, AP, LR), need to transpose --> (2,1,0)
    cnn_path = r'/home/nghiemb/PyMoCo/cnn/3DUNet_SAP'
    wpath_severe = cnn_path + r'/weights/{}/train_n134'.format('combo')
    wpath_moderate = cnn_path + r'/weights/{}/train_n134'.format('combo')
    wpath_mild = cnn_path + r'/weights/{}/train_n134'.format('combo')
    pad_x = int((xp.ceil(m_est.shape[2]/32) * 32 - m_est.shape[2])/2) #along LR
    pad_y = int((xp.ceil(m_est.shape[1]/32) * 32 - m_est.shape[1])/2) #along AP
    pads = [pad_x, pad_y]
    #---------------------------------------------------------------------------
    #Alternating image & motion estimation (coordinate descent)
    rmse_tol = 5.0
    trans_axes = (2,1,0)
    cnn_flag = 0 #turn on / off CNN
    JE_flag = 1 #turn JE algorithm on / off
    thresh = {'severe': 500, 'moderate': 0.1}
    if JE_flag and cnn_flag: #UNet + JE
        spath = spath_root + r'/w_cnn_combo_fixedU'
        max_loops = 50
    elif JE_flag and not cnn_flag: #only JE
        spath = spath_root + r'/wo_cnn_fixedU'
        max_loops = 1000
    elif not JE_flag and cnn_flag: #only UNet
        spath = spath_root + r'/w_only_cnn_alt_fixedU'
        max_loops = 1
    plib.Path(spath).mkdir(parents=True, exist_ok=True)
    xp.save(spath + r'/m_corrupted.npy', m_corrupted)
    #
    JE_params = [m_est_rmse, rmse_tol, max_loops, ME_maxiter, LS_maxiter, \
                    CG_maxiter, CG_tol, CG_atol, CG_mask, batch, mask]
    CNN_params = [cnn_flag, JE_flag, trans_axes, pads, wpath_severe, wpath_moderate, wpath_mild, thresh]
    init_est = [m_est, Mtraj_est]
    fixed_vars = [m_init, s_corrupted, C, U, mask, res, spath, m_GT, R_pad]
    stores = [m_cnn_store, Mtraj_store, m_loss_store]
    m_est, m_loss_store, Mtraj_store, m_cnn_store = rec.JointEst(init_est, fixed_vars, \
                                                                    stores, cnn, \
                                                                    CNN_params, JE_params)
    #
    return spath, m_corrupted, m_est, m_loss_store, Mtraj_store

#%% Run main()
if __name__ == "__main__":
    mpath = r'/home/nghiemb/Data/TWH/MPRAGE_PE1Reordered/Scan20230710'
    # test_case = 'scan6-mild'
    # gt_name = 'scan5-reference-reorder'
    test_case = 'scan3-severe'
    gt_name = 'scan2-reference-reorder'
    dpath = mpath + r'/{}'.format(test_case)
    print('Processing Test Case {}'.format(test_case))
    spath_root = dpath
    spath, m_corrupted, m_final, m_loss_store, Mtraj_store = main(dpath, spath_root, mpath, gt_name)
    xp.save(spath + r"/m_corrupted.npy", m_corrupted)
    xp.save(spath + r"/m_final.npy", m_final)
    xp.save(spath + r"/m_loss_store.npy", m_loss_store)
    xp.save(spath + r"/Mtraj_store.npy", Mtraj_store)


'''
#Generating fully-sampled version of interleaved sampling pattern
#To send to Tim, who is developing Siemens UI for PE1 mods 

import motion.motion_sim as msi
import numpy as np

mpath = r'/home/nghiemb/PyMoCo/data/cc/test/combo/severe_cases/additional_n49'
i = 1
dpath = os.path.join(mpath, 'Test{}'.format(i))
spath_root = dpath

PE1 = 256
m = np.zeros((PE1, 224, 192))

Rs = 1
TR_shot = 32
order = 'interleaved'
U = msi.make_samp(m, Rs, TR_shot, order)

#Exporting the matrix size and PE1 order
#To be sent to Tim
mat_size = np.array([m.shape[2], m.shape[0], m.shape[1]])

inds = []
for i in range(len(U)):
    U_i = U[i]
    inds.append(np.where(U_i[:,0,0] == 1)[0])

inds_stacked = np.hstack(inds)

with open(spath_root + r'/samp_pattern_R{}_PE1_{}_{}shots.txt'.format(Rs, PE1, len(U)),"w") as f:
    f.write("\n".join(" ".join(map(str, x)) for x in (mat_size, inds_stacked)))

'''

'''
#---------------------------------------------------------------------------
#quasiSAMER
max_loops = 20
JE_params = [m_est_rmse, rmse_tol, max_loops, ME_maxiter, LS_maxiter, \
                CG_maxiter, CG_tol, CG_atol, batch, mask]
init_est = [m_GT, Mtraj_est]
fixed_vars = [m_init, s_corrupted, C, U, mask, res, spath, m_GT, R_pad]
stores = [m_cnn_store, Mtraj_store, m_loss_store]

#Motion Estimation step
m_est, m_loss_store, Mtraj_store, m_cnn_store = rec.SAMER(init_est, fixed_vars, stores, JE_params)

#Image Recon step
Mtraj_est = Mtraj_store[-1][0]
A_new = partial(eop._EH_E, C=C, U=U, Mtraj=Mtraj_est, res=res, lamda=0, batch=batch)
b_new = eop.Encode_Adj(s_corrupted, C, U, Mtraj_est, res, batch=batch)
m_out = rec.ImageRecon(A_new, b_new, m_init, mask = mask, maxiter=CG_maxiter, \
                       tol=CG_tol, atol=CG_atol)

#---------------------------------------------------------------------------
#Brute force estimation
RotLR_array = xp.arange(-5, 5.5, 0.5)

f_array = xp.zeros((len(U), RotLR_array.shape[0]))
m_temp = m_corrupted

for shot_ind, U_n in enumerate(U):
    print("Evaluating for Shot {}".format(shot_ind+1))
    for val_ind, val in enumerate(RotLR_array):
        print("Pitch: {} deg".format(val))
        Mtraj_est_n = xp.array([0.,0.,0.,val,0.,0.])
        f_val = rec._f_n(Mtraj_est_n, m_est=m_temp, C=C, res=res, \
                         U_n=U_n, R_pad=R_pad, s_corrupted=s_corrupted)
        f_array = f_array.at[shot_ind, val_ind].set(f_val)

plt.figure()
plt.imshow(f_array, cmap = "gray", extent=[-5,5.5,1,len(U)+1])
plt.xlabel("Rotation about LR")
plt.ylabel("Shot Index")
plt.title("Loss Along Rot-LR, with corrupted image")
plt.show()

fig, axes = plt.subplots(len(U),1)
for shot_ind, ax in enumerate(axes.ravel()):
    ax.plot(RotLR_array, f_array[shot_ind,:])
    ax.set_title("Shot {}".format(shot_ind+1))
#
fig.suptitle("Loss Along Rot-LR for Each Shot")
plt.show()

#---------------------------------------------------------------------------
#Running image recon
Mtraj_est = Mtraj_store[-1][0]
A_new = partial(eop._EH_E, C=C, U=U, Mtraj=Mtraj_est, res=res, lamda=0, batch=batch)
b_new = eop.Encode_Adj(s_corrupted, C, U, Mtraj_est, res, batch=batch)
m_out = rec.ImageRecon(A_new, b_new, m_init, mask = mask, maxiter=CG_maxiter, \
                       tol=CG_tol, atol=CG_atol)

'''


'''
vmax = abs(m_ref[...].flatten()).max()

coil_ind = 15
plot_view(m_n[coil_ind,...], transpose, None, 'sagittal', vmax = vmax)
plot_view(m_ref[coil_ind,...], transpose, None, 'sagittal', vmax = vmax)

plot_view(m_n[coil_ind,...], transpose, None, 'axial', vmax = vmax)
plot_view(m_ref[coil_ind,...], transpose, None, 'axial', vmax = vmax)

plot_view(m_n[coil_ind,...], transpose, None, 'coronal', vmax = vmax)
plot_view(m_ref[coil_ind,...], transpose, None, 'coronal', vmax = vmax)

'''