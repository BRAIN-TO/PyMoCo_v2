"""
MAIN SCRIPT
Running Joint Motion and Image Estimation
on Real Motion-Corrupted Data, with Interleaved PE1 Reordering (acquired July 2023)
"""
import os
import pathlib as plib
from time import time
from functools import partial
import numpy as np

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
    #---------------------------------------
    U = np.load(dpath + r'/samp_order.npy', allow_pickle=1) #LR, AP, SI
    #---------------------------------------------------------------------------
    mask = rec.getMask(C); xp.save(dpath + r'/m_GT_brain_mask.npy', mask)
    res = xp.array([1,1,1])
    #---------------------------------------
    m_GT = xp.load(mpath + r'/{}/img_CG.npy'.format(gt_name))
    maxval = abs(xp.load(mpath + r'/{}/img_rss_trunc.npy'.format(gt_name)).flatten()).max()
    # m_GT /= maxval
    s_corrupted /= maxval
    #---------------------------------------
    #Loading the skull-stripping mask, generated from FreeSurfer SynthStrip tool
    cerebrum_mask = xp.ones(m_GT.shape)
    # cerebrum_slice= 180 #Brian
    cerebrum_slice= 195 #Icaro
    # cerebrum_slice= 200 #Tim
    cerebrum_mask = cerebrum_mask.at[cerebrum_slice:,...].set(0)
    #---------------------------------------
    #Motion trajectory
    R_pad = (10, 10, 10)
    batch = 1
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
    wpath_severe = cnn_path + r'/weights/PE1_AP/Complex/{}/train_n360'.format('combo')
    wpath_moderate = cnn_path + r'/weights/PE1_AP/Complex/{}/train_n360'.format('combo')
    wpath_mild = cnn_path + r'/weights/PE1_AP/Complex/{}/train_n360'.format('combo')
    pad_x = int((xp.ceil(m_est.shape[2]/32) * 32 - m_est.shape[2])/2) #along LR
    pad_y = int((xp.ceil(m_est.shape[1]/32) * 32 - m_est.shape[1])/2) #along AP
    pads = [pad_x, pad_y]
    #---------------------------------------------------------------------------
    #Alternating image & motion estimation (coordinate descent)
    # rmse_tol = 5.0
    rmse_tol = 0.0
    trans_axes = (2,1,0,180)
    cnn_flag = 0 #turn on / off CNN
    JE_flag = 1 #turn JE algorithm on / off
    thresh = {'severe': 500, 'moderate': 0.1}
    if JE_flag and cnn_flag: #UNet + JE
        spath = spath_root + r'/w_cnn_combo_PE1_AP_CorrectMask'
        # max_loops = 50
        max_loops = 20
    elif JE_flag and not cnn_flag: #only JE
        spath = spath_root + r'/wo_cnn_CorrectMask'
        max_loops = 250
    elif not JE_flag and cnn_flag: #only UNet
        spath = spath_root + r'/w_only_cnn_combo_PE1_AP'
        max_loops = 1
    plib.Path(spath).mkdir(parents=True, exist_ok=True)
    xp.save(spath + r'/m_corrupted.npy', m_corrupted)
    #
    #**TRYING OUT TURNING OFF CNN AFTER PROPOSED METHOD HAS CONVERGED
    # cnn_flag = 0 #turn off CNN for additional 100 iterations
    #
    #---------------------------------------------------------------------------
    # #Picking up algorithm
    # m_est = np.load(spath + r'/m_final.npy')
    # m_loss_store = list(np.load(spath + r'/m_loss_store.npy', allow_pickle=1))
    # Mtraj_store = list(np.load(spath + r'/Mtraj_store.npy', allow_pickle=1))
    # Mtraj_est = Mtraj_store[-1][0]
    # m_cnn_store = list(np.load(spath + r'/m_cnn_store.npy', allow_pickle=1))
    # #Rescale
    # dscale = 2
    # U = eop._U_subdivide(U, dscale)
    # xp.save(spath + r'/U.npy', U)
    # Mtraj_est = xp.repeat(Mtraj_est, 2, axis = 0)
    # Mtraj_store = []
    # m_cnn_store = []
    dscale = 1
    #---------------------------------------------------------------------------
    JE_params = [m_est_rmse, rmse_tol, max_loops, ME_maxiter, LS_maxiter, \
                    CG_maxiter, CG_tol, CG_atol, CG_mask, batch, mask]
    CNN_params = [cnn_flag, JE_flag, trans_axes, pads, wpath_severe, wpath_moderate, wpath_mild, thresh]
    init_est = [m_est, Mtraj_est]
    fixed_vars = [m_init, s_corrupted, C, U, dscale, res, spath, m_GT, R_pad, cerebrum_mask]
    stores = [m_cnn_store, Mtraj_store, m_loss_store]
    m_est, m_loss_store, Mtraj_store, m_cnn_store = rec.JointEst(init_est, fixed_vars, \
                                                                    stores, cnn, \
                                                                    CNN_params, JE_params)
    #
    return spath, m_corrupted, m_est, m_loss_store, Mtraj_store

#%% Run main()
if __name__ == "__main__":
    mpath = r'/home/nghiemb/Data/TWH/MPRAGE_PE1Reordered/Scan20230831/Sub1'
    # test_case = 'scan1-FreeMotion-16shots'
    # gt_name = 'scan2-Reference-16shots'
    # test_case = 'scan2-Reference-16shots'
    # gt_name = 'scan2-Reference-16shots'
    # test_case = 'scan3-InstructedMotion-16shots'
    # gt_name = 'scan4-Reference-16shots'
    test_case = 'scan4-Reference-16shots'
    gt_name = 'scan4-Reference-16shots'
    #
    # mpath = r'/home/nghiemb/Data/TWH/MPRAGE_PE1Reordered/Scan20230831/Sub2'
    # # test_case = 'scan2-InstructedMotion-16shots'
    # # gt_name = 'scan1-Reference-16shots'
    # # test_case = 'scan1-Reference-16shots'
    # # gt_name = 'scan1-Reference-16shots'
    # # test_case = 'scan4-FreeMotion-16shots'
    # # gt_name = 'scan3-Reference-16shots'
    # test_case = 'scan3-Reference-16shots'
    # gt_name = 'scan3-Reference-16shots'
    # #
    dpath = mpath + r'/{}'.format(test_case)
    print('Processing Test Case {}'.format(test_case))
    spath_root = dpath
    spath, m_corrupted, m_final, m_loss_store, Mtraj_store = main(dpath, spath_root, mpath, gt_name)
    xp.save(spath + r"/m_corrupted.npy", m_corrupted)
    xp.save(spath + r"/m_final.npy", m_final)
    xp.save(spath + r"/m_loss_store.npy", m_loss_store)
    xp.save(spath + r"/Mtraj_store.npy", Mtraj_store)



'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from recon.recon_op import ImageRecon
import pathlib as plib 

def plot_save(fpath, volume_init, transpose, test_case, subname, crop=0, view = 'axial', vmax = 1.0):
    #Choosing slice orientation
    volume = np.transpose(volume_init, axes = transpose)
    if view == 'sagittal':
        if subname == "Sub1":
            slice = abs(volume[volume.shape[0]//2 + 12,:,:]) #Brian
            slice = rotate(slice, -90)
            if crop:
                slice = slice[40:110, 50:120] #Brian
        elif subname == "Sub2":
            slice = abs(volume[volume.shape[0]//2 + 10,:,:]) #Icaro
            slice = rotate(slice, -90)
            if crop:
                slice = slice[40:110, 150:220] #Icaro
        elif subname == "Sub3":
            slice = abs(volume[volume.shape[0]//2 + 10,:,:]) #Tim
            slice = rotate(slice, -90)
            if crop:
                slice = slice[50:120, 150:220] #Tim
        # slice = rotate(slice, -90)
    elif view == 'coronal':
        if subname == "Sub1":
            slice = abs(volume[:,volume.shape[1]//2-30,:]) #Brian
            slice = rotate(slice, -90)
            if crop:
                slice = slice[90:160, 40:110] #Brian
        elif subname == "Sub2":
            slice = abs(volume[:,volume.shape[1]//2,:]) #Icaro
            slice = rotate(slice, -90)
            if crop:
                slice = slice[90:160, 90:160] #Icaro
        elif subname == "Sub3":
            slice = abs(volume[:,volume.shape[1]//2-30,:]) #Tim
            slice = rotate(slice, -90)
            if crop:
                slice = slice[100:170, 100:170] #Tim
        # slice = rotate(slice, -90)
    elif view == 'axial':
        if subname == "Sub1":
            slice = abs(volume[:,:,volume.shape[2]//2-28]) #Brian
            slice = rotate(slice, 90)
            if crop:
                slice = slice[30:100, 70:140] #Brian
        elif subname == "Sub2":
            slice = abs(volume[:,:,volume.shape[2]//2 - 15]) #Icaro
            slice = rotate(slice, 90)
        elif subname == "Sub3":
            slice = abs(volume[:,:,volume.shape[2]//2 - 20]) #Tim
            slice = rotate(slice, 90)
            if crop:
                slice = slice[170:240, 70:140] #Tim
        # slice = rotate(slice, 90)
    #
    plt.figure()
    plt.imsave(fpath, slice, cmap = "gray", vmax = vmax)


# transpose = (2,1,0) #for in-vivo
# vmax = 0.4 #for in-vivo

# subname = "Sub1"
# # img_path = spath + r'/m_intmd_crop_store'
# img_path = spath + r'/m_intmd_store'
# plib.Path(img_path).mkdir(parents=True, exist_ok=True)

# # for i in range(20):
# for i in [0, 4, 9, 14, 19, 25]:
#     print("Iteration {}".format(i+1))
#     Mtraj_est = Mtraj_store[i][0]
#     A_new = partial(eop._EH_E, C=C, U=U, Mtraj=Mtraj_est, \
#                     res=res, lamda=0, batch=batch)
#     b_new = eop.Encode_Adj(s_corrupted, C, U, Mtraj_est, res, batch=batch)
#     #
#     m_out = ImageRecon(A_new, b_new, m_init, mask = mask, \
#                         maxiter=CG_maxiter, tol=CG_tol, atol=CG_atol)
#     m_intmd = abs(m_out[-1]).astype(xp.float32)
#     spath_temp_sag = img_path + r"/m_iter{}_sag.png".format(i+1)
#     # spath_temp_axl = img_path + r"/m_iter{}_axl.png".format(i+1)
#     # spath_temp_cor = img_path + r"/m_iter{}_cor.png".format(i+1)
#     plot_save(spath_temp_sag, m_intmd, transpose, None, subname, 'sagittal', vmax = vmax)
#     # plot_save(spath_temp_axl, m_intmd, transpose, None, subname, 'axial', vmax = vmax)
#     # plot_save(spath_temp_cor, m_intmd, transpose, None, subname, 'coronal', vmax = vmax)

    
transpose = (2,1,0) #for in-vivo
vmax = 0.4 #for in-vivo

subname = "Sub2"
crop_flag = 1

if crop_flag:
    img_path = spath + r'/m_intmd_crop_store'
else:
    img_path = spath + r'/m_intmd_store'

plib.Path(img_path).mkdir(parents=True, exist_ok=True)

spath_fin_sag = img_path + r"/m_final_sag.png"
spath_fin_axl = img_path + r"/m_final_axl.png"
spath_fin_cor = img_path + r"/m_final_cor.png"
plot_save(spath_fin_sag, m_final, transpose, None, subname, crop = crop_flag, view = 'sagittal', vmax = vmax)
plot_save(spath_fin_axl, m_final, transpose, None, subname, crop = crop_flag, view = 'axial', vmax = vmax)
plot_save(spath_fin_cor, m_final, transpose, None, subname, crop = crop_flag, view = 'coronal', vmax = vmax)

spath_corr_sag = img_path + r"/m_corrupted_sag.png"
spath_corr_axl = img_path + r"/m_corrupted_axl.png"
spath_corr_cor = img_path + r"/m_corrupted_cor.png"
plot_save(spath_corr_sag, m_corrupted, transpose, None, subname, crop = crop_flag, view = 'sagittal', vmax = vmax)
plot_save(spath_corr_axl, m_corrupted, transpose, None, subname, crop = crop_flag, view = 'axial', vmax = vmax)
plot_save(spath_corr_cor, m_corrupted, transpose, None, subname, crop = crop_flag, view = 'coronal', vmax = vmax)

spath_GT_sag = img_path + r"/m_GT_sag.png"
spath_GT_axl = img_path + r"/m_GT_axl.png"
spath_GT_cor = img_path + r"/m_GT_cor.png"
plot_save(spath_GT_sag, m_GT, transpose, None, subname, crop = crop_flag, view = 'sagittal', vmax = vmax)
plot_save(spath_GT_axl, m_GT, transpose, None, subname, crop = crop_flag, view = 'axial', vmax = vmax)
plot_save(spath_GT_cor, m_GT, transpose, None, subname, crop = crop_flag, view = 'coronal', vmax = vmax)


'''

