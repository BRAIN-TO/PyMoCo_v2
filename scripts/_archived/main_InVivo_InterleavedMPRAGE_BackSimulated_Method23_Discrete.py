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
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform" #allows deallocation after object deletion

import encode.encode_op as eop
import recon.recon_op as rec
import cnn.run_unet as cnn
import utils.metrics as mtc
import motion.motion_sim as msi

#-------------------------------------------------------------------------------
def main(sub, dpath, spath_root, mpath, gt_name, flag, cerebrum_slice, spath_alt):
    #---------------------------------------------------------------------------
    #--------------------------Loading Corrupted Data---------------------------
    #---------------------------------------------------------------------------
    #Load data
    # s_corrupted = xp.load(dpath + r'/kdat_trunc.npy') #NC, SI, AP, LR
    if sub == 4:
        C_init = xp.load(dpath + r'/sens.npy')
    else:
        C_init = xp.load(mpath + r'/{}/sens.npy'.format(gt_name))
    C = xp.transpose(C_init, (3,0,1,2))
    mask = rec.getMask(C); xp.save(dpath + r'/m_GT_brain_mask.npy', mask)
    del C_init
    #---------------------------------------
    U = np.load(dpath + r'/samp_order.npy', allow_pickle=1) #LR, AP, SI
    #---------------------------------------------------------------------------
    res = xp.array([1,1,1])
    #---------------------------------------
    m_GT = xp.load(mpath + r'/{}/img_CG.npy'.format(gt_name)) ###I THINK I'VE OVERWRITTEN IMG_CG, SO MAXVAL NOW = 1
    maxval = abs(m_GT.flatten()).max()
    m_GT /= maxval
    # s_corrupted /= maxval
    #---------------------------------------
    #Loading the skull-stripping mask, generated from FreeSurfer SynthStrip tool
    cerebrum_mask = xp.ones(m_GT.shape)
    cerebrum_mask = cerebrum_mask.at[cerebrum_slice:,...].set(0)
    #---------------------------------------
    #Motion trajectory
    R_pad = (10, 10, 10)
    batch = 1
    #---------------------------------------------------------------------------
    #Loading Mtraj estimated from wo_cnn
    Mtraj_GT_store = list(np.load(spath_alt + r'/Mtraj_store.npy', allow_pickle=1))
    Mtraj_GT = Mtraj_GT_store[-1][0]
    #
    s_corrupted = eop.Encode(m_GT, C, U, Mtraj_GT, res, batch=batch)
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
    DC_store = []
    #---------------------------------------
    #Reconstruct image using CG SENSE algorithm
    m_init = eop.Encode_Adj(s_corrupted, C, U, Mtraj_init, res, batch=batch) #E.H*s
    m_corrupted = m_init
    ############################################################################
    ########    RERUNNING FOR SUB 1 - 4 IF PROPER RESCALING   
    subset = [1,2,3,4]
    if sub in subset:
        maxval = xp.max(abs(m_corrupted.flatten()))
        s_corrupted /= maxval
        m_corrupted /= maxval
    ############################################################################
    #---------------------------------------
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
    rmse_tol = 0.0
    ssim_tol = 1.0 #heuristics, found that this corresponds to acceptable correction for R = 2
    trans_axes = (2,1,0,180)
    cnn_flag = flag[0] #turn on / off CNN
    JE_flag = flag[1] #turn JE algorithm on / off
    thresh = {'severe': 500, 'moderate': 0.1}
    if JE_flag and cnn_flag: #UNet + JE
        spath = spath_root + r'/w_cnn_BackCorrupted_July16'
        max_loops = 250
    elif JE_flag and not cnn_flag: #only JE
        spath = spath_root + r'/wo_cnn_BackCorrupted_July16'
        max_loops = 250
    elif not JE_flag and cnn_flag: #only UNet
        spath = spath_root + r'/w_only_cnn_BackCorrupted_July16'
        max_loops = 1
    plib.Path(spath).mkdir(parents=True, exist_ok=True)
    xp.save(spath + r'/m_corrupted.npy', m_corrupted)
    #
    #---------------------------------------------------------------------------
    dscale = 1
    continuity = 0
    grad_tol = 0.0
    JE_params = [m_est_rmse, rmse_tol, m_est_ssim, ssim_tol, max_loops, ME_maxiter, LS_maxiter, \
                    CG_maxiter, CG_tol, CG_atol, CG_mask, batch, mask, continuity, grad_tol]
    CNN_params = [cnn_flag, JE_flag, trans_axes, pads, wpath_severe, wpath_moderate, wpath_mild, thresh]
    init_est = [m_est, Mtraj_est]
    fixed_vars = [m_init, s_corrupted, C, U, dscale, res, spath, m_GT, R_pad, cerebrum_mask]
    #
    DC_norm = rec._f(Mtraj_init, m_est=m_corrupted, C=C, res=res, U=U, R_pad=R_pad, s_corrupted=s_corrupted)
    xp.save(spath + r'/DC_norm.npy', DC_norm)
    #
    stores = [m_cnn_store, Mtraj_store, m_loss_store, DC_store]
    m_est, m_loss_store, Mtraj_store, m_cnn_store = rec.JointEst(init_est, fixed_vars, \
                                                                    stores, cnn, \
                                                                    CNN_params, JE_params)
    return spath, m_corrupted, m_est, m_loss_store, Mtraj_store


#%% Run main()
if __name__ == "__main__":
    #
    mpath_sub1 = r'/home/nghiemb/Data/TWH/MPRAGE_PE1Reordered/Scan20230717'
    dpath_sub1 = mpath_sub1 + r'/scan9-severe-16shots'
    spath_sub1 = [dpath_sub1 + r'/wo_cnn', dpath_sub1 + r'/w_cnn_combo_PE1_AP']
    gt_name_sub1 = 'scan8-reference-16shots'
    cerebrum_slice_sub1 = 180
    paths_sub1 = [mpath_sub1, dpath_sub1, gt_name_sub1, cerebrum_slice_sub1, spath_sub1]
    #
    mpath_sub2 = r'/home/nghiemb/Data/TWH/MPRAGE_PE1Reordered/Scan20230831/Sub1'
    dpath_sub2 = mpath_sub2 + r'/scan3-InstructedMotion-16shots'
    spath_sub2 = [dpath_sub2 + r'/wo_cnn_CorrectMask', dpath_sub2 + r'/w_cnn_combo_PE1_AP_CorrectMask']
    gt_name_sub2 = 'scan4-Reference-16shots'
    cerebrum_slice_sub2 = 195
    paths_sub2 = [mpath_sub2, dpath_sub2, gt_name_sub2, cerebrum_slice_sub2, spath_sub2]
    #
    mpath_sub3 = r'/home/nghiemb/Data/TWH/MPRAGE_PE1Reordered/Scan20230831/Sub2'
    dpath_sub3 = mpath_sub3 + r'/scan2-InstructedMotion-16shots'
    spath_sub3 = [dpath_sub3 + r'/wo_cnn_CorrectMask', dpath_sub3 + r'/w_cnn_combo_PE1_AP_CorrectMask_Rerun-06-19-2024']
    gt_name_sub3 = 'scan1-Reference-16shots'
    cerebrum_slice_sub3 = 200
    paths_sub3 = [mpath_sub3, dpath_sub3, gt_name_sub3, cerebrum_slice_sub3, spath_sub3]
    #
    mpath_sub4 = r'/home/nghiemb/Data/TWH/MPRAGE_PE1Reordered/Scan20230925/Sub1'
    dpath_sub4 = mpath_sub4 + r'/scan2-InstructedMotion-16shots'
    spath_sub4 = [dpath_sub4 + r'/wo_cnn_CorrectMask_Rerun-06-19-2024', dpath_sub4 + r'/w_cnn_combo_PE1_AP_CorrectMask_Rerun-06-19-2024']
    gt_name_sub4 = 'scan1-Reference-16shots'
    cerebrum_slice_sub4 = 180
    paths_sub4 = [mpath_sub4, dpath_sub4, gt_name_sub4, cerebrum_slice_sub4, spath_sub4]
    #
    mpath_sub5 = r'/home/nghiemb/Data/TWH/MPRAGE_PE1Reordered/Scan20231116/Sub5'
    dpath_sub5 = mpath_sub5 + r'/scan3-InstructedMotion-16shots/npy'
    spath_sub5 = [dpath_sub5 + r'/wo_cnn_CorrectMask', dpath_sub5 + r'/w_cnn_combo_PE1_AP_CorrectMask_Rerun-06-19-2024']
    gt_name_sub5 = r'scan1-ReferenceProduct/npy'
    cerebrum_slice_sub5 = 175
    paths_sub5 = [mpath_sub5, dpath_sub5, gt_name_sub5, cerebrum_slice_sub5, spath_sub5]
    #
    mpath_sub6 = r'/home/nghiemb/Data/TWH/MPRAGE_PE1Reordered/Scan20231116/Sub6'
    dpath_sub6 = mpath_sub6 + r'/scan4-InstructedMotion-16shots/npy'
    spath_sub6 = [dpath_sub6 + r'/wo_cnn_CorrectMask', dpath_sub6 + r'/w_cnn_combo_PE1_AP_CorrectMask_Rerun-06-19-2024']
    gt_name_sub6 = r'scan2-ReferenceProduct-16shots-v2/npy'
    cerebrum_slice_sub6 = 195
    paths_sub6 = [mpath_sub6, dpath_sub6, gt_name_sub6, cerebrum_slice_sub6, spath_sub6]
    #
    mpath_sub7 = r'/home/nghiemb/Data/TWH/MPRAGE_PE1Reordered/Scan20231116/Sub7'
    dpath_sub7 = mpath_sub7 + r'/scan3-InstructedMotion-16shots/npy'
    spath_sub7 = [dpath_sub7 + r'/wo_cnn_CorrectMask_MaxIter750/', dpath_sub7 + r'/w_cnn_combo_PE1_AP_CorrectMask_Rerun-06-24-2024']
    gt_name_sub7 = r'scan1-ReferenceProduct-16shots/npy'
    cerebrum_slice_sub7 = 215
    paths_sub7 = [mpath_sub7, dpath_sub7, gt_name_sub7, cerebrum_slice_sub7, spath_sub7]
    #
    mpath_sub8 = r'/home/nghiemb/Data/TWH/MPRAGE_PE1Reordered/Scan20231116/Sub8'
    dpath_sub8 = mpath_sub8 + r'/scan3-InstructedMotion-16shots/npy'
    spath_sub8 = [dpath_sub8 + r'/wo_cnn_CorrectMask_MaxIter500', dpath_sub8 + r'/w_cnn_combo_PE1_AP_CorrectMask_250Iters']
    gt_name_sub8 = r'scan1-ReferenceProduct-16shots/npy'
    cerebrum_slice_sub8 = 190
    paths_sub8 = [mpath_sub8, dpath_sub8, gt_name_sub8, cerebrum_slice_sub8, spath_sub8]
    #
    mpath_sub9 = r'/home/nghiemb/Data/TWH/MPRAGE_PE1Reordered/Scan20231218/Sub9/h5data'
    dpath_sub9 = mpath_sub9 + r'/scan3-InstructedMotion-16shots/npy'
    spath_sub9 = [dpath_sub9 + r'/wo_cnn_CorrectMask_MaxIter750', dpath_sub9 + r'/w_cnn_combo_PE1_AP_CorrectMask_250Iters']
    gt_name_sub9 = r'scan1-ReferenceProduct/npy'
    cerebrum_slice_sub9 = 185
    paths_sub9 = [mpath_sub9, dpath_sub9, gt_name_sub9, cerebrum_slice_sub9, spath_sub9]
    #
    mpath_sub10 = r'/home/nghiemb/Data/TWH/MPRAGE_PE1Reordered/Scan20231218/Sub10/h5data'
    dpath_sub10 = mpath_sub10 + r'/scan3-InstructedMotion-16shots/npy'
    spath_sub10 = [dpath_sub10 + r'/wo_cnn_CorrectMask', dpath_sub10 + r'/w_cnn_combo_PE1_AP_CorrectMask_250Iters']
    gt_name_sub10 = r'scan1-ReferenceProduct/npy'
    cerebrum_slice_sub10 = 190
    paths_sub10 = [mpath_sub10, dpath_sub10, gt_name_sub10, cerebrum_slice_sub10, spath_sub10]
    #
    paths_list = [paths_sub1, paths_sub2, paths_sub3, paths_sub4, paths_sub5, paths_sub6, paths_sub7, paths_sub8, paths_sub9, paths_sub10]
    sub_list = [1,2,3,4,5,6,7,8,9]
    #
    count = 0
    for i, paths in enumerate(paths_list):
        if i in sub_list:
            sub = i+1
            print('Processing Test Case {}'.format(i+1))
            mpath, dpath, gt_name, cerebrum_slice, spath_alt_list = paths
            spath_root = dpath    
            spath_alt = spath_alt_list[1] #using Mtraj estimated from Method 3
            flags = [[1,1], [0,1]] #Running only CNN
            for flag in flags:
                spath, m_corrupted, m_final, m_loss_store, Mtraj_store = main(sub, dpath, spath_root, mpath, gt_name, flag, cerebrum_slice, spath_alt)
                xp.save(spath + r"/m_corrupted.npy", m_corrupted)
                xp.save(spath + r"/m_final.npy", m_final)
                xp.save(spath + r"/m_loss_store.npy", m_loss_store)
                xp.save(spath + r"/Mtraj_store.npy", Mtraj_store)
        else:
            pass
