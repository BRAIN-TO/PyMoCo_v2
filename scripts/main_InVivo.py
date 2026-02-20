"""
MAIN SCRIPT
Running Joint Motion and Image Estimation
on Real Motion-Corrupted Data, with Interleaved PE1 Reordering
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
def main(sub, dpath, flag, cerebrum_slice):
    #---------------------------------------------------------------------------
    #--------------------------Loading Corrupted Data---------------------------
    #---------------------------------------------------------------------------
    #Load data
    s_corrupted = xp.load(dpath + r'/kdat_trunc.npy') #NC, SI, AP, LR
    C_init = xp.load(dpath + r'/sens.npy')
    C = xp.transpose(C_init, (3,0,1,2))
    del C_init
    #---------------------------------------
    mask = rec.getMask(C); xp.save(dpath + r'/m_GT_brain_mask.npy', mask)
    U = np.load(dpath + r'/samp_order.npy', allow_pickle=1)
    res = xp.array([1,1,1]) #spatial resolution
    #---------------------------------------
    m_GT = xp.load(dpath + r'/img_CG.npy')
    maxval = abs(m_GT.flatten()).max()
    m_GT /= maxval
    s_corrupted /= maxval
    #---------------------------------------
    #Manually masking out axial slices below the cerebellum
    cerebrum_mask = xp.ones(m_GT.shape)
    cerebrum_mask = cerebrum_mask.at[cerebrum_slice:,...].set(0)
    #---------------------------------------
    #Motion trajectory
    R_pad = (10, 10, 10) #spatial zero-padding to avoid wrapping during rotations
    batch = 1 #grouping shots together during estimation; batch set to 1 due to memory limits
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
    #Motion-corrupted reconstruction
    A = partial(eop._EH_E, C=C, U=U, Mtraj=Mtraj_est, res=res, lamda = CG_lamda, batch=batch)
    b = eop.Encode_Adj(s_corrupted, C, U, Mtraj_est, res, batch=batch)
    #
    m_corrupted = m_init
    subset = [1,2,3,4] #need to rescale for Subs 1 - 4 
    if sub in subset:
        maxval = xp.max(abs(m_corrupted.flatten()))
        s_corrupted /= maxval
        m_corrupted /= maxval
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
    root = os.getcwd()
    cnn_path = root + r'/cnn/3DUNet_SAP'
    wpath = cnn_path + r'/weights/P1CD' # P1CD = "UNet_A"; P1CDEF = "UNet_B"
    pad_x = int((xp.ceil(m_est.shape[2]/32) * 32 - m_est.shape[2])/2) #along LR
    pad_y = int((xp.ceil(m_est.shape[1]/32) * 32 - m_est.shape[1])/2) #along AP
    pads = [pad_x, pad_y]
    #---------------------------------------------------------------------------
    #Alternating image & motion estimation (coordinate descent)
    rmse_tol = 0.0 #default to impossible RMSE, to max out on iters
    ssim_tol = 1.0 #default to impossible SSIM, to max out on iters
    trans_axes = (2,1,0,180)
    cnn_flag = flag[0] #turn on / off CNN
    JE_flag = flag[1] #turn JE algorithm on / off
    thresh = {'severe': 500, 'moderate': 0.1}
    if JE_flag and cnn_flag: #UNet + JE
        spath = dpath + r'/UNetJE'
        max_loops = 250
    elif JE_flag and not cnn_flag: #only JE
        spath = dpath + r'/JE'
        max_loops = 250 #ie. additional 250 iterations, picking from previous run
    elif not JE_flag and cnn_flag: #only UNet
        spath = dpath + r'/UNet'
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
    CNN_params = [cnn_flag, JE_flag, trans_axes, pads, wpath, wpath, wpath, thresh]
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
    return spath, m_corrupted, m_est, m_loss_store, Mtraj_store

#%% Run main()
if __name__ == "__main__":
    NSUBS = 10
    root = os.getcwd()
    dpath_init = root + r'/data/InVivo'
    cerebrum_slices = [180, 195, 200, 180, 175, 195, 215, 190, 185, 190]
    #
    for sub in range(1,NSUBS+1):
        dpath = dpath_init + r'/Sub{}'.format(sub)
        cerebrum_slice = cerebrum_slices[sub-1]
        flags = [[1,0], [0,1], [1,1]] #flags for using UNet and JE, respectively; [1,1] is UNet+JE
        for flag in flags:
            spath, m_corrupted, m_final, m_loss_store, Mtraj_store = main(sub, dpath, flag, cerebrum_slice)
            xp.save(spath + r"/m_corrupted.npy", m_corrupted)
            xp.save(spath + r"/m_final.npy", m_final)
            xp.save(spath + r"/m_loss_store.npy", m_loss_store)
            xp.save(spath + r"/Mtraj_store.npy", Mtraj_store)

