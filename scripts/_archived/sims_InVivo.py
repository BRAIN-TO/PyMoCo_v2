"""
MAIN SCRIPT
Running Joint Motion and Image Estimation
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

#-------------------------------------------------------------------------------
def main(dpath, spath_root, mpath):    
    #---------------------------------------------------------------------------
    #-----------------------Image Acquisition Simulation------------------------
    #---------------------------------------------------------------------------
    #Load data
    m_GT = xp.load(dpath + r'/rss.npy')
    maxval = abs(m_GT.flatten()).max()
    m_GT /= maxval
    C = xp.load(dpath + r'/sens.npy')
    mask = rec.getMask(C)
    res = xp.array([1,1,1])
    U = xp.load(dpath + r'/U.npy')
    #
    #Motion trajectory
    Mtraj_GT = xp.load(dpath + r'/Mtraj_GT.npy')
    R_pad = (10, 10, 10)
    batch = 1
    #
    #Simulate motion corruption
    s_corrupted = eop.Encode(m_GT, C, U, Mtraj_GT, res, batch=batch) #E.H*s
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
    if CG_mask:
        m_out = rec.ImageRecon(A, b, m_init, mask = mask, maxiter=CG_maxiter, \
                                tol=CG_tol, atol=CG_atol)
    else:
        m_out = rec.ImageRecon(A, b, m_init, maxiter=CG_maxiter, \
                                tol=CG_tol, atol=CG_atol)     
    m_corrupted = mask*m_out[-1]
    m_est_rmse = mtc.evalPE(m_corrupted, m_GT, mask)
    m_est_ssim = mtc.evalSSIM(m_corrupted, m_GT, mask=mask)
    m_loss_store.append([m_est_rmse, m_est_ssim])
    print("RMSE of Corrupted Image: {:.2f} %".format(m_est_rmse))
    print("SSIM of Corrupted Image: {}".format(m_est_ssim))
    m_est = m_corrupted
    #
    #Motion-compensated reconstruction
    A_mc = partial(eop._EH_E, C=C, U=U, Mtraj=Mtraj_GT, res=res, \
                lamda = CG_lamda, batch=batch)
    b_mc = eop.Encode_Adj(s_corrupted, C, U, Mtraj_GT, res, batch=batch)
    if CG_mask:
        m_out_GT = rec.ImageRecon(A_mc, b_mc, m_init, mask = mask, maxiter=CG_maxiter, \
                                tol=CG_tol, atol=CG_atol)
    else:
        m_out_GT = rec.ImageRecon(A_mc, b_mc, m_init, maxiter=CG_maxiter, \
                                tol=CG_tol, atol=CG_atol)  
    m_moco_GT = mask*m_out_GT[-1]
    #---------------------------------------------------------------------------
    #Loading trained CNN model
    # NB. UNet takes in data as [LR, AP, SI]
    # For my Ref Data (SI, AP, LR), need to transpose --> (2,1,0)
    cnn_path = r'/home/nghiemb/PyMoCo/cnn/3DUNet_SAP'
    wpath_severe = cnn_path + r'/weights/{}/train_n134'.format('severe')
    wpath_moderate = cnn_path + r'/weights/{}/train_n134'.format('moderate')
    wpath_mild = cnn_path + r'/weights/{}/train_n134'.format('mild')
    pad_x = int((xp.ceil(m_est.shape[2]/32) * 32 - m_est.shape[2])/2) #along LR
    pad_y = int((xp.ceil(m_est.shape[1]/32) * 32 - m_est.shape[1])/2) #along AP
    pads = [pad_x, pad_y]
    #---------------------------------------------------------------------------
    #Alternating image & motion estimation (coordinate descent)
    rmse_tol = 5.0
    trans_axes = (2,1,0)
    cnn_flag = 1 #turn on / off CNN
    JE_flag = 1 #turn JE algorithm on / off
    thresh = {'severe': 300, 'moderate': 0.01}
    if JE_flag and cnn_flag: #UNet + JE
        spath = spath_root + r'/w_cnn_alt_moderate'
        max_loops = 20
    elif JE_flag and not cnn_flag: #only JE
        spath = spath_root + r'/wo_cnn_alt'
        max_loops = 2000
    elif not JE_flag and cnn_flag: #only UNet
        spath = spath_root + r'/w_only_cnn_alt'
        max_loops = 1
    plib.Path(spath).mkdir(parents=True, exist_ok=True)
    xp.save(spath + r'/m_corrupted.npy', m_corrupted)
    xp.save(spath + r'/m_moco_GT.npy', m_moco_GT)
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
    mpath = r'/home/nghiemb/Data/TWH/MPRAGE_ReferencePoses/InVivo/sub-01/dat'
    test_case = 1
    dpath = mpath + r'/simulations/Test{}'.format(test_case)
    spath_root = dpath
    spath, m_corrupted, m_final, m_loss_store, Mtraj_store = main(dpath, spath_root, mpath)
    xp.save(spath + r"/m_corrupted.npy", m_corrupted)
    xp.save(spath + r"/m_final.npy", m_final)
    xp.save(spath + r"/m_loss_store.npy", m_loss_store)
    xp.save(spath + r"/Mtraj_store.npy", Mtraj_store)
