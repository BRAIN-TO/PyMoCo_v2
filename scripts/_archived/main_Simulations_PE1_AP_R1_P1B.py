"""
MAIN SCRIPT
Running Joint Motion and Image Estimation
Rerunning Simulation Study, now with PE1 along AP and R = 1 (Nov 2023)
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
def main(dpath, spath_root, mpath, gt_name, test_flag):
    #---------------------------------------------------------------------------
    #-----------------------Image Acquisition Simulation------------------------
    #---------------------------------------------------------------------------
    #Loading GT data
    m_GT_init = xp.load(dpath + r'/current_test_GT.npy') #SI, LR, AP
    m_GT = xp.pad(m_GT_init[:,:,:,0,0] + 1j*m_GT_init[:,:,:,1,0], ((1,1), (0,0), (0,0)))
    del m_GT_init
    C = xp.load(dpath + r'/sens.npy')
    #Transpose to reorient as LR, AP, SI
    m_GT = xp.transpose(m_GT, (1,2,0))
    m_GT = m_GT[6:-6, 3:-3, :]
    xp.save(dpath + r'/img_CG.npy', m_GT)
    #---------------------------------------
    Rs = 1
    TR_shot = 16
    U_array = xp.transpose(msi.make_samp(xp.transpose(m_GT, (1,0,2)), Rs, TR_shot, order='interleaved'), (0,2,1,3))
    U = eop._U_Array2List(U_array, m_GT.shape)
    #---------------------------------------------------------------------------
    mask = rec.getMask(C); xp.save(dpath + r'/m_GT_brain_mask.npy', mask)
    res = xp.array([1,1,1])
    #---------------------------------------
    #Set mask to identity for the simulation study
    cerebrum_mask = xp.ones(m_GT.shape)
    #---------------------------------------
    #Motion trajectory
    R_pad = (10, 10, 10)
    batch = 1
    #Generating discrete sinusoidal motion trajectory
    sinusoid_ncycles = 1
    sinusoid_nshots = len(U)
    sinsusoid_amplitude = 1
    Mtraj_shots = (2*xp.pi)*(sinusoid_ncycles / sinusoid_nshots)*xp.arange(0,len(U))
    sinusoid_R_LR = sinsusoid_amplitude*xp.sin(Mtraj_shots)
    sinusoid_T_SI = sinusoid_R_LR / 2 #half the amplitude and reversed sign
    Mtraj_GT = xp.zeros((len(U),6))
    Mtraj_GT = Mtraj_GT.at[:,3].set(sinusoid_R_LR) #rotations about LR axis
    Mtraj_GT = Mtraj_GT.at[:,2].set(-sinusoid_T_SI) #translations along SI axis
    xp.save(dpath + r'/Mtraj.npy', Mtraj_GT)
    #
    s_corrupted = eop.Encode(m_GT, C, U, Mtraj_GT, res, batch=batch)
    xp.save(dpath + r'/s_corrupted.npy', s_corrupted)
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
    wpath_severe = cnn_path + r'/weights/PE1_AP/Complex/{}/train_n360'.format('combo')
    wpath_moderate = cnn_path + r'/weights/PE1_AP/Complex/{}/train_n360'.format('combo')
    wpath_mild = cnn_path + r'/weights/PE1_AP/Complex/{}/train_n360'.format('combo')
    pads = [11,3]
    #---------------------------------------------------------------------------
    #Alternating image & motion estimation (coordinate descent)
    # rmse_tol = 5.0
    rmse_tol = 3.0 #heuristics, found that this corresponds to acceptable correction
    trans_axes = (0,1,2,0) 
    cnn_flag = test_flag[0] #turn on / off CNN
    JE_flag = test_flag[1] #turn JE algorithm on / off
    thresh = {'severe': 500, 'moderate': 0.1}
    if JE_flag and cnn_flag: #UNet + JE
        spath = spath_root + r'/w_cnn_combo_PE1_AP_CorrectMask'
        # max_loops = 250
        max_loops = 250
    elif JE_flag and not cnn_flag: #only JE
        spath = spath_root + r'/wo_cnn_CorrectMask'
        max_loops = 250
    elif not JE_flag and cnn_flag: #only UNet
        spath = spath_root + r'/w_only_cnn_combo_PE1_AP'
        max_loops = 1
    plib.Path(spath).mkdir(parents=True, exist_ok=True)
    xp.save(spath + r'/m_corrupted.npy', m_corrupted)
    #
    #---------------------------------------------------------------------------
    dscale = 1
    continuity = 0
    JE_params = [m_est_rmse, rmse_tol, max_loops, ME_maxiter, LS_maxiter, \
                    CG_maxiter, CG_tol, CG_atol, CG_mask, batch, mask, continuity]
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
    mpath = r'/home/nghiemb/PyMoCo/data/cc/test/PE1_AP/Complex/R1/Paradigm_1B'
    cases = [1,4,5,6,7]
    test_flags = [[1,1], [0,1], [1,0]]
    for case in cases:
        for test_flag in test_flags:
            test_case = 'Test{}'.format(case)
            gt_name = test_case
            # #
            dpath = mpath + r'/{}'.format(test_case)
            print('Processing Test Case {}'.format(test_case))
            spath_root = dpath
            spath, m_corrupted, m_final, m_loss_store, Mtraj_store = main(dpath, spath_root, mpath, gt_name, test_flag)
            xp.save(spath + r"/m_corrupted.npy", m_corrupted)
            xp.save(spath + r"/m_final.npy", m_final)
            xp.save(spath + r"/m_loss_store.npy", m_loss_store)
            xp.save(spath + r"/Mtraj_store.npy", Mtraj_store)

