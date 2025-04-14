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

import jax
import jax.numpy as xp
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="0" #turn off GPU pre-allocation
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import encode.encode_op as eop
import recon.recon_op as rec
import cnn.run_unet as cnn
import utils.metrics as mtc
import motion.motion_sim as msi
import utils.visualize as vis

#-------------------------------------------------------------------------------
def main(dpath, spath_root, mpath, gt_name, test_flag, case):
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
    #
    #---------------------------------------
    TR = 1.6 #T1w MPRAGE acquisition parameter
    Rs = 1 #SENSE acceleration factor
    TR_shot = 16
    print("Simulated motion temporal resolution: {} sec".format(TR * TR_shot))
    U = msi.make_samp(m_GT, Rs, TR_shot, order='interleaved', mode = 'list')
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
    specs_scale = [1, 1] #[r_scale, p_scale]
    #Generating discrete random motion trajectory
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
    #
    motion_lv = 'moderate'
    j = 1; k = 1 #legacy parameters, from training dataset script
    rand_keys = msi._gen_key(60+case, j, k)
    Mtraj_GT = msi._gen_traj(rand_keys, len(U), motion_specs.get(motion_lv), specs_scale)
    # xp.save(dpath + r'/Mtraj.npy', Mtraj_GT)
    #
    s_corrupted = eop.Encode(m_GT, C, U, Mtraj_GT, res, batch=batch)
    # xp.save(dpath + r'/s_corrupted.npy', s_corrupted)
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
    wpath = cnn_path + r'/weights/PE1_AP/Complex/combo/train_n240_interleaved_P1EF_2025-04-02/slices'
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
        spath = spath_root + r'/w_cnn_ReIm_AugmentedTraining_April-11-2025'
        max_loops = 100
    elif JE_flag and not cnn_flag: #only JE
        # spath = spath_root + r'/wo_cnn_100Iters_July-10-2024'
        # max_loops = 100
        spath = spath_root + r'/wo_cnn_ReIm_AugmentedTraining_April-11-2025'
        max_loops = 200
    elif not JE_flag and cnn_flag: #only UNet
        spath = spath_root + r'/w_only_cnn_Mag_AugmentedTraining_April-11-2025'
        max_loops = 1
    plib.Path(spath).mkdir(parents=True, exist_ok=True)
    xp.save(spath + r'/m_corrupted.npy', m_corrupted)
    #
    #---------------------------------------------------------------------------
    dscale = 1
    continuity = 0
    grad_tol = 1e-4 #
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
    mpath = r'/home/nghiemb/PyMoCo/data/cc/test/PE1_AP/Complex/R1/Paradigm_1C'
    cases = [1,4,5,6,7]
    for case in cases:
        test_flags = [[1,0]] #[CNN, JE]
        for test_flag in test_flags:
            test_case = 'Test{}'.format(case)
            gt_name = test_case
            # #
            dpath = mpath + r'/{}'.format(test_case)
            print('Processing Test Case {}'.format(test_case))
            spath_root = dpath
            spath, m_corrupted, m_final, m_loss_store, Mtraj_store = main(dpath, spath_root, mpath, gt_name, test_flag, case)
            xp.save(spath + r"/m_corrupted.npy", m_corrupted)
            xp.save(spath + r"/m_final.npy", m_final)
            xp.save(spath + r"/m_loss_store.npy", m_loss_store)
            xp.save(spath + r"/Mtraj_store.npy", Mtraj_store)


