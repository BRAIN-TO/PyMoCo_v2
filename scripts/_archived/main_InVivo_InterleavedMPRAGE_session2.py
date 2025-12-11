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
    #---------------------------------------
    U = xp.load(dpath + r'/U.npy') #LR, AP, SI
    mask = rec.getMask(C); xp.save(dpath + r'/m_GT_brain_mask.npy', mask)
    res = xp.array([1,1,1])
    #---------------------------------------
    m_GT = xp.load(mpath + r'/{}/img_rss_trunc.npy'.format(gt_name))
    maxval = abs(m_GT.flatten()).max()
    m_GT /= maxval
    s_corrupted /= maxval
    #---------------------------------------
    #Loading the skull-stripping mask, generated from FreeSurfer SynthStrip tool
    # skull_mask = xp.load(dpath + r'/img_stripped_mask.npy')
    skull_mask = xp.ones(m_GT.shape)
    skull_mask = skull_mask.at[175:,:,:].set(0)
    # # #---------------------------------------
    # #Retrospective undersampling
    # U_R3 = xp.load(dpath + r'/U_R3_mask.npy') #mask for retrospective R=2 undersampling
    # s_corrupted *= U_R3
    # U *= U_R3
    #---------------------------------------
    #Motion trajectory
    R_pad = (10, 10, 10)
    batch = 1
    #---------------------------------------------------------------------------
    #------------------JOINT IMAGE RECON AND MOTION ESTIMATION------------------
    #---------------------------------------------------------------------------
    #Initializing update vars
    Mtraj_init = xp.zeros((U.shape[0], 6))
    Mtraj_est = Mtraj_init
    # CG_maxiter = 3 #limit CG_iter to 3 iters for fully-sampled data to prevent artifacts
    CG_maxiter = 5
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
    # #----------------------------------------
    # #If retrospectively undersampling
    # if CG_mask:
    #     m_out = rec.ImageRecon(A, b, m_init, mask = mask, maxiter=CG_maxiter, \
    #                             tol=CG_tol, atol=CG_atol)
    # else:
    #     m_out = rec.ImageRecon(A, b, m_init, maxiter=CG_maxiter, \
    #                             tol=CG_tol, atol=CG_atol)   
    # m_corrupted = mask*m_out[-1]
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
    rmse_tol = 5.0
    trans_axes = (2,1,0,180)
    cnn_flag = 1 #turn on / off CNN
    JE_flag = 1 #turn JE algorithm on / off
    thresh = {'severe': 500, 'moderate': 0.1}
    if JE_flag and cnn_flag: #UNet + JE
        spath = spath_root + r'/w_cnn_combo_PE1_AP_R3_CG5_U256shots'
        max_loops = 100
    elif JE_flag and not cnn_flag: #only JE
        spath = spath_root + r'/wo_cnn_PE1_AP_R3_CG5_U256shots'
        max_loops = 1000
    elif not JE_flag and cnn_flag: #only UNet
        spath = spath_root + r'/w_only_cnn_combo_PE1_AP_R3_CG5_U256shots'
        max_loops = 1
    plib.Path(spath).mkdir(parents=True, exist_ok=True)
    xp.save(spath + r'/m_corrupted.npy', m_corrupted)
    #
    #---------------------------------------------------------------------------
    # #Picking up algorithm
    # import numpy as np
    # m_est = np.load(spath + r'/m_final.npy')
    # m_loss_store = list(np.load(spath + r'/m_loss_store.npy', allow_pickle=1))
    # # m_cnn_store = np.load(dpath + r'/{}/m_cnn_store.npy'.format(cnn_path_names[-1]), allow_pickle=1)
    # Mtraj_store = list(np.load(spath + r'/Mtraj_store.npy', allow_pickle=1))
    # Mtraj_est = Mtraj_store[-1][0]
    # m_cnn_store = list(np.load(spath + r'/m_cnn_store.npy', allow_pickle=1))
    #---------------------------------------------------------------------------
    JE_params = [m_est_rmse, rmse_tol, max_loops, ME_maxiter, LS_maxiter, \
                    CG_maxiter, CG_tol, CG_atol, CG_mask, batch, mask]
    CNN_params = [cnn_flag, JE_flag, trans_axes, pads, wpath_severe, wpath_moderate, wpath_mild, thresh]
    init_est = [m_est, Mtraj_est]
    fixed_vars = [m_init, s_corrupted, C, U, res, spath, m_GT, R_pad, skull_mask]
    stores = [m_cnn_store, Mtraj_store, m_loss_store]
    m_est, m_loss_store, Mtraj_store, m_cnn_store = rec.JointEst(init_est, fixed_vars, \
                                                                    stores, cnn, \
                                                                    CNN_params, JE_params)
    #
    return spath, m_corrupted, m_est, m_loss_store, Mtraj_store

#%% Run main()
if __name__ == "__main__":
    mpath = r'/home/nghiemb/Data/TWH/MPRAGE_PE1Reordered/Scan20230717'
    test_case = 'scan3-mild-8shots'
    gt_name = 'scan2-reference-8shots'
    # mpath = r'/home/nghiemb/Data/TWH/MPRAGE_PE1Reordered/Scan20230710'
    # test_case = 'scan3-severe'
    # gt_name = 'scan2-reference-reorder'
    # # test_case = 'scan6-mild'
    # # gt_name = 'scan5-reference-reorder'
    dpath = mpath + r'/{}'.format(test_case)
    print('Processing Test Case {}'.format(test_case))
    spath_root = dpath
    spath, m_corrupted, m_final, m_loss_store, Mtraj_store = main(dpath, spath_root, mpath, gt_name)
    xp.save(spath + r"/m_corrupted.npy", m_corrupted)
    xp.save(spath + r"/m_final.npy", m_final)
    xp.save(spath + r"/m_loss_store.npy", m_loss_store)
    xp.save(spath + r"/Mtraj_store.npy", Mtraj_store)


'''
#Sampling pattern
# PE1 = 256
# PE2 = 256
# RO = 192
# m = xp.zeros((PE2, PE1, RO))
# Rs = 1
# TR_shot = 32
# order = 'interleaved'
# U = msi.make_samp(m, Rs, TR_shot, order)
# U = xp.transpose(U, axes = (0,2,1,3)) #PE1 was along AP, need to match
# xp.save(dpath + r'/U.npy', U)
'''

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
PE2 = 256
m = np.zeros((PE1, PE2, 192))

Rs = 1
TR_shot = 14
order = 'interleaved'
U_temp = msi.make_samp(m, Rs, TR_shot, order)
U = xp.transpose(U_temp, axes = (0,2,1,3)) 

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
