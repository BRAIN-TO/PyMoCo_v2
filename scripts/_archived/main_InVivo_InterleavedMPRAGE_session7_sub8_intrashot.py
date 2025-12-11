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

def BFGS(f,x0,args,max_it,h,spath):
    '''
    DESCRIPTION
    BFGS Quasi-Newton Method, implemented as described in Nocedal:
    Numerical Optimisation.
    #
    #
    INPUTS:
    f:      function to be optimised 
    x0:     intial guess
    max_it: maximum iterations 
    plot:   if the problem is 2 dimensional, returns 
            a trajectory plot of the optimisation scheme.
    #
    OUTPUTS: 
    x:      the optimal solution of the function f 
    #
    '''
    #Initial values
    d = len(x0) # dimension of problem 
    g = grad(f,x0,args,h) # initial gradient 
    H = xp.eye(d) # initial hessian
    x = x0[:]
    #Set up stores
    x_store = []
    f_store = []
    g_store = []
    #Run BFGS algorithm
    it = 1 
    while xp.linalg.norm(g) > 1e-5: # while gradient is positive
        print("BFGS Iteration {}".format(it), end = '\n')
        if it > max_it: 
            print('Maximum iterations reached!')
            break
        t1 = time()
        p = -H@g # search direction (Newton Method)
        # a = line_search(f,x,p,g) # line search 
        a = 1e-4 #fixing step size
        s = a * p 
        x_new = x + s
        g_new = grad(f,x_new,args,h)
        y = g_new - g 
        y = xp.array([y])
        s = xp.array([s])
        y = xp.reshape(y,(d,1))
        s = xp.reshape(s,(d,1))
        r = 1/(y.T@s)
        li = (xp.eye(d)-(r*((s@(y.T)))))
        ri = (xp.eye(d)-(r*((y@(s.T)))))
        hess_inter = li@H@ri
        H = hess_inter + (r*((s@(s.T)))) # BFGS Update
        g = g_new[:] 
        x = x_new[:]
        it += 1
        #Save to store
        x_store.append(x)
        f_store.append(f(x,args[0],args[1],args[2],args[3],args[4],args[5]))
        g_store.append(g)
        xp.save(spath + r'/opt_out.npy', [x_store, f_store, g_store])
        t2 = time()
        print("Elapsed time for BFGS Iter {}: {} sec".format(it, t2 - t1))
    return x_store, f_store, g_store

def grad(f_init,x,args,h=1e-3):     
    #CENTRAL FINITE DIFFERENCE CALCULATION
    # h = xp.cbrt(xp.finfo(float).eps)
    d = len(x)
    g = xp.zeros(d)
    for i in range(d): 
        #Set-up partial loss func
        TR_ind = int(xp.floor(i/6)) #Index of TR, given 6 DOFs per TR
        U_shot_i = args[3][TR_ind]
        f = partial(f_init, m_est = args[0], \
                    C = args[1], res = args[2], \
                    U_shot = [U_shot_i], \
                    R_pad = args[4], \
                    s_corrupted = args[5])
        #Evaluate finite difference 
        x_for = x.at[i].set(x[i]+h)[TR_ind*6:(TR_ind+1)*6]
        x_back = x.at[i].set(x[i]-h)[TR_ind*6:(TR_ind+1)*6]
        f_for = f(x_for)
        f_back = f(x_back)
        f_dif = (f_for- f_back)/(2*h)
        g = g.at[i].set(f_dif)
        print("Dimension {} -- Finite dif: {}".format(i+1, f_dif), end='\r')
    return g 

def _f_intra(Mtraj_est_n, m_est=None, C=None, res=None, U_shot=None, R_pad=None, s_corrupted=None):
    #Data consistency for a given shot
    n_shots = len(U_shot)
    s_n = eop.Encode(m_est, C, U_shot, Mtraj_est_n.reshape(n_shots, 6), res, R_pad)
    U_shot_full = xp.zeros(s_corrupted.shape)
    for i in range(len(U_shot)):
        U_shot_full += eop._gen_U_n(U_shot[i], m_est.shape)
    DC = s_n.flatten() - (U_shot_full*s_corrupted).flatten()
    return xp.abs(xp.dot(xp.conjugate(DC), DC)) #L2-norm

#-------------------------------------------------------------------------------
def main(dpath, spath_root, mpath, gt_name):
    #---------------------------------------------------------------------------
    #-----------------------Loading Reference Data------------------------
    #---------------------------------------------------------------------------
    # s_corrupted = xp.load(mpath + r'/{}/npy/kdat_trunc.npy'.format(gt_name)) #NC, SI, AP, LR
    # C_init = xp.load(mpath + r'/{}/npy/sens.npy'.format(gt_name))
    # C = xp.transpose(C_init, (3,0,1,2)); del C_init
    # mask = rec.getMask(C); xp.save(mpath + r'/{}/npy/m_GT_brain_mask.npy'.format(gt_name), mask)
    # U = np.load(mpath + r'/{}/npy/samp_order.npy'.format(gt_name), allow_pickle=1) #LR, AP, SI
    # res = xp.array([1,1,1])
    # R_pad = (10, 10, 10)
    # batch = 1
    # Mtraj_init = xp.zeros((len(U), 6))
    # m_GT = eop.Encode_Adj(s_corrupted, C, U, Mtraj_init, res, batch=batch) #E.H*s
    # xp.save(mpath + r'/{}/npy/img_CG.npy'.format(gt_name), m_GT)
    # del s_corrupted, C, mask, U, res, R_pad, batch, Mtraj_init, m_GT
    #---------------------------------------------------------------------------
    #--------------------------Loading Corrupted Data---------------------------
    #---------------------------------------------------------------------------
    #Load data
    s_corrupted = xp.load(dpath + r'/npy/kdat_trunc.npy') #NC, SI, AP, LR
    C_init = xp.load(mpath + r'/{}/npy/sens.npy'.format(gt_name))
    C = xp.transpose(C_init, (3,0,1,2))
    mask = rec.getMask(C); xp.save(dpath + r'/npy/m_GT_brain_mask.npy', mask)
    del C_init
    #---------------------------------------
    U = np.load(dpath + r'/npy/samp_order.npy', allow_pickle=1) #LR, AP, SI
    #---------------------------------------------------------------------------
    res = xp.array([1,1,1])
    #---------------------------------------
    m_GT = xp.load(mpath + r'/{}/npy/img_CG.npy'.format(gt_name))
    maxval = abs(m_GT.flatten()).max()
    m_GT /= maxval
    s_corrupted /= maxval
    #---------------------------------------
    #Loading the skull-stripping mask, generated from FreeSurfer SynthStrip tool
    cerebrum_mask = xp.ones(m_GT.shape)
    cerebrum_slice= 190 
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
    #---------------------------------------
    #Reconstruct image using CG SENSE algorithm
    m_init = eop.Encode_Adj(s_corrupted, C, U, Mtraj_init, res, batch=batch) #E.H*s
    #Motion-corrupted reconstruction
    A = partial(eop._EH_E, C=C, U=U, Mtraj=Mtraj_est, res=res, lamda = CG_lamda, batch=batch)
    b = eop.Encode_Adj(s_corrupted, C, U, Mtraj_est, res, batch=batch)
    #
    m_corrupted = m_init
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
    rmse_tol = 5.0
    trans_axes = (2,1,0,180)
    cnn_flag = 1 #turn on / off CNN
    JE_flag = 1 #turn JE algorithm on / off
    thresh = {'severe': 500, 'moderate': 0.1}
    if JE_flag and cnn_flag: #UNet + JE
        spath = spath_root + r'/npy/w_cnn_combo_PE1_AP_CorrectMask_UpRes32'
        max_loops = 50
    elif JE_flag and not cnn_flag: #only JE
        spath = spath_root + r'/npy/wo_cnn_CorrectMask_UpRes'
        max_loops = 250
    elif not JE_flag and cnn_flag: #only UNet
        spath = spath_root + r'/npy/w_only_cnn_combo_PE1_AP'
        max_loops = 1
    plib.Path(spath).mkdir(parents=True, exist_ok=True)
    xp.save(spath + r'/m_corrupted.npy', m_corrupted)
    #
    #---------------------------------------------------------------------------
    #Picking up algorithm
    m_est = np.load(spath + r'/m_intmd.npy')
    m_loss_store = list(np.load(spath + r'/m_loss_store.npy', allow_pickle=1))
    Mtraj_store = list(np.load(spath + r'/Mtraj_store.npy', allow_pickle=1))
    Mtraj_est = Mtraj_store[-1][0]
    #--------------------------------
    #Setting up shot pattern 
    nPE = m_est.shape[1]
    PE1_list = U[0][1]
    for i in range(1,len(U)):
        PE1_list = xp.concatenate((PE1_list, U[i][1]))
    #
    #
    shot_TRs = 2 #ie. # of sub-shots
    PE_ratio = 16//shot_TRs
    U_full = [] 
    for i in range(nPE//PE_ratio):
        U_full.append([U[0][0], PE1_list[i*PE_ratio:(i+1)*PE_ratio], U[0][2]])
    #
    #--------------------------------
    h = 1e-2
    maxiter = 10
    Mtraj_store_upres = xp.zeros((len(U)*shot_TRs, 6))
    f_store_upres = xp.zeros((len(U), maxiter))
    grad_store_upres = xp.zeros((len(U)*shot_TRs, 6))
    #
    for shot_ind in range(len(U)):
        print("Shot Index {}".format(shot_ind+1))
        U_shot = U_full[shot_ind*shot_TRs:(shot_ind+1)*shot_TRs]
        Mtraj_est_init = Mtraj_est[shot_ind,:]
        Mtraj_est_n = xp.tile(Mtraj_est_init, (shot_TRs,1)).flatten()
        args = [m_est, C, res, U_shot, R_pad, s_corrupted]
        opt_out = BFGS(_f_intra, Mtraj_est_n, args, maxiter, h, spath)
        #
        Mtraj_est_n_new = opt_out[0][-1].reshape(shot_TRs,6)
        f_store_upres = f_store_upres.at[shot_ind].set(xp.asarray(opt_out[1]))
        g_est_n_new = opt_out[2][-1].reshape(shot_TRs,6)
        Mtraj_store_upres = Mtraj_store_upres.at[shot_ind*shot_TRs:(shot_ind+1)*shot_TRs].set(Mtraj_est_n_new)
        grad_store_upres = grad_store_upres.at[shot_ind*shot_TRs:(shot_ind+1)*shot_TRs].set(g_est_n_new)
        xp.save(spath + r'/Mtraj_store_upres.npy', [Mtraj_store_upres, f_store_upres, grad_store_upres])
    #
    A_new = partial(eop._EH_E, C=C, U=U_full, Mtraj=xp.asarray(Mtraj_store_upres), \
                    res=res, lamda=0, batch=batch)
    #
    b_new = eop.Encode_Adj(s_corrupted, C, U_full, xp.asarray(Mtraj_store_upres), res, batch=batch)
    t1 = time()
    m_est = rec.ImageRecon(A_new, b_new, m_init, mask = mask, maxiter=CG_maxiter, \
                        tol=CG_tol, atol=CG_atol)
    t2 = time()
    xp.save(spath + r'/m_upres.npy', m_est[-1])
    print("Elapsed time: {} sec".format(t2 - t1))
    #
    return spath, m_corrupted, m_est, m_loss_store, Mtraj_store

#%% Run main()
if __name__ == "__main__":
    subname = "Sub8"
    mpath = r'/home/nghiemb/Data/TWH/MPRAGE_PE1Reordered/Scan20231116/{}'.format(subname)
    # test_case = 'scan3-InstructedMotion-16shots'
    test_case = 'scan4-FreeMotion-16shots'
    gt_name = 'scan1-ReferenceProduct-16shots'
    dpath = mpath + r'/{}'.format(test_case)
    print('Processing Test Case {}'.format(test_case))
    spath_root = dpath
    spath, m_corrupted, m_final, m_loss_store, Mtraj_store = main(dpath, spath_root, mpath, gt_name)
    xp.save(spath + r"/m_corrupted.npy", m_corrupted)
    xp.save(spath + r"/m_final.npy", m_final)
    xp.save(spath + r"/m_loss_store.npy", m_loss_store)
    xp.save(spath + r"/Mtraj_store.npy", Mtraj_store)

