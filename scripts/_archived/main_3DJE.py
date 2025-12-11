"""
MAIN SCRIPT
Running Joint Motion and Image Estimation
"""
import os
from time import time
from functools import partial

import tensorflow as tf
import jax.numpy as xp

import data.load_data as dat
import motion.motion_sim as msi
import encode.encode_op as eop
import recon.recon_op as rec
import cnn.run_unet as cnn

#%%-----------------------------------------------------------------------------
def add_noise(kdat, sd, mean=0):
    """Adding Gaussian noise to kdat"""
    nc,nx,ny,ns = kdat.shape
    noise_m = xp.random.normal(mean,sd,(nc,nx,ny,ns))
    F = sp.linop.FFT(noise_m.shape, axes=range(-len(noise_m.shape), 0))
    noise_kspace = F*noise_m
    return kdat + noise_kspace, noise_kspace

def evalRMSE(m, m_gt):
    dif2 = (xp.abs(m.flatten()) - xp.abs(m_gt.flatten()))**2
    return xp.sqrt(xp.mean(dif2))

def evalPE(m, m_gt, mask=None): #percent error
    if mask != None:
        m *= mask; m_gt *= mask
    #
    return 100*(evalRMSE(m, m_gt) / evalRMSE(m_gt, xp.zeros(m_gt.shape)))

def evalSSIM(m, m_gt, max_val = 1.0, mask=None): #percent error
    if mask != None:
        m *= mask; m_gt *= mask
    #
    return xp.mean(tf.image.ssim(abs(m_gt), abs(m), max_val).numpy())

#-------------------------------------------------------------------------------
def main(dpath, spath):
    #---------------------------------------------------------------------------
    #-----------------------Image Acquisition Simulation------------------------
    #---------------------------------------------------------------------------
    #Load data
    # dpath = r'./data/cc/test/severe/Test3'
    s_corrupted = xp.load(dpath + r'/s_corrupted.npy')
    C = xp.load(dpath + r'/sens.npy')
    mask = rec.getMask(C)
    res = xp.array([1,1,1])
    U = xp.load(dpath + r'/U.npy')
    m_GT = xp.load(dpath + r'/rss.npy'); m_GT /= xp.max(abs(m_GT.flatten()))
    #Motion trajectory
    Mtraj_GT = xp.load(dpath + r'/Mtraj.npy')
    R_pad = (0,0,0)
    batch = 1
    #---------------------------------------------------------------------------
    #------------------JOINT IMAGE RECON AND MOTION ESTIMATION------------------
    #---------------------------------------------------------------------------
    #Initializing update vars
    Mtraj_init = xp.zeros((U.shape[0], 6))
    Mtraj_est = Mtraj_init
    CG_maxiter = 10
    ME_maxiter = 1 #motion estimation maxiter
    tol = 1e-3
    atol = 0.0
    CG_lamda = 0
    #Initialize stores
    m_loss_store = []
    m_cnn_store = []
    Mtraj_store = []
    ##
    #
    # import numpy as np
    # Mtraj_store = np.load(spath + r"/Mtraj_store_JE.npy", allow_pickle=1).tolist()
    # Mtraj_est = Mtraj_store[-1][0]
    # m_loss_store = np.load(spath + r"/m_loss_store.npy", allow_pickle=1).tolist()
    #---------------------------------------------------------------------------
    #Reconstruct image using CG SENSE algorithm
    m_init = eop.Encode_Adj(s_corrupted, C, U, Mtraj_init, res, batch=batch) #E.H*s
    x0 = m_init
    #
    A = partial(eop._EH_E, C=C, U=U, Mtraj=Mtraj_est, res=res, lamda = 0, batch=batch)
    b = eop.Encode_Adj(s_corrupted, C, U, Mtraj_est, res, batch=batch)
    #
    t1 = time()
    m_out = rec.ImageRecon(A, b, x0, maxiter=CG_maxiter, tol=tol, atol=0.0)
    t2 = time()
    print("Time elapsed: {} sec".format(str(t2 - t1)))
    m_corrupted = m_out[-1]
    m_est_rmse = evalPE(m_corrupted, m_GT, mask)
    m_est_ssim = evalSSIM(m_corrupted, m_GT, mask=mask)
    m_loss_store.append([m_est_rmse, m_est_ssim])
    print("Error: {:.2f} %".format(m_est_rmse))
    m_est = m_corrupted
    # #---------------------------------------------------------------------------
    # #Loading trained CNN model
    mpath = r'/home/nghiemb/PyMoCo/cnn/3DUNet_SAP'
    weights_path_severe = mpath + r'/weights/{}/train_n134'.format('severe')
    weights_path_moderate = mpath + r'/weights/{}/train_n134'.format('moderate')
    weights_path_mild = mpath + r'/weights/{}/train_n134'.format('mild')
    pads = [11,3]
    #---------------------------------------------------------------------------
    #Alternating image recovery & motion estimation (coordinate descent)
    rmse_tol = 3.0
    max_loops = 50
    # max_loops = 2000
    i = 0
    t1 = time()
    while m_est_rmse >= rmse_tol and i <= max_loops:
    # for i in range(max_loops):
        t2 = time()
        print("-----------------------------------------------------------")
        print("Joint Optimization iter:{}".format(i))
        #-----------------------------------------------------------------------
        #Run CNN
        if i < 7:
            print("UNet - Severe")
            m_cnn = cnn.main(m_est, pads, weights_path_severe)
        elif i < 14:
            print("UNet - Moderate")
            m_cnn = cnn.main(m_est, pads, weights_path_moderate)
        else:
            print("UNet - Mild")
            m_cnn = cnn.main(m_est, pads, weights_path_mild)
        m_est = m_cnn*mask
        m_cnn_store.append(m_cnn)
        xp.save(spath + r"/m_cnn_store.npy", m_cnn_store)
        #-----------------------------------------------------------------------
        #Motion Estimation step
        Mtraj_est, Mtraj_loss, Mtraj_grad = rec.MotionEst(Mtraj_est, m_est, C, U, res, s_corrupted, maxiter = ME_maxiter)
        Mtraj_est_loss = [evalPE(Mtraj_est[:,n], Mtraj_GT[:, n]) for n in range(6)]
        Mtraj_store.append((Mtraj_est, Mtraj_est_loss, Mtraj_loss, Mtraj_grad))
        xp.save(spath + r"/Mtraj_store_JE.npy", Mtraj_store)
        #-----------------------------------------------------------------------
        #Image Recovery step
        A_new = partial(eop._EH_E, C=C, U=U, Mtraj=Mtraj_est, res=res, lamda = 0, batch=batch)
        b_new = eop.Encode_Adj(s_corrupted, C, U, Mtraj_est, res, batch=batch)
        #
        m_out = rec.ImageRecon(A_new, b_new, x0, maxiter=CG_maxiter, tol=tol, atol=0.0)
        m_est = m_out[-1]
        #
        m_est_rmse = evalPE(m_est, m_GT, mask)
        m_est_ssim = evalSSIM(m_est, m_GT, mask=mask)
        m_loss_store.append([m_est_rmse,m_est_ssim])
        xp.save(spath + r"/m_loss_store.npy", m_loss_store)
        print("Error: {:.2f} %".format(m_est_rmse))
        t3 = time()
        print("Time elapsed for iter {}: {}sec".format(str(i+1), str(t3 - t2))) #100 iters = 65s
        #-----------------------------------------------------------------------
        i+=1
    print("Total Time elapsed: {} sec".format(time() - t1))
    return m_corrupted, m_est, m_loss_store, Mtraj_store

#%% Run main()
if __name__ == "__main__":
    dpath = os.environ['IN_DIR']
    spath = os.environ['OUT_DIR']
    t1 = time()
    m_corrupted, m_final, m_loss_store, Mtraj_store = main(dpath, spath)
    xp.save(spath + r"/m_corrupted.npy", m_corrupted)
    xp.save(spath + r"/m_final.npy", m_final)
    xp.save(spath + r"/m_loss_store.npy", m_loss_store)
    xp.save(spath + r"/Mtraj_store.npy", Mtraj_store)

# dpath = r'/home/nghiemb/PyMoCo/data/cc/test/severe/Test3'
# spath = r'/home/nghiemb/PyMoCo/data/cc/test/combo/Test3/namer_output1'