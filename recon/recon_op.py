# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 14:47:26 2020

@author: brian
"""
import jax
import jax.numpy as xp
from jax.scipy.optimize import minimize
from time import time
from functools import partial

from optimize.minimize import minimize as minimize_local # only for integrated loss function
import encode.encode_op as eop
import utils.metrics as mtc

from scipy.ndimage import rotate
import scipy.signal as ss

import gc

#%%-----------------------------------------------------------------------------
#----------------------------------IMAGE MASK-----------------------------------
#%%-----------------------------------------------------------------------------

def getMask(C, threshold = 1e-5):
    '''For masking corrupted data to match masking of estimated coil profiles'''
    C_n = C[0,...] #extract a single coil profile
    mask = xp.zeros(C_n.shape, dtype = xp.float32)
    mask = mask.at[xp.abs(C_n)>threshold].set(1)
    return mask

#%%-----------------------------------------------------------------------------
#-----------------------------IMAGE RECONSTRUCTION------------------------------
#%%-----------------------------------------------------------------------------
'''
***TEMPORARY - running into memory issues with directly calling jax.scipy.sparse.linalg***
Currently just copying jax.scipy.sparse.linalg.cg script below
'''

def ImageRecon(A, b, x0, mask=None, maxiter=3, tol=1e-5, atol=0.0):
    """
    ***TEMPORARY - running into memory issues with directly calling jax.scipy.sparse.linalg***
    CG-SENSE image reconstruction; from jax.scipy.sparse.linalg.cg
    """
    if mask == None:
        mask = xp.ones(x0.shape)
    #
    bs = xp.vdot(b,b)
    atol2 = xp.maximum(xp.square(tol) * bs, xp.square(atol))
    r0 = (b - A(x0))*mask
    p0 = r0
    gamma0 = xp.vdot(r0, r0)
    k_ = 0
    #-------------------------------------
    #Initialize input for CG body function
    vals = (x0, r0, gamma0, p0, k_, 0, 0, 0)
    x_store = []
    p_store = []
    g_store = []
    a_store = []
    b_store = []
    #-------------------------------------
    #Helper functions
    def cond_fun(vals):
        _, r, rs, _, k, _, _, _ = vals
        return (rs > atol2) & (k < maxiter)
    #
    def body_fun(vals):
        x, r, gamma, p, k, pAp, alpha, beta_ = vals
        Ap = A(p)
        pAp = xp.vdot(p, Ap)
        alpha = gamma / pAp
        x_ = x + alpha*p
        r_ = r - alpha*Ap
        gamma_ = xp.vdot(r_, r_)
        beta_ = gamma_ / gamma
        p_ = (r_ + beta_ * p)*mask
        return x_, r_, gamma_, p_, k + 1, pAp, alpha, beta_
    #
    #-------------------------------------
    while cond_fun(vals):
        t1 = time()
        print("Iter: {}".format(str(k_+1)))
        vals = body_fun(vals)
        t2 = time()
        print("Time elapsed: {} sec".format(str(t2 - t1)))
        x_, r_, g_, p_, k_, pAp_, a_, b_ = vals
        x_store.append(x_)
        p_store.append(p_)
        g_store.append(g_)
        a_store.append(a_)
        b_store.append(b_)
    #
    x_final, r_final, *_ = vals
    return x_store


#%%-----------------------------------------------------------------------------
#-------------------------------MOTION ESTIMATION-------------------------------
#----------------------------FULL EVAL COST FUNCTION----------------------------
#%%-----------------------------------------------------------------------------

'''
def _f(Mtraj_est_init, m_est=None, C=None, res=None, U=None, R_pad=None, s_corrupted=None):
    Mtraj_est = Mtraj_est_init.reshape(16,6)
    s_temp = eop.Encode(m_est, C, U, Mtraj_est, res, R_pad)
    DC = s_temp.flatten() - s_corrupted.flatten()
    return xp.abs(xp.dot(xp.conjugate(DC), DC)) #L2-norm

import scipy
def _min(Mtraj_est, m_est, C, res, U, R_pad, s_corrupted):
    maxiter = 1
    opts = {'maxiter':maxiter} # max number of iterations
    opt_out = scipy.optimize.minimize(fun=_f, x0=Mtraj_est, \
                                        args=(m_est, C, res, U, R_pad, s_corrupted), \
                                        method='bfgs', options = opts)
    Mtraj_update_n = opt_out.x
    f_val = opt_out.fun
    g_val = opt_out.jac
    return Mtraj_update_n, f_val, g_val
'''

'''
def _f_intra(Mtraj_est_n, m_est=None, C=None, res=None, U_shot=None, R_pad=None, s_corrupted=None):
    #Data consistency for a given shot
    n_shots = len(U_shot)
    s_n = eop.Encode(m_est, C, U_shot, Mtraj_est_n.reshape(n_shots, 6), res, R_pad)
    #
    U_shot_full = xp.zeros(s_corrupted.shape)
    for i in range(len(U_shot)):
        U_shot_full += eop._gen_U_n(U_shot[i], m_est.shape)
    #
    DC = s_n.flatten() - (U_shot_full*s_corrupted).flatten()
    return xp.abs(xp.dot(xp.conjugate(DC), DC)) #L2-norm
'''

'''
import scipy

def _f_intra(Mtraj_est_n, m_est=None, C=None, res=None, U_shot=None, R_pad=None, s_corrupted=None):
    #Data consistency for a given shot
    n_shots = len(U_shot)
    s_n = eop.Encode(m_est, C, U_shot, Mtraj_est_n.reshape(n_shots, 6), res, R_pad)
    #
    U_shot_full = xp.zeros(s_corrupted.shape)
    for i in range(len(U_shot)):
        U_shot_full += eop._gen_U_n(U_shot[i], m_est.shape)
    #
    DC = s_n.flatten() - (U_shot_full*s_corrupted).flatten()
    return xp.abs(xp.dot(xp.conjugate(DC), DC)) #L2-norm

def _min_scipy(Mtraj_est_n, m_est, C, res, U_n, R_pad, s_corrupted, options):
    #Minimizing data consistency wrt motion parameters for given shot
    opt_out = scipy.optimize.minimize(_f_n, Mtraj_est_n, \
                        args=(m_est, C, res, U_n, R_pad, s_corrupted), \
                        method='bfgs', options = options)
    Mtraj_update_n = opt_out.x
    f_val = opt_out.fun
    g_val = opt_out.jac
    return Mtraj_update_n, f_val, g_val
'''

#%%-----------------------------------------------------------------------------
#--------------------------------------UNET-------------------------------------
#%%-----------------------------------------------------------------------------

def rescale_sym(x, max):
	#For data ranging from [-max, max]
	#Output rescaled to [0,1]
	return (x + max) / (2*max) #RESCALE to [0,1]

def unscale_sym(x, max):
    return x*(2*max) - max

def UNet_Mag(m_est, trans_axes, pads, wpath_severe, mask, cnn): #Magnitude Only
    #Update: June 6, 2024
    m_cnn_in_init = xp.transpose(m_est, axes=trans_axes[:3])
    m_cnn_in = rotate(m_cnn_in_init, angle=trans_axes[3], axes=(0,1))
    m_cnn_out_mag = cnn.main(xp.abs(m_cnn_in), pads, wpath_severe + r'/magnitude') #MAGNITUDE UNET
    m_cnn_out = m_cnn_out_mag
    #
    m_est_cnn_init = rotate(m_cnn_out, angle=-trans_axes[3], axes=(0,1))
    m_est_cnn = xp.transpose(m_est_cnn_init, axes=trans_axes[:3])*mask
    m_est = m_est_cnn
    return m_est

# def UNet_MagPhase(m_est, trans_axes, pads, wpath_severe, mask, cnn): #Combined Magnitude and Phase Unet
#     #Update: June 6, 2024
#     m_cnn_in_init = xp.transpose(m_est, axes=trans_axes[:3])
#     m_cnn_in = rotate(m_cnn_in_init, angle=trans_axes[3], axes=(0,1))
#     # m_cnn_out = cnn.main(xp.abs(m_cnn_in), pads, wpath_severe) #MAGNITUDE UNET
#     m_cnn_out_mag = cnn.main(xp.abs(m_cnn_in), pads, wpath_severe + r'/magnitude') #MAGNITUDE UNET
#     m_cnn_out_phase = cnn.main(xp.angle(m_cnn_in), pads, wpath_severe + r'/phase') #PHASE UNET
#     m_cnn_out_phase_unscaled = unscale_sym(m_cnn_out_phase, xp.pi)
#     m_cnn_out = m_cnn_out_mag*xp.exp(1j*m_cnn_out_phase_unscaled)
#     #
#     m_est_cnn_init = rotate(m_cnn_out, angle=-trans_axes[3], axes=(0,1))
#     m_est_cnn = xp.transpose(m_est_cnn_init, axes=trans_axes[:3])*mask
#     # m_est = m_est_cnn*xp.exp(1j*xp.angle(m_est)) #recombine with phase of intermediate input image
#     m_est = m_est_cnn
#     # xp.save(spath + r"/m_cnn_store.npy", m_cnn_store)
#     return m_est

def UNet_ReIm(m_est, trans_axes, pads, wpath_severe, mask, cnn):
    m_cnn_in_init = xp.transpose(m_est, axes=trans_axes[:3])
    m_cnn_in = rotate(m_cnn_in_init, angle=trans_axes[3], axes=(0,1))
    #
    m_cnn_out_real = cnn.main(xp.real(m_cnn_in), pads, wpath_severe + r'/real')
    m_cnn_out_imag = cnn.main(xp.imag(m_cnn_in), pads, wpath_severe + r'/imag')
    m_cnn_out = m_cnn_out_real + 1j*m_cnn_out_imag
    #
    m_est_cnn_init = rotate(m_cnn_out, angle=-trans_axes[3], axes=(0,1))
    m_est_cnn = xp.transpose(m_est_cnn_init, axes=trans_axes[:3])*mask
    m_est = m_est_cnn
    return m_est

# def UNet_ReIm_Sequential(m_est, trans_axes, pads, wpath_severe, mask):
#     m_cnn_in_init = xp.transpose(m_est, axes=trans_axes[:3])
#     m_cnn_in = rotate(m_cnn_in_init, angle=trans_axes[3], axes=(0,1))
#     print('Current Loss value: {}'.format(f_val))
#     if f_val >= thresh['severe']:
#         print("UNet - Severe")
#         m_cnn_out_real = cnn.main(xp.real(m_cnn_in), pads, wpath_severe + r'/real')
#         m_cnn_out_imag = cnn.main(xp.imag(m_cnn_in), pads, wpath_severe + r'/imag')
#         m_cnn_out = m_cnn_out_real + 1j*m_cnn_out_imag
#     elif f_val >= thresh['moderate'] and f_val < thresh['severe']:
#         print("UNet - Moderate")
#         m_cnn_out_real = cnn.main(xp.real(m_cnn_in), pads, wpath_moderate + r'/real')
#         m_cnn_out_imag = cnn.main(xp.imag(m_cnn_in), pads, wpath_moderate + r'/imag')
#         m_cnn_out = m_cnn_out_real + 1j*m_cnn_out_imag
#     elif f_val >= 0 and f_val < thresh['moderate']:
#         print("UNet - Mild")
#         m_cnn_out_real = cnn.main(xp.real(m_cnn_in), pads, wpath_mild + r'/real')
#         m_cnn_out_imag = cnn.main(xp.imag(m_cnn_in), pads, wpath_mild + r'/imag')
#         m_cnn_out = m_cnn_out_real + 1j*m_cnn_out_imag
#     elif f_val < 0: #if diverging
#         print("No UNet")
#         m_cnn_out = m_cnn_in
#     m_est_cnn_init = rotate(m_cnn_out, angle=-trans_axes[3], axes=(0,1))
#     m_est_cnn = xp.transpose(m_est_cnn_init, axes=trans_axes[:3])*mask
#     m_cnn_store.append(m_est_cnn)
#     m_est = m_est_cnn
#     return m_est

#%%-----------------------------------------------------------------------------
#-------------------------------MOTION ESTIMATION-------------------------------
#----------------------------SHOT-WISE COST FUNCTION----------------------------
#%%-----------------------------------------------------------------------------

def _f_n(Mtraj_est_n, m_est=None, C=None, res=None, U_n=None, R_pad=None, s_corrupted=None):
    '''Data consistency for a given shot'''
    T_n = Mtraj_est_n[:3]
    R_n = Mtraj_est_n[3:]
    s_n = eop._E_n(U_n, R_n, T_n, m_est, C, res, R_pad) #evaluate E*m for shot n
    DC = s_n.flatten() - (U_n*s_corrupted).flatten()
    return xp.abs(xp.dot(xp.conjugate(DC), DC)) #L2-norm

@partial(jax.jit, static_argnums = (5,)) #R_pad is static due to use in encode_op._pad
def _min(Mtraj_est_n, m_est, C, res, U_n, R_pad, s_corrupted, options):
    '''Minimizing data consistency wrt motion parameters for given shot'''
    opt_out = minimize(_f_n, Mtraj_est_n, \
    # opt_out = minimize_local(_f_n, Mtraj_est_n, \
                       args=(m_est, C, res, U_n, R_pad, s_corrupted), \
                       method='bfgs', options = options)
    Mtraj_update_n = opt_out.x
    f_val = opt_out.fun
    g_val = opt_out.jac
    return Mtraj_update_n, f_val, g_val

def MotionEst(Mtraj_est, m_est, C, U, dscale, res, s_corrupted, R_pad = (0,0,0), maxiter = 1, ls_maxiter = 10, continuity = 0):
    '''
    Data consistency-based motion estimation
    Using BFGS quasi-Newton algorithm from jax.scipy.optimize.minimize
    '''
    #Constants
    nshots = len(U)
    # opts = {'maxiter':maxiter, 'fixed':0} # max number of iterations
    opts = {'maxiter':maxiter} # max number of iterations
    #Run BFGS algorithm
    Mtraj_out = xp.zeros(Mtraj_est.shape, dtype = Mtraj_est.dtype)
    f_out = [] #loss values
    g_out = [] #gradient of loss function
    print("Shot 1 set as reference (ie. zero motion)")
    for n in range(nshots):
        print("Shot: {}".format(str(n+1)))
        t1 = time()
        U_n = eop._gen_U_n(U[n], m_est.shape)
        if continuity and n>0: #enforce continuity for subsequent shots
                Mtraj_est_n = Mtraj_out[n-1,:] #seed with updated estimate for previous shot
        else:
            Mtraj_est_n = Mtraj_est[n, :] #seed with previous estimate of given shot
        Mtraj_out_n, f_val, g_val = _min(Mtraj_est_n, m_est, C, res, U_n, R_pad, s_corrupted, opts)
        Mtraj_out = Mtraj_out.at[n,:].set(Mtraj_out_n)
        #
        f_out.append(f_val)
        g_out.append(g_val)
        t2 = time()
        print("Time elapsed for shot {}: {} sec".format(str(n+1), t2 - t1))
    # Mtraj_offset = xp.tile(Mtraj_out[:dscale,:], (Mtraj_out.shape[0]//dscale,1))
    Mtraj_offset = Mtraj_out[0,:]
    Mtraj_out -= Mtraj_offset #set first shot to be the reference
    return Mtraj_out, f_out, g_out


#-----------------------------------------------------------------------
#Update June 27, 2024
#Retrospectively evaluating total DC loss 

def _f(Mtraj_est, m_est=None, C=None, res=None, U=None, R_pad=None, s_corrupted=None):
    s_temp = eop.Encode(m_est, C, U, Mtraj_est, res, R_pad)
    DC = s_temp.flatten() - s_corrupted.flatten()
    return xp.abs(xp.dot(xp.conjugate(DC), DC)) #L2-norm

def eval_TotalDC(Mtraj_est, fixed_vars, JE_params):
    x0, s_corrupted, C, U, _, res, _, _, R_pad, _ = fixed_vars
    _, _, _, _, _, _, _, CG_maxiter, CG_tol, CG_atol, CG_mask, batch, mask, _, _ = JE_params
    #
    A_new = partial(eop._EH_E, C=C, U=U, Mtraj=Mtraj_est, res=res, lamda=0, batch=batch)
    b_new = eop.Encode_Adj(s_corrupted, C, U, Mtraj_est, res, batch=batch)
    #
    if CG_mask:
        m_out = ImageRecon(A_new, b_new, x0, mask = mask, maxiter=CG_maxiter, \
                            tol=CG_tol, atol=CG_atol)
    else:
        m_out = ImageRecon(A_new, b_new, x0, maxiter=CG_maxiter, \
                            tol=CG_tol, atol=CG_atol)
    #
    m_est = mask*m_out[-1]
    DC = _f(Mtraj_est, m_est=m_est, C=C, res=res, U=U, R_pad=R_pad, s_corrupted=s_corrupted)
    return DC


#%% ----------------------------------------------------------------------------
# -------------------------------JOINT ESTIMATION-------------------------------
# ---------------------------MULTI-LEVEL OPTIMIZATION---------------------------
# ------------------------------------------------------------------------------
def grad_condition(Mtraj_store, grad_tol=1e-1, filter_window=19):
    JE_loss_vals_shotwise = xp.array([Mtraj_store[i][1] for i in range(len(Mtraj_store))])
    JE_loss_grad_shotwise = JE_loss_vals_shotwise[1:,:,0] - JE_loss_vals_shotwise[:-1,:,0]
    shotval_store = []
    for shotval in range(JE_loss_grad_shotwise.shape[1]):
        JE_loss_grad_shot = JE_loss_grad_shotwise[:,shotval]
        if filter_window != 0:
            JE_loss_grad_shot_filtered = ss.savgol_filter(JE_loss_grad_shot, \
                                                        window_length=filter_window, \
                                                        polyorder=2)
        else:
            JE_loss_grad_shot_filtered = JE_loss_grad_shot
        shotval_store.append(JE_loss_grad_shot_filtered[-1])
    grad_flag = xp.all(abs(xp.array(shotval_store)) < grad_tol)
    return grad_flag

def JointEst(init_est, fixed_vars, stores, cnn, CNN_params, JE_params):
    x0, s_corrupted, C, U, dscale, res, spath, m_GT, R_pad, skull_mask = fixed_vars
    m_est, Mtraj_est = init_est
    m_cnn_store, Mtraj_store, m_loss_store, DC_store = stores
    m_est_rmse, rmse_tol, m_est_ssim, ssim_tol, max_loops, ME_maxiter, LS_maxiter, CG_maxiter, CG_tol, CG_atol, CG_mask, batch, mask, continuity, grad_tol = JE_params
    cnn_flag, JE_flag, trans_axes, pads, wpath_severe, wpath_moderate, wpath_mild, thresh = CNN_params
    grad_flag = 0
    filter_window = 19
    i = 0
    t1 = time()
    while m_est_rmse >= rmse_tol and m_est_ssim <= ssim_tol and i < max_loops and not grad_flag:
        t2 = time()
        print("-----------------------------------------------------------")
        print("Joint Optimization iter:{}".format(i+1))
        # ----------------------------------------------------------------------
        # Update L2-norm of data consistency
        if i == 0:
            f_val = thresh['severe'] #initialize st use UNet_severe
        else:
            f_val = xp.mean(xp.array(Mtraj_loss))
        # Run CNN
        if not JE_flag: #Magnitude only
            m_est = UNet_Mag(m_est, trans_axes, pads, wpath_severe, mask, cnn)
            # m_est = UNet_ReIm(m_est, trans_axes, pads, wpath_severe, mask, cnn)
            # gc.collect() #force garbage collection
        else:
            if cnn_flag:
                m_est = UNet_ReIm(m_est, trans_axes, pads, wpath_severe, mask, cnn)
                # gc.collect() #force garbage collection
            # ----------------------------------------------------------------------
            # Motion Estimation step
            if JE_flag:
                Mtraj_est, Mtraj_loss, Mtraj_grad = MotionEst(Mtraj_est, m_est*skull_mask, C, U, \
                                                                dscale, res, s_corrupted, \
                                                                R_pad = R_pad, \
                                                                maxiter=ME_maxiter, \
                                                                ls_maxiter = LS_maxiter, \
                                                                continuity = continuity)
                Mtraj_loss_out = xp.tile(xp.array(Mtraj_loss)[:,None], (1,6))
                # Mtraj_grad_out = xp.array(Mtraj_grad)
                # Mtraj_store.append((Mtraj_est, Mtraj_loss_out, Mtraj_grad_out))
                Mtraj_store.append((Mtraj_est, Mtraj_loss_out))
                xp.save(spath + r"/Mtraj_store.npy", Mtraj_store)
                if i>=filter_window+1:
                    grad_flag = grad_condition(Mtraj_store, grad_tol, filter_window)
                # ----------------------------------------------------------------------
                # Image Recovery step
                A_new = partial(eop._EH_E, C=C, U=U, Mtraj=Mtraj_est, \
                                res=res, lamda=0, batch=batch)
                b_new = eop.Encode_Adj(s_corrupted, C, U, Mtraj_est, res, batch=batch)
                #
                if CG_mask:
                    m_out = ImageRecon(A_new, b_new, x0, mask = mask, maxiter=CG_maxiter, \
                                        tol=CG_tol, atol=CG_atol)
                else:
                    m_out = ImageRecon(A_new, b_new, x0, maxiter=CG_maxiter, \
                                        tol=CG_tol, atol=CG_atol)
                m_est = mask*m_out[-1]
                xp.save(spath + r"/m_intmd.npy", m_est)
                DC_store.append(eval_TotalDC(Mtraj_est, fixed_vars, JE_params))
                xp.save(spath + r"/DC_store.npy", DC_store)
        #
        m_est_rmse = mtc.evalPE(m_est, m_GT, mask)
        m_est_ssim = mtc.evalSSIM(m_est, m_GT, mask=mask)
        m_loss_store.append([m_est_rmse, m_est_ssim])
        xp.save(spath + r"/m_loss_store.npy", m_loss_store)
        print("NRMSE: {:.2f} %".format(m_est_rmse))
        print("SSIM: {:.2f} %".format(m_est_ssim))
        t3 = time()
        print("Time elapsed for iter {}: {}sec".format(str(i+1), str(t3 - t2)))
        # ----------------------------------------------------------------------
        i += 1
    print("Total Time elapsed: {} sec".format(time() - t1))
    return m_est, m_loss_store, Mtraj_store, m_cnn_store


# def SAMER(init_est, fixed_vars, stores, JE_params):
#     x0, s_corrupted, C, U, dscale, mask, res, spath, m_GT, R_pad = fixed_vars
#     m_est, Mtraj_est = init_est
#     m_cnn_store, Mtraj_store, m_loss_store = stores
#     m_est_rmse, rmse_tol, max_loops, ME_maxiter, LS_maxiter, CG_maxiter, CG_tol, CG_atol, batch, mask = JE_params
#     i = 0
#     t1 = time()
#     while m_est_rmse >= rmse_tol and i <= max_loops:
#         t2 = time()
#         print("-----------------------------------------------------------")
#         print("Joint Optimization iter:{}".format(i+1))
#         # Motion Estimation step
#         Mtraj_est, Mtraj_loss, Mtraj_grad = MotionEst(Mtraj_est, m_est, C, U, \
#                                                       dscale, res, s_corrupted, \
#                                                       R_pad = R_pad, \
#                                                       maxiter=ME_maxiter, \
#                                                       ls_maxiter = LS_maxiter)
#         Mtraj_store.append((Mtraj_est, Mtraj_loss, Mtraj_grad))
#         xp.save(spath + r"/Mtraj_store_SAMER.npy", Mtraj_store)
#         t3 = time()
#         print("Time elapsed for iter {}: {}sec".format(str(i+1), str(t3 - t2)))
#         # ----------------------------------------------------------------------
#         i += 1
#     print("Total Time elapsed: {} sec".format(time() - t1))
#     return m_est, m_loss_store, Mtraj_store, m_cnn_store




'''
#TEMP CODE, April 24 2024
#WANT TO SEE IF I CAN APPLY THE MAGNITUDE-ONLY UNET TO THE REAL AND IMAGINARY COMPONENTS OF THE COMPLEX DATA

m_cnn_in_init = xp.transpose(m_est, axes=trans_axes[:3])
m_cnn_in = rotate(m_cnn_in_init, angle=trans_axes[3], axes=(0,1))

#TRY APPLING THE MAGNITUDE UNET ON THE REAL AND IMAGINARY COMPONENTS OF THE COMPLEX IMAGE
m_cnn_out_re = cnn.main(xp.real(m_cnn_in), pads, wpath_severe + r'/magnitude') #MAGNITUDE UNET
m_cnn_out_im = cnn.main(xp.imag(m_cnn_in), pads, wpath_severe + r'/magnitude') #MAGNITUDE UNET

m_cnn_out = m_cnn_out_re + 1j*m_cnn_out_im
m_est_cnn_init = rotate(m_cnn_out, angle=-trans_axes[3], axes=(0,1))
m_est_cnn = xp.transpose(m_est_cnn_init, axes=trans_axes[:3])*mask

m_cnn_store.append(m_est_cnn)
m_est = m_est_cnn

import matplotlib.pyplot as plt

img_plot = abs(xp.imag(m_corrupted[m_est.shape[0]//2,:,:]))
img_plot = abs(xp.imag(m_corrupted[:,m_est.shape[1]//2,:]))
img_plot = abs(xp.imag(m_corrupted[:,:,m_est.shape[2]//2]))

plt.figure()
plt.imshow(img_plot, cmap = "gray")
plt.show()


'''
