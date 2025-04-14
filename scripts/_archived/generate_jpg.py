"""
MAIN SCRIPT
Running Joint Motion and Image Estimation
"""
import os
import pathlib as plib
from time import time
from functools import partial

from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

import numpy as np
from scipy.ndimage import rotate

import jax.numpy as xp
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="0" #turn off GPU pre-allocation

import encode.encode_op as eop
import recon.recon_op as rec
import cnn.run_unet as cnn
import utils.metrics as mtc

#-------------------------------------------------------------------------------
#Helper Functions
def float2int8(img):
    '''Source: https://stackoverflow.com/questions/53235638/how-should-i-convert-a-float32-image-to-an-uint8-image'''
    img = abs(img)
    vmin = img.min()
    vmax = img.max() - vmin
    img_int32 = ((img - vmin)/vmax) * (2**8 - 1)
    return img_int32.astype(np.uint8)

def gen_img(volume_init, transpose, view = 'axial', vmax = 1.0):
    #Choosing slice orientation
    volume = np.transpose(volume_init, axes = transpose)
    if view == 'sagittal':
        slice = abs(volume[volume.shape[0]//2 + 5,:,:])
    elif view == 'coronal':
        slice = abs(volume[:,volume.shape[1]//2,:])
    elif view == 'axial':
        # slice = abs(volume[:,:,volume.shape[2]//2])
        slice = abs(volume[:,:,120]) #subjects 2, 3, 4, 5
        # slice = abs(volume[:,:,115]) #subjects 1, 6, 7
    #
    #Changing display range
    slice_newvmax = slice.at[xp.where(slice > vmax)].set(vmax) #threshold vmax
    slice_newdisp = slice_newvmax / vmax #reset display range to [0,1]
    slice_rot = rotate(slice_newdisp, angle = -90)
    slice_uint8 = np.asarray(float2int8(slice_rot))
    img = Image.fromarray(slice_uint8)
    return img

#-------------------------------------------------------------------------------
def main(dpath, spath_root, mpath):
    #---------------------------------------------------------------------------
    #-----------------------Image Acquisition Simulation------------------------
    #---------------------------------------------------------------------------
    #Load data
    s_corrupted = xp.load(dpath + r'/s_corrupted.npy') #NC, SI, AP, LR
    C = xp.load(dpath + r'/sens.npy')
    mask = rec.getMask(C)
    res = xp.array([1,1,1])
    U = xp.load(dpath + r'/U.npy')
    m_GT = xp.load(dpath + r'/rss.npy')
    maxval = abs(m_GT.flatten()).max()
    m_GT /= maxval
    #Motion trajectory
    Mtraj_GT = xp.load(dpath + r'/Mtraj.npy')
    R_pad = (10, 10, 10)
    batch = 1
    #
    #Load Mtraj trajectories for UNet-only, JE, and Unet + JE
    prefix = 'wo_cnn'
    # prefix = 'w_cnn_alt'
    spath = spath_root + r'/{}/GIF'.format(prefix)
    plib.Path(spath).mkdir(parents=True, exist_ok=True)
    #
    Mtraj_store_init = np.load(dpath + r'/{}/Mtraj_store.npy'.format(prefix), allow_pickle = 1)
    #---------------------------------------------------------------------------
    #------------------JOINT IMAGE RECON AND MOTION ESTIMATION------------------
    #---------------------------------------------------------------------------
    #Initializing update vars
    Mtraj_init = xp.zeros((U.shape[0], 6))
    Mtraj_store = [Mtraj_init] + [Mtraj_store_init[i][0] for i in range(len(Mtraj_store_init))]
    CG_maxiter = 10 #limit CG_iter to 3 iters for fully-sampled data to prevent artifacts
    CG_tol = 1e-7 #relative tolerance
    CG_atol = 1e-4 #absolute tolerance
    CG_lamda = 0
    CG_mask = 0 #turn on for in-vivo dataset, turn off for CC dataset
    #Initialize stores
    m_loss_store = []
    #Image display variables
    transpose = (0,1,2)
    vmax = 0.5
    m_init = eop.Encode_Adj(s_corrupted, C, U, Mtraj_init, res, batch=batch) #E.H*s
    #---------------------------------------------------------------------------
    #Reconstruct image using CG SENSE algorithm
    start = 0
    stop = len(Mtraj_store)
    for iter, Mtraj_temp in enumerate(Mtraj_store[start:stop]):
        iter += start
        print('Iteration {}'.format(iter))
        A = partial(eop._EH_E, C=C, U=U, Mtraj=Mtraj_temp, res=res, \
                    lamda = CG_lamda, batch=batch)
        b = eop.Encode_Adj(s_corrupted, C, U, Mtraj_temp, res, batch=batch)
        if CG_mask:
            m_out = rec.ImageRecon(A, b, m_init, mask = mask, maxiter=CG_maxiter, \
                                    tol=CG_tol, atol=CG_atol)
        else:
            m_out = rec.ImageRecon(A, b, m_init, maxiter=CG_maxiter, \
                                    tol=CG_tol, atol=CG_atol)     
        m_est = mask*m_out[-1]
        # m_est_rmse = mtc.evalPE(m_est, m_GT, mask)
        # m_est_ssim = mtc.evalSSIM(m_est, m_GT, mask=mask)
        # m_loss_store.append([m_est_rmse, m_est_ssim])
        # print("RMSE of Corrupted Image: {:.2f} %".format(m_est_rmse))
        # print("SSIM of Corrupted Image: {}".format(m_est_ssim))
        #
        #Saving output
        im = gen_img(m_est, transpose, 'axial', vmax)
        # fnt = ImageFont.truetype(size = 10)
        # d = ImageDraw.Draw(im)
        # d.text((5, 5), "Iteration {}".format(iter+1), font = fnt, fill = (255))
        im.save(spath + r"/Iteration{}.png".format(iter))
    return spath, m_corrupted, m_est, m_loss_store, Mtraj_store

#%% Run main()
if __name__ == "__main__":
    # mpath = r'/home/nghiemb/PyMoCo/data/cc/test/combo/severe_cases/additional_n49'
    mpath = r'/home/nghiemb/PyMoCo/data/cc/test/combo/extreme_cases'
    i = 5
    print('Processing Test Case {}'.format(i))
    dpath = os.path.join(mpath, 'Test{}'.format(i))
    spath_root = dpath
    spath, m_corrupted, m_final, m_loss_store, Mtraj_store = main(dpath, spath_root, mpath)
    xp.save(spath + r"/m_corrupted.npy", m_corrupted)
    xp.save(spath + r"/m_final.npy", m_final)
    xp.save(spath + r"/m_loss_store.npy", m_loss_store)
    xp.save(spath + r"/Mtraj_store.npy", Mtraj_store)

