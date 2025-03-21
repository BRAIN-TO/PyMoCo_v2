import jax.numpy as xp
import tensorflow as tf

# import skimage as sk
# import skimage.metrics as skiqm

#----------------------------------------------------------
def evalRMSE(m, m_gt):
    dif2 = (xp.abs(m.flatten()) - xp.abs(m_gt.flatten()))**2
    return xp.sqrt(xp.mean(dif2))

def evalPE(m, m_gt, mask=None): #percent error
    if mask != None:
        m *= mask; m_gt *= mask
    #
    return 100*(evalRMSE(m, m_gt) / evalRMSE(m_gt, xp.zeros(m_gt.shape)))

def evalRMSE_ROI(m, m_gt, mask):
    dif2 = (xp.abs(m.flatten()) - xp.abs(m_gt.flatten()))**2
    dif2_ROI = dif2[xp.where(mask.flatten() == 1)]
    return xp.sqrt(xp.mean(dif2_ROI))

def evalPE_ROI(m, m_gt, mask): #percent error
    RMSE_1 = evalRMSE_ROI(m, m_gt, mask)
    RMSE_2 = evalRMSE_ROI(m_gt, xp.zeros(m_gt.shape), mask)
    return 100*(RMSE_1 / RMSE_2)

#----------------------------------------------------------
def evalSSIM(m, m_gt, mask=None): #percent error
    max_val = abs(m).flatten().max()
    if mask != None:
        m *= mask; m_gt *= mask
    #
    return xp.mean(tf.image.ssim(abs(m_gt), abs(m), max_val).numpy())

#----------------------------------------------------------
def bounding_box(mask, tol = 1e-2):
    x_proj = xp.where(xp.sum(mask, axis = (1,2))<tol,0,1) #binarized projection
    y_proj = xp.where(xp.sum(mask, axis = (0,2))<tol,0,1)
    z_proj = xp.where(xp.sum(mask, axis = (0,1))<tol,0,1)
    #
    x_inds = xp.where(x_proj == 1)[0]
    y_inds = xp.where(y_proj == 1)[0]
    z_inds = xp.where(z_proj == 1)[0]
    #
    x_min = x_inds[0]; x_max = x_inds[-1]
    y_min = y_inds[0]; y_max = y_inds[-1]
    z_min = z_inds[0]; z_max = z_inds[-1]
    return x_min, x_max, y_min, y_max, z_min, z_max


def evalSSIM_bbox(m, m_gt, mask, tol = 1e-2):
    bbox = bounding_box(mask, tol)
    m_bbox = m[bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]]
    m_GT_bbox = m_gt[bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]]
    return evalSSIM(m_bbox, m_GT_bbox)