# import numpy as np
import jax.numpy as xp
import scipy.io
import math
import glob
import re

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import model_from_json

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#-------------------------------------------------------------------------------
#Helper functions
def slice_mode(array, mode): #for generating datasets for adjacent
    if mode == 'current':
        array = array[:,1:-1,:,:]
    elif mode == 'before':
        array = array[:,:-2,:,:]
    elif mode == 'after':
        array = array[:,2:,:,:]
    return array

def truncate_dat(m_out, flag1 = 0, flag2 = 0):
	if m_out.shape[0] == 180: #hard-coding LR dimension for Calgary-Campinas dataset
		m_out = m_out[5:-5,...]; flag1 = 1 #choosing to crop outlier 180-LR dim to 170
	if m_out.shape[1] > 224: #hard-coding RO dimension for Calgary-Campinas dataset
		offset = (m_out.shape[1] - 224)//2
		m_out = m_out[:,offset:-offset,:]; flag2 = 1
	return m_out.astype('float32'), flag1, flag2 #single precision to minimize mem

def vol2slice(array): #transform array of volumes to AXIAL slices
    array = xp.transpose(array, axes = (0,3,1,2)) #Nsubjects, SI, LR, AP
    array = array.reshape((array.shape[0] * array.shape[1],array.shape[2], array.shape[3]))
    return array

def gen_AdjSlice(array, shape_val = (1,256,192,224), mode = 'current'):
    array_reshape = array.reshape(shape_val)
    array_crop = slice_mode(array_reshape, mode)
    array_slices = array_crop.reshape((array_crop.shape[0]*array_crop.shape[1],\
									   array_crop.shape[2], array_crop.shape[3]))
    return array_slices

def pad_dat(array, pad_x, pad_y):
    array_pad = xp.pad(array, ((0,0), (pad_x,pad_x), (pad_y,pad_y)))
    return array_pad

#------------------------------------------
def _preprocess(m_in, pads, shape_val = (1,256,192,224)):
	#shape_val: (nsubjects, SI, LR, AP)
	# m_trunc, flag1, flag2 = truncate_dat(abs(m_in))
	m_trunc, flag1, flag2 = truncate_dat(m_in) #for complex-valued UNet, want to preserve sign
	m_slices = vol2slice(m_trunc[None,...]) #transform volumes to stack of slices
	m_pad = pad_dat(m_slices, pads[0], pads[1])[..., None] #Need to add 4th dimension
	shape_val = m_pad.shape; shape_val = (shape_val[3], shape_val[0], shape_val[1], shape_val[2])
	slices_current = gen_AdjSlice(m_pad, shape_val = shape_val, mode = 'current')[...,None]
	slices_after = gen_AdjSlice(m_pad, shape_val = shape_val, mode = 'after')[...,None]
	slices_before = gen_AdjSlice(m_pad, shape_val = shape_val, mode = 'before')[...,None]
	return slices_current, slices_after, slices_before, flag1, flag2

def _postprocess(array_in, pads, flag1, flag2):
	array_3d = array_in[...,0] #remove 4th dim
	pad_x_init = pads[0]; pad_x_final = array_3d.shape[1] - pads[0]
	pad_y_init = pads[1]; pad_y_final = array_3d.shape[2] - pads[1]
	array_unpad = array_3d[:,pad_x_init:pad_x_final, pad_y_init:pad_y_final]
	array_out = xp.transpose(array_unpad, axes=(1,2,0))
	array_out = xp.pad(array_out, ((0,0), (0,0), (1,1)))
	if flag1: #TO DO - currently hardcoded for Calgary-Campinas dataset
		array_out = xp.pad(array_out, ((5,5), (0,0), (0,0)))
	if flag2: #TO DO - currently hardcoded for Calgary-Campinas dataset
		array_out = xp.pad(array_out, ((0,0), (16,16), (0,0)))
	return array_out

#------------------------------------------
def load_model(path_weight, md = 'lstm'):
	json_file = open(path_weight+r"/model_"+md+".json", 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	loaded_model.load_weights(path_weight+r"/model_"+md+".h5")
	print("Loaded model from disk")
	return loaded_model

#-------------------------------------------------------------------------------
def main(m_in, pads, weights_path, complex_flag = 0):
	#NB. Takes m_in as [LR, AP, SI]
	print('Reading Data ... ')
	#Load corrupted
	shape_val = (1, m_in.shape[2], m_in.shape[0], m_in.shape[1])
	# scale = abs(m_in).flatten().max() #Need to scale input st max val = 1
	scale = 1 #TEMP
	if complex_flag: #UPDATE April 2025, patch for complex-valued UNet; need to rewrite more efficiently
		test_current_re, test_after_re, test_before_re, flag1_re, flag2_re = _preprocess(m_in[..., 0] / scale, pads, shape_val)
		test_current_im, test_after_im, test_before_im, flag1_im, flag2_im = _preprocess(m_in[..., 1] / scale, pads, shape_val)
		test_current = xp.concatenate((test_current_re[..., None], test_current_im[..., None]), axis = 3)
		test_after = xp.concatenate((test_after_re[..., None], test_after_im[..., None]), axis = 3)
		test_before = xp.concatenate((test_before_re[..., None], test_before_im[..., None]), axis = 3)
	else:
		test_current, test_after, test_before, flag1, flag2 = _preprocess(m_in / scale, pads, shape_val)
	#---------------------------------------------------------------------------
	# Load the model
	print('Loading Model Weights')
	model = load_model(weights_path, 'CorrectionUNet_')
	print('---------------------------------')
	print('Evaluate Model on Testing Set ...')
	print('---------------------------------')
	m_pred = model.predict([test_before, test_current, test_after])
	#---------------------------------------------------------------------------
	#Post process the output (unpad, reshape)
	if complex_flag: #UPDATE April 2025, patch for complex-valued UNet; need to rewrite more efficiently
		m_corrected_re = _postprocess(m_pred[..., 0:1] * scale, pads, flag1_re, flag2_re)
		m_corrected_im = _postprocess(m_pred[..., 1:2] * scale, pads, flag1_im, flag2_im)
		m_corrected = xp.concatenate((m_corrected_re[..., None], m_corrected_im[..., None]), axis = 3)
	else:
		m_corrected = _postprocess(m_pred * scale, pads, flag1, flag2)
	#
	print('---------------------------------')
	print('Inference Completed')
	print('---------------------------------')
	return m_corrected
