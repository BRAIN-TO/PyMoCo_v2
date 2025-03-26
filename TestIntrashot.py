"""
Generate Training Dataset with Simulated Motion Correction

Test data will be provided upon request
"""

import os
import pathlib as plib
from time import time
from functools import partial
import numpy as np

import jax
import jax.numpy as xp
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="0" #turn off GPU pre-allocation
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

os.environ['CUDA_VISIBLE_DEVICES'] = '' #TEMPORARY force to use CPU
print(jax.numpy.ones(3).device()) # TFRT_CPU_0

import encode.encode_op as eop
import recon.recon_op as rec
import cnn.run_unet as cnn
import utils.metrics as mtc
import motion.motion_sim as msi

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#Helper Functions
def _gen_traj_dof(rand_key, motion_lv, dof, nshots, motion_specs):
    '''
    Input:
        rand_key=jax.random.PRNGKey object,
        motion_lv={'mild','moderate','severe'},
        dof={'Tx','Ty','Tz','Rx','Ry','Rz'}
        nshots=int # of motion states
    Output:
        xp.array of motion trajectory for a given DOF
    '''
    p_val = motion_specs[motion_lv][dof][1]
    p_array = xp.array([p_val/2, 1-p_val, p_val/2])
    opts = xp.array([-1,0,1]) #move back, stay, move fwd
    maxval = motion_specs[motion_lv][dof][0]
    minval = maxval / 2
    array = jax.random.choice(rand_key, a = opts, shape=(nshots-1,), p = p_array) #binary array
    array = xp.concatenate((xp.array([0]), array)) #ensure first motion state is origin
    vals = jax.random.uniform(rand_key, shape=(nshots,),minval=minval, maxval=maxval) #displacements
    return xp.cumsum(array * vals) #absolute value of motion trajectory

def _gen_traj(rand_keys, motion_lv, nshots, motion_specs):
    '''
    Input:
        rand_key=jax.random.PRNGKey object,
        motion_lv={'mild','moderate','severe'},
        nshots=int # of motion states
    Output:
        xp.array of motion trajectory across all 6 DOFs
    '''
    out_array = xp.zeros((nshots, 6))
    out_array = out_array.at[:,0].set(_gen_traj_dof(rand_keys[0], motion_lv, 'Tx', nshots, motion_specs))
    out_array = out_array.at[:,1].set(_gen_traj_dof(rand_keys[1], motion_lv, 'Ty', nshots, motion_specs))
    out_array = out_array.at[:,2].set(_gen_traj_dof(rand_keys[2], motion_lv, 'Tz', nshots, motion_specs))
    out_array = out_array.at[:,3].set(_gen_traj_dof(rand_keys[3], motion_lv, 'Rx', nshots, motion_specs))
    out_array = out_array.at[:,4].set(_gen_traj_dof(rand_keys[4], motion_lv, 'Ry', nshots, motion_specs))
    out_array = out_array.at[:,5].set(_gen_traj_dof(rand_keys[5], motion_lv, 'Rz', nshots, motion_specs))
    return out_array

def _gen_seq(i,j,k,dof):
    a1 = (i+j+dof+1)**2 + (5*j)**2 + (17*i)**2 + (k*1206)**2 #including the exponent to guarantee different random value than training dataset
    return a1

def _gen_key(i, j, k):
    return [jax.random.PRNGKey(_gen_seq(i,j,k,dof)) for dof in range(6)]

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#Set up motion sim specs
# motion_lv_list = ['mild', 'moderate', 'severe']

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

motion_specs = {'mild':moderate_specs,'moderate':severe_specs1,\
                'large':severe_specs2,'extreme':severe_specs3}

motion_lv_list = ['moderate', 'large']

#-------------------------------------------------------------------------------
#-------------------------Image Acquisition Simulation--------------------------
#-------------------------------------------------------------------------------
#Load data
mpath = r'/home/nghiemb/PyMoCo' ####CHANGE TO MAIN WORKING DIRECTORY
dpath = r'/home/nghiemb/Data/CC' ####CHANGE TO DIRECTORY OF CLEAN DATA
cnn_path = mpath + r'/cnn/3DUNet_SAP'
spath = cnn_path + r'/weights/PE1_AP/Complex/{}/train_n360'.format('combo')
m_files = sorted(os.listdir(os.path.join(dpath,'m_complex'))) #alphanumeric order
C_files = sorted(os.listdir(os.path.join(dpath, 'sens')))

nsims = 2 # number of motion simulations per subject per motion level
IQM_store = []

#-------------------------------------------------------------------------------
t1 = time()
count = 1

i = 0

t3 = time()
print("Subject {}".format(str(i+1)))
#---------------------------------------------------------------------------
#Load data
m_fname = os.path.join(dpath,'m_complex',m_files[i])
C_fname = os.path.join(dpath,'sens',C_files[i])
m_GT = xp.load(m_fname)
C = xp.load(C_fname); C = xp.transpose(C, axes = (3,2,0,1))
res = xp.array([1,1,1])

m_GT = m_GT / xp.max(abs(m_GT.flatten())) #rescale
mask = rec.getMask(C)
# plib.Path(os.path.join(dpath,'mask')).mkdir(parents=True, exist_ok=True)
# mask_name = os.path.join(dpath,'mask','mask_' + m_files[i][10:])
# xp.save(mask_name, mask)
#---------------------------------------------------------------------------
#Sampling pattern, for Calgary-Campinas brain data (12 coils, [PE:218,RO:256,SL:170])
PE1 = m_GT.shape[0] #LR
PE2 = m_GT.shape[1] #AP
RO = m_GT.shape[2] #SI
#
Rs = 1
TR_shot = 16
order = 'interleaved'
# U_array = xp.transpose(msi.make_samp(xp.transpose(m_GT, (1,0,2)), \
#                                 Rs, TR_shot, order=order), (0,2,1,3)).astype('int16')
# U = eop._U_Array2List(U_array, m_GT.shape); del U_array
# np.save(r'/home/nghiemb/Data/CC/U_list_16TR.npy', U)

U = np.load(r'/home/nghiemb/Data/CC/U_list_16TR.npy', allow_pickle=1)

#---------------------------------------------------------------------------
#Generate sliding window
import itertools

def _U_Array2List(U, m_shape):
    U_list = []
    for i in range(U.shape[0]):
        RO_temp = xp.arange(0, m_shape[0])
        PE1_temp = xp.where(U[i,0,:,0] == 1)[0]
        PE2_temp = xp.arange(0, m_shape[2])
        U_list.append([RO_temp, PE1_temp, PE2_temp])
    return U_list

def _gen_U_n(U_vals, m_shape):
    #Lazy evaluation of sampling pattern
    U_RO = xp.zeros(m_shape[0]); U_RO = U_RO.at[U_vals[0]].set(1) 
    U_PE1 = xp.zeros(m_shape[1]); U_PE1 = U_PE1.at[U_vals[1]].set(1)
    U_PE2 = xp.zeros(m_shape[2]); U_PE2 = U_PE2.at[U_vals[2]].set(1)    
    return np.multiply.outer(U_RO, xp.outer(U_PE1, U_PE2))


PE1_combined_init = [list(U[i][1]) for i in range(len(U))]
PE1_combined = list(itertools.chain.from_iterable(PE1_combined_init))

strides = [0,8] #takes value between [0, 15]
# strides = [0,4,8,12] #takes value between [0, 15]
# strides = [0,2,4,6,8,10,12,14] #takes value between [0, 15]
PE1_sliding_window = []
for shot_nominal_iter in range(len(U)):
    for stride in strides:
        window_temp = PE1_combined[TR_shot*shot_nominal_iter+stride:TR_shot*(shot_nominal_iter+1)+stride]
        PE1_sliding_window.append(window_temp)

U_sliding_window = []
U_RO_vals = U[0][0]
U_PE2_vals = U[0][2]
for i in range(len(PE1_sliding_window)):
    U_PE1_vals = np.asarray(PE1_sliding_window[i])
    U_vals_temp = np.asarray([U_RO_vals, U_PE1_vals, U_PE2_vals])
    U_sliding_window.append(U_vals_temp)


# U_temp = _gen_U_n(U_sliding_window[9], m_GT.shape)


#---------------------------------------------------------------------------
#Loading in vivo data with continuous head motion
mpath_sub1 = r'/home/nghiemb/Data/TWH/MPRAGE_PE1Reordered/Scan20231204/Sub1/h5data'
dpath_sub1 = mpath_sub1 + r'/scan1-ContinuousMotion-16shots/npy'
gt_name_sub1 = r'scan1-ContinuousMotion-16shots/npy'
cerebrum_slice_sub1 = 190
paths_sub1 = [mpath_sub1, dpath_sub1, gt_name_sub1, cerebrum_slice_sub1]



s_corrupted = xp.load(dpath + r'/kdat_trunc.npy') #NC, SI, AP, LR
if sub == 4:
    C_init = xp.load(dpath + r'/sens.npy')
else:
    C_init = xp.load(mpath + r'/{}/sens.npy'.format(gt_name))
C = xp.transpose(C_init, (3,0,1,2))
mask = rec.getMask(C); xp.save(dpath + r'/m_GT_brain_mask.npy', mask)
del C_init
#---------------------------------------
U = np.load(dpath + r'/samp_order.npy', allow_pickle=1) #LR, AP, SI
#---------------------------------------------------------------------------
res = xp.array([1,1,1])
#---------------------------------------
# 
# maxval = abs(m_GT.flatten()).max()
# m_GT /= maxval
# s_corrupted /= maxval
try:
    m_GT = xp.load(mpath + r'/{}/img_CG.npy'.format(gt_name)) ###I THINK I'VE OVERWRITTEN IMG_CG, SO MAXVAL NOW = 1
except:
    m_GT = xp.ones(s_corrupted.shape[1:]) #BYPASSING LOADING M_GT
maxval = abs(m_GT.flatten()).max()
m_GT /= maxval
s_corrupted /= maxval
#---------------------------------------
#Loading the skull-stripping mask, generated from FreeSurfer SynthStrip tool
cerebrum_mask = xp.ones(m_GT.shape)
cerebrum_mask = cerebrum_mask.at[cerebrum_slice:,...].set(0)
#---------------------------------------
#Motion trajectory
R_pad = (10, 10, 10)
batch = 1

