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
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="0" #turn off GPU pre-allocation
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

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
mpath = r'/home/nghiemb/PyMoCo' ##CHANGE TO MAIN WORKING DIRECTORY
dpath = r'/home/nghiemb/PyMoCo/data' ##CHANGE TO DIRECTORY OF CLEAN DATA
cnn_path = mpath + r'/cnn/3DUNet_SAP'
spath = cnn_path + r'/weights/PE1_AP/Complex/{}/train_n360'.format('combo')
m_files = sorted(os.listdir(os.path.join(dpath,'m_complex'))) #alphanumeric order
C_files = sorted(os.listdir(os.path.join(dpath, 'sens')))

nsims = 2 # number of motion simulations per subject per motion level
IQM_store = []
#-------------------------------------------------------------------------------
t1 = time()
count = 1
for i in range(len(m_files)):
    t3 = time()
    print("Subject {}".format(str(i+1)))
    #---------------------------------------------------------------------------
    #Load data
    m_fname = os.path.join(dpath,'m_complex',m_files[i])
    C_fname = os.path.join(dpath,'sens',C_files[i])
    m_GT = xp.load(m_fname)
    C = xp.load(C_fname); C = xp.transpose(C, axes = (3,2,0,1))
    res = xp.array([1,1,1])
    #
    m_GT = m_GT / xp.max(abs(m_GT.flatten())) #rescale
    mask = rec.getMask(C)
    plib.Path(os.path.join(dpath,'mask')).mkdir(parents=True, exist_ok=True)
    mask_name = os.path.join(dpath,'mask','mask_' + m_files[i][10:])
    xp.save(mask_name, mask)
    #---------------------------------------------------------------------------
    #Sampling pattern, for Calgary-Campinas brain data (12 coils, [PE:218,RO:256,SL:170])
    PE1 = m_GT.shape[0] #LR
    PE2 = m_GT.shape[1] #AP
    RO = m_GT.shape[2] #SI
    #
    Rs = 1
    TR_shot = 16
    order = 'interleaved'
    U_array = xp.transpose(msi.make_samp(xp.transpose(m_GT, (1,0,2)), \
                                    Rs, TR_shot, order=order), (0,2,1,3)).astype('int16')
    U = eop._U_Array2List(U_array, m_GT.shape)
    #---------------------------------------------------------------------------
    #Generate motion trace
    for j, motion_lv in enumerate(motion_lv_list):
        for k in range(nsims):
            print("Sim {} for Subject {}".format(str(j+1), str(i+1)))
            rand_keys = _gen_key(i, j, k)
            Mtraj_GT = _gen_traj(rand_keys, motion_lv, len(U), motion_specs)
            Mtraj_init = xp.zeros((len(U), 6))
            #-------------------------------------------------------------------
            R_pad = (10,10,10) #zero-pad image before rotations to prevent wrap-arounds; pads are automatically removed after rotations
            batch = 1
            s_corrupted = eop.Encode(m_GT, C, U, Mtraj_GT, res, batch=batch)
            #-------------------------------------------------------------------
            #Reconstruct image via EH, since data is fully-sampled
            m_corrupted = eop.Encode_Adj(s_corrupted, C, U, Mtraj_init, res, batch=batch)
            m_corrupted_PE = mtc.evalPE(m_corrupted, m_GT, mask=mask)
            m_corrupted_SSIM = mtc.evalSSIM(m_corrupted, m_GT, mask=mask)
            m_corrupted_loss = [m_corrupted_PE, m_corrupted_SSIM]
            IQM_store.append(m_corrupted_loss)
            xp.save(spath + r'/IQM_store.npy', IQM_store)
            #
            #save the filename, motion trajectory, simulated k-space and image
            output = [m_files[i], Mtraj_GT, s_corrupted, m_corrupted, m_corrupted_loss, U]
            s_path_temp = os.path.join(spath, motion_lv)
            plib.Path(s_path_temp).mkdir(parents=True, exist_ok=True)
            xp.save(s_path_temp + r'/train_dat{}.npy'.format((i+1)+(k*67)), output)
    t4 = time()
    print("Time elapsed for Subject {}: {} sec".format(str(i+1), str(t4 - t3)))
    #
#

print("Finished simulating training data")
t2 = time()
print("Total elapsed time: {} min".format(str((t2 - t1)/60)))



