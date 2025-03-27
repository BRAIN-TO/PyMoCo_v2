# -*- coding: utf-8 -*-
"""
Created on Sun Mar 7 9:15:00 2021

@author: brian
"""
"""Simulating Motion"""
import jax
import jax.numpy as xp
import warnings


#-------------------------------------------------------------------------------
#Functions for generating random motion trajectories
def _gen_traj_dof(rand_key, dof, nshots, motion_spec, specs_scale):
    '''
    Input:
        rand_key=jax.random.PRNGKey object,
        motion_lv={'mild','moderate','severe'},
        dof={'Tx','Ty','Tz','Rx','Ry','Rz'}
        nshots=int # of motion states
    Output:
        xp.array of motion trajectory for a given DOF
    '''
    p_val = motion_spec[dof][1]*specs_scale[1]
    p_array = xp.array([p_val/2, 1-p_val, p_val/2])
    opts = xp.array([-1,0,1]) #move back, stay, move fwd
    maxval = motion_spec[dof][0]*specs_scale[0]
    minval = maxval / 2
    array = jax.random.choice(rand_key, a = opts, shape=(nshots-1,), p = p_array) #binary array
    array = xp.concatenate((xp.array([0]), array)) #ensure first motion state is origin
    vals = jax.random.uniform(rand_key, shape=(nshots,),minval=minval, maxval=maxval) #displacements
    return xp.cumsum(array * vals) #absolute value of motion trajectory

def _gen_traj(rand_keys, nshots, motion_spec, specs_scale=[1,1]):
    '''
    Input:
        rand_key=jax.random.PRNGKey object,
        motion_lv={'mild','moderate','severe'},
        nshots=int # of motion states
    Output:
        xp.array of motion trajectory across all 6 DOFs
    '''
    out_array = xp.zeros((nshots, 6))
    out_array = out_array.at[:,0].set(_gen_traj_dof(rand_keys[0], 'Tx', nshots, motion_spec, specs_scale))
    out_array = out_array.at[:,1].set(_gen_traj_dof(rand_keys[1], 'Ty', nshots, motion_spec, specs_scale))
    out_array = out_array.at[:,2].set(_gen_traj_dof(rand_keys[2], 'Tz', nshots, motion_spec, specs_scale))
    out_array = out_array.at[:,3].set(_gen_traj_dof(rand_keys[3], 'Rx', nshots, motion_spec, specs_scale))
    out_array = out_array.at[:,4].set(_gen_traj_dof(rand_keys[4], 'Ry', nshots, motion_spec, specs_scale))
    out_array = out_array.at[:,5].set(_gen_traj_dof(rand_keys[5], 'Rz', nshots, motion_spec, specs_scale))
    return out_array

def _gen_seq(i,j,k,dof):
    a1 = (i+j+dof+1)**2 + (5*j)**2 + (17*i)**2 + (k*1206)**2 #including the exponent to guarantee different random value than training dataset
    return a1

def _gen_key(i, j, k):
    return [jax.random.PRNGKey(_gen_seq(i,j,k,dof)) for dof in range(6)]


#-------------------------------------------------------------------------------
#Functions for generating sampling pattern
def seq_order(U_sum,m,Rs,TR_shot,nshots,mode='array'):
    '''Sequential k-space sampling order'''
    if mode == 'array':
        U_seq = xp.zeros((nshots, m.shape[0], m.shape[1], m.shape[2]))
        for i in range(nshots):
            print("Shot {}".format(i))
            ind_start = i*(Rs*TR_shot)
            if i == (nshots-1):
                ind_end = -1
            else:
                ind_end = (i+1)*(Rs*TR_shot)
            val = U_sum[ind_start:ind_end,...]
            U_seq = U_seq.at[i,ind_start:ind_end,...].set(val)
            #
        #
    elif mode == 'list':
        U_seq = []
        for i in range(nshots):
            print("Shot {}".format(i))
            ind_start = i*(Rs*TR_shot)
            ind_end = (i+1)*(Rs*TR_shot)
            #
            RO_temp = xp.arange(0, m.shape[0])
            PE1_temp = xp.arange(ind_start, ind_end)
            PE2_temp = xp.arange(0, m.shape[2])
            U_seq.append([RO_temp, PE1_temp, PE2_temp])
            #
        #
    return U_seq

def int_order(U_sum,m,Rs,TR_shot,nshots,mode='array'):
    '''Interleaved k-space sampling order'''
    if mode == 'array':
        U_int = xp.zeros((nshots, m.shape[0], m.shape[1], m.shape[2]))
        for i in range(nshots):
            interval = Rs*nshots
            ind_start = i*Rs
            ind_end = ind_start + TR_shot*interval
            for j in range(ind_start,ind_end,interval):
                U_int = U_int.at[i,j,:,:].set(1)
            #
        #
    elif mode == 'list':
        U_int = []
        for i in range(nshots):
            interval = Rs*nshots
            ind_start = i*Rs
            ind_end = ind_start + TR_shot*interval
            RO_temp = xp.arange(0, m.shape[0])
            PE1_temp = xp.arange(ind_start, ind_end,interval)
            PE2_temp = xp.arange(0, m.shape[2])
            U_int.append([RO_temp, PE1_temp, PE2_temp])
            #
        #
    return U_int

def make_samp(m, Rs, TR_shot, order='interleaved', tile_dims = None, mode = 'array', PE_indices = [1,0]):
    #Base sampling pattern
    PE1_index, PE2_index = PE_indices
    U_sum = xp.zeros(m.shape)
    U_sum = U_sum.at[::Rs,...].set(1) #cumulative sampling, with R = 2
    nshots = int(xp.round(m.shape[PE1_index]/(Rs*TR_shot)))
    #---------------------------------------------------------------------------
    #Generating different sampling orderings
    if order == "sequential":
        U = seq_order(U_sum, m, Rs, TR_shot, nshots, mode)
    elif order == "interleaved":
        U = int_order(U_sum, m, Rs, TR_shot, nshots, mode)
    else:
        warnings.warn("Error: sampling order not yet implemented; defaulting to sequential order")
        U = seq_order(U_sum, m, Rs, TR_shot, nshots, mode)
    #
    return U


#-------------------------------------------------------------------------------
#Functions for altering existing sampling pattern
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

def _U_subdivide(U, dscale):
    #Subdivide U into finer temporal resolution
    U_temp = []
    for n in range(len(U)):
        RO_temp = U[n][0]
        PE2_temp = U[n][2]
        PE1_temp = U[n][1]
        for m in range(dscale):
            ind1 = m*PE1_temp.shape[0]//dscale
            ind2 = (m+1)*PE1_temp.shape[0]//dscale
            if len(PE1_temp[ind1:ind2])==0: #if exceeded number of PE1 steps in the shot
                pass
            else:
                U_temp.append([RO_temp, PE1_temp[ind1:ind2], PE2_temp])
        #
    return U_temp   

def _U_combine(U, upscale):
    U_temp = []
    upscale_inds = xp.arange(0,len(U), upscale)
    for i, ind in enumerate(upscale_inds):
        start = ind
        if i == len(upscale_inds)-1:
            end = len(U)
        else:
            end = ind+2
        RO_temp = U[i][0]
        PE2_temp = U[i][2]
        PE1_temp = []
        for j in range(start,end):
            PE1_temp.append(U[j][1])
        PE1_temp = xp.asarray(PE1_temp).flatten()
        U_temp.append([RO_temp, PE1_temp, PE2_temp])
    return U_temp