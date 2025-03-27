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
        U = seq_order(U_sum, m, Rs, TR_shot, nshots, mode, PE_indices)
    elif order == "interleaved":
        U = int_order(U_sum, m, Rs, TR_shot, nshots, mode, PE_indices)
    else:
        warnings.warn("Error: sampling order not yet implemented; defaulting to sequential order")
        U = seq_order(U_sum, m, Rs, TR_shot, nshots, mode, PE_indices)
    #
    return U
