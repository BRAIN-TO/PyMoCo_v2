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
def seq_order(U_sum,m,Rs,TR_shot,nshots):
    '''Sequential k-space sampling order'''
    U_seq = xp.zeros((nshots, m.shape[0], m.shape[1], m.shape[2]))
    for i in range(nshots):
        ind_start = i*(Rs*TR_shot)
        if i == (nshots-1):
            ind_end = -1
        else:
            ind_end = (i+1)*(Rs*TR_shot)
        val = U_sum[ind_start:ind_end,...]
        U_seq = U_seq.at[i,ind_start:ind_end,...].set(val)
        #
    #
    return U_seq

def int_order(U_sum,m,Rs,TR_shot,nshots):
    '''Interleaved k-space sampling order'''
    U_int = xp.zeros((nshots, m.shape[0], m.shape[1], m.shape[2]))
    for i in range(nshots):
        interval = Rs*nshots
        ind_start = i*Rs
        ind_end = ind_start + TR_shot*interval
        for j in range(ind_start,ind_end,interval):
            U_int = U_int.at[i,j,:,:].set(1)
        #
    #
    return U_int

def dis_order(m,Rs,tile_dims):
    '''Random checkered sampling order, adapted from Cordero-Grande et al 2020'''
    PE_dim, SL_dim, PE_inds, SL_inds = tile_dims
    RO_dim = m.shape[1]
    PE_red = PE_dim // Rs #reduced dim
    nshots = PE_red * SL_dim
    #
    order_init = xp.arange(PE_red * SL_dim)
    tile_init = xp.zeros(nshots)
    ##
    U_dis = xp.zeros((nshots, PE_dim * PE_inds, SL_dim * SL_inds))
    for i in range(PE_inds):
        PE_start = i * PE_dim; PE_end = (i+1) * PE_dim
        for j in range(SL_inds):
            SL_start = j * SL_dim; SL_end = (j+1) * SL_dim
            #
            key = jax.random.PRNGKey((j+1)*(i+1))
            order_new = jax.random.shuffle(key, order_init)
            for k in range(nshots):
                print("PE seg {}, SL seg {}, Shot {}".format(i+1, j+1, k+1), end = '\r')
                tile_new = tile_init.at[order_new[k]].set(1).reshape((PE_red, SL_dim))
                tile_out = xp.zeros((PE_dim, SL_dim))
                tile_out = tile_out.at[0,:].set(tile_new[0,:])
                tile_out = tile_out.at[2,:].set(tile_new[1,:])
                tile_out = tile_out.at[4,:].set(tile_new[2,:])
                U_dis = U_dis.at[k,PE_start:PE_end, SL_start:SL_end].set(tile_out)
            #
        #
    #
    U_dis = xp.repeat(U_dis[:,:,None,:], repeats = RO_dim, axis = 2)
    U_out = U_dis[:,:m.shape[0],:,:m.shape[2]]
    #
    return U_out

def make_samp(m, Rs, TR_shot, order='interleaved', tile_dims = None):
    #Base sampling pattern
    U_sum = xp.zeros(m.shape)
    U_sum = U_sum.at[::Rs,...].set(1) #cumulative sampling, with R = 2
    nshots = int(xp.round(m.shape[0]/(Rs*TR_shot)))
    #---------------------------------------------------------------------------
    #Generating different sampling orderings
    if order == "sequential":
        U = seq_order(U_sum, m, Rs, TR_shot, nshots)
    elif order == "interleaved":
        U = int_order(U_sum, m, Rs, TR_shot, nshots)
    elif order == "disorder":
        U = dis_order(m,Rs,tile_dims)
    else:
        warnings.warn("Error: sampling order not yet implemented; defaulting to sequential order")
        U = seq_order(U_sum, m, Rs, TR_shot, nshots)
    #
    return U
