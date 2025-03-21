"""
Defining motion-aware signal encoding model
Subclassing sigpy.Linop
"""

import jax
import jax.numpy as xp
from jax.numpy.fft import fftshift, ifftshift, fftn, ifftn
from jax import vmap, grad, jit
from jax.lax import fori_loop
from functools import partial
from itertools import zip_longest
import numpy as np

#%%-----------------------------------------------------------------------------
#--------------------------------HELPER FUNCTIONS-------------------------------
#%%-----------------------------------------------------------------------------

def _farray(ishape, res, axis):
    '''Compute kspace coordinates along axis'''
    return xp.fft.fftshift(xp.fft.fftfreq(ishape[axis], d = res[axis]))

def _fgrid(ishape, res):
    '''Compute grid of kspace coordinates'''
    fx_array = _farray(ishape, res, 0)
    fy_array = _farray(ishape, res, 1)
    fz_array = _farray(ishape, res, 2)
    fx_grid, fy_grid, fz_grid = xp.meshgrid(fx_array, fy_array, fz_array, indexing = 'ij')
    return fx_grid, fy_grid, fz_grid

def _iarray(ishape, res, axis):
    '''Compute image-space coordinates along axis'''
    return xp.arange(-ishape[axis]//2, ishape[axis]//2)*res[axis]

def _igrid(ishape, res):
    '''Compute image-space coordinates'''
    x_array = _iarray(ishape, res, 0)
    y_array = _iarray(ishape, res, 1)
    z_array = _iarray(ishape, res, 2)
    x_grid, y_grid, z_grid = xp.meshgrid(x_array, y_array, z_array, indexing = 'ij')
    return x_grid, y_grid, z_grid

def _fft(input, axes):
    in_shift = xp.fft.ifftshift(input, axes = axes)
    in_fft = xp.fft.fftn(in_shift, axes = axes, norm = "ortho")
    return xp.fft.fftshift(in_fft, axes = axes)

def _ifft(input, axes):
    in_shift = xp.fft.ifftshift(input, axes = axes)
    in_ifft = xp.fft.ifftn(in_shift, axes = axes, norm = "ortho")
    return xp.fft.fftshift(in_ifft, axes = axes)

#%%-----------------------------------------------------------------------------
#------------------------------TRANSLATION OPERATOR-----------------------------
#%%-----------------------------------------------------------------------------
#Helper functions for translation
def _phaseRamp(D, fgrid, axis):
    '''Compute phase ramp for given axis'''
    phase = fgrid[axis] * D[axis]
    return xp.exp(-2j*xp.pi*phase)

def _trans1D(D, fgrid, axis, input):
    '''Apply translation (phase ramp) for given axis'''
    ramp = _phaseRamp(D, fgrid, axis)
    ft = _fft(input, (axis,))
    pm = ft * ramp
    out = _ifft(pm, (axis,))
    return out

#-------------------------------------------------------------------------------
# @jit
def Translate(input, D, res, mode='fwd'):
    '''Implementing 3D Translation via k-space Linear Phase Ramps (Cordero-Grande et al, 2016)'''
    fgrid = _fgrid(input.shape, res)
    if mode == 'inv':
        D = -D
    Tx = _trans1D(D, fgrid, 0, input)
    TyTx = _trans1D(D, fgrid, 1, Tx)
    TzTyTx = _trans1D(D, fgrid, 2, TyTx)
    return TzTyTx

#%%-----------------------------------------------------------------------------
#-------------------------------ROTATION OPERATOR-------------------------------
#%%-----------------------------------------------------------------------------
#Helper functions for rotation
def _pad(input, pad):
    '''Add symmetric padding to input'''
    output = xp.pad(input, ((pad[0],pad[0]), (pad[1],pad[1]), (pad[2],pad[2])))
    return output

def _unpad_inds(pad):
    '''Output start and stop slice indices for unpad'''
    start = pad
    if pad == 0:
        stop = None
    else:
        stop = -pad
    return start, stop

def _unpad(input, pad):
    '''Remove input's symmetric padding'''
    indx_start, indx_stop = _unpad_inds(pad[0])
    indy_start, indy_stop = _unpad_inds(pad[1])
    indz_start, indz_stop = _unpad_inds(pad[2])
    output = input[indx_start:indx_stop, indy_start:indy_stop, indz_start:indz_stop]
    return output

def _deg2rad(val): #convert to rad
    return val * (xp.pi / 180)

def _phase_tan(R_i, fgrid_i, igrid_i):
    phase = -xp.tan(_deg2rad(R_i/2)) * xp.multiply(fgrid_i, igrid_i)
    return xp.exp(-2j*xp.pi*phase)

def _phase_sin(R_i, fgrid_i, igrid_i):
    phase = xp.sin(_deg2rad(R_i)) * xp.multiply(fgrid_i, igrid_i)
    return xp.exp(-2j*xp.pi*phase)

def _shear_tan(R_i, fgrid_i, igrid_i, tan_axis, input):
    #Compute nonlinear phase ramp for shearing along given axis
    phase = _phase_tan(R_i, fgrid_i, igrid_i)
    ft = _fft(input, (tan_axis,))
    pm = ft * phase
    out = _ifft(pm, (tan_axis,))
    return out

def _shear_sin(R_i, fgrid_i, igrid_i, sin_axis, input):
    #Compute nonlinear phase ramp for shearing along given axis
    phase = _phase_sin(R_i, fgrid_i, igrid_i)
    ft = _fft(input, (sin_axis,))
    pm = ft * phase
    out = _ifft(pm, (sin_axis,))
    return out

def _rot1D(R, axis, fgrids, igrids, axes, input): #3-pass shear decomposition
    R_i = R[axis]
    S_tan1 = _shear_tan(R_i, fgrids[axes[0]], igrids[axes[1]], axes[0], input)
    S_sin = _shear_sin(R_i, fgrids[axes[1]], igrids[axes[0]], axes[1], S_tan1)
    S_tan2 = _shear_tan(R_i, fgrids[axes[0]], igrids[axes[1]], axes[0], S_sin)
    return S_tan2

#-------------------------------------------------------------------------------
# @partial(jit, static_argnums = (3,))
def Rotate(input, R, res, pad, mode='fwd'):
    '''Implementing 9-Pass Shear Decomposition of 3D Rotation (Unser et al, 1995)'''
    m_pad = _pad(input, pad)
    fgrids = _fgrid(m_pad.shape, res)
    igrids = _igrid(m_pad.shape, res)
    if mode=='fwd':
        Rx = _rot1D(R, 0, fgrids, igrids, [1,2], m_pad)
        RyRx = _rot1D(R, 1, fgrids, igrids, [2,0], Rx)
        RzRyRx = _rot1D(R, 2, fgrids, igrids, [0,1], RyRx)
        out = _unpad(RzRyRx, pad)
    elif mode=='inv': #reverse order of rotation application (rotations aren't commutative)
        Rz = _rot1D(-R, 2, fgrids, igrids, [0,1], m_pad)
        RyRz = _rot1D(-R, 1, fgrids, igrids, [2,0], Rz)
        RxRyRz = _rot1D(-R, 0, fgrids, igrids, [1,2], RyRz)
        out = _unpad(RxRyRz, pad)
    #
    return out

#%%-----------------------------------------------------------------------------
#--------------------------------ENCODING MODEL---------------------------------
#%%-----------------------------------------------------------------------------
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

def _E_n(U_n, R_n, T_n, m=None, C=None, res=None, R_pad=None): #Apply forward encoding operator for single shot
    #Apply FWD encoding
    Rm = Rotate(m, R_n, res, R_pad)
    TRm = Translate(Rm, T_n, res)
    CTRm = C * TRm
    FCTRm = _fft(CTRm, (1,2,3))
    s_n = U_n * FCTRm
    return s_n

def _EH_n(U_n, R_n, T_n, s=None, C=None, res=None, R_pad=None): #Apply inverse encoding operator for single shot
    #Apply INV
    Us = xp.conj(U_n)*s
    FUs = _ifft(Us, (1,2,3))
    CFUs = xp.sum(xp.conj(C) * FUs, axis = 0)
    TCFUs = Translate(CFUs, T_n, res, mode='inv')
    m_n = Rotate(TCFUs, R_n, res, R_pad, mode='inv')
    return m_n

def _Omega(U_n, m):
    #Retrieves subimage recon of given k-space shot
    Fm = _fft(m, (0,1,2))
    UFm = U_n * Fm
    IFUFm = _ifft(UFm, (0,1,2))
    return IFUFm

def _E_n_alt(U_n, R_n, T_n, m=None, C=None, res=None, R_pad=None): #Apply forward encoding operator for single shot
    #Apply FWD encoding
    Om = _Omega(U_n, m) #subimage recon of n-th k-space shot
    ROm = Rotate(Om, R_n, res, R_pad)
    TROm = Translate(ROm, T_n, res)
    CTROm = C * TROm
    s_n = _fft(CTROm, (1,2,3))
    return s_n


# @partial(jit, static_argnums = (5,)) #R_pad is static argument due to explicit ref in pad / unpad
# def _E_vmap(input, C, res, U_n, R_n, R_pad, T_n):
#     fmap = vmap(_E_n, in_axes = (0, 0, 0, None, None, None, None), out_axes = 0)
#     return fmap(U_n, R_n, T_n, input, C, res, R_pad)

# @partial(jit, static_argnums = (5,)) #R_pad is static argument due to explicit ref in pad / unpad
# def _EH_vmap(input, C, res, U_n, R_n, R_pad, T_n):
#     fmap = vmap(_EH_n, in_axes = (0, 0, 0, None, None, None, None), out_axes = 0)
#     return fmap(U_n, R_n, T_n, input, C, res, R_pad)

# @partial(jit, static_argnums = (5,)) #R_pad is static argument due to explicit ref in pad / unpad
def Encode(input, C, U, Mtraj, res, R_pad = (0,0,0), batch = 1):
    '''
    Defining signal encoding model, with rotation and translation operators
    Parallelized across shots; batch size needs to be tuned to memory availability
    IN: image (m), coil sensitivities (C), undersampling mask (U), motion trajectory (Mtraj)
    OUT: signal (s)
    '''
    nshots = len(U)
    #Vectorize U, R, T input for E
    # U_vmap = xp.array_split(U, len(U) // batch, axis = 0)
    # R_vmap = xp.array_split(Mtraj[:, 3:], Mtraj.shape[0] // batch, axis = 0)
    # T_vmap = xp.array_split(Mtraj[:, :3], Mtraj.shape[0] // batch, axis = 0)
    # #Run vectorized E
    # s_out = xp.zeros(C.shape, dtype = C.dtype)
    # for (U_n, R_n, T_n) in zip(U_vmap, R_vmap, T_vmap):
    #     s_vmap = _E_vmap(input, C, res, U_n, R_n, R_pad, T_n)
    #     s_out += xp.sum(s_vmap, axis=0)
    # #
    s_out = xp.zeros(C.shape, dtype = C.dtype)
    for n in range(nshots):
        U_n = _gen_U_n(U[n], input.shape)
        s_out += _E_n(U_n, Mtraj[n,3:], Mtraj[n,:3], input, C, res, R_pad)
    return s_out

# @partial(jit, static_argnums = (5,)) #R_pad is static argument due to explicit ref in pad / unpad
def Encode_Adj(input, C, U, Mtraj, res, R_pad = (0,0,0), batch = 1):
    '''
    Defining adjoint of signal encoding model, with rotation and translation operators
    Parallelized across shots; batch size needs to be tuned to memory availability
    IN: signal (s), coil sensitivities (C), undersampling mask (U), motion trajectory (Mtraj)
    OUT: image (m)
    '''
    nshots = len(U)
    # #Vectorize U, R, T input for EH
    # U_vmap = xp.array_split(U, U.shape[0] // batch, axis = 0)
    # R_vmap = xp.array_split(Mtraj[:, 3:], Mtraj.shape[0] // batch, axis = 0)
    # T_vmap = xp.array_split(Mtraj[:, :3], Mtraj.shape[0] // batch, axis = 0)
    # #Run vectorized EH
    # m_out = xp.zeros(C.shape[1:], dtype = C.dtype)
    # for (U_n, R_n, T_n) in zip(U_vmap, R_vmap, T_vmap):
    #     m_vmap = _EH_vmap(input, C, res, U_n, R_n, R_pad, T_n)
    #     m_out += xp.sum(m_vmap, axis=0)
    # #
    m_out = xp.zeros(C.shape[1:], dtype = C.dtype)
    for n in range(nshots):
        U_n = _gen_U_n(U[n], input.shape[1:])
        m_out += _EH_n(U_n, Mtraj[n,3:], Mtraj[n,:3], input, C, res, R_pad)
    return m_out

#%%-----------------------------------------------------------------------------
# @partial(jit, static_argnums = (6,)) #R_pad is static argument due to explicit ref in pad / unpad
def _EH_E(input, C=None, U=None, Mtraj=None, res=None, lamda = 0, R_pad = (0,0,0), batch = 1):
    '''Applying EHE, for use in recon.ImageRecon (CG SENSE)'''
    nshots = len(U)
    #Vectorize U, R, T input for E
    # U_vmap = xp.array_split(U, len(U) // batch, axis = 0)
    # R_vmap = xp.array_split(Mtraj[:, 3:], Mtraj.shape[0] // batch, axis = 0)
    # T_vmap = xp.array_split(Mtraj[:, :3], Mtraj.shape[0] // batch, axis = 0)
    # #Run vectorized E
    # s_out = xp.zeros(C.shape, dtype = C.dtype)
    # for (U_n, R_n, T_n) in zip(U_vmap, R_vmap, T_vmap):
    #     s_vmap = _E_vmap(input, C, res, U_n, R_n, R_pad, T_n)
    #     s_out += xp.sum(s_vmap, axis=0)
    # #
    # #
    s_out = xp.zeros(C.shape, dtype = C.dtype)
    for n in range(nshots):
        U_n = _gen_U_n(U[n], input.shape)
        s_out += _E_n(U_n, Mtraj[n,3:], Mtraj[n,:3], input, C, res, R_pad)
    # #Run vectorized EH
    # m_out = xp.zeros(input.shape[1:], dtype = input.dtype)
    # for (U_n, R_n, T_n) in zip(U_vmap, R_vmap, T_vmap):
    #     m_vmap = _EH_vmap(s_out, C, res, U_n, R_n, R_pad, T_n)
    #     m_out += xp.sum(m_vmap, axis=0)
    # #
    m_out = xp.zeros(C.shape[1:], dtype = C.dtype)
    for n in range(nshots):
        U_n = _gen_U_n(U[n], s_out.shape[1:])
        m_out += _EH_n(U_n, Mtraj[n,3:], Mtraj[n,:3], s_out, C, res, R_pad)
    #
    return m_out + lamda * xp.ones(m_out.shape, dtype=m_out.dtype)
