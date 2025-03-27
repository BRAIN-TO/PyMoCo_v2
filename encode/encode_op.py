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

import motion.motion_sim as msi

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
def Encode(input, C, U, Mtraj, res, R_pad = (0,0,0), batch = 1):
    '''
    Defining signal encoding model, with rotation and translation operators
    Parallelized across shots; batch size needs to be tuned to memory availability
    IN: image (m), coil sensitivities (C), undersampling mask (U), motion trajectory (Mtraj)
    OUT: signal (s)
    '''
    nshots = len(U)
    s_out = xp.zeros(C.shape, dtype = C.dtype)
    for n in range(nshots):
        print("Shot {}".format(n+1), end='\r')
        U_n = msi._gen_U_n(U[n], input.shape)
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
    m_out = xp.zeros(C.shape[1:], dtype = C.dtype)
    for n in range(nshots):
        print("Shot {}".format(n+1), end='\r')
        U_n = msi._gen_U_n(U[n], input.shape[1:])
        m_out += _EH_n(U_n, Mtraj[n,3:], Mtraj[n,:3], input, C, res, R_pad)
    return m_out

#%%-----------------------------------------------------------------------------
# @partial(jit, static_argnums = (6,)) #R_pad is static argument due to explicit ref in pad / unpad
def _EH_E(input, C=None, U=None, Mtraj=None, res=None, lamda = 0, R_pad = (0,0,0), batch = 1):
    '''Applying EHE, for use in recon.ImageRecon (CG SENSE)'''
    nshots = len(U)
    s_out = xp.zeros(C.shape, dtype = C.dtype)
    for n in range(nshots):
        U_n = msi._gen_U_n(U[n], input.shape)
        s_out += _E_n(U_n, Mtraj[n,3:], Mtraj[n,:3], input, C, res, R_pad)
    m_out = xp.zeros(C.shape[1:], dtype = C.dtype)
    for n in range(nshots):
        U_n = msi._gen_U_n(U[n], s_out.shape[1:])
        m_out += _EH_n(U_n, Mtraj[n,3:], Mtraj[n,:3], s_out, C, res, R_pad)
    #
    return m_out + lamda * xp.ones(m_out.shape, dtype=m_out.dtype)
