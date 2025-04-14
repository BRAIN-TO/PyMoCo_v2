"""
MAIN SCRIPT
Running Joint Motion and Image Estimation
"""
import os
import pathlib as plib
from time import time
from functools import partial

import jax.numpy as xp
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="0" #turn off GPU pre-allocation
# os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']='platform'
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# import jax.profiler

import encode.encode_op as eop
import recon.recon_op as rec
import cnn.run_unet as cnn
import utils.metrics as mtc

#-------------------------------------------------------------------------------
def main(dpath, spath_root, mpath, cnn_path_names):
    #---------------------------------------------------------------------------
    #-----------------------Image Acquisition Simulation------------------------
    #---------------------------------------------------------------------------
    #Load data
    s_corrupted = xp.load(dpath + r'/s_corrupted.npy') #NC, SI, AP, LR
    C = xp.load(dpath + r'/sens.npy')
    mask = rec.getMask(C)
    res = xp.array([1,1,1])
    U = xp.load(dpath + r'/U.npy')
    # m_GT = xp.load(r'/home/nghiemb/Data/TWH/MPRAGE_ReferencePoses/InVivo/sub-01/dat/reference/img_rss_trunc_shift.npy')
    m_GT = xp.load(dpath + r'/rss.npy')
    maxval = abs(m_GT.flatten()).max()
    m_GT /= maxval
    #Motion trajectory
    # Mtraj_GT = xp.load(dpath + r'/Mtraj_GT.npy')
    Mtraj_GT = xp.load(dpath + r'/Mtraj.npy')
    R_pad = (10, 10, 10)
    batch = 1
    #---------------------------------------------------------------------------
    #------------------JOINT IMAGE RECON AND MOTION ESTIMATION------------------
    #---------------------------------------------------------------------------
    #Initializing update vars
    Mtraj_init = xp.zeros((U.shape[0], 6))
    Mtraj_est = Mtraj_init
    CG_maxiter = 10 #limit CG_iter to 3 iters for fully-sampled data to prevent artifacts
    ME_maxiter = 1 #motion estimation maxiter
    LS_maxiter = 20 #line search maxiter for BFGS algorithm
    CG_tol = 1e-7 #relative tolerance
    CG_atol = 1e-4 #absolute tolerance
    CG_lamda = 0
    CG_mask = 0 #turn on for in-vivo dataset, turn off for CC dataset
    #Initialize stores
    m_loss_store = []
    m_cnn_store = []
    Mtraj_store = []
    # # #---------------------------------------------------------------------------
    # #Picking up algorithm
    # import numpy as np
    # m_loss_store = list(np.load(dpath + r'/{}/m_loss_store.npy'.format(cnn_path_names[-1]), allow_pickle=1))
    # # m_cnn_store = np.load(dpath + r'/{}/m_cnn_store.npy'.format(cnn_path_names[-1]), allow_pickle=1)
    # Mtraj_store_initial = list(np.load(dpath + r'/{}/Mtraj_store.npy'.format(cnn_path_names[-1]), allow_pickle=1))
    # Mtraj_est = Mtraj_store_initial[-1][0]
    #---------------------------------------------------------------------------
    #Reconstruct image using CG SENSE algorithm
    m_init = eop.Encode_Adj(s_corrupted, C, U, Mtraj_init, res, batch=batch) #E.H*s
    A = partial(eop._EH_E, C=C, U=U, Mtraj=Mtraj_est, res=res, \
                lamda = CG_lamda, batch=batch)
    b = eop.Encode_Adj(s_corrupted, C, U, Mtraj_est, res, batch=batch)
    if CG_mask:
        m_out = rec.ImageRecon(A, b, m_init, mask = mask, maxiter=CG_maxiter, \
                                tol=CG_tol, atol=CG_atol)
    else:
        m_out = rec.ImageRecon(A, b, m_init, maxiter=CG_maxiter, \
                                tol=CG_tol, atol=CG_atol)     
    m_corrupted = mask*m_out[-1]
    m_est_rmse = mtc.evalPE(m_corrupted, m_GT, mask)
    m_est_ssim = mtc.evalSSIM(m_corrupted, m_GT, mask=mask)
    m_loss_store.append([m_est_rmse, m_est_ssim])
    print("RMSE of Corrupted Image: {:.2f} %".format(m_est_rmse))
    print("SSIM of Corrupted Image: {}".format(m_est_ssim))
    m_est = m_corrupted
    # jax.profiler.save_device_memory_profile()
    #---------------------------------------------------------------------------
    #Loading trained CNN model
    # NB. UNet takes in data as [LR, AP, SI]
    # For my Ref Data (SI, AP, LR), need to transpose --> (2,1,0)
    cnn_path = r'/home/nghiemb/PyMoCo/cnn/3DUNet_SAP'
    # wpath_severe = cnn_path + r'/weights/{}/train_n134'.format('severe')
    # wpath_moderate = cnn_path + r'/weights/{}/train_n134'.format('moderate')
    # wpath_mild = cnn_path + r'/weights/{}/train_n134'.format('mild')
    wpath_severe = cnn_path + r'/weights/{}/train_n134'.format(cnn_path_names[0])
    wpath_moderate = cnn_path + r'/weights/{}/train_n134'.format(cnn_path_names[1])
    wpath_mild = cnn_path + r'/weights/{}/train_n134'.format(cnn_path_names[2])
    # wpath_severe = cnn_path + r'/weights/{}/train_n134'.format('combo')
    # wpath_moderate = cnn_path + r'/weights/{}/train_n134'.format('combo')
    # wpath_mild = cnn_path + r'/weights/{}/train_n134'.format('combo')
    # pad_x = int((xp.ceil(m_est.shape[2]/32) * 32 - m_est.shape[2])/2) #along LR
    # pad_y = int((xp.ceil(m_est.shape[1]/32) * 32 - m_est.shape[1])/2) #along AP
    # pads = [pad_x, pad_y]
    pads = [11,3]
    #---------------------------------------------------------------------------
    #Alternating image & motion estimation (coordinate descent)
    # rmse_tol = 3.0
    rmse_tol = 5.0
    # trans_axes = (2,1,0)
    trans_axes = (0,1,2)
    cnn_flag = 0 #turn on / off CNN
    JE_flag = 1 #turn JE algorithm on / off
    thresh = {'severe': 200, 'moderate': 50}
    if JE_flag and cnn_flag: #UNet + JE
        # spath = spath_root + r'/w_cnn_alt'
        # spath = spath_root + r'/w_cnn_mild_only'
        spath = spath_root + r'/{}'.format(cnn_path_names[-1])
        # max_loops = 100
        max_loops = 50
    elif JE_flag and not cnn_flag: #only JE
        spath = spath_root + r'/wo_cnn'
        # spath = spath_root + r'/wo_cnn_cntd' #continued for additional 2000 iterations
        max_loops = 2000 - len(m_loss_store)
    elif not JE_flag and cnn_flag: #only UNet
        # spath = spath_root + r'/w_only_cnn_combo'
        spath = spath_root + r'/{}'.format(cnn_path_names[-1])
        max_loops = 1
    plib.Path(spath).mkdir(parents=True, exist_ok=True)
    xp.save(spath + r'/m_corrupted.npy', m_corrupted)
    #
    JE_params = [m_est_rmse, rmse_tol, max_loops, ME_maxiter, LS_maxiter, \
                    CG_maxiter, CG_tol, CG_atol, CG_mask, batch, mask]
    CNN_params = [cnn_flag, JE_flag, trans_axes, pads, wpath_severe, wpath_moderate, wpath_mild, thresh]
    init_est = [m_est, Mtraj_est]
    fixed_vars = [m_init, s_corrupted, C, U, mask, res, spath, m_GT, R_pad]
    stores = [m_cnn_store, Mtraj_store, m_loss_store]
    m_est, m_loss_store, Mtraj_store, m_cnn_store = rec.JointEst(init_est, fixed_vars, \
                                                                    stores, cnn, \
                                                                    CNN_params, JE_params)
    #
    return spath, m_corrupted, m_est, m_loss_store, Mtraj_store

#%% Run main()
if __name__ == "__main__":
    # mpath = os.environ['M_DIR']
    # dpath = os.environ['IN_DIR']
    # spath = os.environ['OUT_DIR']
    mpath = r'/home/nghiemb/PyMoCo/data/cc/test/combo/severe_cases/additional_n49'
    # mpath = r'/home/nghiemb/PyMoCo/data/cc/test/combo/moderate_cases'
    # mpath = r'/home/nghiemb/PyMoCo/data/cc/test/combo/severe_cases'
    # mpath = r'/home/nghiemb/PyMoCo/data/cc/test/combo/extreme_cases'
    # cnn_path_store = [['mild', 'mild', 'mild', 'w_cnn_mild_only'], \
    #                     ['moderate', 'moderate', 'moderate', 'w_cnn_moderate_only'], \
    #                     ['severe', 'severe', 'severe', 'w_cnn_severe_only'], \
    #                     ['severe', 'moderate', 'mild', 'w_cnn_alt'], \
    #                     ['combo', 'combo', 'combo', 'w_cnn_combo']]
    cnn_path_store = [['severe', 'moderate', 'mild', 'wo_cnn']]
    base_case = [1,4,5,6,7]
    cases = []
    cases += [base_case[j]+(7*i) for i in range(7) for j in range(len(base_case))]
    # for i in range(1, 50):
    cases = [1]
    for cnn_path_names in cnn_path_store:
        for i in cases:
            print('Processing Test Case {}'.format(i))
            dpath = os.path.join(mpath, 'Test{}'.format(i))
            spath_root = dpath
            spath, m_corrupted, m_final, m_loss_store, Mtraj_store = main(dpath, spath_root, mpath, cnn_path_names)
            xp.save(spath + r"/m_corrupted.npy", m_corrupted)
            xp.save(spath + r"/m_final.npy", m_final)
            xp.save(spath + r"/m_loss_store.npy", m_loss_store)
            xp.save(spath + r"/Mtraj_store.npy", Mtraj_store)


'''
#Generating fully-sampled version of interleaved sampling pattern
#To send to Tim, who is developing Siemens UI for PE1 mods 

import motion.motion_sim as msi
import numpy as np

mpath = r'/home/nghiemb/PyMoCo/data/cc/test/combo/severe_cases/additional_n49'
i = 1
dpath = os.path.join(mpath, 'Test{}'.format(i))
spath_root = dpath

PE1 = 256
m = np.zeros((PE1, 224, 192))

Rs = 1
TR_shot = 32
order = 'interleaved'
U = msi.make_samp(m, Rs, TR_shot, order)

#Exporting the matrix size and PE1 order
#To be sent to Tim
mat_size = np.array([m.shape[2], m.shape[0], m.shape[1]])

inds = []
for i in range(len(U)):
    U_i = U[i]
    inds.append(np.where(U_i[:,0,0] == 1)[0])

inds_stacked = np.hstack(inds)

with open(spath_root + r'/samp_pattern_R{}_PE1_{}_{}shots.txt'.format(Rs, PE1, len(U)),"w") as f:
    f.write("\n".join(" ".join(map(str, x)) for x in (mat_size, inds_stacked)))

'''