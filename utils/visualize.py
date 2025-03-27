# import jax.numpy as xp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False


def plot_views(img, vmax = 1.0):
    if vmax == "auto": #if auto, set as max val of volume
        vmax = abs(img.flatten().detach().cpu()).max()
    #
    fig, axes = plt.subplots(1,3)
    for i, ax in enumerate(axes):
        if i==0:
            ax.imshow(img[img.shape[0]//2,:,:], cmap = "gray", vmax = vmax)
        if i==1:
            ax.imshow(img[:,img.shape[1]//2,:], cmap = "gray", vmax = vmax)
        if i==2:
            ax.imshow(img[:,:,img.shape[2]//2], cmap = "gray", vmax = vmax)
        #
    plt.show()

def plot_Mtraj(Mtraj_GT, Mtraj, img_dims, rescale = 0):
    Nx, Ny, Nz = img_dims
    if rescale:
        Tx_scale = (Nx/2)
        Ty_scale = (Ny/2)
        Tz_scale = (Nz/2)
        R_scale = 1/(np.pi/180)
    else:
        Tx_scale = 1; Ty_scale = 1; Tz_scale = 1
        R_scale = 1
    #
    T_GT = Mtraj_GT[:,:3]
    R_GT = Mtraj_GT[:,3:]
    T = Mtraj[:,:3]
    R = Mtraj[:,3:]
    plt.figure()
    plt.plot(T_GT[:,0]*Tx_scale, '--r', alpha = 0.75, label="Tx - GT")
    plt.plot(T_GT[:,1]*Ty_scale, '--b', alpha = 0.75, label="Ty - GT")
    plt.plot(T_GT[:,2]*Tz_scale, '--g', alpha = 0.75, label="Tz - GT")
    plt.plot(T[:,0]*Tx_scale, 'r', label="Tx")
    plt.plot(T[:,1]*Ty_scale, 'b', label="Ty")
    plt.plot(T[:,2]*Tz_scale, 'g', label="Tz")
    # plt.legend(loc="lower left")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=5)
    plt.ylabel("Translations (mm)")
    plt.xlabel("Shot Index")
    plt.title("Estimated Motion Trajectories: Translations")
    plt.show()
    #
    plt.figure()
    plt.plot(R_GT[:,0]*R_scale, '--r', alpha = 0.75, label="Rx - GT")
    plt.plot(R_GT[:,1]*R_scale, '--b', alpha = 0.75, label="Ry - GT")
    plt.plot(R_GT[:,2]*R_scale, '--g', alpha = 0.75, label="Rz - GT")
    plt.plot(R[:,0]*R_scale, 'r', label="Rx")
    plt.plot(R[:,1]*R_scale, 'b', label="Ry")
    plt.plot(R[:,2]*R_scale, 'g', label="Rz")
    # plt.legend(loc="upper left")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=5)
    plt.ylabel("Rotations (deg)")
    plt.xlabel("Shot Index")
    plt.title("Estimated Motion Trajectories: Rotations")
    plt.show()

