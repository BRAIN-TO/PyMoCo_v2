# Joint Motion and Image Estimation
_A Python package for retrospective motion correction (RMC) of head MRI._

- RMC is carried out through multicoil data consistency-driven joint motion and image estimation.  
- This work is based on and extends the contributions of [Haskell et al., 2019](https://doi.org/10.1002/mrm.27771) and [Cordero-Grande et al., 2018](https://doi.org/10.1002/mrm.26796). 

In brief, PyMoCo elevates the Network-assisted Motion Estimation and Reconstruction (NAMER) approach to 3D MR acquisitions and incorporates state-of-the-art deep neural networks (UNet, [Al Masni et al., 2022](https://doi.org/10.1016/j.neuroimage.2022.119411)) for motion artifact removal. 

The implementation itself is in Python and accelerated by efficent use of GPU computation using [Jax](https://docs.jax.dev/en/latest/)

If you are using this toolbox, please reference the following publication:

Nghiem, B., Wu, Z., Kashyap, S., Kasper, L., Uludağ, K., 2026. A network-assisted joint image and motion estimation approach for robust 3D MRI motion correction across severity levels. _Magnetic Resonance in Medicine 95_, 363–381. https://doi.org/10.1002/mrm.70052

# Setting up conda env
This repo was developed and tested on a server with CUDA v11.4. The conda environment can be created from the environment.yml file, or installed from scratch with the following commands:

```bash
#!/bin/bash
conda create -n PyMoCo_env python==3.9.7
conda activate PyMoCo_env

pip install scipy==1.10.* 
pip install matplotlib==3.5.1
pip install protobuf==3.20.*

conda install cudatoolkit #installs pkgs/main/linux-64::cudatoolkit-11.8.0-h6a678d5_0
conda install cudnn #installs pkgs/main/linux-64::cudnn-8.9.2.26-cuda11_0

pip install jax==0.2.25 jaxlib==0.1.75+cuda11.cudnn82 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

pip install tensorflow-gpu==2.6.2 keras==2.6.0 #Al-Masni et al SAP UNET

```
