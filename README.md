# Joint Motion and Image Estimation
_A Python package for retrospective motion correction (RMC) of head MRI._

- RMC is carried out through multicoil data consistency-driven joint motion and image estimation.  
- This work is based on and extends the contributions of [Haskell et al., 2019](https://doi.org/10.1002/mrm.27771) and [Cordero-Grande et al., 2018](https://doi.org/10.1002/mrm.26796). 

In brief, PyMoCo elevates the Network-assisted Motion Estimation and Reconstruction (NAMER) approach to 3D MR acquisitions and incorporates state-of-the-art deep neural networks (UNet, [Al Masni et al., 2022](https://doi.org/10.1016/j.neuroimage.2022.119411)) for motion artifact removal. 

The implementation itself is in Python and accelerated by efficent use of GPU computation using [Jax](https://docs.jax.dev/en/latest/)

If you are using this toolbox, please reference the following publication:

Nghiem, B., Wu, Z., Kashyap, S., Kasper, L., Uludağ, K., 2026. A network-assisted joint image and motion estimation approach for robust 3D MRI motion correction across severity levels. _Magnetic Resonance in Medicine 95_, 363–381. https://doi.org/10.1002/mrm.70052

