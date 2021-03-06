## Real-time-3D-Perception

<img src="img/labeled_tsdf-1.png" width="300px">    <img src="img/labeled_tsdf-2.png" width="300px"> 

<img src="img/Reconstruction_and_segmentation_result-3.png" width="300px">    <img src="img/Reconstruction_and_segmentation_result-4.png" width="300px"> 


This is a real-time 3D perception system that includes 3D reconstruction, instance segmentation, and real-time rendering with a multi-resolution voxel space.

Thanks to [bundlefusion](https://github.com/niessner/BundleFusion) for its great contribution, some of our code comes from them.

Project video is available [here](https://www.zhihu.com/zvideo/1517576595287597056).

More information about this project can be found in our paper (not yet published).

## Installation
The code was developed under VS2017, and tested with an intel Realsense D435i.

We tested our code on:
- DirectX SDK June 2010
- NVIDIA CUDA 11.5 (>=8.0)
- Realsense SDK2.0
- our research library mLib, a git submodule in external/mLib [here](https://github.com/niessner/mLib/tree/ac6b9e9d1da1df00a2293da64a9f146c123fa2ca)
- mLib external libraries can be downloaded [here](http://kaldir.vc.in.tum.de/mLib/mLibExternal.zip)
- Intel OpenVINO 2021.4.689
- Opencv with OpenVINO

Optional:
- Kinect SDK (2.0 and above)
- Prime sense SDK



## Contact:
If you have any questions, please email us at zhu_hao_wei@163.com.


