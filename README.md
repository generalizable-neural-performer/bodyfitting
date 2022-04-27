# BodyFitting - A Multi-view SMPLx Optimization Framework

Bodyfitting is the SMPLx fitting tool in "Generalizable Neural Performer: Learning Robust Radiance Fields for Human Novel View Synthesis" and GeneBody Dataset. 

This toolbox can register SMPLx from calibrated motion capture images, as well as the synthetic meshes; SMPL+D and texture fitting is also provided.

## Setup environment
The project is built on python3.6 and torch 1.2, you can set up the environment as:
``` bash
conda env create -n bodyfitting python=3.6
pip install -r requirements.txt
cd thirdparty/neural_renderer && python setup.py install
cd thirdparty/mesh_grid && python setup.py install
```
Note: *neural_renderer* in this repo is a modified version for texture fitting. *mesh_grid* is our implemetation of mesh closest point.

## Data Download
Please download the SMPLx model from [here](https://smpl-x.is.tue.mpg.de/), and HMR model from [here](https://hkustconnect-my.sharepoint.com/:u:/g/personal/wchengad_connect_ust_hk/EXuFgaiOuMRMh8O_oNzc3DYBlSyxsVOPWNA-Qn3m4PV-zA?e=Aie8nd), and put them in the *data* folder.

## 2D keypoint detector
We require Openpose for 2D keypoint detection, please build the cpp version from the [instructions](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation/0_index.md#compiling-and-running-openpose-from-source).

Other alts such as [MMPose](https://github.com/open-mmlab/mmpose) also fits this framework, as long as appropriate joint mapping is assigned.

## Demos

### Motion capture data fitting
You can fit your motion capture data using the following command, choose `smpl_type` and `use_mask` to enable silhouette fitting if human segmentation is given.
```bash
python apps/genebody_fitting.py --target_dir path_to_images --openpose_dir path_to_openpose --output_dir path_to_output --smpl_type smplx --use_mask --tasks openpose smplify output
```
You can find the SMPL obj file and optimized parameters in path_to_output.

### Mesh data fitting 
You can fit your Render People mesh data using the following command. To perform SMPL+D and texture fitting, please add `smpld` and `texfit` in tasks list.
```bash
python apps/rp_fitting.py --target_dir path_to_images --info_dir path_to_csv_file --openpose_dir path_to_openpose --output_dir path_to_output --smpl_type smplx --smpl_uv_dir ./smpl_uv --tasks openpose smplify smpld texfit output 
```

### Texture fitting
Texture fitting takes opitmized SMPL+D as input, and optimize the texture image by comparing the rendered image of groudtruth mesh and SMPL+D mesh. The process and results are as follows:


<p align="center"><img src="./texfit_proc.gif" width="30%"> <img src="./texfit_comp.gif" width="60%"></p>
<p align="center">Left: Texture optimization process, Right: Comparison of ground truth mesh and textured SMPL+D</p>


## Citation
If you find this repo useful for your work, please cite the follow technical paper
```
@article{
    author = {Wei, Cheng and Su, Xu and Jingtan, Piao and Wayne, Wu and Chen, Qian and Kwan-Yee, Lin and Hongsheng, Li},
    title = {Generalizable Neural Performer: Learning Robust Radiance Fields for Human Novel View Synthesis},
    publisher = {arXiv},
    year = {2022},
  }

@inproceedings{SMPL-X:2019,
  title = {Expressive Body Capture: {3D} Hands, Face, and Body from a single Image},
  author = {Pavlakos, Georgios and Choutas, Vasileios and Ghorbani, Nima and Bolkart, Timo and Osman, Ahmed A. A. and Tzionas, Dimitrios and Black, Michael J.},
  booktitle = {Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {10975--10985},
  year = {2019}
}
```