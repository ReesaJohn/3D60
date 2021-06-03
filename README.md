# create_h5.py

An easily modifiable python script to generate an h5 file out of the Matterport, Stanford, and Sun datasets. 

Make sure to install the following libraries before running this file.

- [OpenCV](https://pypi.org/project/opencv-python/)
- [numpy](https://numpy.org/)
- [easydict](https://pypi.org/project/easydict/)
- [matplotlib](https://matplotlib.org/)
- [torchvision](https://pypi.org/project/torchvision/)

Make sure your image/exr directories are one file level above the script. If it isn't modify the script. 
If you are creating a custom .txt file for splitting the data, please keep in mind that the script changes directories to one file level above.
The comments within the file should help guide you to create the h5 file structure you want.

Currently the code as is will only use the Matterport dataset that it assumes is one file level above to create an h5 structured like so:

- train
	- image_paths: relational paths of the color images (as listed in the text files used to determine the splits)
	- colors: numpy array of the color image (scaled 0-1, RGB formatted)
	- depths: numpy array of the depth maps (mostly scaled 0 to 255; large numbers correlate to closer areas; contains large negative numbers around some reflective surfaces)
- test
	- image_paths: relational paths of the color images (as listed in the text files used to determine the splits)
	- colors: numpy array of the color image (scaled 0-1, RGB formatted)
	- depths: numpy array of the depth maps (mostly scaled 0 to 255; large numbers correlate to closer areas; contains large negative numbers around some reflective surfaces)
- val
	- image_paths: relational paths of the color images (as listed in the text files used to determine the splits)
	- colors: numpy array of the color image (scaled 0-1, RGB formatted)
	- depths: numpy array of the depth maps (mostly scaled 0 to 255; large numbers correlate to closer areas; contains large negative numbers around some reflective surfaces)

Additionally every group of 3 images correspond to the same the area where the first image is taken from a center-left to the second and third images, the second image is taken 0.26 meters to the right of the first image, and the third image is taken 0.26 meters above the first image. However every group of three images is a randomly ordered.

To modify any behavior of this code, please look at the file and read its comments.


[![OmniDepth](http://img.shields.io/badge/OmniDepth-arxiv.1807.09620-critical.svg?style=plastic)](https://arxiv.org/pdf/1807.09620.pdf)
[![Conference](http://img.shields.io/badge/ECCV-2018-blue.svg?style=plastic)](https://eccv2018.org/)
[![Project Page](http://img.shields.io/badge/Project-Page-blueviolet.svg?style=plastic)](https://vcl3d.github.io/3D60/)

[![Spherical View Synthesis](http://img.shields.io/badge/SphericalViewSynthesis-arxiv.1909.08112-critical.svg?style=plastic)](https://arxiv.org/pdf/1909.08112.pdf)
[![Conference](http://img.shields.io/badge/3DV-2019-blue.svg?style=plastic)](http://3dv19.gel.ulaval.ca/)
[![Project Page](http://img.shields.io/badge/Project-Page-blueviolet.svg?style=plastic)](https://vcl3d.github.io/SphericalViewSynthesis/)

[![Surface Regression](http://img.shields.io/badge/SurfaceRegression-arxiv.1909.07043-critical.svg?style=plastic)](https://arxiv.org/pdf/1909.07043.pdf)
[![Conference](http://img.shields.io/badge/3DV-2019-blue.svg?style=plastic)](http://3dv19.gel.ulaval.ca/)
[![Project Page](http://img.shields.io/badge/Project-Page-blueviolet.svg?style=plastic)](https://vcl3d.github.io/HyperSphereSurfaceRegression/)

# converter.py
A very simple script to convert the Windows' slashes ('\') to that of Unix's ('/'). Please note that the script assumes to run on a Windows system so modify accordingly. Please modify the file's `path` and `output_path` variables to change which files it wants to convert.

# 3D60 Toolset
A set of tools for working with the [3D60 dataset](https://vcl3d.github.io/3D60/):
 - PyTorch data loaders
 - Dataset splits generation scripts

The **3D60** dataset was generated by ray-casting existing 3D datasets, making it a derivative of:
- [Matterport3D](https://niessner.github.io/Matterport/) __\[[1](#M3D)\]__, 
- [Stanford2D3D](http://buildingparser.stanford.edu/dataset.html) __\[[2](#S2D3D)\]__, and,
- [SunCG](https://sscnet.cs.princeton.edu/) __\[[3](#SunCG)\]__  

## Requirements
This code has been tested with:
- [PyTorch 1.0.1](https://pytorch.org/get-started/previous-versions/)
- [Python 3.7.4](https://www.python.org/downloads/release/python-374/)
- [CUDA 9.2](https://developer.nvidia.com/cuda-92-download-archive)

Besides PyTorch, the following Python packages are needed:
- [OpenCV](https://pypi.org/project/opencv-python/)
- [numpy](https://numpy.org/)
- [easydict](https://pypi.org/project/easydict/)
- [matplotlib](https://matplotlib.org/)
- [torchvision](https://pypi.org/project/torchvision/)
- [Visdom](https://github.com/facebookresearch/visdom)

## Data Loading
An example data loading usage can be found in [`visualize_dataset.py`](./visualize_dataset.py) where the dataset is loaded and visualized using visdom.

Given that **3D60** can be used in a variety of machine learning tasks, data loading can be modified w.r.t which `datasets`, `image_types` and `placements` (for stereo viewpoints) will be loaded by constructing customized dataset iterators:

```python
dataset_iterator = ThreeD60.get_datasets(".//splits//3D60_train.txt",\
	    datasets=["suncg", "m3d", "s2d3d"],\
	    placements=[ThreeD60.Placements.CENTER, ThreeD60.Placements.RIGHT, ThreeD60.Placements.UP],\
	    image_types=[ThreeD60.ImageTypes.COLOR, ThreeD60.ImageTypes.DEPTH, ThreeD60.ImageTypes.NORMAL],\
	    longitudinal_rotation=True)
```
In addition, randomized horizontal rotation augmentations can also be triggered with the `longitudinal_rotation` flag.

The returned iterator can be used to construct a PyTorch DataLoader:
```python
dataset_loader = torch.utils.data.DataLoader(dataset_iterator,\
  batch_size=32, shuffle=True, pin_memory=False, num_workers=4)
```
Specific image tensors can be extracted from the returned dictionary with the `extract_image` function:

```python
for i, b in  enumerate(dataset_loader):
	center_color_image = ThreeD60.extract_image(b, ThreeD60.Placements.CENTER, ThreeD60.ImageTypes.COLOR)
	center_depth_map = ThreeD60.extract_image(b, ThreeD60.Placements.CENTER, ThreeD60.ImageTypes.DEPTH)
	center_normal_map = ThreeD60.extract_image(b, ThreeD60.Placements.CENTER, ThreeD60.ImageTypes.NORMAL)
```

## Splits

### Published
A set of published splits are provided which are used in the corresponding works:
- [360D ECCV18](./splits/eccv18/) (used in OmniDepth [[4](#OmniDepth)])
- [3D60 3DV19](./splits/3dv19/) (with a smaller synthetic part used in [[5](#SVS)] & [[6](#HSSR)])
- [v1](./splits/v1/) (with a larger synthetic part) 

These splits rely on the official splits of the real datasets -- _i.e._ Matterport3D and Stanford2D3D ([fold#1](http://buildingparser.stanford.edu/dataset.html#splits)) -- but use a random selection of scenes from the synthetic SunCG dataset. 

### Custom

We also offer a set of scripts to generate new splits:
 - [`calculate_statistics.py`](./splits/calculate_statistics.py)

>This script calculates a depth value distribution histogram w.r.t. a `--max_depth` argument value (default: *10.0m*) as well as the percentage of values lower than *0.5m* and over *5.0m* for each of the datasets. 
>The resulting `.csv` files are saved in the `--stats_path` argument folder (default: `./splits/`), one for each rendered 3D dataset and prefixed with their `codename`: `"m3d", "s2d3d" and "suncg"`.
> The paths containing the rendered data for each dataset are provided with the `--m3d_path`, `--s2d3d_path` and `--suncg_path` for Matterport3D, Stanford2D3D and SunCG respectively. 
 - [`find_outliers.py`](./splits/find_outliers.py)
> This script has two modes based on the `--action` argument:
> - _`'calc'`_: Finds and saves outlier renders based on a set of heuristics w.r.t. their near (`--lower_threshold`) and far (`--upper_threshold`) depth value distributions. 
> When the percentage of total pixels of each examined depth map exceeds specific percentage bounds either under the near value threshold or over the far value threshold, it is considered as an outlier or bad render.
> The depth maps are located in each dataset's respective folder provided by the `codename` prefixed arguments `--m3d_path`, `--s2d3d_path` and `--suncg_path` similar to [`calculate_statistics.py`](./splits/calculate_statistics.py).
> This can happen due to incomplete scans, scanning artifacts and errors, unfinished 3D modeling or missing assets.
> Different lower and upper bounds can be set for each dataset as their typical depth distribution values differ. 
> They are fractional percentages, set using `codename` prefixed lower and upper bound arguments: 
>   - `--m3d_lower_bound` and `--m3d_upper_bound` for Matterport3D,
>   - `--s2d3d_lower_bound` and `--s2d3d_upper_bound` for Stanford2D3D, and
>   - `--suncg_lower_bound` and `--suncg_upper_bound` for SunCG.
>
>   The resulting `.csv` files contain the file names of the outlier renders, prefixed with each dataset's `codename` and saved in the `--outliers_path` argument folder (default: `./splits/`). 
> - _`'save'`_: Stores the calculated outliers read from the `--outliers_path` argument folder in `.png` images saved in the same path. The images contain multiple tiled outliers for quick visual inspection of the results.
 - [`create_splits.py`](./splits/create_splits.py)
> This script creates the split (_i.e._ `train`, `test` and `val`) files that in turn contain the filenames of each rendered modality and placement for each viewpoint. 
> The split files are prefixed with the `--name` argument and saved in the `--outliers_path` folder, where the outlier files are also read from.
> The script has the train/test/validation set splits from [Matterport3D](https://github.com/niessner/Matterport/tree/master/tasks/benchmark) and [Stanford2D3D's fold#1](http://buildingparser.stanford.edu/dataset.html#splits) hardcoded in it, but uses a random selection for SunCG.
> It scans each dataset's folder, as provided by the `codename` prefixed arguments `--m3d_path`, `--s2d3d_path` and `--suncg_path` similar to [`calculate_statistics.py`](./splits/calculate_statistics.py) and [`find_outliers.py`](./splits/find_outliers.py), and ignores the outliers as read from the available files for each dataset.  

The splits available in [./splits/v1](./splits/v1/) were generated by running the aforementioned scripts in order and using their default parameters. However, they can also be used to create new splits using different parameterisations for outlier rejection, based on custom distance thresholds, or by ignoring specific datasets. 
*If any of the `codename` prefixed arguments `--m3d_path`, `--s2d3d_path` and `--suncg_path` are not provided, they are skipped from all steps/scripts, and thus, single dataset splits can also be generated (_i.e._ for leave-one-out experiments).*

**Important Note**: Taking this into account, consistency between experiments may not always be possible without using the aforementioned splits or when involving custom splits that contain synthetic samples. However, the realistic (_i.e._ Matterport3D and Stanford2D3D) samples test and validation sets should be consistent between published and custom splits.

## References
<a name="M3D"/> __\[[1](https://niessner.github.io/Matterport/)\]__ Chang, A., Dai, A., Funkhouser, T., Halber, M., Niessner, M., Savva, M., Song, S., Zeng, A. and Zhang, Y. (2017). [Matterport3d: Learning from rgb-d data in indoor environments](https://arxiv.org/pdf/1709.06158.pdf). In Proceedings of the International Conference on 3D Vision (3DV).

<a name="S2D3D"/> __\[[2](http://buildingparser.stanford.edu/dataset.html)\]__ Armeni, I., Sax, S., Zamir, A.R. and Savarese, S., 2017. [Joint 2d-3d-semantic data for indoor scene understanding](https://arxiv.org/pdf/1702.01105.pdf). arXiv preprint arXiv:1702.01105.

<a name="SunCG"/> __\[[3](https://sscnet.cs.princeton.edu/)\]__ Song, S., Yu, F., Zeng, A., Chang, A.X., Savva, M. and Funkhouser, T., 2017. [Semantic scene completion from a single depth image](http://openaccess.thecvf.com/content_cvpr_2017/papers/Song_Semantic_Scene_Completion_CVPR_2017_paper.pdf). In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

<a name="OmniDepth"/>__[[4](https://vcl.iti.gr//360-dataset/)]__ Zioulis, N.__\*__, Karakottas, A.__\*__, Zarpalas, D., and Daras, P. (2018). [Omnidepth: Dense depth estimation for indoors spherical panoramas](https://arxiv.org/pdf/1807.09620.pdf). In Proceedings of the European Conference on Computer Vision (ECCV).

<a name="SVS"/>__[[5](https://vcl3d.github.io/SphericalViewSynthesis/)]__ Zioulis, N., Karakottas, A., Zarpalas, D., Alvarez, F., and Daras, P. (2019). [Spherical View Synthesis for Self-Supervised 360<sup>o</sup> Depth Estimation](https://arxiv.org/pdf/1909.08112.pdf). In Proceedings of the International Conference on 3D Vision (3DV).

<a name="HSSR"/>__[[6](https://vcl3d.github.io/HyperSphereSurfaceRegression/)]__ Karakottas, A., Zioulis, N., Samaras, S., Ataloglou, D., Gkitsas, V., Zarpalas, D., and Daras, P. (2019). [360<sup>o</sup> Surface Regression with a Hyper-sphere Loss](https://arxiv.org/pdf/1909.07043.pdf). In Proceedings of the International Conference on 3D Vision (3DV).
