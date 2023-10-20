# Open Place Recognition library

## Installation

### Pre-requisites

- The library requires PyTorch~=1.13 and MinkowskiEngine library to be installed manually. See [PyTorch website](https://pytorch.org/get-started/previous-versions/) and [MinkowskiEngine repository](https://github.com/NVIDIA/MinkowskiEngine) for the detailed instructions.

- Another option is to use the suggested Dockerfile. The following commands should be used to build, start and enter the container:

  1. Build the image

      ```bash
      bash docker/build.sh
      ```

  2. Start the container with the datasets directory mounted:

      ```bash
      bash docker/start.sh [DATASETS_DIR]
      ```

  3. Enter the container (if needed):

      ```bash
      bash docker/into.sh
      ```

### Library installation

- After the pre-requisites are met, install the Open Place Recognition library with the following command:

    ```bash
    pip install .
    ```

## Package Structure

### opr.datasets

Subpackage containing dataset classes and functions.

Usage example:

```python
from opr.datasets import OxfordDataset

train_dataset = OxfordDataset(
    dataset_root="/home/docker_opr/Datasets/pnvlad_oxford_robotcar_full/",
    subset="train",
    data_to_load=["image_stereo_centre", "pointcloud_lidar"]
)
```

The iterator will return a dictionary with the following keys:
- `"idx"`: index of the sample in the dataset, single number Tensor
- `"utm"`: UTM coordinates of the sample, Tensor of shape `(2)`
- (optional) `"image_stereo_centre"`: image Tensor of shape `(C, H, W)`
- (optional) `"pointcloud_lidar_feats"`: point cloud features Tensor of shape `(N, 1)`
- (optional) `"pointcloud_lidar_coords"`: point cloud coordinates Tensor of shape `(N, 3)`

## License

[MIT License](./LICENSE) (**_the license is subject to change in future versions_**)
