<p align="center">
  <h1 align="center">RaCo: Ranking and Covariance<br>for Practical Learned Keypoints</h1>
  <p align="center">
    <a href="https://www.linkedin.com/in/abhiramshenoi/">Abhiram Shenoi</a>
    ·
    <a href="https://www.linkedin.com/in/philipplindenberger/">Philipp Lindenberger</a>
    ·
    <a href="https://psarlin.com/">Paul-Edouard&nbsp;Sarlin</a>
    ·
    <a href="https://www.microsoft.com/en-us/research/people/mapoll/">Marc&nbsp;Pollefeys</a>
  </p>
  <h2 align="center">
    3DV 2026<br>
    <a href="https://arxiv.org/pdf/2602.15755" align="center">paper</a> | 
    <a href="#demo" align="center">demos</a> |
    <a href="#pytorch-hub-usage-guide" align="center">PyTorch Hub</a>
  </h2>
</p>

## Abstract

We introduce **RaCo**, a lightweight neural network designed to learn robust and versatile keypoints suitable for a variety of 3D computer vision tasks. The model integrates three key components: the repeatable keypoint detector, a differentiable ranker to maximize matches with a limited number of keypoints, and a covariance estimator to quantify spatial uncertainty in metric scale.

Trained on perspective image crops only, RaCo operates without the need for covisible image pairs. It achieves strong rotational robustness through extensive data augmentation, even without the use of computationally expensive equivariant network architectures. The method is evaluated on several challenging datasets, where it demonstrates state-of-the-art performance in keypoint repeatability and two-view matching, particularly under large in-plane rotations.

Ultimately, RaCo provides an effective and simple strategy to independently estimate keypoint ranking and metric covariance without additional labels, detecting interpretable and repeatable interest points.

<p align="center">
  <img src="assets/pipeline.png" alt="RaCo Pipeline" width="600">
</p>


## Features

RaCo is a rotationally equivariant keypoint detector that identifies stable and repeatable keypoints in images. Refer the paper for a quantitative evaluation of its rotational equivariance.


<div align="center">
  <video src="https://github.com/user-attachments/assets/a41394b5-b2d3-4341-a036-afc0fa2eb6e0">
</div>

<details>
<summary>RaCo's Outputs (click to expand)</summary>
RaCo outputs the following for each input image:

- **Keypoints**: 2D coordinates of detected keypoints in pixel space.
- **Keypoint Scores**: Detection confidence scores for each keypoint.
- **Ranker Scores**: Ranking scores for each keypoint indicating their reliability for matching. To subsample keypoints based on ranking, select the top-k keypoints with the highest ranker scores.
- **Covariances**: 2x2 covariance matrices representing the uncertainty of each keypoint's location.

<p align="center">
  <img src="assets/raco_outputs.png" alt="Covariance Estimation" width="600">
</p>
</details>

<details>
<summary>Keypoint Ranking (click to expand)</summary>
The ranking module is trained to produce a ranking score for each keypoint, which is consistent across different viewpoints and image transformations. The ranking score is more reliable than the detection score for selecting keypoints for matching as it is trained to maintain a consistent order of keypoints across images.

Here the repeatability of keypoints selected based on ranking scores is shown to be higher than those selected based on detection scores. Refer the paper for a quantitative evaluation of the ranking module.

<p align="center">
  <img src="assets/ranking_example.png" alt="Keypoint Ranking" width="600">
</p>
</details>

<details>
<summary>Covariance Estimation (click to expand)</summary>
The covariance estimator predicts the uncertainty of keypoint locations in 2D pixel space. This is useful for applications that require a measure of confidence in keypoint detections.

<p align="center">
  <img src="assets/covariance_example.png" alt="Covariance Estimation" width="600">
</p>
</details>

## Installation

```bash
git clone https://github.com/cvg/RaCo.git
cd RaCo && pip install -e .
```
This will install PyTorch>=1.9.1, OpenCV, and Kornia.

## Quick Start

```python
import torch
from raco import RaCo
from raco.utils import load_image

# Load an image
device = "cuda" if torch.cuda.is_available() else "cpu"
image = load_image("assets/i_castle.png")

# Initialize a RaCo extractor with the default configuration
extractor = RaCo()

# Extract keypoints
output = extractor.extract(image)
print("Output keys:", [k for k in output.keys()])

# Access results
keypoints = output["keypoints"]  # (B, N, 2) - keypoint coordinates
scores = output["keypoint_scores"]  # (B, N) - detection confidence
ranker_scores = output["ranker_scores"]  # (B, N) - ranking scores
covariances = output["covariances"]  # (B, N, 2, 2) - uncertainty estimates
```

## PyTorch Hub Usage Guide

Load RaCo directly from PyTorch Hub without cloning the repository.
Pass in image tensors normalized to `[0, 1]` range and with shape
`(B, 3, H, W)` for RGB or `(B, 1, H, W)` for grayscale images.

Ensure all the dependencies listed in `hubconf.py` are installed in your environment to use the model from PyTorch Hub.

```python
import torch

# Load RaCo with pretrained weights
model = torch.hub.load('cvg/RaCo', 'raco', pretrained=True)
model.eval()

# Extract features from an image
image = torch.rand(1, 3, 480, 640)  # Your image tensor (values in [0, 1])
with torch.no_grad():
    output = model.extract(image)

print(output.keys())
# Output: dict_keys(['keypoints', 'keypoint_scores', 'ranker_scores', 
#                    'covariances', 'image_size'])
```

<details>
<summary>Configuration with PyTorch Hub (click to expand)</summary>
You can also customize the configuration when loading from PyTorch Hub:

```python
model = torch.hub.load(
    'cvg/RaCo',
    'raco',
    pretrained=True,
    max_num_keypoints=1024,
    detection_threshold=0.0001,
    ranker=True,
    covariance_estimator=True
)
```
Refer to the [Advanced Configuration](#advanced-configuration) section for details on all available parameters.
</details>

## Matching Performance with LightGlue

We evaluate the performance of RaCo keypoints, matched using ALIKED descriptors and LightGlue+, a new, performant variant of LightGlue trained for RaCo keypoints.
We report the Area Under the recall Curve (AUC) of the estimated homography or relative pose.


HPatches, 1024 keypoints per image, AUC at 1px / 3px / 5px:

| Methods | DLT | PoseLib |
| :--- | :--- | :--- |
| **SuperPoint + SuperGlue** | 32.1 / 65.0 / 75.7 | 37.0 / 68.2 / 78.7 |
| **SuperPoint + LightGlue** | 35.1 / 67.2 / 77.6 | 37.1 / 67.4 / 77.8 |
| **ALIKED + LightGlue** | 33.8 / 66.1 / 76.6 | 37.0 / 68.8 / 78.9 |
| **RaCo + LightGlue+** | **40.4** / **71.1** / **80.5** | **44.7** / **73.4** / **82.3** |

Megadepth-1500, 2048 keypoints per image, ACU at 5° / 10° / 20°:

| Methods | OpenCV | PoseLib |
| :--- | :--- | :--- |
| **SuperPoint + SuperGlue** | 48.7 / 65.6 / 79.0 | 64.8 / 77.9 / 87.0 |
| **SuperPoint + LightGlue** | 51.0 / 68.1 / 80.7 | 66.8 / **79.3** / **87.9** |
| **ALIKED + LightGlue** | **52.3** / **68.8** / **81.0** | 66.4 / 79.0 / 87.5 |
| **RaCo + LightGlue+** | 51.9 / 68.3 / 80.8 | **67.3** / **79.3** / 87.4 |

ScanNet-1500, 2048 keypoints per image, AUC at 5° / 10° / 20°:

| Methods | OpenCV | PoseLib |
| :--- | :--- | :--- |
| **SuperPoint + SuperGlue** | **17.9** / **35.4** / 49.5 | **22.7** / 39.5 / 54.3 |
| **SuperPoint + LightGlue** | 17.8 / 34.0 / **52.0** | 21.9 / **39.8** / **55.7** |
| **DISK + LightGlue** | 9.0 / 18.1 / 29.2 | 12.1 / 23.1 / 35.0 |
| **ALIKED + LightGlue** | 15.4 / 31.2 / 47.5 | 19.6 / 36.5 / 52.8 |
| **RaCo + LightGlue+** | 17.1 / 33.7 / 50.0 | 21.7 / 39.2 / 55.2 |

## Demo

### Extracting Keypoints, Ranker Scores, and Covariances
Check out the [`demo.ipynb`](demo.ipynb) notebook for a complete walkthrough showing:

- Basic keypoint extraction
- Visualization of detected keypoints
- Uncertainty estimation with covariance ellipses
- Visualization of the ranking of keypoints

### Matching with LightGlue+
We additionally show how to easily use RaCo's keypoints with ALIKED descriptors and LightGlue for two-view matching in the [`matching_demo.ipynb`](matching_demo.ipynb) notebook.

## Advanced configuration

<details>
<summary>Details of all parameters (click to expand)</summary>

- `weights`: Path or URL to pretrained weights. Can be a local file path or a URL (e.g., `"https://github.com/cvg/RaCo/releases/download/v1.0.0/raco.pth"`). Set to `None` to skip loading pretrained weights. Default: official pretrained weights.
- `max_num_keypoints`: Maximum number of keypoints to extract per image. Default: 512.
- `nms_radius`: Radius for non-maximum suppression (must be odd). Larger values result in more spread out keypoints. Default: 3.
- `subpixel_sampling`: Enable subpixel refinement of keypoint locations for higher accuracy. Default: True.
- `subpixel_temp`: Temperature parameter for subpixel refinement softmax. Lower values make the refinement more focused. Default: 0.5.
- `detection_threshold`: Minimum keypoint score threshold. Keypoints with keypoint scores below this value are filtered out. Set to -1 to disable filtering. Default: -1 (disabled).
- `ranker`: Enable the ranking module to predict the ranker scores. Default: True.
- `covariance_estimator`: Enable covariance prediction for 2D spatial keypoint uncertainty estimation. Default: True.

</details>

The training and evaluation code will be released in [Glue Factory](https://github.com/cvg/glue-factory).

## Citation

If you use RaCo in your research, please cite:

```bibtex
@inproceedings{shenoi2026raco,
  title={{RaCo: Ranking and Covariance for Practical Learned Keypoints}},
  author={Shenoi, Abhiram and Lindenberger, Philipp and Sarlin, Paul-Edouard and Pollefeys, Marc},
  booktitle={International Conference on 3D Vision},
  year={2026},
}
```

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

