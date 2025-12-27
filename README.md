# RaCo: Ranking and Covariance for Practical Learned Keypoints

*README.md is WIP*

RaCo is a state-of-the-art computer vision library for keypoint detection and feature extraction. It provides robust keypoint detection with uncertainty estimation through covariance prediction.

## Features

## Installation

### From source

```bash
git clone https://github.com/cvg/RaCo.git
cd RaCo
pip install -e .
```

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

## Demo

Check out the [`demo.ipynb`](demo.ipynb) notebook for a complete walkthrough showing:

- Basic keypoint extraction
- Visualization of detected keypoints
- Uncertainty estimation with covariance ellipses
- Visualization of the ranking of keypoints

## Advanced configuration

<details>
<summary>[Detail of all parameters - click to expand]</summary>

- `weights`: Path or URL to pretrained weights. Can be a local file path (e.g., `"raco/raco.pth"`) or a URL (e.g., `"https://github.com/cvg/RaCo/releases/download/v1.0.0/raco.pth"`). Set to `None` to skip loading pretrained weights. Default: `"raco/raco.pth"`.
- `max_num_keypoints`: Maximum number of keypoints to extract per image. Default: 512.
- `nms_radius`: Radius for non-maximum suppression (must be odd). Larger values result in more spread out keypoints. Default: 3.
- `subpixel_sampling`: Enable subpixel refinement of keypoint locations for higher accuracy. Default: True.
- `subpixel_temp`: Temperature parameter for subpixel refinement softmax. Lower values make the refinement more focused. Default: 0.5.
- `detection_threshold`: Minimum keypoint score threshold. Keypoints with keypoint scores below this value are filtered out. Set to -1 to disable filtering. Default: -1 (disabled).
- `ranker`: Enable the ranking module to predict the ranker scores. Default: True.
- `covariance_estimator`: Enable covariance prediction for 2D spatial keypoint uncertainty estimation. Default: True.

</details>

## Requirements

- Python >= 3.8
- PyTorch >= 1.9.1
- OpenCV
- NumPy
- Matplotlib
- Kornia >= 0.6.11

## Citation

If you use RaCo in your research, please cite:

```bibtex
@inproceedings{shenoi2025raco,
  title={{RaCo}: Ranking and Covariance for Practical Learned Keypoints},
  author={Shenoi, Abhiram and Lindenberger, Philipp and Sarlin, Paul-Edouard and Pollefeys, Marc},
  booktitle={Thirteenth International Conference on 3D Vision},
  year={2025},
  url={https://openreview.net/forum?id=BWtdgrdcBH}
}
```

## License

This project is licensed under the Apache License 2.0 see the [LICENSE](LICENSE) file for details.

