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

## Visualization

### Basic Keypoint Visualization

```python
from raco import viz2d
import matplotlib.pyplot as plt

# Visualize keypoints
ax = viz2d.plot_images([image])
viz2d.plot_keypoints([output["keypoints"][0]], axes=ax)

plt.suptitle("RaCo Keypoints", fontsize=16, y=0.95)
plt.tight_layout()
```

## Demo

Check out the [`demo.ipynb`](demo.ipynb) notebook for a complete walkthrough showing:

- Basic keypoint extraction
- Visualization of detected keypoints
- Uncertainty estimation with covariance ellipses
- Visualization of the ranking of keypoints

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
@article{raco2025,
  title={RaCo: Ranking and Covariance for Practical Learned Keypoints},
  author={},
  journal={},
  year={2025}
}
```

## License

This project is licensed under the Apache License 2.0 see the [LICENSE](LICENSE) file for details.

