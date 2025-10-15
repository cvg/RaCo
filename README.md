# RaCo: Ranking and Covariance for Practical Learned Keypoints

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
image = load_image("path/to/your/image.jpg")

# Initialize the extractor
extractor = RaCo()

# Extract keypoints
output = extractor.extract(image)

# Access results
keypoints = output["keypoints"]  # (N, 2) - keypoint coordinates
scores = output["keypoint_scores"]  # (N,) - detection confidence
ranker_scores = output["ranker_scores"]  # (N,) - ranker confidence
covariances = output["covariances"]  # (N, 2, 2) - uncertainty estimates
```

## Visualization

```python
from raco import viz2d

# Visualize keypoints
ax = viz2d.plot_images([image])
viz2d.plot_keypoints(keypoints, axes=ax)
```

## Configuration

RaCo can be configured with various parameters:

```python
extractor = RaCo(
    max_num_keypoints=1024,
    detection_threshold=0.005,
    nms_radius=3,
    subpixel_sampling=True,
    ranker=True,
    covariance_estimator=True
)
```

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
@article{raco2024,
  title={RaCo: Ranking and Covariance for Robust Keypoint Detection},
  author={},
  journal={},
  year={2025}
}
```
