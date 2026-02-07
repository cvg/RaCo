"""
PyTorch Hub configuration for RaCo

Usage:
    import torch

    # Load RaCo model with default configuration
    model = torch.hub.load('cvg/RaCo', 'raco', pretrained=True)

    # Load RaCo model without pretrained weights
    model = torch.hub.load('cvg/RaCo', 'raco', pretrained=False)

    # Load with custom configuration
    model = torch.hub.load('cvg/RaCo', 'raco', pretrained=True,
                          max_num_keypoints=1024, nms_radius=5)
"""

dependencies = ["torch", "torchvision"]


def raco(pretrained=True, **kwargs):
    """
    RaCo feature extractor for keypoint detection with ranking and uncertainty estimation.

    Args:
        pretrained (bool): If True, loads pretrained weights from the official
                          release. Default: True
        **kwargs: Additional configuration parameters:
            - max_num_keypoints (int): Maximum number of keypoints to extract.
              Default: 512
            - nms_radius (int): Radius for non-maximum suppression.
              Default: 3
            - subpixel_sampling (bool): Enable subpixel refinement.
              Default: True
            - subpixel_temp (float): Temperature for subpixel refinement.
              Default: 0.5
            - detection_threshold (float): Threshold for keypoint detection.
              Default: -1
            - ranker (bool): Enable ranking head. Default: True
            - covariance_estimator (bool): Enable covariance estimation.
              Default: True

    Returns:
        RaCo model instance

    Example:
        >>> import torch
        >>> model = raco(pretrained=True)
        >>> model.eval()
        >>>
        >>> # Extract features from an image
        >>> image = torch.randn(1, 3, 480, 640)  # Example input
        >>> with torch.no_grad():
        >>>     output = model.extract(image)
        >>>
        >>> print(output.keys())
        >>> # dict_keys(['keypoints', 'keypoint_scores', 'ranker_scores',
        >>> #             'covariances', 'image_size'])
    """
    from raco import RaCo

    # Set weights path if pretrained
    if pretrained:
        if "weights" not in kwargs:
            kwargs["weights"] = (
                "https://github.com/cvg/RaCo/releases/download/v1.0.0/raco.pth"
            )
    else:
        kwargs["weights"] = None

    model = RaCo(**kwargs)
    return model
