"""
RaCo (Ranking and Covariance) Feature Extractor
"""

import logging
from types import SimpleNamespace
from typing import Optional

import torch
from importlib.resources import files
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from .raco_model import RacoModel
from .utils import Extractor

logger = logging.getLogger(__name__)

# Configuration constants
DEFAULT_DETECTION_THRESHOLD = -1
DEFAULT_NMS_RADIUS = 3
DEFAULT_MAX_KEYPOINTS = 512
DEFAULT_SUBPIXEL_TEMP = 0.5
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class TanhTimesN(nn.Module):
    """
    Custom activation function that applies N * tanh(x) to the input.
    Ensures output is in the range [-N, N].

    Args:
        N: Scaling factor for the tanh output (must be positive)
    """

    def __init__(self, N: float = 1.0):
        super().__init__()
        if N <= 0:
            raise ValueError(f"N must be positive, got {N}")
        self.N = N

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.N * torch.tanh(x)

    def extra_repr(self) -> str:
        return f"N={self.N}"


def _get_grid(
    B: int,
    H: int,
    W: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Generate normalized coordinate grid for batch of images.

    Args:
        B: Batch size
        H: Height
        W: Width
        device: Target device

    Returns:
        Grid tensor of shape (B, H*W, 2) with normalized coordinates in [-1, 1]
    """
    x1_n = torch.meshgrid(
        *[torch.linspace(-1 + 1 / n, 1 - 1 / n, n, device=device) for n in (B, H, W)],
        indexing="ij",
    )
    x1_n = torch.stack((x1_n[2], x1_n[1]), dim=-1).reshape(B, H * W, 2)
    return x1_n


def _extract_patches_from_inds(
    x: torch.Tensor,
    inds: torch.Tensor,
    patch_size: int,
) -> torch.Tensor:
    """
    Extract patches from tensor at specified indices.

    Args:
        x: Input tensor of shape (B, H, W)
        inds: Indices tensor of shape (B, N)
        patch_size: Size of patches to extract (must be odd)

    Returns:
        Patches tensor of shape (B, patch_size**2, N)
    """
    if patch_size % 2 == 0:
        raise ValueError(f"patch_size must be odd, got {patch_size}")

    B, H, W = x.shape
    B, N = inds.shape
    unfolder = nn.Unfold(kernel_size=patch_size, padding=patch_size // 2, stride=1)
    unfolded_x: torch.Tensor = unfolder(x[:, None])  # B x K_H * K_W x H * W
    patches = torch.gather(
        unfolded_x,
        dim=2,
        index=inds[:, None, :].expand(B, patch_size**2, N),
    )  # B x K_H * K_W x N
    return patches


def _covariance_matrix_from_cholesky_elements(
    cholesky_elements_vec: torch.Tensor,
) -> torch.Tensor:
    """
    Converts a vector of Cholesky factor elements (L11, L21, L22) of shape (..., 3)
    to a covariance matrix of shape (..., 2, 2).
    L = [[L11, 0], [L21, L22]]
    Sigma = L @ L.T
    Args:
        cholesky_elements_vec: Tensor of shape (..., 3) where the last dimension
                               contains [L11, L21, L22]. L11 and L22 are assumed
                               to be positive (e.g., from exp/softplus activation).
    Returns:
        Tensor of shape (..., 2, 2) representing the covariance matrix.
    """
    L11 = cholesky_elements_vec[..., 0]
    L21 = cholesky_elements_vec[..., 1]
    L22 = cholesky_elements_vec[..., 2]

    # Create the lower triangular L matrix (batch-wise) The elements L11 and L22 must be positive.
    L = torch.zeros(
        cholesky_elements_vec.shape[:-1] + (2, 2),
        device=cholesky_elements_vec.device,
    )
    L[..., 0, 0] = L11
    L[..., 1, 0] = L21
    L[..., 1, 1] = L22

    # Compute Sigma = L @ L.T
    if cholesky_elements_vec.dim() > 1:  # If there's a batch or N dimension
        original_shape = cholesky_elements_vec.shape[:-1]
        L_flat = L.view(-1, 2, 2)
        cov_matrix_flat = torch.bmm(L_flat, L_flat.transpose(-1, -2))
        return cov_matrix_flat.view(original_shape + (2, 2))
    else:  # Single matrix case
        return torch.matmul(L, L.transpose(-1, -2))


def _to_pixel_coords(
    normalized_coords: torch.Tensor,
    h: int,
    w: int,
) -> torch.Tensor:
    """
    Convert normalized coordinates [-1, 1] to pixel coordinates.

    Args:
        normalized_coords: Tensor of shape (..., 2) with normalized coordinates
        h: Image height
        w: Image width

    Returns:
        Pixel coordinates tensor of same shape
    """
    if normalized_coords.shape[-1] != 2:
        raise ValueError(f"Expected shape (..., 2), but got {normalized_coords.shape}")
    pixel_coords = torch.stack(
        (
            w * (normalized_coords[..., 0] + 1) / 2,
            h * (normalized_coords[..., 1] + 1) / 2,
        ),
        dim=-1,
    )
    return pixel_coords


def _compute_subpixel_offsets(
    raw_logits: torch.Tensor,
    inds: torch.Tensor,
    nms_radius: int,
    H: int,
    W: int,
    subpixel_temp: float = DEFAULT_SUBPIXEL_TEMP,
) -> torch.Tensor:
    """Compute subpixel offsets for keypoints using local patch softmax."""
    B = raw_logits.shape[0]
    device = raw_logits.device

    offsets = _get_grid(B, nms_radius, nms_radius, device=device).reshape(
        B, nms_radius**2, 2
    )
    offsets[..., 0] = offsets[..., 0] * nms_radius / W
    offsets[..., 1] = offsets[..., 1] * nms_radius / H

    keypoint_patch_scores = _extract_patches_from_inds(
        raw_logits.squeeze(1), inds, nms_radius
    )
    keypoint_patch_probs = (keypoint_patch_scores / subpixel_temp).softmax(dim=1)
    keypoint_offsets = torch.einsum("bkn, bkd ->bnd", keypoint_patch_probs, offsets)
    return keypoint_offsets


class RaCo(Extractor):
    default_conf = {
        "name": "raco",
        "weights": files("raco") / "raco.pth",
        "trainable": False,
        "max_num_keypoints": DEFAULT_MAX_KEYPOINTS,
        "nms_radius": DEFAULT_NMS_RADIUS,
        "subpixel_sampling": True,
        "detection_threshold": DEFAULT_DETECTION_THRESHOLD,
        "ranker": True,
        "covariance_estimator": True,
    }
    
    preprocess_conf = {
        "resize": None,
    }

    def __init__(self, **conf) -> None:
        """Initialize the RaCo model with given configuration."""
        super().__init__(**conf)
        
        self.conf = SimpleNamespace(**{**self.default_conf, **conf})

        # Validate configuration
        if self.conf.nms_radius % 2 == 0:
            raise ValueError(f"nms_radius must be odd, got {self.conf.nms_radius}")
        if self.conf.max_num_keypoints <= 0:
            raise ValueError(
                f"max_num_keypoints must be positive, got {self.conf.max_num_keypoints}"
            )

        # Setup ImageNet normalization
        self.normalizer = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

        # Initialize model
        self.model = RacoModel(
            {
                "ranker": self.conf.ranker,
                "covariance_estimator": self.conf.covariance_estimator,
            }
        )

        # Load pretrained weights if specified
        if self.conf.weights is not None:
            try:
                state_dict = torch.load(self.conf.weights, map_location="cpu")["model"]
                self.load_state_dict(state_dict, strict=True)
                logger.info(f"[RaCo] Loaded weights from {self.conf.weights}")
            except Exception as e:
                logger.error(f"Failed to load weights from {self.conf.weights}: {e}")
                raise

    def _sampling(
        self,
        keypoint_probs: torch.Tensor,
        nms_radius: int,
        num_kpts: Optional[int] = None,
        raw_logits: Optional[torch.Tensor] = None,
        subpixel: bool = False,
        subpixel_temp: float = DEFAULT_SUBPIXEL_TEMP,
    ) -> tuple:
        """Sample keypoints using NMS and optional subpixel refinement."""
        if num_kpts is None:
            num_kpts = self.conf.max_num_keypoints

        B, C, H, W = keypoint_probs.size()
        device = keypoint_probs.device

        # Generate coordinate grid
        grid = _get_grid(B, H, W, device=device).reshape(B, H * W, 2)

        # Apply NMS
        if nms_radius % 2 != 1:
            raise ValueError("nms_radius should be odd")
        max_pooled = F.max_pool2d(
            keypoint_probs, nms_radius, stride=1, padding=nms_radius // 2
        )
        keypoint_probs = keypoint_probs * (keypoint_probs == max_pooled)

        # Sample top keypoints
        inds = torch.topk(keypoint_probs.reshape(B, H * W), k=num_kpts).indices
        kps = torch.gather(grid, dim=1, index=inds[..., None].expand(B, num_kpts, 2))

        # Compute subpixel refinement if requested
        if subpixel and raw_logits is not None:
            keypoint_offsets = _compute_subpixel_offsets(
                raw_logits, inds, nms_radius, H, W, subpixel_temp
            )
            kps_subpixel = kps + keypoint_offsets
            kps_subpixel = _to_pixel_coords(kps_subpixel, H, W) - 0.5

        # Convert to pixel coordinates
        kps = _to_pixel_coords(kps, H, W) - 0.5

        # Convert to flat indices
        idxs = kps[..., 1] * W + kps[..., 0]
        idxs = idxs.long()
        idxs = torch.clamp(idxs, min=0, max=W * H - 1)

        if subpixel and raw_logits is not None:
            kps = kps_subpixel

        return idxs, kps

    def forward(self, data: dict) -> dict:
        """Forward pass through the RaCo model."""
        # Preprocess image
        image = data["image"]
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)  # Convert to 3-channel greyscale
        image = self.normalizer(image)

        # Forward through model
        raw_score_map, ranker_map, cov_maps = self.model(image)

        # Compute probability maps using batchwise global softmax normalization
        logx = nn.functional.log_softmax(raw_score_map.flatten(1), dim=1).reshape(
            raw_score_map.size()
        )
        x = torch.exp(logx)

        # Sample keypoints
        kps = None
        idxs, kps = self._sampling(
            keypoint_probs=x,
            nms_radius=self.conf.nms_radius,
            raw_logits=raw_score_map,
            subpixel=self.conf.subpixel_sampling,
        )

        B, _, H, W = x.size()
        probs = x.view(B, -1).gather(1, idxs.view(B, -1))

        # Apply detection threshold if configured
        if self.conf.detection_threshold > 0:
            mask = probs > self.conf.detection_threshold
            idxs = idxs[mask].view(idxs.shape[0], -1)
            probs = probs[mask].view(probs.shape[0], -1)

        # Convert indices to coordinates
        xs = idxs % W
        ys = idxs // W
        keypoints = torch.stack([xs, ys], dim=-1).float() if kps is None else kps

        # Build output dictionary, excluding None values
        out_dict = {
            # Points, add 0.5 to center the keypoints
            "keypoints": keypoints + 0.5,  # (B, N, 2)
            "keypoint_scores": probs,  # (B, N)
        }

        # Add optional outputs only if they exist
        if self.conf.ranker and ranker_map is not None:
            # Higher the ranker score, better the keypoint
            ranker_scores = ranker_map.reshape(B, -1).gather(1, idxs.view(B, -1))
            out_dict["ranker_scores"] = ranker_scores.view(B, -1)  # (B, N)

        if self.conf.covariance_estimator and cov_maps is not None:
            # Apply activations to ensure L11 and L22 are positive
            # cov_maps has shape (B, 3, H, W) with [L11_prime, L21, L22_prime]
            var_activation = nn.Softplus()

            processed_cov_maps = torch.stack(
                [
                    var_activation(cov_maps[:, 0, ...]),  # L11
                    cov_maps[:, 1, ...],  # L21 (no constraint)
                    var_activation(cov_maps[:, 2, ...]),  # L22
                ],
                dim=1,
            )
            # Sample the cholesky elements at the keypoints
            cholesky_scores = (
                processed_cov_maps.view(B, 3, -1)
                .gather(2, idxs.unsqueeze(1).expand(B, 3, -1))
                .permute(0, 2, 1)
            )  # (B, N, 3)

            covariances = _covariance_matrix_from_cholesky_elements(cholesky_scores)

            out_dict["covariances"] = covariances  # (B, N, 2, 2)

        return out_dict


if __name__ == "__main__":
    model = RaCo(RaCo.default_conf)
    print(model)
