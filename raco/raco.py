"""
RaCo (Ranking and Covariance) Feature Extractor
"""

from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from .raco_model import RacoModel
from .utils import Extractor


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
        *[torch.linspace(-1, 1, n, device=device) for n in (B, H, W)],
        indexing="ij",
    )
    x1_n = torch.stack((x1_n[2], x1_n[1]), dim=-1).reshape(B, H * W, 2)
    return x1_n


def _extract_patches_from_indices(
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
    N = inds.shape[1]
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
    L11, L21, L22 = torch.unbind(cholesky_elements_vec, dim=-1)
    zeros = torch.zeros_like(L11)

    # L = [[L11, 0], [L21, L22]]
    L = torch.stack(
        [torch.stack([L11, zeros], dim=-1), torch.stack([L21, L22], dim=-1)], dim=-2
    )

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
    Convert normalized coordinates [-1, 1] to pixel coordinates [0, W-1] x [0, H-1].

    Args:
        normalized_coords: Tensor of shape (..., 2) with normalized coordinates
        h: Image height
        w: Image width

    Returns:
        Pixel coordinates tensor of same shape, in range [0, W-1] x [0, H-1]
    """
    if normalized_coords.shape[-1] != 2:
        raise ValueError(f"Expected shape (..., 2), but got {normalized_coords.shape}")
    pixel_coords = torch.stack(
        (
            (w - 1) * (normalized_coords[..., 0] + 1) / 2,
            (h - 1) * (normalized_coords[..., 1] + 1) / 2,
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
    subpixel_temp: float = 0.5,
) -> torch.Tensor:
    """Compute subpixel offsets for keypoints using local patch softmax.

    Returns offsets in pixel coordinates (not normalized).
    """
    B = raw_logits.shape[0]
    device = raw_logits.device

    offset_range = torch.linspace(
        -(nms_radius - 1) / 2, (nms_radius - 1) / 2, nms_radius, device=device
    )
    offset_grid = torch.meshgrid(offset_range, offset_range, indexing="ij")
    offsets = torch.stack((offset_grid[1], offset_grid[0]), dim=-1).reshape(
        nms_radius**2, 2
    )
    offsets = offsets.unsqueeze(0).expand(B, -1, -1)  # (B, nms_radius**2, 2)

    keypoint_patch_scores = _extract_patches_from_indices(
        raw_logits.squeeze(1), inds, nms_radius
    )
    keypoint_patch_probs = (keypoint_patch_scores / subpixel_temp).softmax(dim=1)
    keypoint_offsets = torch.einsum("bkn, bkd ->bnd", keypoint_patch_probs, offsets)
    return keypoint_offsets


def _sample_at_keypoints(
    feature_map: torch.Tensor,
    keypoints: torch.Tensor,
    H: int,
    W: int,
    use_subpixel: bool,
) -> torch.Tensor:
    """
    Sample feature map values at keypoint locations using either bilinear interpolation
    or direct indexing.

    Args:
        feature_map: Feature map of shape (B, C, H, W)
        keypoints: Keypoint locations in pixel coordinates
                   [0, W-1] x [0, H-1], shape (B, N, 2)
        H: Feature map height
        W: Feature map width
        use_subpixel: If True, use bilinear interpolation; if False, use direct indexing

    Returns:
        Sampled features of shape (B, N, C) if C > 1, or (B, N) if C == 1
    """
    B, C = feature_map.shape[:2]

    if use_subpixel:
        grid_coords = torch.stack(
            [
                2.0 * keypoints[..., 0] / (W - 1) - 1.0,  # x
                2.0 * keypoints[..., 1] / (H - 1) - 1.0,  # y
            ],
            dim=-1,
        ).unsqueeze(2)  # (B, N, 1, 2)

        sampled = F.grid_sample(
            feature_map,
            grid_coords,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).squeeze(-1)  # (B, C, N)
    else:
        # For integer keypoints, use direct indexing
        idxs = (
            torch.round(keypoints[..., 1]).long() * W
            + torch.round(keypoints[..., 0]).long()
        )
        idxs = torch.clamp(idxs, min=0, max=W * H - 1)
        sampled = feature_map.view(B, C, -1).gather(
            2, idxs.unsqueeze(1).expand(B, C, -1)
        )  # (B, C, N)

    sampled = sampled.permute(0, 2, 1)  # (B, N, C)
    return sampled.squeeze(-1) if C == 1 else sampled  # (B, N) or (B, N, C)


class RaCo(Extractor):
    default_conf = {
        "name": "raco",
        # "weights": "https://github.com/cvg/RaCo/releases/download/v1.0.0/raco.pth",
        "weights": "raco/raco.pth",
        "max_num_keypoints": 512,
        "nms_radius": 3,
        "subpixel_sampling": True,
        "subpixel_temp": 0.5,
        "detection_threshold": -1,
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

        if self.conf.nms_radius % 2 == 0:
            raise ValueError(f"nms_radius must be odd, got {self.conf.nms_radius}")
        if self.conf.max_num_keypoints <= 0:
            raise ValueError(
                f"max_num_keypoints must be positive, got {self.conf.max_num_keypoints}"
            )

        self.normalizer = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )  # ImageNet normalization

        self.model = RacoModel(
            {
                "ranker": self.conf.ranker,
                "covariance_estimator": self.conf.covariance_estimator,
            }
        )

        # Initialize Softplus activation for covariance processing
        if self.conf.covariance_estimator:
            self.var_activation = nn.Softplus()

        if self.conf.weights is not None:
            try:
                # Check if weights is a URL or local path
                if isinstance(self.conf.weights, str) and self.conf.weights.startswith(
                    ("http://", "https://")
                ):
                    state_dict = torch.hub.load_state_dict_from_url(
                        self.conf.weights,
                        map_location="cpu",
                        progress=True,
                        weights_only=True,
                    )
                else:
                    state_dict = torch.load(
                        self.conf.weights, map_location="cpu", weights_only=True
                    )
                self.load_state_dict(state_dict, strict=True)
                print(f"[RaCo] Loaded weights from {self.conf.weights}")
            except Exception as e:
                print(f"Failed to load weights from {self.conf.weights}: {e}")
                raise

    def _sampling(
        self,
        keypoint_probs: torch.Tensor,
        nms_radius: int,
        num_kpts: Optional[int] = None,
        raw_logits: Optional[torch.Tensor] = None,
        subpixel: bool = False,
        subpixel_temp: Optional[float] = None,
    ) -> torch.Tensor:
        """Sample keypoints using NMS and optional subpixel refinement.

        Returns keypoints in pixel coordinates [0, W-1] x [0, H-1].
        """
        # Modified from DaD https://github.com/Parskatt/dad

        if num_kpts is None:
            num_kpts = self.conf.max_num_keypoints
        if subpixel_temp is None:
            subpixel_temp = self.conf.subpixel_temp

        B, C, H, W = keypoint_probs.size()
        device = keypoint_probs.device

        grid = _get_grid(B, H, W, device=device).reshape(B, H * W, 2)

        # Apply NMS
        max_pooled = F.max_pool2d(
            keypoint_probs, nms_radius, stride=1, padding=nms_radius // 2
        )
        keypoint_probs = keypoint_probs * (keypoint_probs == max_pooled)

        inds = torch.topk(keypoint_probs.reshape(B, H * W), k=num_kpts).indices
        kpts = torch.gather(grid, dim=1, index=inds[..., None].expand(B, num_kpts, 2))

        kpts = _to_pixel_coords(kpts, H, W)

        if subpixel and raw_logits is not None:
            keypoint_offsets = _compute_subpixel_offsets(
                raw_logits, inds, nms_radius, H, W, subpixel_temp
            )
            kpts = kpts + keypoint_offsets

        return kpts

    def forward(self, data: dict) -> dict:
        """Forward pass through the RaCo model."""
        # Preprocess image
        image = data["image"]
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)  # Convert to 3-channel greyscale
        image = self.normalizer(image)

        raw_score_map, ranker_map, cholesky_maps = self.model(image)

        # Compute probability maps using batchwise global softmax normalization
        keypoint_probs = F.softmax(raw_score_map.flatten(1), dim=1).reshape(
            raw_score_map.size()
        )

        keypoints = self._sampling(
            keypoint_probs=keypoint_probs,
            nms_radius=self.conf.nms_radius,
            raw_logits=raw_score_map,
            subpixel=self.conf.subpixel_sampling,
        )

        B, _, H, W = keypoint_probs.size()

        # Sample scores at keypoint locations
        probs = _sample_at_keypoints(
            keypoint_probs, keypoints, H, W, self.conf.subpixel_sampling
        )  # (B, N)

        if self.conf.detection_threshold > 0:
            mask = probs > self.conf.detection_threshold
            keypoints = keypoints[mask].view(keypoints.shape[0], -1, 2)
            probs = probs[mask].view(probs.shape[0], -1)

        out_dict = {
            "keypoints": keypoints + 0.5,  # (B, N, 2) in pixel coordinates
            "keypoint_scores": probs,  # (B, N)
        }

        if self.conf.ranker and ranker_map is not None:
            # Higher the ranker score, better the keypoint
            ranker_scores = _sample_at_keypoints(
                ranker_map, keypoints, H, W, self.conf.subpixel_sampling
            )  # (B, N)
            out_dict["ranker_scores"] = ranker_scores

        if self.conf.covariance_estimator and cholesky_maps is not None:
            # Apply activations to ensure L11 and L22 are positive
            # cholesky_maps has shape (B, 3, H, W) with [L11_prime, L21, L22_prime]
            processed_cholesky_maps = torch.stack(
                [
                    self.var_activation(cholesky_maps[:, 0]),  # L11
                    cholesky_maps[:, 1],  # L21 (no constraint)
                    self.var_activation(cholesky_maps[:, 2]),  # L22
                ],
                dim=1,
            )
            cholesky_scores = _sample_at_keypoints(
                processed_cholesky_maps, keypoints, H, W, self.conf.subpixel_sampling
            )  # (B, N, 3)
            covariances = _covariance_matrix_from_cholesky_elements(cholesky_scores)
            out_dict["covariances"] = covariances  # (B, N, 2, 2)

        return out_dict
