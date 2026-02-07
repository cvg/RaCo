"""
RaCo (Ranking and Covariance) Feature Extractor
"""

from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from .utils import ImagePreprocessor


class InputPadder:
    """Pads images such that dimensions are divisible by 8"""

    def __init__(self, h: int, w: int, divis_by: int = 8):
        self.ht = h
        self.wd = w
        pad_ht = (((self.ht // divis_by) + 1) * divis_by - self.ht) % divis_by
        pad_wd = (((self.wd // divis_by) + 1) * divis_by - self.wd) % divis_by
        self._pad = [
            pad_wd // 2,
            pad_wd - pad_wd // 2,
            pad_ht // 2,
            pad_ht - pad_ht // 2,
        ]

    def pad(self, x: torch.Tensor):
        assert x.ndim == 4
        return F.pad(x, self._pad, mode="replicate")

    def unpad(self, x: torch.Tensor):
        assert x.ndim == 4
        ht = x.shape[-2]
        wd = x.shape[-1]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0] : c[1], c[2] : c[3]]


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
    ):
        super().__init__()
        self.gate = nn.SELU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.gate(self.bn1(self.conv1(x)))  # B x in_channels x H x W
        x = self.gate(self.bn2(self.conv2(x)))  # B x out_channels x H x W
        return x


class ResBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.gate = nn.SELU(inplace=True)
        self.match_dims = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=1,
            stride=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.gate(out)

        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.match_dims(identity)

        out += identity
        out = self.gate(out)

        return out


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)


def conv3x3(in_planes, out_planes, stride=1, kernel_size=3):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
        bias=False,
    )


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


class RaCo(nn.Module):
    default_conf = {
        "name": "raco",
        "weights": "https://github.com/cvg/RaCo/releases/download/v1.0.0/raco.pth",
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
        super().__init__()

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

        # Model architecture based on ALIKED-n16 https://github.com/Shiaoming/ALIKED)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.AvgPool2d(kernel_size=4, stride=4)
        self.gate = nn.SELU(inplace=True)

        c1, c2, c3, c4 = 16, 32, 64, 128
        self.block1 = ConvBlock(3, c1)
        self.block2 = ResBlock(c1, c2)
        self.block3 = ResBlock(c2, c3)
        self.block4 = ResBlock(c3, c4)
        dim = c4

        self.conv1 = conv1x1(c1, dim // 4)
        self.conv2 = conv3x3(c2, dim // 4)
        self.conv3 = conv3x3(c3, dim // 4)
        self.conv4 = conv3x3(c4, dim // 4)
        self.upsample2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        self.upsample8 = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True)
        self.upsample32 = nn.Upsample(
            scale_factor=32, mode="bilinear", align_corners=True
        )

        self.score_head = nn.Sequential(
            conv1x1(dim, 8),
            nn.SELU(inplace=True),
            conv3x3(8, 4),
            nn.SELU(inplace=True),
            conv3x3(4, 4),
            nn.SELU(inplace=True),
            conv3x3(4, 1),
        )

        # Ranker head
        if self.conf.ranker:
            ranker_dim = 12
            ranker_layers = [ResBlock(3, ranker_dim)]
            ranker_layers += [ResBlock(ranker_dim, ranker_dim) for _ in range(8)]
            ranker_layers += [
                nn.Conv2d(
                    ranker_dim,
                    1,
                    kernel_size=5,
                    padding=2,
                    bias=True,
                    padding_mode="reflect",
                )
            ]
            self.ranker_head = nn.Sequential(*ranker_layers)

        # Covariance estimator head
        if self.conf.covariance_estimator:
            modules = []
            in_channels = dim
            for out_channels in [64, 32, 32]:
                modules.append(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False,
                        padding_mode="reflect",
                    )
                )
                modules.append(nn.LeakyReLU(inplace=True))
                in_channels = out_channels
            # Output 3 channels for Cholesky elements [L11, L21, L22]
            modules.append(
                nn.Conv2d(
                    in_channels=32,
                    out_channels=3,
                    kernel_size=1,
                    bias=True,
                    padding_mode="reflect",
                )
            )
            self.covariance_estimator_head = nn.Sequential(*modules)
            self.var_activation = nn.Softplus()

        if self.conf.weights is not None:
            # Load pretrained weights from URL or local path
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
        # Preprocess image
        image = data["image"]
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)  # Convert to 3-channel greyscale
        image = self.normalizer(image)

        div_by = 2**5
        padder = InputPadder(image.shape[-2], image.shape[-1], div_by)
        x = padder.pad(image)

        # Feature extraction
        x1 = self.block1(x)  # B x c1 x H x W
        x2 = self.pool2(x1)
        x2 = self.block2(x2)  # B x c2 x H/2 x W/2
        x3 = self.pool4(x2)
        x3 = self.block3(x3)  # B x c3 x H/8 x W/8
        x4 = self.pool4(x3)
        x4 = self.block4(x4)  # B x dim x H/32 x W/32

        # Feature aggregation
        x1 = self.gate(self.conv1(x1))  # B x dim//4 x H x W
        x2 = self.gate(self.conv2(x2))  # B x dim//4 x H//2 x W//2
        x3 = self.gate(self.conv3(x3))  # B x dim//4 x H//8 x W//8
        x4 = self.gate(self.conv4(x4))  # B x dim//4 x H//32 x W//32
        x2_up = self.upsample2(x2)  # B x dim//4 x H x W
        x3_up = self.upsample8(x3)  # B x dim//4 x H x W
        x4_up = self.upsample32(x4)  # B x dim//4 x H x W
        x1234 = torch.cat([x1, x2_up, x3_up, x4_up], dim=1)

        # Score head
        raw_score_map = self.score_head(x1234)
        raw_score_map = padder.unpad(raw_score_map)

        # Ranker head
        ranker_map = None
        if self.conf.ranker:
            ranker_map = self.ranker_head(x)
            ranker_map = padder.unpad(ranker_map)

        # Covariance estimator head
        cholesky_maps = None
        if self.conf.covariance_estimator:
            cholesky_maps = self.covariance_estimator_head(x1234)
            cholesky_maps = padder.unpad(cholesky_maps)
            # Apply softplus only to diagonal elements (L11 and L22), not L21
            cholesky_maps = torch.stack(
                [
                    self.var_activation(cholesky_maps[:, 0]),  # L11
                    cholesky_maps[:, 1],  # L21 (no constraint)
                    self.var_activation(cholesky_maps[:, 2]),  # L22
                ],
                dim=1,
            )

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
            # Cholesky maps already have softplus applied to L11 and L22
            cholesky_scores = _sample_at_keypoints(
                cholesky_maps, keypoints, H, W, self.conf.subpixel_sampling
            )  # (B, N, 3)
            covariances = _covariance_matrix_from_cholesky_elements(cholesky_scores)
            out_dict["covariances"] = covariances  # (B, N, 2, 2)

        return out_dict

    @torch.no_grad()
    def extract(self, img: torch.Tensor, **conf) -> dict:
        """Perform extraction with online resizing.

        Args:
            img: Input image tensor of shape (C, H, W) or (B, C, H, W)
            **conf: Additional preprocessing configuration (e.g., resize settings)

        Returns:
            Dictionary containing extracted features with scaled coordinates
        """
        if img.dim() == 3:
            img = img[None]  # add batch dim
        assert img.dim() == 4 and img.shape[0] == 1
        shape = img.shape[-2:][::-1]
        img, scales = ImagePreprocessor(**{**self.preprocess_conf, **conf})(img)
        feats = self.forward({"image": img})
        feats["image_size"] = torch.tensor(shape)[None].to(img).float()
        feats["keypoints"] = (feats["keypoints"] + 0.5) / scales[None] - 0.5

        # Scale covariances if present
        if "covariances" in feats:
            scales_mat = torch.diag(scales).to(img)
            feats["covariances"] = (
                scales_mat[None]
                @ feats["covariances"]
                @ scales_mat[None].transpose(-1, -2)
            )
        return feats
