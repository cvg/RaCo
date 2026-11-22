"""Export the RaCo keypoint detector to ONNX for COLMAP (onnx/check.py checks it).

Exports the detector-only graph (score_head -> NMS -> top-K -> subpixel
refinement, without the ranker or covariance heads):

- raco_detector.onnx / raco_detector_fp16.onnx (batch size 1, dynamic H, W)
    image          [1, 3, H, W] float32 in [0, 1]
    max_keypoints  scalar int64, runtime cap on detected keypoints (>= 1)
  ->
    keypoints      [1, K, 2] float32, pixel coordinates with origin at the
                   top-left image corner (center of top-left pixel is (0.5, 0.5))
    scores         [1, K] float32, keypoint probabilities at the keypoints

The image is padded symmetrically to a multiple of 32 (replicate mode) and the
score map is unpadded inside the graph, matching RaCo.forward() for any (H, W).

Graph optimizations (matching RaCo.forward() on images with >= max_keypoints
NMS peaks; when fewer peaks exist, output count is capped to min(max_keypoints,
num_maxima) instead of padding with zero-score non-peaks):
- Conv2d + BatchNorm2d folding in float32 before optional fp16 conversion.
- Native ONNX Selu instead of decomposed Elu + Mul.
- Factoring score_head[0] (1x1 conv, 128->8) into four 32->8 projections before
  bilinear upsampling, reducing upsampled channels 4x and removing the
  128-channel full-resolution concatenation.
- Compacting nms_radius x nms_radius (default 3x3) NMS local maxima with
  NonZero before TopK (speeds up ONNX Runtime CUDA TopK ~8x on high-resolution
  images).
- Direct nms_radius x nms_radius neighbor indexing for subpixel refinement
  instead of nn.Unfold over the full image.

With --fp16, the convolutional backbone and score head run in float16 while
ImageNet normalization, global softmax, NMS, top-K, subpixel refinement, and
score sampling remain in float32. The I/O types stay float32.

Requires onnx and onnxscript.

Usage (from the root of the repository):
    python onnx/export.py          # fp32 detector
    python onnx/export.py --fp16   # fp16 detector (for CUDA)
The models are written to onnx/models/ by default. onnx/check.py checks them
against PyTorch.
"""

import argparse
import copy
import hashlib
import sys
import time
from pathlib import Path

import onnx
import onnxscript
import torch
import torch.nn.functional as F
from torch import nn

# Also run from a clone of the repository without installing it.
sys.path.append(str(Path(__file__).resolve().parent.parent))
from raco import RaCo  # noqa: E402
from raco.raco import InputPadder, _sample_at_keypoints  # noqa: E402

DEFAULT_WEIGHTS = RaCo.default_conf["weights"]
MIN_OPSET = 18

# PyTorch nn.SELU constants.
SELU_ALPHA = 1.6732632423543772
SELU_GAMMA = 1.0507009873554805


@torch.library.custom_op("raco::selu", mutates_args=())
def native_selu(x: torch.Tensor) -> torch.Tensor:
    return F.selu(x)


@native_selu.register_fake
def _(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


class OnnxSELU(nn.Module):
    """nn.SELU exported as a single native ONNX Selu operator."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.raco.selu(x)


def replace_selu(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.SELU):
            setattr(module, name, OnnxSELU())
        else:
            replace_selu(child)


@torch.library.custom_op("raco::pad_to_multiple", mutates_args=())
def pad_to_multiple(x: torch.Tensor, divisor: int) -> torch.Tensor:
    """Symmetrically replicate-pad (B, C, H, W) so H and W are divisible by divisor."""
    return InputPadder(x.shape[-2], x.shape[-1], divisor).pad(x).clone()


@pad_to_multiple.register_fake
def _(x: torch.Tensor, divisor: int) -> torch.Tensor:
    ctx = torch.library.get_ctx()
    # Express padded dims as explicit multiples of divisor so torch.export can
    # prove divisibility through AvgPool2d(2/4) without specializing on H % 32.
    h_blocks = ctx.new_dynamic_size(min=26, max=120)
    w_blocks = ctx.new_dynamic_size(min=26, max=120)
    torch._check(h_blocks >= 26)
    torch._check(w_blocks >= 26)
    torch._check(h_blocks <= 120)
    torch._check(w_blocks <= 120)
    return x.new_empty(x.shape[0], x.shape[1], divisor * h_blocks, divisor * w_blocks)


@torch.library.custom_op("raco::unpad_to", mutates_args=())
def unpad_to(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Undo pad_to_multiple(ref) on a 4-D tensor x with the same spatial padding."""
    h, w = ref.shape[-2], ref.shape[-1]
    pad_t = (x.shape[-2] - h) // 2
    pad_l = (x.shape[-1] - w) // 2
    return x[..., pad_t : pad_t + h, pad_l : pad_l + w].clone()


@unpad_to.register_fake
def _(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return x.new_empty(x.shape[0], x.shape[1], ref.shape[2], ref.shape[3])


@torch.library.custom_op("raco::upsample_bilinear", mutates_args=())
def upsample_bilinear(x: torch.Tensor, scale: int) -> torch.Tensor:
    """Bilinear upsample by an integer factor with align_corners=True."""
    return F.interpolate(
        x, scale_factor=float(scale), mode="bilinear", align_corners=True
    )


@upsample_bilinear.register_fake
def _(x: torch.Tensor, scale: int) -> torch.Tensor:
    return x.new_empty(x.shape[0], x.shape[1], x.shape[2] * scale, x.shape[3] * scale)


@torch.library.custom_op("raco::nms_topk_coords", mutates_args=())
def nms_topk_coords(
    probs: torch.Tensor, mask: torch.Tensor, max_k: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select top-K (h_inds, w_inds) among NMS local maxima (mask == True).

    Computes row/column coordinates in exact int64 (avoiding float32 floor
    division precision loss on images with > 2^24 pixels).
    """
    w = probs.shape[-1]
    probs_flat = probs.reshape(-1)
    nz = torch.nonzero(mask.reshape(-1), as_tuple=False).squeeze(-1)
    k = min(int(max_k.item()), int(nz.shape[0]))
    _, top_i = torch.topk(probs_flat[nz], k=k, dim=0, largest=True, sorted=True)
    orig_inds = nz[top_i]
    h_inds = torch.div(orig_inds, w, rounding_mode="floor")
    w_inds = orig_inds - h_inds * w
    return h_inds.unsqueeze(0), w_inds.unsqueeze(0)


@nms_topk_coords.register_fake
def _(
    probs: torch.Tensor, mask: torch.Tensor, max_k: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    ctx = torch.library.get_ctx()
    # min=2 avoids 0/1 broadcast specialization during torch.export tracing;
    # runtime k=1 remains valid in the exported ONNX graph.
    k_sym = ctx.new_dynamic_size(min=2)
    torch._check(k_sym >= 2)
    return (
        probs.new_empty((1, k_sym), dtype=torch.int64),
        probs.new_empty((1, k_sym), dtype=torch.int64),
    )


def make_custom_translation_table(opset: int) -> dict:
    op = onnxscript.values.Opset("", opset)

    def native_selu_onnx(x):
        return op.Selu(x, alpha=SELU_ALPHA, gamma=SELU_GAMMA)

    def pad_to_multiple_onnx(x, divisor: int):
        shape = op.Shape(x)
        h = op.Slice(shape, [2], [3])
        w = op.Slice(shape, [3], [4])
        d = op.Constant(value_ints=[divisor])
        two = op.Constant(value_ints=[2])
        zero = op.Constant(value_ints=[0, 0])
        pad_h = op.Mod(op.Sub(d, op.Mod(h, d)), d)
        pad_w = op.Mod(op.Sub(d, op.Mod(w, d)), d)
        pad_t = op.Div(pad_h, two)
        pad_b = op.Sub(pad_h, pad_t)
        pad_l = op.Div(pad_w, two)
        pad_r = op.Sub(pad_w, pad_l)
        pads = op.Concat(zero, pad_t, pad_l, zero, pad_b, pad_r, axis=0)
        return op.Pad(x, pads, mode="edge")

    def unpad_to_onnx(x, ref):
        x_shape = op.Shape(x)
        ref_shape = op.Shape(ref)
        x_hw = op.Slice(x_shape, [2], [4])
        ref_hw = op.Slice(ref_shape, [2], [4])
        two = op.Constant(value_ints=[2, 2])
        starts = op.Div(op.Sub(x_hw, ref_hw), two)
        ends = op.Add(starts, ref_hw)
        axes = op.Constant(value_ints=[2, 3])
        return op.Slice(x, starts, ends, axes)

    def upsample_bilinear_onnx(x, scale: int):
        scales = op.Constant(value_floats=[1.0, 1.0, float(scale), float(scale)])
        return op.Resize(
            x,
            None,
            scales,
            mode="linear",
            coordinate_transformation_mode="align_corners",
        )

    def nms_topk_coords_onnx(probs, mask, max_k):
        ax0 = op.Constant(value_ints=[0])
        one = op.Constant(value_ints=[1])
        neg1 = op.Constant(value_ints=[-1])
        width = op.Slice(op.Shape(probs), [3], [4])
        probs_flat = op.Reshape(probs, neg1)
        mask_flat = op.Reshape(mask, neg1)
        nz = op.Squeeze(op.NonZero(mask_flat), ax0)
        num_maxima = op.Shape(nz)
        k = op.Min(op.Reshape(max_k, one), num_maxima)
        cand_vals = op.Gather(probs_flat, nz, axis=0)
        _, top_i = op.TopK(cand_vals, k, axis=0, largest=1, sorted=1)
        orig_inds = op.Gather(nz, top_i, axis=0)
        h_inds = op.Div(orig_inds, width)
        w_inds = op.Sub(orig_inds, op.Mul(h_inds, width))
        return op.Unsqueeze(h_inds, ax0), op.Unsqueeze(w_inds, ax0)

    return {
        torch.ops.raco.selu.default: native_selu_onnx,
        torch.ops.raco.pad_to_multiple.default: pad_to_multiple_onnx,
        torch.ops.raco.unpad_to.default: unpad_to_onnx,
        torch.ops.raco.upsample_bilinear.default: upsample_bilinear_onnx,
        torch.ops.raco.nms_topk_coords.default: nms_topk_coords_onnx,
    }


class RaCoDetectorONNX(nn.Module):
    """Detector-only RaCo graph (batch size 1, dynamic H, W, and max_keypoints)."""

    def __init__(self, raco: RaCo, fp16: bool = False):
        super().__init__()
        self.raco = copy.deepcopy(raco).eval()
        self.fp16 = fp16
        self.nms_radius = int(self.raco.conf.nms_radius)
        self.subpixel_sampling = bool(self.raco.conf.subpixel_sampling)
        self.subpixel_temp = float(self.raco.conf.subpixel_temp)

        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

        # Subpixel local patch offsets matching _compute_subpixel_offsets().
        r = self.nms_radius // 2
        offset_range = torch.arange(-r, r + 1, dtype=torch.int64)
        gy, gx = torch.meshgrid(offset_range, offset_range, indexing="ij")
        self.register_buffer("patch_dy", gy.reshape(1, -1, 1))
        self.register_buffer("patch_dx", gx.reshape(1, -1, 1))
        self.register_buffer(
            "patch_offsets",
            torch.stack((gx, gy), dim=-1).reshape(1, -1, 1, 2).to(torch.float32),
        )

        # Fold Conv2d + BatchNorm2d in float32 before optional fp16 conversion.
        for block in (
            self.raco.block1,
            self.raco.block2,
            self.raco.block3,
            self.raco.block4,
        ):
            block.conv1 = nn.utils.fuse_conv_bn_eval(block.conv1, block.bn1)
            block.bn1 = nn.Identity()
            block.conv2 = nn.utils.fuse_conv_bn_eval(block.conv2, block.bn2)
            block.bn2 = nn.Identity()

        # Factor score_head[0] (1x1 Conv 128->8) across the four scales before
        # upsampling: W @ cat([x1, U2(x2), U8(x3), U32(x4)]) ==
        # W1(x1) + U2(W2(x2)) + U8(W3(x3)) + U32(W4(x4)).
        first_conv = self.raco.score_head[0]
        assert isinstance(first_conv, nn.Conv2d) and first_conv.kernel_size == (1, 1)
        in_per_scale = first_conv.in_channels // 4
        out_ch = first_conv.out_channels
        self.score_projs = nn.ModuleList()
        for i in range(4):
            proj = nn.Conv2d(
                in_per_scale,
                out_ch,
                kernel_size=1,
                bias=(i == 0 and first_conv.bias is not None),
            )
            with torch.no_grad():
                proj.weight.copy_(
                    first_conv.weight[:, i * in_per_scale : (i + 1) * in_per_scale]
                )
                if proj.bias is not None:
                    proj.bias.copy_(first_conv.bias)
            self.score_projs.append(proj)
        self.score_tail = nn.Sequential(*list(self.raco.score_head.children())[1:])
        self.raco.score_head = nn.Identity()

        replace_selu(self.raco)
        replace_selu(self.score_tail)

        if fp16:
            for mod in (
                self.raco.block1,
                self.raco.block2,
                self.raco.block3,
                self.raco.block4,
                self.raco.conv1,
                self.raco.conv2,
                self.raco.conv3,
                self.raco.conv4,
                self.score_projs,
                self.score_tail,
            ):
                mod.half()

    def _subpixel_offsets(
        self,
        raw_score_map: torch.Tensor,
        h_inds: torch.Tensor,
        w_inds: torch.Tensor,
        w: int,
    ) -> torch.Tensor:
        r = self.nms_radius // 2
        padded = F.pad(raw_score_map, (r, r, r, r), mode="constant", value=0.0)
        w_pad = w + 2 * r
        nb_inds = (h_inds.unsqueeze(1) + r + self.patch_dy) * w_pad + (
            w_inds.unsqueeze(1) + r + self.patch_dx
        )  # (1, K_p, N)
        patch_scores = torch.gather(
            padded.reshape(1, -1), dim=1, index=nb_inds.reshape(1, -1)
        ).reshape(1, self.nms_radius**2, -1)
        patch_probs = (patch_scores / self.subpixel_temp).softmax(dim=1)
        return (patch_probs.unsqueeze(-1) * self.patch_offsets).sum(dim=1)

    def forward(
        self, image: torch.Tensor, max_keypoints: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h, w = image.shape[-2], image.shape[-1]
        x = (image - self.mean) / self.std
        x = torch.ops.raco.pad_to_multiple(x, 32)

        if self.fp16:
            x = x.half()

        m = self.raco
        x1 = m.block1(x)
        x2 = m.block2(m.pool2(x1))
        x3 = m.block3(m.pool4(x2))
        x4 = m.block4(m.pool4(x3))

        s1 = self.score_projs[0](m.gate(m.conv1(x1)))
        s2 = torch.ops.raco.upsample_bilinear(self.score_projs[1](m.gate(m.conv2(x2))), 2)
        s3 = torch.ops.raco.upsample_bilinear(self.score_projs[2](m.gate(m.conv3(x3))), 8)
        s4 = torch.ops.raco.upsample_bilinear(
            self.score_projs[3](m.gate(m.conv4(x4))), 32
        )
        raw_score_map = self.score_tail(s1 + s2 + s3 + s4)

        if self.fp16:
            raw_score_map = raw_score_map.float()

        raw_score_map = torch.ops.raco.unpad_to(raw_score_map, image)

        # Global softmax, NMS, top-K, subpixel refinement, and score sampling in fp32.
        keypoint_probs = F.softmax(raw_score_map.flatten(1), dim=1).reshape(1, 1, h, w)
        max_pooled = F.max_pool2d(
            keypoint_probs,
            self.nms_radius,
            stride=1,
            padding=self.nms_radius // 2,
        )
        nms_mask = keypoint_probs == max_pooled
        h_inds, w_inds = torch.ops.raco.nms_topk_coords(
            keypoint_probs, nms_mask, max_keypoints
        )
        kpts = torch.stack([w_inds.float(), h_inds.float()], dim=-1)

        if self.subpixel_sampling:
            kpts = kpts + self._subpixel_offsets(raw_score_map, h_inds, w_inds, w)

        scores = _sample_at_keypoints(keypoint_probs, kpts, h, w, self.subpixel_sampling)
        return kpts + 0.5, scores


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_dims_dynamic(path: Path) -> None:
    """Clear intermediate shape annotations and name the dynamic dimensions."""
    model = onnx.load(str(path))
    del model.graph.value_info[:]
    for value in model.graph.input:
        if value.name == "image":
            for idx, name in ((2, "height"), (3, "width")):
                dim = value.type.tensor_type.shape.dim[idx]
                dim.Clear()
                dim.dim_param = name
    for value in model.graph.output:
        if value.name in ("keypoints", "scores"):
            dim = value.type.tensor_type.shape.dim[1]
            dim.Clear()
            dim.dim_param = "num_keypoints"
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def export_detector(
    raco: RaCo, path: Path, opset: int = MIN_OPSET, fp16: bool = False
) -> None:
    if opset < MIN_OPSET:
        raise ValueError(f"ONNX opset must be >= {MIN_OPSET}, got {opset}")
    model = RaCoDetectorONNX(raco, fp16=fp16).eval()
    image = torch.rand(1, 3, 491, 653)
    max_keypoints = torch.tensor(512, dtype=torch.int64)
    h_dim = torch.export.Dim("height", min=32, max=16384)
    w_dim = torch.export.Dim("width", min=32, max=16384)
    torch.onnx.export(
        model,
        (image, max_keypoints),
        str(path),
        input_names=["image", "max_keypoints"],
        output_names=["keypoints", "scores"],
        dynamic_shapes={"image": {2: h_dim, 3: w_dim}, "max_keypoints": None},
        opset_version=opset,
        dynamo=True,
        external_data=False,
        custom_translation_table=make_custom_translation_table(opset),
    )
    make_dims_dynamic(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="Path or URL.")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(__file__).resolve().parent / "models",
    )
    parser.add_argument("--opset", type=int, default=MIN_OPSET)
    parser.add_argument(
        "--nms_radius",
        type=int,
        default=RaCo.default_conf["nms_radius"],
    )
    parser.add_argument(
        "--subpixel_temp",
        type=float,
        default=RaCo.default_conf["subpixel_temp"],
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Cast the convolutional backbone and score head to float16 (for CUDA).",
    )
    args = parser.parse_args()
    if args.opset < MIN_OPSET:
        parser.error(f"--opset must be >= {MIN_OPSET}, got {args.opset}")
    args.model_name = "raco_detector" + ("_fp16" if args.fp16 else "") + ".onnx"
    return args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raco = RaCo(
        weights=args.weights,
        nms_radius=args.nms_radius,
        subpixel_temp=args.subpixel_temp,
        ranker=False,
        covariance_estimator=False,
    ).eval()
    path = args.output_dir / args.model_name
    t = time.time()
    export_detector(raco, path, opset=args.opset, fp16=args.fp16)
    print(
        f"Exported {path} in {time.time() - t:.1f}s "
        f"({path.stat().st_size / 1e6:.2f}MB, sha256={sha256sum(path)})"
    )


if __name__ == "__main__":
    main()
