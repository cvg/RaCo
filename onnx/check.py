"""Check RaCo detector ONNX models against the PyTorch implementation.

Runs ONNX Runtime and PyTorch (with the given checkpoint) on asset images at
multiple resolutions (including non-multiples of 32) and keypoint budgets, and
compares detected keypoints and scores.

Requires onnx, onnxruntime (onnxruntime-gpu on GPU), numpy, opencv, and torch.

Usage (from the root of the repository):
    python onnx/check.py --cpu --model onnx/models/raco_detector.onnx
    python onnx/check.py --model onnx/models/raco_detector_fp16.onnx --timing
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

# Also run from a clone of the repository without installing it.
sys.path.append(str(Path(__file__).resolve().parent.parent))
from raco import RaCo  # noqa: E402
from raco.utils import load_image  # noqa: E402

ASSETS = Path(__file__).resolve().parent.parent / "assets"
# Include both original sizes and non-multiple-of-32 resizes to exercise padding.
CHECK_IMAGES = [
    ("i_castle.png", None),
    ("v_grace_1.ppm", None),
    ("v_grace_3.ppm", 777),
]

# Tolerances for float32 on CPU / float32 on CUDA / float16 on CUDA.
# Because top-K selects peaks from a dense score map, small numerical
# differences (TF32 or fp16) can swap a tiny fraction of near-tied keypoints
# near the K-th rank boundary. We therefore measure matched-keypoint accuracy
# (by nearest neighbor) and overlap fraction at 0.5px.
TOLERANCES = {
    "cpu": dict(px_median=1e-4, px_p99=1e-3, overlap=0.999, score_rel_median=1e-4),
    "cuda": dict(px_median=1e-3, px_p99=2e-2, overlap=0.99, score_rel_median=1e-3),
    "cuda_fp16": dict(px_median=5e-3, px_p99=5e-2, overlap=0.97, score_rel_median=5e-3),
}


def is_fp16_model(path: Path) -> bool:
    model = onnx.load(str(path), load_external_data=False)
    return any(
        init.data_type == onnx.TensorProto.FLOAT16 for init in model.graph.initializer
    )


def onnx_providers(cpu: bool) -> list[str]:
    if not cpu and "CUDAExecutionProvider" in ort.get_available_providers():
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def time_session(session, feeds: dict, warmup: int = 3, runs: int = 15) -> float:
    for _ in range(warmup):
        session.run(None, feeds)
    t = time.time()
    for _ in range(runs):
        session.run(None, feeds)
    return (time.time() - t) * 1000.0 / runs


def time_pytorch(
    model: RaCo, image: torch.Tensor, warmup: int = 3, runs: int = 15
) -> float:
    device = image.device
    with torch.no_grad():
        for _ in range(warmup):
            model({"image": image})
        if device.type == "cuda":
            torch.cuda.synchronize()
        t = time.time()
        for _ in range(runs):
            model({"image": image})
        if device.type == "cuda":
            torch.cuda.synchronize()
    return (time.time() - t) * 1000.0 / runs


def compare_outputs(
    kpts_pt: np.ndarray,
    scores_pt: np.ndarray,
    kpts_ort: np.ndarray,
    scores_ort: np.ndarray,
    tol: dict,
) -> tuple[bool, str]:
    assert kpts_pt.shape == kpts_ort.shape, (kpts_pt.shape, kpts_ort.shape)
    assert scores_pt.shape == scores_ort.shape, (scores_pt.shape, scores_ort.shape)
    assert kpts_ort.dtype == np.float32, kpts_ort.dtype
    assert scores_ort.dtype == np.float32, scores_ort.dtype
    if not (np.isfinite(kpts_ort).all() and np.isfinite(scores_ort).all()):
        return False, "non-finite ONNX output"

    # Nearest-neighbor matching from PyTorch keypoints to ONNX keypoints.
    diff = kpts_pt[:, None, :] - kpts_ort[None, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    nn_idx = np.argmin(dists, axis=1)
    nn_dist = dists[np.arange(len(kpts_pt)), nn_idx]

    matched = nn_dist <= 0.5
    overlap = float(matched.mean())
    if matched.any():
        px_median = float(np.median(nn_dist[matched]))
        px_p99 = float(np.quantile(nn_dist[matched], 0.99))
        rel_err = np.abs(scores_ort[nn_idx[matched]] - scores_pt[matched]) / np.maximum(
            np.abs(scores_pt[matched]), 1e-12
        )
        score_rel_median = float(np.median(rel_err))
    else:
        px_median = px_p99 = score_rel_median = float("inf")

    if int(matched.sum()) >= 2:
        matched_pt_idx = np.flatnonzero(matched)
        matched_ort_idx = nn_idx[matched]
        rank_corr = float(np.corrcoef(matched_pt_idx, matched_ort_idx)[0, 1])
    else:
        rank_corr = 1.0

    ok = (
        overlap >= tol["overlap"]
        and px_median <= tol["px_median"]
        and px_p99 <= tol["px_p99"]
        and score_rel_median <= tol["score_rel_median"]
        and rank_corr >= 0.99
    )
    msg = (
        f"overlap@0.5px={overlap:.2%} "
        f"px: median={px_median:.2e} p99={px_p99:.2e} "
        f"score_rel_med={score_rel_median:.2e} rank_r={rank_corr:.4f}"
    )
    return ok, msg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", type=Path, required=True, help="Detector ONNX model.")
    parser.add_argument(
        "--weights",
        default=RaCo.default_conf["weights"],
        help="PyTorch checkpoint (path or URL).",
    )
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
    parser.add_argument("--num_keypoints", default="512,2048")
    parser.add_argument("--timing", action="store_true", help="Report latency.")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    providers = onnx_providers(args.cpu)
    device = "cuda" if providers[0].startswith("CUDA") else "cpu"
    fp16 = is_fp16_model(args.model)
    print(f"ONNX Runtime {ort.__version__}, providers: {providers}, fp16: {fp16}")
    if fp16 and device == "cpu":
        raise SystemExit("The float16 model requires CUDA.")

    if device == "cuda" and hasattr(ort, "preload_dlls"):
        ort.preload_dlls(cuda=True, cudnn=True)

    session = ort.InferenceSession(str(args.model), providers=providers)
    if device == "cuda" and "CUDAExecutionProvider" not in session.get_providers():
        active = session.get_providers()
        raise SystemExit(
            f"CUDAExecutionProvider failed to initialize (active: {active})."
        )
    model = (
        RaCo(
            weights=args.weights,
            nms_radius=args.nms_radius,
            subpixel_temp=args.subpixel_temp,
            ranker=False,
            covariance_estimator=False,
        )
        .eval()
        .to(device)
    )

    tol = TOLERANCES[device + ("_fp16" if fp16 else "")]
    num_keypoints_list = [int(k) for k in args.num_keypoints.split(",")]
    all_ok = True

    for img_idx, (name, resize) in enumerate(CHECK_IMAGES):
        img = load_image(ASSETS / name, resize=resize)[None]
        h, w = img.shape[-2:]
        print(f"[{name} ({w}x{h})]")
        img_dev = img.to(device)
        img_np = img.numpy()
        # Exercise K=1 on the first image to verify broadcast/squeeze behavior
        # without inflating --timing across every resolution.
        ks = (
            [1, *num_keypoints_list]
            if img_idx == 0 and 1 not in num_keypoints_list
            else num_keypoints_list
        )
        for k in ks:
            model.conf.max_num_keypoints = k
            with torch.no_grad():
                pred = model({"image": img_dev})
            kpts_pt = pred["keypoints"][0].cpu().numpy()
            scores_pt = pred["keypoint_scores"][0].cpu().numpy()

            feeds = {"image": img_np, "max_keypoints": np.array(k, dtype=np.int64)}
            kpts_ort, scores_ort = session.run(None, feeds)
            ok, msg = compare_outputs(kpts_pt, scores_pt, kpts_ort[0], scores_ort[0], tol)
            if args.timing and k != 1:
                ort_ms = time_session(session, feeds)
                pt_ms = time_pytorch(model, img_dev)
                msg += f" pt={pt_ms:.1f}ms ort={ort_ms:.1f}ms"
            print(f"  K={k:4d}: {msg} {'OK' if ok else 'FAIL'}")
            all_ok &= ok

    print("CHECK", "PASSED" if all_ok else "FAILED")
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
