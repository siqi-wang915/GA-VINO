from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Avoid writing Matplotlib/Fontconfig caches into the user's home directory on
# compute nodes where that directory may be read-only.
PLOT_CACHE_ROOT = Path(tempfile.gettempdir()) / "hole_gpu_plot_cache"
PLOT_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(PLOT_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PLOT_CACHE_ROOT / "xdg"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm

from shared.common.paths import build_paths, normalize_case_name


SUPPORTED_CASES = (
    "hole_um_up",
    "hole_um_grfp",
    "holes2_um_up",
    "holes2_um_grfp",
    "lshape_grfm_up",
    "lshape_grfm_grfp",
    "lshape_grfm_up_065_085",
    "annular_sector_grfm_grfp",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot CPU and GPU FEM transverse displacement fields side by side."
        )
    )
    parser.add_argument(
        "--case-name",
        choices=SUPPORTED_CASES,
        default="hole_um_up",
        help="Case family (default: hole_um_up).",
    )
    parser.add_argument(
        "--case-tag",
        default="case_0000",
        help="Case tag, for example case_0000 (default: case_0000).",
    )
    parser.add_argument(
        "--cpu-path",
        type=Path,
        default=None,
        help="Optional explicit CPU FEM .npz path.",
    )
    parser.add_argument(
        "--gpu-path",
        type=Path,
        default=None,
        help="Optional explicit GPU FEM .npz path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=THIS_FILE.parent / "outputs" / "comparisons",
        help="Figure output directory.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG resolution (default: 300).",
    )
    return parser.parse_args()


def normalize_case_tag(value: str) -> str:
    tag = Path(str(value).strip()).stem
    for suffix in ("_gpu", "_fem"):
        if tag.endswith(suffix):
            tag = tag[: -len(suffix)]
    if not tag:
        raise ValueError("--case-tag must not be empty")
    return tag


def load_result(path: Path, label: str) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} result does not exist: {path}")
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    missing = [
        key for key in ("query_xyz", "faces", "disp_scalar") if key not in data
    ]
    if missing:
        raise KeyError(f"{label} result {path} is missing keys: {missing}")
    return data


def validate_and_extract(
    cpu: dict[str, np.ndarray], gpu: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cpu_xyz = np.asarray(cpu["query_xyz"], dtype=np.float64)
    gpu_xyz = np.asarray(gpu["query_xyz"], dtype=np.float64)
    cpu_faces = np.asarray(cpu["faces"], dtype=np.int64)
    gpu_faces = np.asarray(gpu["faces"], dtype=np.int64)
    cpu_w = np.asarray(cpu["disp_scalar"], dtype=np.float64).reshape(-1)
    gpu_w = np.asarray(gpu["disp_scalar"], dtype=np.float64).reshape(-1)

    if cpu_xyz.ndim != 2 or cpu_xyz.shape[1] < 2:
        raise ValueError(f"Invalid CPU query_xyz shape: {cpu_xyz.shape}")
    if cpu_xyz.shape != gpu_xyz.shape or not np.allclose(
        cpu_xyz, gpu_xyz, rtol=0.0, atol=1.0e-12
    ):
        raise ValueError(
            "CPU and GPU query points are not aligned. Compare results from "
            "the same case_name and case_tag."
        )
    if cpu_faces.shape != gpu_faces.shape or not np.array_equal(
        cpu_faces, gpu_faces
    ):
        raise ValueError(
            "CPU and GPU triangle connectivity differs. Compare results from "
            "the same case_name and case_tag."
        )
    if cpu_w.shape[0] != cpu_xyz.shape[0] or gpu_w.shape[0] != gpu_xyz.shape[0]:
        raise ValueError("disp_scalar length does not match query point count")
    if not np.all(np.isfinite(cpu_w)) or not np.all(np.isfinite(gpu_w)):
        raise ValueError("CPU or GPU displacement contains NaN/Inf")

    return cpu_xyz[:, :2], cpu_faces, cpu_w, gpu_w


def shared_color_scale(cpu_w: np.ndarray, gpu_w: np.ndarray):
    vmin = float(min(np.min(cpu_w), np.min(gpu_w)))
    vmax = float(max(np.max(cpu_w), np.max(gpu_w)))
    if np.isclose(vmin, vmax):
        pad = max(abs(vmin), 1.0) * 1.0e-12
        vmin -= pad
        vmax += pad
    scale = max(abs(vmin), abs(vmax), np.finfo(float).eps)
    sign_tolerance = 1.0e-10 * scale
    if vmin < -sign_tolerance and vmax > sign_tolerance:
        return TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax), "RdBu_r"
    # Do not let round-off values near zero force a diverging scale or create
    # visually prominent boundary artifacts.
    if vmin >= -sign_tolerance:
        vmin = 0.0
    if vmax <= sign_tolerance:
        vmax = 0.0
    return Normalize(vmin=vmin, vmax=vmax), "viridis"


def scalar_value(data: dict[str, np.ndarray], key: str) -> float | None:
    if key not in data:
        return None
    value = np.asarray(data[key]).reshape(-1)
    return float(value[0]) if value.size else None


def make_figure(
    *,
    xy: np.ndarray,
    faces: np.ndarray,
    cpu_w: np.ndarray,
    gpu_w: np.ndarray,
    case_name: str,
    case_tag: str,
    cpu: dict[str, np.ndarray],
    gpu: dict[str, np.ndarray],
    png_path: Path,
    pdf_path: Path,
    dpi: int,
) -> dict[str, float]:
    difference = gpu_w - cpu_w
    cpu_norm = float(np.linalg.norm(cpu_w))
    relative_l2 = float(np.linalg.norm(difference) / max(cpu_norm, np.finfo(float).eps))
    rmse = float(np.sqrt(np.mean(difference**2)))
    max_abs_error = float(np.max(np.abs(difference)))

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.5,
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
            "savefig.bbox": "tight",
        }
    )

    triangulation = mtri.Triangulation(xy[:, 0], xy[:, 1], triangles=faces)
    norm, cmap = shared_color_scale(cpu_w, gpu_w)

    fig, axes = plt.subplots(
        1, 2, figsize=(7.0, 3.15), sharex=True, sharey=True, constrained_layout=True
    )
    panel_data = (("CPU FEM", cpu_w), ("GPU FEM", gpu_w))
    mappable = None
    for ax, (label, values) in zip(axes, panel_data):
        mappable = ax.tripcolor(
            triangulation,
            values,
            shading="gouraud",
            cmap=cmap,
            norm=norm,
            rasterized=True,
        )
        ax.set_title(f"{label}\nmax $|w|$ = {np.max(np.abs(values)):.6e}")
        ax.set_xlabel("$x$")
        ax.set_aspect("equal", adjustable="box")
        ax.margins(0.01)
    axes[0].set_ylabel("$y$")

    assert mappable is not None
    colorbar = fig.colorbar(mappable, ax=axes, shrink=0.88, pad=0.025)
    colorbar.set_label("Transverse displacement $w$")
    colorbar.formatter.set_powerlimits((-3, 3))
    colorbar.update_ticks()

    fig.suptitle(
        f"{case_name} / {case_tag}   "
        f"relative $L_2$ = {relative_l2:.3e}, "
        f"max error = {max_abs_error:.3e}",
        fontsize=10.5,
        fontweight="bold",
    )

    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=dpi)
    fig.savefig(pdf_path)
    plt.close(fig)

    metrics = {
        "relative_l2": relative_l2,
        "rmse": rmse,
        "max_abs_error": max_abs_error,
        "cpu_max_abs_w": float(np.max(np.abs(cpu_w))),
        "gpu_max_abs_w": float(np.max(np.abs(gpu_w))),
    }
    cpu_time = scalar_value(cpu, "fem_solve_time_sec")
    gpu_time = scalar_value(gpu, "fem_solve_time_sec")
    if cpu_time is not None:
        metrics["cpu_solve_time_sec"] = cpu_time
    if gpu_time is not None:
        metrics["gpu_solve_time_sec"] = gpu_time
    return metrics


def main() -> None:
    args = parse_args()
    case_name = normalize_case_name(args.case_name)
    case_tag = normalize_case_tag(args.case_tag)
    case_prefix = f"{case_name}_"
    if case_tag.startswith(case_prefix):
        case_tag = case_tag[len(case_prefix) :]
    paths = build_paths(case_name, root=PROJECT_ROOT, create=False)

    cpu_path = (
        args.cpu_path.resolve()
        if args.cpu_path is not None
        else paths["fem_prediction_root"] / f"{case_tag}_fem.npz"
    )
    if args.gpu_path is not None:
        gpu_path = args.gpu_path.resolve()
    else:
        prediction_dir = THIS_FILE.parent / "outputs" / "predictions"
        named_gpu_path = prediction_dir / f"{case_name}_{case_tag}_gpu.npz"
        legacy_gpu_path = prediction_dir / f"{case_tag}_gpu.npz"
        gpu_path = named_gpu_path if named_gpu_path.is_file() else legacy_gpu_path
    output_dir = args.output_dir.resolve() / case_name
    output_stem = output_dir / f"{case_name}_{case_tag}_cpu_gpu_displacement"

    cpu = load_result(cpu_path, "CPU FEM")
    gpu = load_result(gpu_path, "GPU FEM")
    xy, faces, cpu_w, gpu_w = validate_and_extract(cpu, gpu)
    metrics = make_figure(
        xy=xy,
        faces=faces,
        cpu_w=cpu_w,
        gpu_w=gpu_w,
        case_name=case_name,
        case_tag=case_tag,
        cpu=cpu,
        gpu=gpu,
        png_path=output_stem.with_suffix(".png"),
        pdf_path=output_stem.with_suffix(".pdf"),
        dpi=args.dpi,
    )

    print(f"[Compare] CPU input : {cpu_path}")
    print(f"[Compare] GPU input : {gpu_path}")
    print(f"[Compare] relative L2 error : {metrics['relative_l2']:.6e}")
    print(f"[Compare] RMSE              : {metrics['rmse']:.6e}")
    print(f"[Compare] max abs error     : {metrics['max_abs_error']:.6e}")
    if "cpu_solve_time_sec" in metrics:
        print(f"[Compare] CPU solve time    : {metrics['cpu_solve_time_sec']:.6f} s")
    if "gpu_solve_time_sec" in metrics:
        print(f"[Compare] GPU solve time    : {metrics['gpu_solve_time_sec']:.6f} s")
    print(f"[Compare] PNG output: {output_stem.with_suffix('.png')}")
    print(f"[Compare] PDF output: {output_stem.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
