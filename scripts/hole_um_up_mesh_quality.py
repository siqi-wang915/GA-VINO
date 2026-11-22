from __future__ import annotations

"""End-to-end fixed-size mesh-quality study for ``hole_um_up``.

Quality level 1 is the existing ``hole_um_up`` case.  Levels 2--4 use the
same samples, boundary vertices, topology, and mesh counts, with successively
larger boundary-safe interior jitter.
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np


_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from NO.problem.hole_mesh_convergence import mesh_quality_statistics
from NO.problem.hole_mesh_quality import QUALITY_LEVELS, quality_case_name
from shared.common.data_io import load_dataset, load_test_tags, load_train_tags
from shared.common.paths import build_paths
from shared.prep.hole_um_up_mesh_quality import prepare_hole_um_up_mesh_quality


SOURCE_CASE_NAME = "hole_um_up"
FIXED_STUDY_EPOCHS = 500
# Keep the fixed-schedule level-1 model separate from the ordinary early-stop
# hole_um_up baseline, which remains the result used in the main paper table.
SOURCE_METHOD_NAME = "mesh_quality_500"


def _result_paths(
    *,
    case_name: str,
    root: Path,
    method_name: Optional[str] = None,
) -> Dict[str, Path]:
    """Return NO/compare paths for a base or method-specific result."""
    paths = build_paths(case_name, root=root, create=True)
    if method_name is None:
        return paths
    method_case = f"{case_name}_{method_name}"
    no_output_root = root / "NO" / "outputs" / method_case
    compare_root = root / "compare" / method_case
    no_prediction_root = no_output_root / "predictions"
    no_prediction_root.mkdir(parents=True, exist_ok=True)
    compare_root.mkdir(parents=True, exist_ok=True)
    paths = dict(paths)
    paths.update(
        {
            "no_output_root": no_output_root,
            "no_prediction_root": no_prediction_root,
            "compare_root": compare_root,
            "final_metrics_npz_path": compare_root / "final_test_error_metrics.npz",
            "final_metrics_csv_path": compare_root / "final_test_error_metrics.csv",
        }
    )
    return paths
METRIC_KEYS = (
    "l2_error",
    "h1_semi_error",
    "energy_error",
    "theta_l2_error",
    "curvature_error",
)


def _read_csv(path: Path) -> list[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Required result table not found: {path}")
    with open(path, "r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _mean(rows: Iterable[Dict[str, Any]], key: str) -> float:
    values = []
    for row in rows:
        value = row.get(key, None)
        if value not in (None, ""):
            values.append(float(value))
    return float(np.mean(values)) if values else float("nan")


def _std(rows: Iterable[Dict[str, Any]], key: str) -> float:
    values = []
    for row in rows:
        value = row.get(key, None)
        if value not in (None, ""):
            values.append(float(value))
    return float(np.std(values)) if values else float("nan")


def _source_quality_rows(root: Path) -> list[Dict[str, Any]]:
    train, test, _ = load_dataset(case_name=SOURCE_CASE_NAME, root=root)
    rows = []
    for split, samples in (("train", train), ("test", test)):
        for index, sample in enumerate(samples):
            verts = np.asarray(sample["verts_xy"], dtype=np.float64)
            faces = np.asarray(sample["faces"], dtype=np.int64)
            rows.append(
                {
                    "case_name": SOURCE_CASE_NAME,
                    "split": split,
                    "sample_index": index,
                    "sample_id": int(sample.get("sample_id", index)),
                    "quality_level": 1,
                    "jitter_amplitude": 0.0,
                    "n_vertices": len(verts),
                    "n_faces": len(faces),
                    **mesh_quality_statistics(verts, faces),
                }
            )
    return rows


def run_fem_quality_cases(
    *,
    root: Optional[str | Path] = None,
    levels: Sequence[int] = QUALITY_LEVELS,
    overwrite: bool = False,
    include_train: bool = False,
) -> Dict[str, Any]:
    """Solve test FEM references; optionally solve train cases as well."""
    from FEM.solvers.registry import solve_cases

    results = {}
    for level in levels:
        case_name = quality_case_name(level)
        tags = list(load_test_tags(case_name=case_name, root=root))
        if include_train:
            tags.extend(load_train_tags(case_name=case_name, root=root))
        print("=" * 80)
        print(f"[Mesh quality] FEM {case_name}: {len(tags)} cases")
        results[case_name] = solve_cases(
            case_name=case_name,
            case_tags=tags,
            root=root,
            overwrite=overwrite,
            n_load_steps=1,
        )
        if results[case_name].get("failed"):
            raise RuntimeError(
                f"FEM failed for {case_name}: {results[case_name]['failed'][:3]}"
            )
    return results


def train_quality_cases(
    *,
    root: Optional[str | Path] = None,
    levels: Sequence[int] = QUALITY_LEVELS,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    lr: Optional[float] = None,
    eval_fem_every: Optional[int] = None,
    device: str = "auto",
) -> Dict[str, Any]:
    # Device selection occurs when NO.common.core is imported.
    os.environ["NO_DEVICE"] = str(device)
    from NO.train.train_hole_um_up_mr import run_train_hole_um_up

    if epochs not in (None, FIXED_STUDY_EPOCHS):
        raise ValueError(
            f"Mesh-quality training is fixed at {FIXED_STUDY_EPOCHS} epochs; "
            f"got --epochs {epochs}."
        )
    override: Dict[str, Any] = {
        "epochs": FIXED_STUDY_EPOCHS,
        "max_epochs": FIXED_STUDY_EPOCHS,
        "early_stopping": False,
    }
    if batch_size is not None:
        override["batch_size"] = int(batch_size)
    if lr is not None:
        override["lr"] = float(lr)
    if eval_fem_every is not None:
        override["eval_fem_every"] = int(eval_fem_every)

    # Level 1 is the original mesh. Re-train it under the same fixed schedule
    # as the degraded levels instead of retaining an early-stopped baseline.
    print("=" * 80)
    print(f"[Mesh quality] Training {SOURCE_CASE_NAME} (level 1)")
    results = {
        SOURCE_CASE_NAME: run_train_hole_um_up(
            root=root,
            config={**override, "method_name": SOURCE_METHOD_NAME},
            case_name=SOURCE_CASE_NAME,
        )
    }
    for level in levels:
        case_name = quality_case_name(level)
        print("=" * 80)
        print(f"[Mesh quality] Training {case_name}")
        results[case_name] = run_train_hole_um_up(
            root=root, config=override, case_name=case_name
        )
    return results


def summarize_quality_study(
    *,
    root: Optional[str | Path] = None,
    levels: Sequence[int] = QUALITY_LEVELS,
    require_training_results: bool = True,
) -> Dict[str, str]:
    root_path = Path(root).resolve() if root is not None else _PROJECT_ROOT
    out_root = root_path / "compare" / "hole_um_up_mesh_quality"
    out_root.mkdir(parents=True, exist_ok=True)

    entries = [
        (1, SOURCE_CASE_NAME, _source_quality_rows(root_path), SOURCE_METHOD_NAME)
    ]
    for level in levels:
        case_name = quality_case_name(level)
        paths = _result_paths(
            case_name=case_name, root=root_path, method_name=None
        )
        entries.append(
            (
                level,
                case_name,
                _read_csv(paths["data_root"] / "mesh_quality_summary.csv"),
                None,
            )
        )

    summary_rows = []
    all_quality_rows = []
    for level, case_name, quality_rows, method_name in entries:
        all_quality_rows.extend(quality_rows)
        test_quality = [row for row in quality_rows if row["split"] == "test"]
        paths = _result_paths(
            case_name=case_name, root=root_path, method_name=method_name
        )
        metrics_path = paths["final_metrics_csv_path"]
        if metrics_path.exists():
            metrics = _read_csv(metrics_path)
        elif require_training_results:
            raise FileNotFoundError(
                f"Training/comparison results are missing for {case_name}: {metrics_path}"
            )
        else:
            metrics = []

        row: Dict[str, Any] = {
            "quality_level": level,
            "case_name": case_name,
            "n_test_meshes": len(test_quality),
            "n_vertices_mean": _mean(test_quality, "n_vertices"),
            "n_faces_mean": _mean(test_quality, "n_faces"),
            "jitter_amplitude_mean": _mean(test_quality, "jitter_amplitude"),
            "quality_min_mean": _mean(test_quality, "quality_min"),
            "quality_p05_mean": _mean(test_quality, "quality_p05"),
            "quality_mean_mean": _mean(test_quality, "quality_mean"),
            "min_angle_min_deg_mean": _mean(test_quality, "min_angle_min_deg"),
            "min_angle_p05_deg_mean": _mean(test_quality, "min_angle_p05_deg"),
            "area_cv_mean": _mean(test_quality, "area_cv"),
            "n_test_results": len(metrics),
        }
        for key in METRIC_KEYS:
            row[f"{key}_mean"] = _mean(metrics, key)
            row[f"{key}_std"] = _std(metrics, key)
        summary_rows.append(row)

    summary_csv = out_root / "mesh_quality_training_summary.csv"
    all_meshes_csv = out_root / "mesh_quality_all_samples.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    keys = []
    for row in all_quality_rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(all_meshes_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(all_quality_rows)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    q05 = np.asarray([row["quality_p05_mean"] for row in summary_rows])
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    axes[0].plot(
        [row["quality_level"] for row in summary_rows],
        q05,
        "o-",
        linewidth=2.0,
    )
    axes[0].set_xlabel("Mesh quality level (1 = existing best)")
    axes[0].set_ylabel(r"Mean test-mesh $q_{5\%}$")
    for key, label, marker in (
        ("l2_error_mean", r"$w$ relative $L^2$", "o"),
        ("theta_l2_error_mean", r"$\theta$ relative $L^2$", "s"),
        ("energy_error_mean", "Energy error", "^"),
    ):
        values = np.asarray([row[key] for row in summary_rows], dtype=np.float64)
        if np.any(np.isfinite(values)):
            axes[1].plot(q05, values, marker=marker, linewidth=2.0, label=label)
    axes[1].set_xlabel(r"Mean test-mesh $q_{5\%}$")
    axes[1].set_ylabel("Mean test error")
    if axes[1].lines:
        axes[1].set_yscale("log")
        axes[1].legend()
    for axis in axes:
        axis.grid(True, alpha=0.3)
    fig.tight_layout()
    plot_path = out_root / "mesh_quality_effect_on_training.png"
    from shared.compare.metrics import safe_savefig
    safe_savefig(fig, plot_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[Mesh quality] summary: {summary_csv}")
    return {
        "output_root": str(out_root),
        "summary_csv": str(summary_csv),
        "all_meshes_csv": str(all_meshes_csv),
        "plot": str(plot_path),
    }


def _parse_levels(text: str) -> tuple[int, ...]:
    levels = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    if not levels or any(level not in QUALITY_LEVELS for level in levels):
        raise ValueError(f"Levels must be selected from {QUALITY_LEVELS}, got {levels}")
    return levels


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare, FEM-solve, train, and summarize fixed-size hole_um_up "
            "mesh-quality variants. Existing hole_um_up is quality level 1."
        )
    )
    parser.add_argument(
        "action", choices=("prepare", "fem", "train", "summarize", "all")
    )
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--levels", type=str, default="2,3,4")
    parser.add_argument("--jitter-max", type=float, default=0.95)
    parser.add_argument("--candidate-count", type=int, default=81)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument(
        "--epochs", type=int, default=None,
        help=f"Fixed at {FIXED_STUDY_EPOCHS} for this study; other values are rejected.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--eval-fem-every", type=int, default=None)
    parser.add_argument("--include-train-fem", action="store_true")
    parser.add_argument("--no-overwrite", action="store_true")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()

    levels = _parse_levels(args.levels)
    root = Path(args.root).resolve() if args.root else _PROJECT_ROOT
    overwrite = not args.no_overwrite

    if args.action in ("prepare", "all"):
        prepare_hole_um_up_mesh_quality(
            root=root,
            overwrite=overwrite,
            config={
                "quality_levels": levels,
                "jitter_max": args.jitter_max,
                "candidate_count": args.candidate_count,
                "seed": args.seed,
            },
        )
        summarize_quality_study(
            root=root, levels=levels, require_training_results=False
        )
    if args.action in ("fem", "all"):
        run_fem_quality_cases(
            root=root,
            levels=levels,
            overwrite=overwrite,
            include_train=args.include_train_fem,
        )
    if args.action in ("train", "all"):
        train_quality_cases(
            root=root,
            levels=levels,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            eval_fem_every=args.eval_fem_every,
            device=args.device,
        )
        summarize_quality_study(root=root, levels=levels)
    elif args.action == "summarize":
        summarize_quality_study(root=root, levels=levels)


if __name__ == "__main__":
    main()
