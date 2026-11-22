from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from NO.problem.hole_mesh_convergence import (
    MESH_DENSITY_RATIOS,
    density_case_name,
    mesh_quality_statistics,
)
from shared.common.data_io import load_dataset, load_npz_dict
from shared.common.paths import build_paths, fem_pred_npz_path, no_pred_npz_path
from shared.prep.hole_um_up_mesh_convergence import prepare_hole_um_up_mesh_convergence


METRIC_KEYS = (
    "node_l2_w",
    "node_l2_theta",
    "l2_error",
    "h1_semi_error",
    "energy_error",
    "theta_l2_error",
    "curvature_error",
)

SOURCE_CASE_NAME = "hole_um_up"
FIXED_STUDY_EPOCHS = 500
# Keep the fixed-schedule 100% reference separate from the normal early-stop
# baseline in NO/outputs/hole_um_up and compare/hole_um_up.
SOURCE_METHOD_NAME = "mesh_convergence_500"


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


def _relative_node_l2(no: np.ndarray, fem: np.ndarray) -> tuple[float, float]:
    """Same unweighted retained-node metric as the PC-density study."""
    no = np.asarray(no, dtype=np.float64)
    fem = np.asarray(fem, dtype=np.float64)
    if no.shape != fem.shape or no.ndim != 2 or no.shape[1] < 3:
        raise ValueError(f"NO/FEM nodal arrays are not aligned: {no.shape} vs {fem.shape}")
    w = float(np.linalg.norm(no[:, 0] - fem[:, 0]) / max(np.linalg.norm(fem[:, 0]), 1e-30))
    theta = float(
        np.linalg.norm(no[:, 1:3] - fem[:, 1:3])
        / max(np.linalg.norm(fem[:, 1:3]), 1e-30)
    )
    return w, theta


def _read_csv(path: Path) -> list[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Required result table not found: {path}")
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _ensure_final_metrics_table(
    *,
    case_name: str,
    root: Path,
    paths: Dict[str, Any],
    method_name: Optional[str] = None,
) -> Path:
    """Create a missing standard NO--FEM comparison table from saved outputs.

    The 100% row uses the fixed-schedule method-specific ``hole_um_up``
    result. Older runs may contain its checkpoint/prediction files but predate
    the final comparison CSV, so create that table lazily here.
    """
    metrics_path = Path(paths["final_metrics_csv_path"])
    if metrics_path.exists():
        return metrics_path

    from shared.compare.compare import compare_cases

    print(
        f"[Mesh convergence] Missing final metrics for {case_name}; "
        "building them from existing NO predictions and FEM references."
    )
    compare_cases(
        case_name=case_name,
        root=root,
        deformation_scale=10.0,
        max_plot_cases=5,
        method_name=method_name,
    )
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Could not create {metrics_path}. Ensure {case_name} has both "
            "NO prediction files and FEM reference files for every test tag."
        )
    return metrics_path


def _mean(rows: Iterable[Dict[str, str]], key: str) -> float:
    values = [float(row[key]) for row in rows if key in row and row[key] not in ("", None)]
    return float(np.mean(values)) if values else float("nan")


def _std(rows: Iterable[Dict[str, str]], key: str) -> float:
    values = [float(row[key]) for row in rows if key in row and row[key] not in ("", None)]
    return float(np.std(values)) if values else float("nan")


def _source_mesh_rows(*, root: Path) -> list[Dict[str, Any]]:
    """Build the 100% quality rows directly from the original dataset."""
    train_samples, test_samples, _ = load_dataset(case_name=SOURCE_CASE_NAME, root=root)
    rows: list[Dict[str, Any]] = []
    for split, samples in (("train", train_samples), ("test", test_samples)):
        for i, sample in enumerate(samples):
            verts = np.asarray(sample["verts_xy"], dtype=np.float64)
            faces = np.asarray(sample["faces"], dtype=np.int64)
            pc = np.asarray(sample["pc"])
            hole = np.asarray(sample["hole_xy"])
            n_outer = len(pc) - len(hole)
            quality = mesh_quality_statistics(verts, faces)
            rows.append(
                {
                    "split": split,
                    "sample_index": i,
                    "sample_id": int(sample.get("sample_id", i)),
                    "target_density": 1.0,
                    "actual_vertex_density": 1.0,
                    "actual_element_density": 1.0,
                    "element_density_relative_error": 0.0,
                    "target_boundary_density": 1.0,
                    "actual_boundary_density": 1.0,
                    "relative_mesh_size": 1.0,
                    "mesh_lc": float("nan"),
                    "mesh_calibration_attempts": 0,
                    "n_vertices": len(verts),
                    "n_faces": len(faces),
                    "n_boundary_points": len(pc),
                    "source_n_vertices": len(verts),
                    "source_n_faces": len(faces),
                    "outer_n_per_edge": n_outer // 4,
                    "hole_n_points": len(hole),
                    **quality,
                }
            )
    return rows


def summarize_prepared_meshes(
    *,
    root: Optional[str | Path] = None,
    ratios=MESH_DENSITY_RATIOS,
) -> Dict[str, str]:
    """Combine mesh counts/quality before any NO model has been trained."""
    root_path = Path(root).resolve() if root is not None else _PROJECT_ROOT
    out_root = root_path / "compare" / "hole_um_up_mesh_convergence"
    out_root.mkdir(parents=True, exist_ok=True)
    all_rows: list[Dict[str, Any]] = []
    summary_rows: list[Dict[str, Any]] = []
    entries = []
    for ratio in ratios:
        ratio = float(ratio)
        case_name = density_case_name(ratio)
        paths = build_paths(case_name, root=root_path, create=True)
        rows = _read_csv(Path(paths["data_root"]) / "mesh_quality_summary.csv")
        entries.append((ratio, case_name, rows))
    entries.append((1.0, SOURCE_CASE_NAME, _source_mesh_rows(root=root_path)))

    for ratio, case_name, rows in entries:
        for row in rows:
            all_rows.append({"case_name": case_name, "density": ratio, **row})
        summary_rows.append(
            {
                "case_name": case_name,
                "target_density": ratio,
                "n_meshes": len(rows),
                "n_vertices_mean": _mean(rows, "n_vertices"),
                "n_faces_mean": _mean(rows, "n_faces"),
                "n_boundary_points_mean": _mean(rows, "n_boundary_points"),
                "actual_element_density_mean": _mean(rows, "actual_element_density"),
                "element_density_relative_error_mean": _mean(
                    rows, "element_density_relative_error"
                ),
                "actual_boundary_density_mean": _mean(rows, "actual_boundary_density"),
                "relative_mesh_size_mean": _mean(rows, "relative_mesh_size"),
                "quality_min_mean": _mean(rows, "quality_min"),
                "quality_p05_mean": _mean(rows, "quality_p05"),
                "quality_mean": _mean(rows, "quality_mean"),
                "min_angle_min_deg_mean": _mean(rows, "min_angle_min_deg"),
                "min_angle_p05_deg_mean": _mean(rows, "min_angle_p05_deg"),
                "area_cv_mean": _mean(rows, "area_cv"),
            }
        )

    all_csv = out_root / "prepared_mesh_quality_all_samples.csv"
    summary_csv = out_root / "prepared_mesh_quality_summary.csv"
    for path, rows in ((all_csv, all_rows), (summary_csv, summary_rows)):
        keys = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    density = 100.0 * np.asarray([r["target_density"] for r in summary_rows])
    vertices = np.asarray([r["n_vertices_mean"] for r in summary_rows])
    faces = np.asarray([r["n_faces_mean"] for r in summary_rows])
    q05 = np.asarray([r["quality_p05_mean"] for r in summary_rows])
    a05 = np.asarray([r["min_angle_p05_deg_mean"] for r in summary_rows])
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    axes[0].plot(density, vertices, "o-", label="vertices")
    axes[0].plot(density, faces, "s-", label="triangles")
    axes[0].set_xlabel("Target vertex density (%)")
    axes[0].set_ylabel("Mean count")
    axes[0].legend()
    ax2 = axes[1].twinx()
    axes[1].plot(density, q05, "o-", color="tab:blue", label=r"$q_{5\%}$")
    ax2.plot(density, a05, "s--", color="tab:orange", label="angle 5%")
    axes[1].set_xlabel("Target vertex density (%)")
    axes[1].set_ylabel(r"Mean normalized quality $q_{5\%}$", color="tab:blue")
    ax2.set_ylabel(r"Mean 5th-percentile minimum angle ($^\circ$)", color="tab:orange")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plot_path = out_root / "prepared_mesh_counts_and_quality.png"
    fig.savefig(plot_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[Mesh convergence] prepared-mesh summary: {summary_csv}")
    return {
        "all_samples_csv": str(all_csv),
        "summary_csv": str(summary_csv),
        "plot": str(plot_path),
    }


def summarize_convergence(
    *,
    root: Optional[str | Path] = None,
    ratios=MESH_DENSITY_RATIOS,
) -> Dict[str, Any]:
    root_path = Path(root).resolve() if root is not None else _PROJECT_ROOT
    out_root = root_path / "compare" / "hole_um_up_mesh_convergence"
    out_root.mkdir(parents=True, exist_ok=True)
    summary_rows: list[Dict[str, Any]] = []
    case_rows: list[Dict[str, Any]] = []

    entries = [(float(ratio), density_case_name(float(ratio))) for ratio in ratios]
    entries.append((1.0, SOURCE_CASE_NAME))
    source_quality_rows = _source_mesh_rows(root=root_path)

    for ratio, case_name in entries:
        method_name = SOURCE_METHOD_NAME if ratio == 1.0 else None
        paths = _result_paths(
            case_name=case_name, root=root_path, method_name=method_name
        )
        metrics = _read_csv(
            _ensure_final_metrics_table(
                case_name=case_name,
                root=root_path,
                paths=paths,
                method_name=method_name,
            )
        )
        # NO and the per-density FEM reference are already aligned at the same
        # query points. The coarse references were interpolated during prepare.
        aligned_metrics = []
        for metric_row in metrics:
            metric_row = dict(metric_row)
            tag = metric_row["case_tag"]
            no_data = load_npz_dict(no_pred_npz_path(paths, tag))
            fem_data = load_npz_dict(fem_pred_npz_path(paths, tag))
            node_w, node_theta = _relative_node_l2(
                no_data["pred_local"], fem_data["fem_local"]
            )
            metric_row["node_l2_w"] = node_w
            metric_row["node_l2_theta"] = node_theta
            aligned_metrics.append(metric_row)
        metrics = aligned_metrics
        quality = (
            source_quality_rows
            if ratio == 1.0
            else _read_csv(Path(paths["data_root"]) / "mesh_quality_summary.csv")
        )
        quality_test = [row for row in quality if row["split"] == "test"]
        row: Dict[str, Any] = {
            "case_name": case_name,
            "target_density": ratio,
            "actual_vertex_density_mean": _mean(quality_test, "actual_vertex_density"),
            "actual_element_density_mean": _mean(quality_test, "actual_element_density"),
            "element_density_relative_error_mean": _mean(
                quality_test, "element_density_relative_error"
            ),
            "actual_boundary_density_mean": _mean(quality_test, "actual_boundary_density"),
            "relative_mesh_size_mean": _mean(quality_test, "relative_mesh_size"),
            "n_vertices_mean": _mean(quality_test, "n_vertices"),
            "n_faces_mean": _mean(quality_test, "n_faces"),
            "n_boundary_points_mean": _mean(quality_test, "n_boundary_points"),
            "quality_min_mean": _mean(quality_test, "quality_min"),
            "quality_p05_mean": _mean(quality_test, "quality_p05"),
            "quality_mean": _mean(quality_test, "quality_mean"),
            "min_angle_min_deg_mean": _mean(quality_test, "min_angle_min_deg"),
            "min_angle_p05_deg_mean": _mean(quality_test, "min_angle_p05_deg"),
            "area_cv_mean": _mean(quality_test, "area_cv"),
            "n_test_cases": len(metrics),
        }
        for key in METRIC_KEYS:
            row[f"{key}_mean"] = _mean(metrics, key)
            row[f"{key}_std"] = _std(metrics, key)
        summary_rows.append(row)
        for metric_row in metrics:
            combined = dict(metric_row)
            combined["case_name"] = case_name
            combined["target_density"] = ratio
            case_rows.append(combined)

    # Observed order between adjacent levels, using h ~ N_element^(-1/2).
    by_faces = sorted(range(len(summary_rows)), key=lambda i: summary_rows[i]["n_faces_mean"])
    for key in METRIC_KEYS:
        order_key = f"{key}_observed_order"
        for i in by_faces:
            summary_rows[i][order_key] = float("nan")
        for coarse_i, fine_i in zip(by_faces[:-1], by_faces[1:]):
            coarse = summary_rows[coarse_i]
            fine = summary_rows[fine_i]
            e0, e1 = float(coarse[f"{key}_mean"]), float(fine[f"{key}_mean"])
            h0 = 1.0 / np.sqrt(float(coarse["n_faces_mean"]))
            h1 = 1.0 / np.sqrt(float(fine["n_faces_mean"]))
            if e0 > 0.0 and e1 > 0.0 and h0 > h1:
                fine[order_key] = float(np.log(e0 / e1) / np.log(h0 / h1))

    summary_csv = out_root / "mesh_convergence_summary.csv"
    cases_csv = out_root / "mesh_convergence_all_cases.csv"
    summary_npz = out_root / "mesh_convergence_summary.npz"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    with open(cases_csv, "w", newline="", encoding="utf-8") as f:
        keys = []
        for row in case_rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(case_rows)
    np.savez_compressed(
        summary_npz,
        **{
            key: np.asarray(
                [row[key] for row in summary_rows],
                dtype=str if key == "case_name" else np.float64,
            )
            for key in summary_rows[0]
        },
    )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.asarray([row["n_faces_mean"] for row in summary_rows], dtype=np.float64)
    order = np.argsort(x)
    x = x[order]
    for key, ylabel in (
        ("node_l2_w", r"Retained-node relative $L^2$ error of $w$"),
        ("node_l2_theta", r"Retained-node relative $L^2$ error of $\theta$"),
        ("l2_error", r"Relative $L^2$ error of $w$"),
        ("h1_semi_error", r"Relative $H^1$ seminorm error of $w$"),
        ("energy_error", "Relative energy error"),
        ("theta_l2_error", r"Relative $L^2$ error of $\theta$"),
        ("curvature_error", "Relative curvature error"),
    ):
        y = np.asarray([row[f"{key}_mean"] for row in summary_rows], dtype=np.float64)[order]
        yerr = np.asarray([row[f"{key}_std"] for row in summary_rows], dtype=np.float64)[order]
        fig, ax = plt.subplots(figsize=(7.0, 5.0))
        ax.errorbar(x, y, yerr=yerr, marker="o", linewidth=2.0, capsize=4)
        ax.set_xlabel("Mean number of triangular elements")
        ax.set_ylabel(ylabel)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_root / f"{key}_mesh_convergence.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    density = 100.0 * np.asarray(
        [summary_rows[i]["actual_vertex_density_mean"] for i in order], dtype=np.float64
    )
    q05 = np.asarray([summary_rows[i]["quality_p05_mean"] for i in order], dtype=np.float64)
    min_angle = np.asarray(
        [summary_rows[i]["min_angle_p05_deg_mean"] for i in order], dtype=np.float64
    )
    axes[0].plot(density, q05, "o-", linewidth=2.0)
    axes[0].set_xlabel("Retained vertex density (%)")
    axes[0].set_ylabel(r"Mean mesh $q_{5\%}$")
    axes[1].plot(density, min_angle, "o-", linewidth=2.0)
    axes[1].set_xlabel("Retained vertex density (%)")
    axes[1].set_ylabel(r"Mean 5th-percentile minimum angle ($^\circ$)")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    quality_plot = out_root / "mesh_quality_across_densities.png"
    fig.savefig(quality_plot, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"[Mesh convergence] summary: {summary_csv}")
    print(f"[Mesh convergence] all cases: {cases_csv}")
    return {
        "output_root": str(out_root),
        "summary_csv": str(summary_csv),
        "cases_csv": str(cases_csv),
        "summary_npz": str(summary_npz),
        "quality_plot": str(quality_plot),
    }


def train_all_densities(
    *,
    root: Optional[str | Path] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    lr: Optional[float] = None,
    eval_fem_every: Optional[int] = None,
    device: str = "auto",
    ratios=MESH_DENSITY_RATIOS,
) -> Dict[str, Any]:
    # Import training only after the CLI device choice has been applied because
    # NO.common.core selects the shared torch device at import time.
    os.environ["NO_DEVICE"] = str(device)
    # The generic hole energy trainer lives in the unified MR entry point.
    # ``train_hole_um_up.py`` was removed when the thickness cases were
    # consolidated into ``train_hole_um_up_mr.py``.
    from NO.train.train_hole_um_up_mr import run_train_hole_um_up

    if epochs not in (None, FIXED_STUDY_EPOCHS):
        raise ValueError(
            f"Mesh-convergence training is fixed at {FIXED_STUDY_EPOCHS} epochs; "
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
    # Re-train the 100% baseline under the same fixed schedule as every
    # coarser mesh, rather than reusing a checkpoint selected by early stop.
    print("=" * 80)
    print(f"[Mesh convergence] Training {SOURCE_CASE_NAME} (100%)")
    results = {
        SOURCE_CASE_NAME: run_train_hole_um_up(
            root=root,
            config={**override, "method_name": SOURCE_METHOD_NAME},
            case_name=SOURCE_CASE_NAME,
        )
    }
    for ratio in ratios:
        case_name = density_case_name(float(ratio))
        print("=" * 80)
        print(f"[Mesh convergence] Training {case_name}")
        results[case_name] = run_train_hole_um_up(
            root=root,
            config=override,
            case_name=case_name,
        )
    results["summary"] = summarize_convergence(root=root, ratios=ratios)
    return results


def _parse_ratios(text: str):
    ratios = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    allowed = set(MESH_DENSITY_RATIOS)
    if not ratios or any(ratio not in allowed for ratio in ratios):
        raise ValueError(f"Ratios must be selected from {sorted(allowed)}, got {ratios}")
    return ratios


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare and train independent Gmsh meshes at 5/12.5/25/50/75%, then "
            "summarize them together with the existing hole_um_up 100% result."
        )
    )
    parser.add_argument("action", choices=("prepare", "train", "summarize", "all"))
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--ratios", type=str, default="0.05,0.125,0.25,0.50,0.75")
    parser.add_argument(
        "--epochs", type=int, default=None,
        help=f"Fixed at {FIXED_STUDY_EPOCHS} for this study; other values are rejected.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--eval-fem-every", type=int, default=None)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help=(
            "Training device. 'auto' validates one real CUDA operation and "
            "falls back to CPU when the installed wheel cannot run on the GPU."
        ),
    )
    parser.add_argument("--no-overwrite", action="store_true")
    args = parser.parse_args()
    ratios = _parse_ratios(args.ratios)
    root = Path(args.root).resolve() if args.root else _PROJECT_ROOT

    if args.action in ("prepare", "all"):
        prepare_hole_um_up_mesh_convergence(
            root=root,
            overwrite=not args.no_overwrite,
            config={"density_ratios": ratios},
        )
        summarize_prepared_meshes(root=root, ratios=ratios)
    if args.action in ("train", "all"):
        train_all_densities(
            root=root,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            eval_fem_every=args.eval_fem_every,
            device=args.device,
            ratios=ratios,
        )
    elif args.action == "summarize":
        summarize_convergence(root=root, ratios=ratios)


if __name__ == "__main__":
    main()
