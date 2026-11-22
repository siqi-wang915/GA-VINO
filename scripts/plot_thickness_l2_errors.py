from __future__ import annotations

"""Plot MR/KL displacement and rotation L2 errors across plate thicknesses."""

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np


_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from shared.prep.hole_um_up_mr import DEFAULT_CONFIG, THICKNESS_CASES


METHODS = {
    "mr": {
        "legend_name": "MR",
        "result_suffix": "",
    },
    "kl": {
        "legend_name": "KL",
        "result_suffix": "_kl",
    },
}

METRICS = {
    "w": {
        "column": "l2_error",
        "legend_quantity": r"$w$ $L^2$ norm",
    },
    "theta": {
        "column": "theta_l2_error",
        "legend_quantity": r"$\theta$ $L^2$ norm",
    },
}

# Match the four-color sequence used by the FEM mesh-convergence plot.
CURVE_COLORS = {
    ("mr", "w"): "#1f77b4",
    ("mr", "theta"): "#ff7f0e",
    ("kl", "w"): "#2ca02c",
    ("kl", "theta"): "#d62728",
}

# Match the four marker shapes used by the FEM mesh-convergence curves.
CURVE_MARKERS = {
    ("mr", "w"): "o",
    ("mr", "theta"): "D",
    ("kl", "w"): "s",
    ("kl", "theta"): "^",
}


def _read_metrics(path: Path) -> list[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Required final metric table was not found: {path}")
    with open(path, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Final metric table is empty: {path}")
    return rows


def _finite_values(rows: Iterable[Dict[str, str]], key: str, path: Path) -> np.ndarray:
    values = np.asarray(
        [float(row[key]) for row in rows if row.get(key) not in (None, "")],
        dtype=np.float64,
    )
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError(f"Column {key!r} has no finite values in {path}")
    return values


def collect_thickness_summary(root: Path) -> list[Dict[str, Any]]:
    """Collect test-set mean/std errors for every MR and KL thickness run."""
    plate_width = float(DEFAULT_CONFIG["lx"])
    rows: list[Dict[str, Any]] = []

    for method, method_info in METHODS.items():
        for thickness_suffix, definition in THICKNESS_CASES.items():
            case_name = str(definition["case_name"])
            result_name = f"{case_name}{method_info['result_suffix']}"
            metrics_path = root / "compare" / result_name / "final_test_error_metrics.csv"
            metrics_rows = _read_metrics(metrics_path)
            thickness = float(definition["thickness"])
            row: Dict[str, Any] = {
                "method": method.upper(),
                "thickness_suffix": thickness_suffix,
                "case_name": case_name,
                "result_name": result_name,
                "thickness": thickness,
                "plate_width": plate_width,
                "thickness_to_width_ratio": thickness / plate_width,
                "n_test_cases": len(metrics_rows),
            }
            for metric_name, metric_info in METRICS.items():
                values_pct = 100.0 * _finite_values(
                    metrics_rows, str(metric_info["column"]), metrics_path
                )
                row[f"{metric_name}_l2_error_mean_pct"] = float(np.mean(values_pct))
                row[f"{metric_name}_l2_error_std_pct"] = float(np.std(values_pct))
            rows.append(row)

    return sorted(
        rows,
        key=lambda row: (str(row["method"]), float(row["thickness_to_width_ratio"])),
    )


def _write_summary(rows: list[Dict[str, Any]], output_root: Path) -> tuple[Path, Path]:
    summary_csv = output_root / "mr_kl_thickness_l2_error_summary.csv"
    summary_npz = output_root / "mr_kl_thickness_l2_error_summary.npz"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        summary_npz,
        **{
            key: np.asarray(
                [row[key] for row in rows],
                dtype=str if isinstance(rows[0][key], str) else np.float64,
            )
            for key in rows[0]
        },
    )
    return summary_csv, summary_npz


def plot_thickness_l2_errors(
    *,
    root: str | Path | None = None,
    output_dir: str | Path | None = None,
    dpi: int = 220,
) -> Dict[str, str]:
    root_path = Path(root).resolve() if root is not None else _PROJECT_ROOT
    output_root = (
        Path(output_dir).resolve()
        if output_dir is not None
        else root_path / "compare" / "hole_um_up_thickness_study"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    rows = collect_thickness_summary(root_path)
    summary_csv, summary_npz = _write_summary(rows, output_root)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, LogLocator, LogFormatterMathtext

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    x_ticks = None
    plotted_y_values: list[float] = []
    for method, method_info in METHODS.items():
        method_rows = sorted(
            (row for row in rows if row["method"] == method.upper()),
            key=lambda row: float(row["thickness_to_width_ratio"]),
        )
        x = np.asarray(
            [row["thickness_to_width_ratio"] for row in method_rows], dtype=np.float64
        )
        if x_ticks is None:
            x_ticks = x
        for metric_name, metric_info in METRICS.items():
            y = np.asarray(
                [row[f"{metric_name}_l2_error_mean_pct"] for row in method_rows],
                dtype=np.float64,
            )
            plotted_y_values.extend(y.tolist())
            curve_color = CURVE_COLORS[(method, metric_name)]
            ax.plot(
                x,
                y,
                color=curve_color,
                marker=CURVE_MARKERS[(method, metric_name)],
                linestyle="-",
                linewidth=2.0,
                markersize=6.0,
                label=(
                    f"{metric_info['legend_quantity']} "
                    f"({method_info['legend_name']})"
                ),
            )

    if x_ticks is None:
        raise RuntimeError("No thickness-study points were collected.")
    ax.set_xlabel("Thickness-to-width ratio")
    ax.set_ylabel("Error (%)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(x_ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    y_min = min(plotted_y_values)
    y_max = max(plotted_y_values)
    lowest_decade = 10.0 ** np.floor(np.log10(y_min))
    ax.set_ylim(lowest_decade / 1.2, y_max * 1.2)
    ax.grid(False, which="both")
    ax.grid(True, which="major", axis="x", color="0.75", alpha=0.55)
    ax.grid(True, which="major", axis="y", color="0.75", alpha=0.55)
    ax.legend(frameon=True)
    fig.tight_layout()
    plot_path = output_root / "mr_kl_l2_error_vs_thickness_ratio.png"
    fig.savefig(plot_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)

    print(f"[Thickness study] plot: {plot_path}")
    print(f"[Thickness study] summary: {summary_csv}")
    return {
        "plot": str(plot_path),
        "summary_csv": str(summary_csv),
        "summary_npz": str(summary_npz),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot MR and KL deflection/rotation L2 errors against the "
            "thickness-to-width ratio."
        )
    )
    parser.add_argument("--root", type=str, default=None, help="Project root.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: compare/hole_um_up_thickness_study).",
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    plot_thickness_l2_errors(
        root=args.root,
        output_dir=args.output_dir,
        dpi=args.dpi,
    )
