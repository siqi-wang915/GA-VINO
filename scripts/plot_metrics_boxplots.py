from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _pretty_case_name(name: str) -> str:
    """Make folder names more readable in plots."""
    name = name.replace("annular_sector", "Annular-sector")
    name = name.replace("ood_from", "OOD from")
    name = name.replace("holes2", "two-hole-temp")
    name = name.replace("hole", "single-hole")
    name = name.replace("two-hole-temp", "two-hole")

    name = name.replace("grfm", "GRF material")
    name = name.replace("grfp", "GRF pressure")
    name = name.replace("um", "uniform material")
    name = name.replace("up", "uniform pressure")
    name = name.replace("lshape", "L-shape")
    return name.replace("_", " ")


def _find_metric_columns(df: pd.DataFrame) -> dict[str, str]:
    """
    Try to find metric columns robustly.
    Expected project columns are usually:
        l2_error, theta_l2_error, h1_semi_error, moment_l2_error,
        shear_l2_error, energy_error
    """
    cols = {c.lower(): c for c in df.columns}

    candidates = {
        "L2": ["l2_error", "relative_l2_error", "l2"],
        "ThetaL2": ["theta_l2_error", "rotation_l2_error", "theta_l2"],
        "H1": ["h1_semi_error", "h1_error", "relative_h1_error", "h1"],
        "MomentL2": ["moment_l2_error", "moment_l2", "m_l2_error", "m_l2"],
        "ShearL2": ["shear_l2_error", "shear_l2", "q_l2_error", "q_l2"],
        "Energy": ["energy_error", "eerr", "relative_energy_error", "energy"],
    }

    found = {}
    for label, names in candidates.items():
        for n in names:
            if n.lower() in cols:
                found[label] = cols[n.lower()]
                break

    return found


def _is_baseline_case(case_name: str) -> bool:
    """Baseline folders are not part of the default method-only plots."""
    baseline_suffixes = ("_mse", "_fno")
    return case_name.endswith(baseline_suffixes)


def plot_one_case(
    csv_path: Path,
    out_dir: Path,
    dpi: int = 300,
    log_y: bool = False,
) -> dict | None:
    case_name = csv_path.parent.name
    df = pd.read_csv(csv_path)

    metric_cols = _find_metric_columns(df)
    if not metric_cols:
        print(f"[Skip] No metric columns found in {csv_path}")
        print(f"       Columns: {list(df.columns)}")
        return None

    # Fixed order follows the manuscript's Table 3: displacement, rotation,
    # displacement H1, moment, shear, and energy errors.
    metric_order = ["L2", "ThetaL2", "H1", "MomentL2", "ShearL2", "Energy"]

    # Formula-style error norms used in the manuscript. The plotted values are
    # expressed as percentages; the axis title carries the percentage unit.
    label_map = {
        "L2": r"$\Vert e_w \Vert_{L^2}$",
        "ThetaL2": r"$\Vert e_\theta \Vert_{L^2}$",
        "H1": r"$\Vert e_w \Vert_{H^1}$",
        "MomentL2": r"$\Vert e_M \Vert_{L^2}$",
        "ShearL2": r"$\Vert e_Q \Vert_{L^2}$",
        "Energy": r"$\Vert e \Vert_E$",
    }

    # Colors corresponding to the reference figure, from left to right.
    color_map = {
        "L2": "#c8bde9",
        "ThetaL2": "#9fc4e8",
        "H1": "#f2a0ad",
        "MomentL2": "#f4c58b",
        "ShearL2": "#9bd3c7",
        "Energy": "#add8b0",
    }

    labels = []
    data = []
    colors = []
    used_metric_names = []

    for metric_name in metric_order:
        if metric_name not in metric_cols:
            continue

        col = metric_cols[metric_name]
        arr = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)

        if arr.size == 0:
            continue

        # Metric files store relative errors as fractions; plot percentages.
        arr = arr * 100.0

        # Log scale cannot display zero or negative values.
        if log_y:
            arr = arr[arr > 0.0]
            if arr.size == 0:
                continue

        labels.append(label_map[metric_name])
        data.append(arr)
        colors.append(color_map[metric_name])
        used_metric_names.append(metric_name)

    if not data:
        print(f"[Skip] Empty metric values in {csv_path}")
        return None

    out_dir.mkdir(parents=True, exist_ok=True)

    # Keep the original figure footprint despite the two extra metrics.  The
    # boxes are narrower and use tighter custom positions to avoid a wider
    # final figure.
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    box_positions = np.arange(len(data), dtype=float) * 0.70

    bp = ax.boxplot(
        data,
        labels=labels,
        positions=box_positions,
        showmeans=True,
        patch_artist=True,
        widths=0.42,
        medianprops={
            "color": "black",
            "linewidth": 1.6,
        },
        meanprops={
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markeredgewidth": 1.2,
            "markersize": 5,
        },
        boxprops={
            "linewidth": 1.2,
            "edgecolor": "black",
        },
        whiskerprops={
            "linewidth": 1.2,
            "color": "black",
        },
        capprops={
            "linewidth": 1.2,
            "color": "black",
        },
        flierprops={
            "marker": "o",
            "markersize": 3.5,
            "markerfacecolor": "none",
            "markeredgecolor": "gray",
            "alpha": 0.75,
        },
    )
    ax.set_xlim(box_positions[0] - 0.35, box_positions[-1] + 0.35)

    # Apply per-metric colors.
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.70)
        patch.set_edgecolor("black")

    # Optional: color outliers by their corresponding box color.
    # Matplotlib returns one flier artist per box.
    for flier, color in zip(bp["fliers"], colors):
        flier.set_markeredgecolor(color)
        flier.set_markerfacecolor("none")
        flier.set_alpha(0.85)

    ax.set_ylabel("Error (%)", fontsize=14)

    # If you want the title, uncomment the next line.
    # ax.set_title(_pretty_case_name(case_name), fontsize=14)

    if log_y:
        ax.set_yscale("log")

    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.tick_params(axis="x", labelsize=13)
    ax.tick_params(axis="y", labelsize=12)

    # Make border lines clearer.
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)

    fig.tight_layout()

    out_path = out_dir / f"{case_name}_metrics_boxplot.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"[Saved] {out_path}")

    summary = {
        "case": case_name,
        "csv": str(csv_path),
        "figure": str(out_path),
    }

    for metric_name, arr in zip(used_metric_names, data):
        summary[f"{metric_name}_mean"] = float(np.mean(arr))
        summary[f"{metric_name}_std"] = float(np.std(arr))
        summary[f"{metric_name}_median"] = float(np.median(arr))

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Project root",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="compare/*/final_test_error_metrics.csv",
        help="Glob pattern for final metrics csv files.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="compare/_metric_plots",
        help="Output directory for figures.",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Deprecated compatibility option; plots are always percentages.",
    )
    parser.add_argument(
        "--percent",
        action="store_true",
        help="Compatibility option; plots are always percentages.",
    )
    parser.add_argument(
        "--log-y",
        action="store_true",
        help="Use logarithmic y-axis.",
    )
    parser.add_argument(
        "--include-baselines",
        action="store_true",
        help="Also plot baseline result folders such as *_mse and *_fno.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = root / args.out_dir
    all_csv_files = sorted(root.glob(args.pattern))
    csv_files = []
    skipped_baselines = []
    for csv_path in all_csv_files:
        case_name = csv_path.parent.name
        if not args.include_baselines and _is_baseline_case(case_name):
            skipped_baselines.append(csv_path)
            continue
        csv_files.append(csv_path)

    if not csv_files:
        raise FileNotFoundError(
            f"No method metrics files found with pattern: {args.pattern} in {root}"
        )

    print(f"[Info] Found {len(csv_files)} method metrics files.")
    if skipped_baselines:
        skipped = ", ".join(p.parent.name for p in skipped_baselines)
        print(f"[Info] Skipped baseline folders: {skipped}")

    summaries = []
    for csv_path in csv_files:
        summary = plot_one_case(
            csv_path=csv_path,
            out_dir=out_dir,
            log_y=args.log_y,
        )
        if summary is not None:
            summaries.append(summary)

    if summaries:
        summary_df = pd.DataFrame(summaries)
        summary_csv = out_dir / "metrics_plot_summary.csv"
        summary_df.to_csv(summary_csv, index=False)
        print(f"[Saved] {summary_csv}")


if __name__ == "__main__":
    main()
