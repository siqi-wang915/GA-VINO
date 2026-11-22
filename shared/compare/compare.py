from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from shared.common.data_io import load_npz_dict, load_test_tags
from shared.common.paths import (
    build_paths,
    case_npz_path,
    fem_pred_npz_path,
    no_pred_npz_path,
)
from shared.compare.metrics import (
    compute_case_error_metrics,
    generate_case_compare_figures,
    save_metrics_table,
    summarize_metric_rows,
)


# -----------------------------------------------------------------------------
# Method-specific path helpers
# -----------------------------------------------------------------------------

def _is_base_method(method_name: Optional[str]) -> bool:
    """
    Return True if the comparison should use the original/default output folder:
        NO/outputs/<case>/
        compare/<case>/

    Any non-base method name will use:
        NO/outputs/<case>_<method_name>/
        compare/<case>_<method_name>/
    """
    if method_name is None:
        return True

    name = str(method_name).strip().lower()
    return name in {
        "",
        "none",
        "base",
        "default",
        "energy",
        "energy_min",
        "energy_minimization",
    }


def _method_case_name(case_name: str, method_name: Optional[str]) -> str:
    """Return the folder name for a method-specific output case."""
    if _is_base_method(method_name):
        return str(case_name)
    return f"{case_name}_{method_name}"


def _build_compare_io_paths(
    *,
    case_name: str,
    root: Optional[str | Path],
    method_name: Optional[str],
    create: bool = True,
) -> Dict[str, Any]:
    """
    Build paths for final NO-FEM comparison.

    Important convention
    --------------------
    The physical case data and FEM reference results are always read from the
    base case folder:

        shared/cases/<case>/
        FEM/outputs/<case>/

    The NO predictions and compare outputs are read/written from the method
    folder when method_name is not a base method:

        NO/outputs/<case>_<method_name>/predictions/
        compare/<case>_<method_name>/

    Example
    -------
    case_name="square", method_name="boundary_pc":

        case/FEM:
            shared/cases/square/
            FEM/outputs/square/

        NO/compare:
            NO/outputs/square_boundary_pc/predictions/
            compare/square_boundary_pc/
    """
    base_paths = build_paths(case_name, root=root, create=create)
    project_root = Path(base_paths["project_root"])

    method_case = _method_case_name(case_name, method_name)

    if _is_base_method(method_name):
        no_prediction_root = Path(
            base_paths.get(
                "no_prediction_root",
                Path(base_paths["no_output_root"]) / "predictions",
            )
        )
        compare_root = Path(base_paths["compare_root"])
        final_metrics_npz_path = Path(base_paths["final_metrics_npz_path"])
        final_metrics_csv_path = Path(base_paths["final_metrics_csv_path"])
    else:
        no_output_root = project_root / "NO" / "outputs" / method_case
        no_prediction_root = no_output_root / "predictions"
        compare_root = project_root / "compare" / method_case
        final_metrics_npz_path = compare_root / "final_test_error_metrics.npz"
        final_metrics_csv_path = compare_root / "final_test_error_metrics.csv"

    if create:
        no_prediction_root.mkdir(parents=True, exist_ok=True)
        compare_root.mkdir(parents=True, exist_ok=True)

    return {
        "base_paths": base_paths,
        "method_case": method_case,
        "no_prediction_root": no_prediction_root,
        "compare_root": compare_root,
        "final_metrics_npz_path": final_metrics_npz_path,
        "final_metrics_csv_path": final_metrics_csv_path,
    }


def _no_pred_path_for_method(
    *,
    base_paths: Dict[str, Path],
    no_prediction_root: Path,
    method_name: Optional[str],
    tag: str,
) -> Path:
    """Return the NO prediction path for either base or method-specific output."""
    if _is_base_method(method_name):
        return no_pred_npz_path(base_paths, tag)
    return Path(no_prediction_root) / f"{tag}_no.npz"


# -----------------------------------------------------------------------------
# Case selection helpers
# -----------------------------------------------------------------------------

def _select_representative_cases_by_l2(
    rows: List[Dict[str, Any]],
) -> List[Tuple[str, str, float]]:
    """
    Select best / median / worst cases according to relative L2 error.

    Returns
    -------
    reps:
        List of (label, case_tag, l2_error). Duplicate tags are removed while
        preserving the order: best -> median -> worst. This matters only when
        the number of compared cases is very small.
    """
    if not rows:
        return []

    sorted_rows = sorted(rows, key=lambda r: float(r["l2_error"]))
    n = len(sorted_rows)

    candidates = [
        ("best", sorted_rows[0]),
        ("median", sorted_rows[n // 2]),
        ("worst", sorted_rows[-1]),
    ]

    reps: List[Tuple[str, str, float]] = []
    used_tags = set()

    for label, row in candidates:
        tag = str(row["case_tag"])
        if tag in used_tags:
            continue
        used_tags.add(tag)
        reps.append((label, tag, float(row["l2_error"])))

    return reps


def _select_regular_plot_cases(
    compared: List[str],
    *,
    max_plot_cases: Optional[int],
    excluded_tags: set[str],
) -> List[str]:
    """
    Select the normal plotted cases.

    The normal figures are selected from the beginning of the test-set order.
    If one overlaps with best/median/worst, it is skipped and selection
    continues, avoiding duplicate plots.
    """
    if max_plot_cases is None:
        return [tag for tag in compared if tag not in excluded_tags]

    if int(max_plot_cases) <= 0:
        return []

    selected: List[str] = []
    for tag in compared:
        if tag in excluded_tags:
            continue
        selected.append(tag)
        if len(selected) >= int(max_plot_cases):
            break

    return selected



def _plot_one_case(
    *,
    compare_root: Path,
    tag: str,
    case_data: Dict[str, Any],
    no_data: Dict[str, Any],
    fem_data: Dict[str, Any],
    deformation_scale: float,
    filename_prefix: str,
) -> None:
    generate_case_compare_figures(
        case_data=case_data,
        no_data=no_data,
        fem_data=fem_data,
        displacement_out_path=compare_root / f"{filename_prefix}_disp_compare.png",
        rotation_out_path=compare_root / f"{filename_prefix}_rot_compare.png",
        resultants_out_path=compare_root / f"{filename_prefix}_resultants_compare.png",
        deformation_scale=deformation_scale,
    )


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def compare_cases(
    *,
    case_name: str,
    case_tags: Optional[List[str]] = None,
    root: Optional[str | Path] = None,
    deformation_scale: float = 10.0,
    max_plot_cases: Optional[int] = None,
    method_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compare NO predictions with FEM references.

    Parameters
    ----------
    case_name:
        Base physical case name, e.g. square, hole, holes2, lshape.

    method_name:
        None / "energy" uses the original output folders:
            NO/outputs/<case>/predictions/
            compare/<case>/

        Non-base names use method-specific folders:
            NO/outputs/<case>_<method_name>/predictions/
            compare/<case>_<method_name>/

        Examples:
            boundary_pc, all_pc, weak_residual, feonet_weak, mixed_loss.

    Behavior
    --------
    1. Metrics are computed for all available test cases.
    2. Displacement, rotation, and a 5x3 resultant comparison figure are
       generated for the first max_plot_cases cases in test-set order.
    3. The same figures are generated for the best, median, and worst cases
       according to relative displacement L2 error.
    4. If a normal case overlaps with best/median/worst, the normal selection
       skips it and continues to the next test case, avoiding duplicate plots.
    """
    io_paths = _build_compare_io_paths(
        case_name=case_name,
        root=root,
        method_name=method_name,
        create=True,
    )

    base_paths: Dict[str, Path] = io_paths["base_paths"]
    no_prediction_root: Path = io_paths["no_prediction_root"]
    compare_root: Path = io_paths["compare_root"]

    if case_tags is None:
        case_tags = load_test_tags(case_name=case_name, root=root)

    compared: List[str] = []
    skipped: List[str] = []
    rows: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # First pass: compute metrics for all available cases.
    # ------------------------------------------------------------------
    for tag in list(case_tags):
        # Case and FEM always come from the base physical case.
        cpath = case_npz_path(base_paths, tag)
        fpath = fem_pred_npz_path(base_paths, tag)

        # NO prediction comes from the selected method folder.
        npath = _no_pred_path_for_method(
            base_paths=base_paths,
            no_prediction_root=no_prediction_root,
            method_name=method_name,
            tag=tag,
        )

        if not cpath.exists() or not npath.exists() or not fpath.exists():
            skipped.append(tag)
            continue

        case_data = load_npz_dict(cpath)
        no_data = load_npz_dict(npath)
        fem_data = load_npz_dict(fpath)

        row = compute_case_error_metrics(
            case_tag=tag,
            case_data=case_data,
            no_data=no_data,
            fem_data=fem_data,
        )
        rows.append(row)
        compared.append(tag)

    if rows:
        save_metrics_table(
            rows=rows,
            npz_path=io_paths["final_metrics_npz_path"],
            csv_path=io_paths["final_metrics_csv_path"],
        )

    summary = summarize_metric_rows(rows)

    # ------------------------------------------------------------------
    # Second pass: select and generate figures.
    # ------------------------------------------------------------------
    representative_cases = _select_representative_cases_by_l2(rows)
    representative_tags = {tag for _, tag, _ in representative_cases}

    regular_plot_tags = _select_regular_plot_cases(
        compared,
        max_plot_cases=max_plot_cases,
        excluded_tags=representative_tags,
    )

    plotted_regular: List[str] = []
    plotted_representative: Dict[str, Dict[str, Any]] = {}

    def _load_triplet(tag: str):
        cpath = case_npz_path(base_paths, tag)
        fpath = fem_pred_npz_path(base_paths, tag)
        npath = _no_pred_path_for_method(
            base_paths=base_paths,
            no_prediction_root=no_prediction_root,
            method_name=method_name,
            tag=tag,
        )
        return load_npz_dict(cpath), load_npz_dict(npath), load_npz_dict(fpath)

    # Normal first-N plots, with overlap removed by shifting to later cases.
    for tag in regular_plot_tags:
        try:
            case_data, no_data, fem_data = _load_triplet(tag)
        except Exception:
            if tag not in skipped:
                skipped.append(tag)
            continue

        _plot_one_case(
            compare_root=compare_root,
            tag=tag,
            case_data=case_data,
            no_data=no_data,
            fem_data=fem_data,
            deformation_scale=deformation_scale,
            filename_prefix=tag,
        )
        plotted_regular.append(tag)

    # Best / median / worst plots according to relative L2 error.
    for label, tag, l2_error in representative_cases:
        try:
            case_data, no_data, fem_data = _load_triplet(tag)
        except Exception:
            if tag not in skipped:
                skipped.append(tag)
            continue

        filename_prefix = f"{label}_l2_{tag}"
        _plot_one_case(
            compare_root=compare_root,
            tag=tag,
            case_data=case_data,
            no_data=no_data,
            fem_data=fem_data,
            deformation_scale=deformation_scale,
            filename_prefix=filename_prefix,
        )

        plotted_representative[label] = {
            "case_tag": tag,
            "l2_error": float(l2_error),
            "disp_figure": str(compare_root / f"{filename_prefix}_disp_compare.png"),
            "rot_figure": str(compare_root / f"{filename_prefix}_rot_compare.png"),
            "resultants_figure": str(compare_root / f"{filename_prefix}_resultants_compare.png"),
        }

    plotted_all = plotted_regular + [v["case_tag"] for v in plotted_representative.values()]

    print("=" * 80)
    if summary:
        for k, v in summary.items():
            print(f"[Compare] {k}: {v:.6e}")
    print("=" * 80)

    return {
        "status": "done" if compared else "skipped",
        "method_case": io_paths["method_case"],
        "compared": compared,
        "plotted": plotted_all,
        "plotted_regular": plotted_regular,
        "plotted_representative": plotted_representative,
        "skipped": skipped,
        "summary": summary,
        "compare_root": str(compare_root),
        "metrics_csv": str(io_paths["final_metrics_csv_path"]) if rows else None,
        "metrics_npz": str(io_paths["final_metrics_npz_path"]) if rows else None,
    }
