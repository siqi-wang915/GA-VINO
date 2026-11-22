from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[1]

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from shared.common.data_io import load_test_tags
from shared.common.paths import build_paths, normalize_case_name, supported_cases

# -----------------------------------------------------------------------------
# Case-name helpers
# -----------------------------------------------------------------------------

def _supported_case_names() -> List[str]:
    return list(supported_cases())


def _normalize_case_name(case_name: str) -> str:
    return normalize_case_name(case_name)


# -----------------------------------------------------------------------------
# Preparation
# -----------------------------------------------------------------------------

def prepare_case(
    *,
    case_name: str,
    root: Optional[str | Path] = None,
    overwrite: bool = True,
) -> Dict[str, Any]:
    case = _normalize_case_name(case_name)

    hole_mr_selectors = {
        "hole_um_up": "base",
        "hole_um_up_t005": "t005",
        "hole_um_up_t0158": "t0158",
        "hole_um_up_t00158": "t00158",
        "hole_um_up_t0005": "t0005",
    }
    if case in hole_mr_selectors:
        from shared.prep.hole_um_up_mr import prepare_hole_um_up_mr
        return prepare_hole_um_up_mr(
            hole_mr_selectors[case], root=root, overwrite=overwrite
        )

    if case.startswith("hole_um_up_mesh_"):
        from NO.problem.hole_mesh_convergence import MESH_DENSITY_RATIOS, density_case_name
        from shared.prep.hole_um_up_mesh_convergence import prepare_hole_um_up_mesh_density

        ratio_by_case = {density_case_name(r): r for r in MESH_DENSITY_RATIOS}
        return prepare_hole_um_up_mesh_density(
            ratio=ratio_by_case[case], root=root, overwrite=overwrite
        )

    if case == "hole_um_grfp":
        from shared.prep.hole_um_grfp import prepare_hole_um_grfp
        return prepare_hole_um_grfp(root=root, overwrite=overwrite)
    
    if case == "holes2_um_up":
        from shared.prep.holes2_um_up import prepare_holes2_um_up
        return prepare_holes2_um_up(root=root, overwrite=overwrite)

    if case == "holes2_um_grfp":
        from shared.prep.holes2_um_grfp import prepare_holes2_um_grfp
        return prepare_holes2_um_grfp(root=root, overwrite=overwrite)

    if case == "lshape_grfm_up":
        from shared.prep.lshape_grfm_up import prepare_lshape_grfm_up
        return prepare_lshape_grfm_up(root=root, overwrite=overwrite)

    if case == "lshape_grfm_grfp":
        from shared.prep.lshape_grfm_grfp import prepare_lshape_grfm_grfp
        return prepare_lshape_grfm_grfp(root=root, overwrite=overwrite)

    if case == "lshape_um_grfp":
        from shared.prep.lshape_um_grfp import prepare_lshape_um_grfp
        return prepare_lshape_um_grfp(root=root, overwrite=overwrite)
    
    if case == "lshape_grfm_up_065_085":
        from shared.prep.lshape_grfm_up_065_085 import prepare_lshape_grfm_up_065_085
        return prepare_lshape_grfm_up_065_085(root=root, overwrite=overwrite)

    if case == "annular_sector_grfm_grfp":
        from shared.prep.annular_sector_grfm_grfp import prepare_annular_sector_grfm_grfp
        return prepare_annular_sector_grfm_grfp(root=root, overwrite=overwrite)

    raise NotImplementedError(
        f"Stage-1 preparation for case='{case}' is not implemented yet. "
        f"Currently implemented: {', '.join(_supported_case_names())}."
    )


# -----------------------------------------------------------------------------
# FEM solve
# -----------------------------------------------------------------------------

def run_fem_for_case(
    *,
    case_name: str,
    root: Optional[str | Path] = None,
    overwrite: bool = False,
    n_load_steps: int = 1,
    max_cases: Optional[int] = None,
    include_train: bool = True,
) -> Dict[str, Any]:
    case = _normalize_case_name(case_name)
    root_path = Path(root).resolve() if root is not None else _PROJECT_ROOT

    # Mesh-convergence cases use the existing 100% FEM solution restricted to
    # retained original vertices.  Their preparation step already exports the
    # aligned reference files, so no new FEM solve or interpolation is needed.
    if case.startswith("hole_um_up_mesh_"):
        paths = build_paths(case, root=root_path, create=True)
        refs = sorted(paths["fem_prediction_root"].glob("*_fem.npz"))
        if not refs:
            raise FileNotFoundError(
                f"No restricted 100% FEM references found for {case}. "
                "Run preparation without --skip-prepare first."
            )
        print(
            f"[Stage1][Mesh convergence] Reusing {len(refs)} restricted "
            f"hole_um_up 100% FEM reference files for {case}."
        )
        return {
            "status": "restricted_source_fem_references",
            "case_name": case,
            "n_reference_files": len(refs),
            "fem_prediction_root": str(paths["fem_prediction_root"]),
        }

    # 3. 先获取测试集标签
    test_tags = load_test_tags(
        case_name=case,
        root=root_path,
    )

    # 4. 合并标签逻辑
    all_tags = list(test_tags)
    if include_train:
        from shared.common.data_io import load_train_tags # 确保这个函数已定义
        train_tags = load_train_tags(case_name=case, root=root_path)
        all_tags.extend(train_tags)

    # 如果有 max_cases 限制（通常用于调试）
    if max_cases is not None:
        all_tags = all_tags[:max_cases]

    print("=" * 80)
    print("[Stage1] Running FEM")
    print(f"[Stage1] case        : {case}")
    print(f"[Stage1] n_cases     : {len(all_tags)} (test: {len(test_tags)})")
    print(f"[Stage1] overwrite   : {overwrite}")
    print("=" * 80)

    from FEM.solvers.registry import solve_cases

    fem_info = solve_cases(
        case_name=case,
        case_tags=all_tags,
        root=root_path,
        overwrite=overwrite,
        n_load_steps=n_load_steps,
    )

    return fem_info


# -----------------------------------------------------------------------------
# Full Stage-1 pipeline
# -----------------------------------------------------------------------------

def run_stage1(
    *,
    case_name: str,
    root: Optional[str | Path] = None,
    overwrite: bool = False,
    skip_prepare: bool = False,
    skip_fem: bool = False,
    n_load_steps: int = 1,
    max_cases: Optional[int] = None,
    include_train: bool = True,
) -> Dict[str, Any]:
    case = _normalize_case_name(case_name)
    root_path = Path(root).resolve() if root is not None else _PROJECT_ROOT
    paths = build_paths(case, root=root_path, create=True)

    print("=" * 80)
    print("[Stage1] Prepare dataset and run FEM")
    print(f"[Stage1] project root: {root_path}")
    print(f"[Stage1] case        : {case}")
    print(f"[Stage1] data root   : {paths['data_root']}")
    print(f"[Stage1] case root   : {paths['case_root']}")
    print(f"[Stage1] fem outputs : {paths['fem_prediction_root']}")
    print("=" * 80)

    prepare_info = None
    fem_info = None

    if not skip_prepare:
        prepare_info = prepare_case(
            case_name=case,
            root=root_path,
            overwrite=overwrite,
        )
    else:
        print("[Stage1] skip_prepare=True, existing dataset/case files will be used.")

    if not skip_fem:
        fem_info = run_fem_for_case(
            case_name=case,
            root=root_path,
            overwrite=overwrite,
            n_load_steps=n_load_steps,
            max_cases=max_cases,
            include_train=include_train,
        )
    else:
        print("[Stage1] skip_fem=True, FEM will not be executed.")

    print("=" * 80)
    print("[Stage1] Finished")
    print("=" * 80)

    return {
        "case_name": case,
        "paths": {k: str(v) for k, v in paths.items()},
        "prepare_info": prepare_info,
        "fem_info": fem_info,
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 1: generate samples, export FEM cases, and run FEM."
    )

    parser.add_argument(
        "--case",
        type=str,
        required=True,
        choices=supported_cases(),
        help="Canonical case name from shared.common.paths.supported_cases().",
    )

    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Project root. If omitted, inferred from this script location.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing dataset, case files, and FEM results.",
    )

    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Skip dataset/case generation and use existing files.",
    )

    parser.add_argument(
        "--skip-fem",
        action="store_true",
        help="Skip FEM solve after preparing cases.",
    )

    parser.add_argument(
        "--n-load-steps",
        type=int,
        default=1,
        help="Number of FEM load steps. Default is 1.",
    )

    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Only run FEM for the first N test cases. Useful for debugging.",
    )

    parser.add_argument(
        "--test-only",
        action="store_true",
        help=(
            "Run only FEM test cases. Use with --skip-prepare --overwrite to "
            "refresh native moment/shear reference fields without regenerating "
            "the dataset or solving training cases."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    run_stage1(
        case_name=args.case,
        root=args.root,
        overwrite=args.overwrite,
        skip_prepare=args.skip_prepare,
        skip_fem=args.skip_fem,
        n_load_steps=args.n_load_steps,
        max_cases=args.max_cases,
        include_train=not args.test_only,
    )


if __name__ == "__main__":
    main()
