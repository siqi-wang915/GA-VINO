from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from shared.common.paths import normalize_case_name


SolverFn = Callable[..., Dict[str, Any]]


def _get_solver_map() -> Dict[str, SolverFn]:
    from FEM.solvers.hole import solve_hole
    from FEM.solvers.holes2 import solve_holes2
    from FEM.solvers.lshape import solve_lshape
    from FEM.solvers.annular_sector import solve_annular_sector

    return {

        "hole_um_up": solve_hole,
        "hole_um_up_t005": solve_hole,
        "hole_um_up_t0158": solve_hole,
        "hole_um_up_t00158": solve_hole,
        "hole_um_up_t0005": solve_hole,
        "hole_um_up_mesh_quality_02": solve_hole,
        "hole_um_up_mesh_quality_03": solve_hole,
        "hole_um_up_mesh_quality_04": solve_hole,
        "hole_um_grfp": solve_hole,

        "holes2_um_up": solve_holes2,
        "holes2_um_grfp": solve_holes2,

        "lshape_grfm_up": solve_lshape,
        "lshape_grfm_grfp": solve_lshape,
        "lshape_um_grfp": solve_lshape,
        "lshape_grfm_up_065_085": solve_lshape,
        "annular_sector_grfm_grfp": solve_annular_sector,
    }


def supported_fem_cases() -> List[str]:
    return sorted(_get_solver_map().keys())


def solve_cases(
    *,
    case_name: str,
    case_tags: Optional[List[str]] = None,
    root: Optional[str | Path] = None,
    overwrite: bool = False,
    n_load_steps: int = 1,
    **kwargs,
) -> Dict[str, Any]:
    case = normalize_case_name(case_name)

    solver_map = _get_solver_map()

    if case not in solver_map:
        supported = ", ".join(sorted(solver_map.keys()))
        raise ValueError(
            f"FEM solver for case='{case}' is not implemented yet. "
            f"Currently implemented FEM cases: {supported}"
        )

    solver = solver_map[case]

    return solver(
        case_name=case,
        case_tags=case_tags,
        root=root,
        overwrite=overwrite,
        n_load_steps=n_load_steps,
        **kwargs,
    )


if __name__ == "__main__":
    info = solve_cases(
        case_name="hole_um_up",
        overwrite=False,
        n_load_steps=1,
    )
    print(info)
