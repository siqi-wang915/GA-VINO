from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from FEM.solvers.lshape import solve_lshape


CASE_NAME = "annular_sector_grfm_grfp"
SOLVER_NAME = "annular_sector_rm_dl_linear_plate_fem"


def solve_annular_sector(
    *,
    case_name: str = CASE_NAME,
    case_tags: Optional[List[str]] = None,
    root: Optional[str | Path] = None,
    overwrite: bool = False,
    n_load_steps: int = 1,
    **kwargs,
) -> Dict[str, Any]:
    """
    Solve annular-sector Mindlin-Reissner plate cases.

    Geometry:
        r_inner <= r <= r_outer
        theta_start <= theta <= theta_end
        theta_end = 90 deg
        theta_start = 90 deg - theta_span

    Boundary condition:
        x = 0 radial edge is clamped;
        inner arc, outer arc, and the other radial edge are free.

    Material:
        Supports elementwise Gaussian random fields through E_elem and nu_elem.

    Load:
        Random nonuniform transverse line load on the theta_start radial edge.
    """
    return solve_lshape(
        case_name=case_name,
        case_tags=case_tags,
        root=root,
        overwrite=overwrite,
        n_load_steps=n_load_steps,
        **kwargs,
    )
