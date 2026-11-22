from __future__ import annotations

"""Dataset entry point for the Kirchhoff-Love hole-plate experiments.

The geometry, material, pressure and FEM input cases are theory-independent,
so this module deliberately reuses the five ``hole_um_up_mr`` datasets.  Only
the training potential changes in ``train_hole_um_up_kl.py``.
"""

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

from shared.prep.hole_um_up_mr import (
    THICKNESS_CASES,
    prepare_hole_um_up_mr,
    resolve_thickness_case,
)


def prepare_hole_um_up_kl(
    selector: str = "base",
    *,
    root: Optional[str | Path] = None,
    overwrite: bool = True,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Prepare one of the five shared physical cases for KL training."""
    suffix, definition = resolve_thickness_case(selector)
    print(
        f"[hole_um_up_kl] suffix={suffix}, h={definition['thickness']}; "
        "reusing the physical dataset/FEM case name "
        f"{definition['case_name']}"
    )
    return prepare_hole_um_up_mr(
        suffix, root=root, overwrite=overwrite, config=config
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a shared physical hole case for Kirchhoff-Love training."
    )
    parser.add_argument(
        "suffix", nargs="?", default="base",
        choices=tuple(THICKNESS_CASES),
        help="Thickness suffix.",
    )
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--no-overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    prepare_hole_um_up_kl(
        args.suffix, root=args.root, overwrite=not args.no_overwrite
    )
