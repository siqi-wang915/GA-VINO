from __future__ import annotations

import argparse
import sys
from pathlib import Path


_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[1]

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from shared.common.paths import supported_cases
from shared.compare.compare import compare_cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Postprocess existing NO/FEM prediction files: update the final "
            "per-sample metric table and generate displacement, rotation, and "
            "P1-reconstructed bending-moment/transverse-shear comparison "
            "figures. No training or FEM solve is performed; resultants on "
            "both sides are reconstructed from their saved vertex fields."
        )
    )
    parser.add_argument("--case", required=True, choices=supported_cases())
    parser.add_argument(
        "--method",
        default=None,
        help=(
            "NO method folder suffix, for example mse or fno. Omit for the "
            "base energy model."
        ),
    )
    parser.add_argument(
        "--max-plot-cases",
        type=int,
        default=5,
        help=(
            "Number of regular test samples to plot in addition to unique "
            "best/median/worst samples (default: 5)."
        ),
    )
    parser.add_argument(
        "--deformation-scale",
        type=float,
        default=10.0,
        help="Scale used only by the existing displacement comparison figure.",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Project root; defaults to the repository containing this script.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    info = compare_cases(
        case_name=args.case,
        root=args.root,
        method_name=args.method,
        max_plot_cases=args.max_plot_cases,
        deformation_scale=args.deformation_scale,
    )
    print(f"[Postprocess] compared samples: {len(info['compared'])}")
    print(f"[Postprocess] plotted samples : {len(info['plotted'])}")
    print(f"[Postprocess] metrics CSV     : {info['metrics_csv']}")
    print(f"[Postprocess] output folder   : {info['compare_root']}")


if __name__ == "__main__":
    main()
