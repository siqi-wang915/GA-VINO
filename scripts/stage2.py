from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[1]

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from shared.common.paths import normalize_case_name, supported_cases


def run_stage2(
    *,
    case_name: str,
    root: Optional[str | Path] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    lr: Optional[float] = None,
    eval_fem_every: Optional[int] = None,
    method: Optional[str] = None,
) -> Dict[str, Any]:
    raw_case = str(case_name).strip().lower().replace("-", "_")
    method_key = None if method is None else str(method).strip().lower()

    # Convenience aliases for supervised FNO baselines. The underlying data,
    # FEM references, and comparison case remain the physical L-shape case.
    if raw_case in {"lshape_grfm_up_fno", "lshape_fno"}:
        raw_case = "lshape_grfm_up"
        method_key = "fno"
    elif raw_case in {"lshape_grfm_grfp_fno", "lshape_grfp_fno"}:
        raw_case = "lshape_grfm_grfp"
        method_key = "fno"
    elif raw_case in {"hole_um_up_mse", "hole_mse"}:
        raw_case = "hole_um_up"
        method_key = "mse"
    elif raw_case in {"hole_um_up_pino_strong", "hole_pino_strong"}:
        raw_case = "hole_um_up"
        method_key = "pino_strong"
    elif raw_case in {"hole_um_up_pino_strong_data", "hole_pino_strong_data"}:
        raw_case = "hole_um_up"
        method_key = "pino_strong_data"
    elif raw_case in {"hole_um_up_energy_data", "hole_energy_data"}:
        raw_case = "hole_um_up"
        method_key = "energy_data"
    elif raw_case in {"hole_um_grfp_mse", "hole_grfp_mse"}:
        raw_case = "hole_um_grfp"
        method_key = "mse"
    elif raw_case in {"holes2_um_up_mse", "holes2_mse"}:
        raw_case = "holes2_um_up"
        method_key = "mse"
    elif raw_case in {"holes2_um_grfp_mse", "holes2_grfp_mse"}:
        raw_case = "holes2_um_grfp"
        method_key = "mse"
    elif raw_case in {"annular_sector_grfm_grfp_mse", "annular_sector_mse", "sector_grfm_grfp_mse"}:
        raw_case = "annular_sector_grfm_grfp"
        method_key = "mse"

    case = normalize_case_name(raw_case)
    root_path = Path(root).resolve() if root is not None else _PROJECT_ROOT

    config: Dict[str, Any] = {}

    if epochs is not None:
        config["epochs"] = int(epochs)

    if batch_size is not None:
        config["batch_size"] = int(batch_size)

    if lr is not None:
        config["lr"] = float(lr)

    if eval_fem_every is not None:
        config["eval_fem_every"] = int(eval_fem_every)

    if method_key in {"fno", "supervised_fno", "data_fno"}:
        if case == "lshape_grfm_up":
            from NO.train.train_lshape_grfm_up_fno import run_train_lshape_grfm_up_fno
            return run_train_lshape_grfm_up_fno(root=root_path, config=config)
        if case == "lshape_grfm_grfp":
            from NO.train.train_lshape_grfm_grfp_fno import run_train_lshape_grfm_grfp_fno
            return run_train_lshape_grfm_grfp_fno(root=root_path, config=config)
        else:
            raise NotImplementedError(
                f"FNO stage2 is currently implemented only for lshape_grfm_up "
                f"and lshape_grfm_grfp, got case='{case}'."
            )

    if method_key in {"kl", "kirchhoff_love", "kirchhoff-love"}:
        hole_kl_selectors = {
            "hole_um_up": "base",
            "hole_um_up_t005": "t005",
            "hole_um_up_t0158": "t0158",
            "hole_um_up_t00158": "t00158",
            "hole_um_up_t0005": "t0005",
        }
        if case in hole_kl_selectors:
            from NO.train.train_hole_um_up_kl import run_train_hole_um_up_kl
            return run_train_hole_um_up_kl(
                hole_kl_selectors[case], root=root_path, config=config
            )
        raise NotImplementedError(
            "Kirchhoff-Love stage2 is implemented only for the five "
            "hole_um_up thickness cases."
        )

    if method_key in {"mse", "supervised_mse", "data_mse"}:
        if case == "hole_um_up":
            from NO.train.train_hole_um_up_mse import run_train_hole_um_up_mse
            return run_train_hole_um_up_mse(root=root_path, config=config)

        if case == "hole_um_grfp":
            from NO.train.train_hole_um_grfp_mse import run_train_hole_um_grfp_mse
            return run_train_hole_um_grfp_mse(root=root_path, config=config)

        if case == "holes2_um_up":
            from NO.train.train_holes2_um_up_mse import run_train_holes2_um_up_mse
            return run_train_holes2_um_up_mse(root=root_path, config=config)

        if case == "holes2_um_grfp":
            from NO.train.train_holes2_um_grfp_mse import run_train_holes2_um_grfp_mse
            return run_train_holes2_um_grfp_mse(root=root_path, config=config)

        if case == "annular_sector_grfm_grfp":
            from NO.train.train_annular_sector_grfm_grfp_mse import run_train_annular_sector_grfm_grfp_mse
            return run_train_annular_sector_grfm_grfp_mse(root=root_path, config=config)

        raise NotImplementedError(
            f"MSE stage2 is currently implemented only for hole/hole2/annular-sector cases, "
            f"got case='{case}'."
        )

    if method_key in {"pino_strong", "strong", "strong_physics"}:
        if case == "hole_um_up":
            from NO.train.train_hole_um_up_physics import run_train_hole_um_up_pino_strong
            return run_train_hole_um_up_pino_strong(root=root_path, config=config)
        raise NotImplementedError("The PINO strong-form baseline is implemented only for hole_um_up.")

    if method_key in {"pino_strong_data", "strong_data", "strong_physics_data"}:
        if case == "hole_um_up":
            from NO.train.train_hole_um_up_physics import run_train_hole_um_up_pino_strong_data
            return run_train_hole_um_up_pino_strong_data(root=root_path, config=config)
        raise NotImplementedError("The PINO strong-form + data baseline is implemented only for hole_um_up.")

    if method_key in {"energy_data", "variational_data"}:
        if case == "hole_um_up":
            from NO.train.train_hole_um_up_physics import run_train_hole_um_up_energy_data
            return run_train_hole_um_up_energy_data(root=root_path, config=config)
        raise NotImplementedError("The variational energy + data baseline is implemented only for hole_um_up.")

    hole_mr_selectors = {
        "hole_um_up": "base",
        "hole_um_up_t005": "t005",
        "hole_um_up_t0158": "t0158",
        "hole_um_up_t00158": "t00158",
        "hole_um_up_t0005": "t0005",
    }
    if case in hole_mr_selectors:
        from NO.train.train_hole_um_up_mr import run_train_hole_um_up_mr
        return run_train_hole_um_up_mr(
            hole_mr_selectors[case], root=root_path, config=config
        )

    if case.startswith("hole_um_up_mesh_"):
        from NO.train.train_hole_um_up_mr import run_train_hole_um_up
        return run_train_hole_um_up(
            root=root_path,
            config=config,
            case_name=case,
        )

    if case == "hole_um_grfp":
        from NO.train.train_hole_um_grfp import run_train_hole_um_grfp
        return run_train_hole_um_grfp(root=root_path, config=config)
    
    if case == "holes2_um_up":
        from NO.train.train_holes2_um_up import run_train_holes2_um_up
        return run_train_holes2_um_up(root=root_path, config=config)

    if case == "holes2_um_grfp":
        from NO.train.train_holes2_um_grfp import run_train_holes2_um_grfp
        return run_train_holes2_um_grfp(root=root_path, config=config)

    if case == "lshape_grfm_up":
        from NO.train.train_lshape_grfm_up import run_train_lshape_grfm_up
        return run_train_lshape_grfm_up(root=root_path, config=config)

    if case == "lshape_grfm_grfp":
        from NO.train.train_lshape_grfm_grfp import run_train_lshape_grfm_grfp
        return run_train_lshape_grfm_grfp(root=root_path, config=config)

    if case == "lshape_um_grfp":
        from NO.train.train_lshape_um_grfp import run_train_lshape_um_grfp
        return run_train_lshape_um_grfp(root=root_path, config=config)

    if case == "annular_sector_grfm_grfp":
        from NO.train.train_annular_sector_grfm_grfp import run_train_annular_sector_grfm_grfp
        return run_train_annular_sector_grfm_grfp(root=root_path, config=config)
    

    raise NotImplementedError(
        f"Stage2 training for case='{case}' is not implemented yet. "
        f"Currently implemented: {', '.join(supported_cases())}."
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 2: train NO with FEM metrics and generate final comparison."
    )

    parser.add_argument(
        "--case",
        type=str,
        required=True,
        choices=supported_cases() + [
            "hole_um_up_mse",
            "hole_mse",
            "hole_um_up_pino_strong",
            "hole_pino_strong",
            "hole_um_up_pino_strong_data",
            "hole_pino_strong_data",
            "hole_um_up_energy_data",
            "hole_energy_data",
            "hole_um_grfp_mse",
            "hole_grfp_mse",
            "holes2_um_up_mse",
            "holes2_mse",
            "holes2_um_grfp_mse",
            "holes2_grfp_mse",
            "annular_sector_mse",
            "annular_sector_grfm_grfp_mse",
            "sector_grfm_grfp_mse",
            "lshape_grfm_up_fno",
            "lshape_fno",
            "lshape_grfm_grfp_fno",
            "lshape_grfp_fno",
        ],
        help=(
            "Case name. Use lshape_grfm_up_fno / lshape_grfm_grfp_fno, "
            "or --case <lshape case> --method fno for the supervised FNO baseline. "
            "Use *_mse aliases or --method mse for supervised FEM-MSE baselines."
        ),
    )

    parser.add_argument(
        "--method",
        type=str,
        default=None,
        choices=["energy", "kl", "kirchhoff_love", "fno", "supervised_fno", "data_fno", "mse", "supervised_mse", "data_mse", "pino_strong", "pino_strong_data", "energy_data"],
        help="Training method override. For hole_um_up, pino_strong, pino_strong_data, and energy_data run the new physics baselines.",
    )

    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Project root. If omitted, inferred from this script location.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override default training epochs.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override default batch size.",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override default learning rate.",
    )

    parser.add_argument(
        "--eval-fem-every",
        type=int,
        default=None,
        help="Compute FEM metrics every N epochs. Default is case config.",
    )

    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Training device; auto validates that the installed CUDA wheel can execute on this GPU.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["NO_DEVICE"] = args.device

    run_stage2(
        case_name=args.case,
        root=args.root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        eval_fem_every=args.eval_fem_every,
        method=args.method,
    )


if __name__ == "__main__":
    main()
