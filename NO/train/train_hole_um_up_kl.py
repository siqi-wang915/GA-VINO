from __future__ import annotations

"""Kirchhoff-Love, bending-only training for the five hole thickness cases."""

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from NO.train.train_hole_um_up_mr import (
    run_train_hole_um_up,
)
from shared.prep.hole_um_up_mr import THICKNESS_CASES, resolve_thickness_case


KL_TRAIN_CONFIG: Dict[str, Any] = {
    # Training schedule.  Edit these values here to tune KL independently of MR.
    "batch_size": 2,
    "epochs": 500,
    "max_epochs": 5000,
    "lr": 5.0e-4,
    "early_stopping": True,
    "early_stopping_min_epochs": 100,
    "early_stopping_patience": 50,
    "early_stopping_min_delta": 1.0e-3,
    "grad_clip_norm": None,
    "loss_scale": 1.0,
    # Model architecture.  KL predicts displacement only; the trainer enforces
    # one output channel and uses this single displacement normalization scale.
    "model_config": {
        "branch_args": {
            "input_channels": 3, "out_c": 64, "width": 64,
            "d_hidden": [64, 64], "n_point": 256, "n_sample": 18,
            "radius": 0.20, "num_heads": 4, "cross_attn_layers": 2,
            "self_attn_layers": 2,
        },
        "trunk_args": {
            "embed_dim": 64, "scalar_dim": 64, "num_heads": 4,
            "cross_attn_layers": 4, "out_std": [8.0e-2],
        },
    },
    # Evaluation and output settings.
    "eval_fem_every": 1,
    "deformation_scale": 10.0,
    "max_plot_cases": 5,
    # Kirchhoff--Love theory settings: rotations are derived from w, and the
    # transverse-shear energy is identically zero.
    "method_name": "kl",
    "include_shear_energy": False,
    "plate_theory": "Kirchhoff-Love",
}


# Per-thickness overrides.  The thin cases retain the original energy scaling,
# gradient clipping, and early-stopping-driven schedule, but are hard-capped at
# KL_TRAIN_CONFIG["max_epochs"].
KL_THICKNESS_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "base": {"loss_scale": 1.0, "epochs": 500, "grad_clip_norm": None},
    "t005": {"loss_scale": 1.0e3, "epochs": None, "grad_clip_norm": 5.0},
    "t0158": {"loss_scale": 10.0 ** 1.5, "epochs": None, "grad_clip_norm": 5.0},
    "t00158": {"loss_scale": 10.0 ** 4.5, "epochs": None, "grad_clip_norm": 5.0},
    "t0005": {"loss_scale": 1.0e6, "epochs": None, "grad_clip_norm": 5.0},
}


def train_config_for_thickness_kl(selector: str = "base") -> Tuple[str, Dict[str, Any]]:
    """Return the explicitly defined KL configuration for one thickness case."""
    suffix, _ = resolve_thickness_case(selector)
    cfg = deepcopy(KL_TRAIN_CONFIG)
    cfg.update(deepcopy(KL_THICKNESS_OVERRIDES[suffix]))
    return suffix, cfg


def run_train_hole_um_up_kl(
    selector: str = "base",
    *,
    root: Optional[str | Path] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Train one thickness using bending energy only (no transverse shear)."""
    suffix, cfg = train_config_for_thickness_kl(selector)
    _, definition = resolve_thickness_case(suffix)
    if config is not None:
        cfg.update(config)

    print(
        f"[train_hole_um_up_kl] suffix={suffix}, "
        f"loss_scale={cfg['loss_scale']}, transverse_shear_energy=False"
    )
    return run_train_hole_um_up(
        root=root, config=cfg, case_name=str(definition["case_name"])
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a five-thickness hole case with Kirchhoff-Love energy."
    )
    parser.add_argument(
        "suffix", nargs="?", default="base", choices=tuple(THICKNESS_CASES)
    )
    parser.add_argument("--root", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_train_hole_um_up_kl(args.suffix, root=args.root)
