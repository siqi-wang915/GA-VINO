from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import sys

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from NO.train.train_lshape_grfm_grfp import run_train_lshape_grfm_grfp


CASE_NAME = "lshape_um_grfp"


DEFAULT_TRAIN_CONFIG: Dict[str, Any] = {
    "batch_size": 16,
    "epochs": 500,
    "lr": 5.0e-4,
    "grad_clip_norm": 1.0,
    "loss_scale": 1.0e5,

    # Stop when the monitored energy loss has no relative improvement of at
    # least 0.1% for 50 epochs. Training always runs for at least 100 epochs.
    "early_stopping": False,
    "early_stopping_min_epochs": 100,
    "early_stopping_patience": 50,
    "early_stopping_min_delta": 1.0e-3,

    # Case-specific GINOT-like architecture parameters.
    "model_config": {
        # Geometry encoder input: boundary point cloud [x, y, z].
        "branch_args": {
            "input_channels": 3,
            "out_c": 64,
            "width": 64,
            "d_hidden": [64, 64],
            "n_point": 256,
            "n_sample": 18,
            "radius": 0.20,
            "num_heads": 4,
            "cross_attn_layers": 2,
            "self_attn_layers": 2,
        },

        # Random-pressure encoder input: [x, y, z, p].
        "load_branch_args": {
            "input_channels": 4,
            "out_c": 96,
            "width": 64,
            "d_hidden": [64, 64],
            "n_point": 256,
            "n_sample": 36,
            "radius": 0.20,
            "num_heads": 4,
            "cross_attn_layers": 2,
            "self_attn_layers": 2,
        },

        # Kept explicit for model/checkpoint compatibility. This case sets
        # use_material_field=False, so this branch is skipped at runtime.
        "material_branch_args": {
            "input_channels": 5,
            "out_c": 96,
            "width": 64,
            "d_hidden": [64, 64],
            "n_point": 256,
            "n_sample": 36,
            "radius": 0.20,
            "num_heads": 4,
            "cross_attn_layers": 2,
            "self_attn_layers": 2,
        },

        # Solution decoder. The scalar encoder uses scalar_dim=64 and receives
        # [0, 0, normalized_mean_pressure] for this uniform-material case.
        "trunk_args": {
            "embed_dim": 64,
            "scalar_dim": 64,
            "num_heads": 4,
            "cross_attn_layers": 4,
            "use_nerf_query": True,
            "out_std": [1.0e-1, 3.0e-2, 3.0e-2],
        },
    },

    # FEM metric frequency. 1 means evaluate FEM error every epoch.
    "eval_fem_every": 1,

    # Final visualization.
    "deformation_scale": 10.0,
    "max_plot_cases": 5,

    "method_name": None,
}


def run_train_lshape_um_grfp(
    *,
    root: Optional[str | Path] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Train, predict, evaluate, and plot the uniform-material GRF-load case."""
    cfg = dict(DEFAULT_TRAIN_CONFIG)
    if config is not None:
        cfg.update(config)

    return run_train_lshape_grfm_grfp(
        root=root,
        config=cfg,
        case_name=CASE_NAME,
    )


if __name__ == "__main__":
    run_train_lshape_um_grfp()
