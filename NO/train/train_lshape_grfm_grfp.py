from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import sys

import torch
from torch.utils.data import DataLoader

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[2]

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from NO.common.core import (
    PlateSampleDataset,
    mr_plate_ginot_minimal_configs,
    plate_collate_fn,
)
from NO.common.evaluator import export_no_predictions
from NO.common.node_l2_evaluator import NodeL2FEMMetricEvaluator as FEMMetricEvaluator
from NO.common.trainer import build_energy_output_paths, train_with_fem_metrics
from shared.common.data_io import load_dataset, load_test_tags
from shared.common.paths import build_paths
from shared.compare.compare import compare_cases


DEFAULT_TRAIN_CONFIG: Dict[str, Any] = {
    "batch_size": 16,
    "epochs": 500,  
    "lr": 5.0e-4,
    "grad_clip_norm": 1.0,
    "loss_scale": 1.0e5,

    # Stop when the monitored energy loss has no relative improvement of at
    # least 0.1% for 50 epochs. Training always runs for at least 100 epochs.
    "early_stopping": True,
    "early_stopping_min_epochs": 100,
    "early_stopping_patience": 50,
    "early_stopping_min_delta": 1.0e-3,

    # Case-specific GINOT-like architecture parameters.

    "model_config": {
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

        "load_branch_args": {
            "input_channels": 4,   # [x, y, z, p]
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

        "material_branch_args": {
            "input_channels": 5,   # 必须是 5: [x,y,z,E,nu]
            "out_c": 96,
            "width": 64,
            "d_hidden": [64, 64],
            "n_point": 256,
            "n_sample": 36,
            "radius": 0.2,
            "num_heads": 4,
            "cross_attn_layers": 2,
            "self_attn_layers": 2,
        },

        "trunk_args": {
            "embed_dim": 64,
            "scalar_dim": 64,
            "num_heads": 4,
            "cross_attn_layers": 4,
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


def run_train_lshape_grfm_grfp(
    *,
    root: Optional[str | Path] = None,
    config: Optional[Dict[str, Any]] = None,
    case_name: str = "lshape_grfm_grfp",
) -> Dict[str, Any]:
    paths = build_paths(case_name, root=root, create=True)

    cfg = dict(DEFAULT_TRAIN_CONFIG)
    if config is not None:
        cfg.update(config)

    method_name = None
    method_paths = build_energy_output_paths(
        case_name=case_name,
        root=root,
        method_name=method_name,
        create=True,
    )


    torch.manual_seed(42)

    train_samples, test_samples, manifest = load_dataset(
        case_name=case_name,
        root=root,
    )


    test_tags = load_test_tags(
        case_name=case_name,
        root=root,
    )

    padding_value = float(mr_plate_ginot_minimal_configs()["padding_value"])

    train_ds = PlateSampleDataset(train_samples)
    test_ds = PlateSampleDataset(test_samples)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        collate_fn=lambda b: plate_collate_fn(b, padding_value=padding_value),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        collate_fn=lambda b: plate_collate_fn(b, padding_value=padding_value),
    )

    fem_evaluator = FEMMetricEvaluator(
        case_name=case_name,
        case_tags=test_tags,
        root=root,
        padding_value=padding_value,
    )

    operator_model, history, fem_error_history = train_with_fem_metrics(
        case_name=case_name,
        train_loader=train_loader,
        test_loader=test_loader,
        fem_evaluator=fem_evaluator,
        root=root,
        epochs=int(cfg["epochs"]),
        lr=float(cfg["lr"]),
        eval_fem_every=int(cfg["eval_fem_every"]),
        grad_clip_norm=float(cfg["grad_clip_norm"]) if cfg.get("grad_clip_norm", None) is not None else None,
        loss_scale=float(cfg.get("loss_scale", 1.0)),
        model_config=cfg.get("model_config", None),
        method_name=method_name,
        early_stopping=bool(cfg["early_stopping"]),
        early_stopping_min_epochs=int(cfg["early_stopping_min_epochs"]),
        early_stopping_patience=int(cfg["early_stopping_patience"]),
        early_stopping_min_delta=float(cfg["early_stopping_min_delta"]),
    )

    saved_tags = export_no_predictions(
        operator_model=operator_model,
        data_loader=test_loader,
        case_name=case_name,
        case_tags=test_tags,
        root=root,
        padding_value=padding_value,
        method_name=method_name,
    )

    compare_info = compare_cases(
        case_name=case_name,
        case_tags=saved_tags,
        root=root,
        deformation_scale=float(cfg["deformation_scale"]),
        max_plot_cases=int(cfg["max_plot_cases"]),
        method_name=method_name,
    )

    print("=" * 80)
    print(f"[Stage2][{case_name}] Finished")
    print("=" * 80)

    return {
        "case_name": case_name,
        "method_name": method_name,
        "paths": {k: str(v) for k, v in paths.items()},
        "method_paths": {k: str(v) for k, v in method_paths.items()},
        "manifest": manifest,
        "history": history,
        "fem_error_history": fem_error_history,
        "saved_no_tags": saved_tags,
        "compare_info": compare_info,
    }


if __name__ == "__main__":
    run_train_lshape_grfm_grfp()
