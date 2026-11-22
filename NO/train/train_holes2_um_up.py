from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import sys
import json

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

    # Stop when the monitored energy loss has no relative improvement of at
    # least 0.1% for 50 epochs. Training always runs for at least 100 epochs.
    "early_stopping": True,
    "early_stopping_min_epochs": 100,
    "early_stopping_patience": 50,
    "early_stopping_min_delta": 1.0e-3,

    # Case-specific GINOT-like architecture parameters.
    "model_config": {
        "branch_args": {
            "input_channels": 3, #xyz
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
            "input_channels": 4,   # 必须是 4: [x,y,z,p]
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
            "out_std": [8.0e-2, 2.0e-2, 2.0e-2],
        },
    },

    # FEM metric frequency. 1 means evaluate FEM error every epoch.
    "eval_fem_every": 1,

    # Final visualization.
    "deformation_scale": 10.0,
    "max_plot_cases": 5,

    "method_name": None,
}

def _to_jsonable(obj):
    """
    Convert common Python / torch / numpy / pathlib objects to JSON-safe objects.
    """
    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, torch.Tensor):
        arr = obj.detach().cpu()
        if arr.numel() == 1:
            return float(arr.reshape(-1)[0].item())
        return arr.tolist()

    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]

    try:
        import numpy as np

        if isinstance(obj, np.ndarray):
            if obj.shape == ():
                return obj.item()
            return obj.tolist()

        if isinstance(obj, (np.integer,)):
            return int(obj)

        if isinstance(obj, (np.floating,)):
            return float(obj)
    except Exception:
        pass

    return obj


def _sample_scalar(sample: Dict[str, Any] | None, key: str, default=None):
    """
    Read one scalar field from a sample dictionary.

    Works for torch tensor, numpy array, Python scalar.
    """
    if sample is None or key not in sample:
        return default

    value = sample[key]

    try:
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu().reshape(-1)[0].item())

        import numpy as np

        arr = np.asarray(value)
        if arr.size > 0:
            return float(arr.reshape(-1)[0])
    except Exception:
        return default

    return default


def _get_output_root_from_method_paths(method_paths: Dict[str, Any]) -> Path:
    """
    Robustly infer the NO output root from build_energy_output_paths() output.

    Expected in your current project:
        method_paths["no_output_root"]

    Fallback:
        parent directory of method_paths["no_prediction_root"].
    """
    if "no_output_root" in method_paths:
        return Path(method_paths["no_output_root"])

    if "output_root" in method_paths:
        return Path(method_paths["output_root"])

    if "no_prediction_root" in method_paths:
        return Path(method_paths["no_prediction_root"]).parent

    raise KeyError(
        "Cannot find NO output root in method_paths. "
        "Expected one of: no_output_root, output_root, no_prediction_root."
    )


def _save_run_metadata(
    *,
    method_paths: Dict[str, Any],
    cfg: Dict[str, Any],
    manifest: Dict[str, Any],
    train_samples,
    test_samples,
    history: Dict[str, Any],
    case_name: str,
    method_name,
) -> Path:
    """
    Save metadata needed for future OOD evaluation.

    For hole -> holes2 OOD prediction, this file records:
        1. source model architecture;
        2. checkpoint paths;
        3. data normalization / load settings;
        4. input-output schema.
    """
    if len(train_samples) > 0:
        ref_sample = train_samples[0]
    elif len(test_samples) > 0:
        ref_sample = test_samples[0]
    else:
        ref_sample = None

    data_config = {
        "E": _sample_scalar(ref_sample, "E", None),
        "nu": _sample_scalar(ref_sample, "nu", None),
        "thickness": _sample_scalar(ref_sample, "thickness", None),

        "pressure_min": _sample_scalar(ref_sample, "pressure_min", None),
        "pressure_max": _sample_scalar(ref_sample, "pressure_max", None),

        # For Gaussian random-field pressure, these fields should exist in the new samples.
        "pressure_mean": _sample_scalar(
            ref_sample,
            "pressure_target_mean",
            _sample_scalar(ref_sample, "pressure_mean", None),
        ),
        "pressure_std": _sample_scalar(ref_sample, "pressure_std", None),
        "pressure_corr_len": _sample_scalar(ref_sample, "pressure_corr_len", None),
        "pressure_n_modes": _sample_scalar(ref_sample, "pressure_n_modes", None),

        "geometry_type": str(ref_sample.get("geometry_type", "")) if ref_sample is not None else "",
        "load_type": str(ref_sample.get("load_type", "")) if ref_sample is not None else "",
        "material_type": str(ref_sample.get("material_type", "uniform")) if ref_sample is not None else "uniform",

        "lx": _sample_scalar(ref_sample, "lx", None),
        "ly": _sample_scalar(ref_sample, "ly", None),
        "center_min": _sample_scalar(ref_sample, "center_min", None),
        "center_max": _sample_scalar(ref_sample, "center_max", None),
        "mean_radius": _sample_scalar(ref_sample, "mean_radius", None),
        "r_min": _sample_scalar(ref_sample, "r_min", None),
        "r_max": _sample_scalar(ref_sample, "r_max", None),
        "roughness": _sample_scalar(ref_sample, "roughness", None),
        "n_modes": _sample_scalar(ref_sample, "n_modes", None),
    }

    metadata = {
        "source_case": case_name,
        "method_name": method_name,

        "checkpoint": {
            "best_model_path": str(method_paths["best_model_path"]),
            "final_model_path": str(method_paths["final_model_path"]),
        },

        # This is the most important part for OOD reconstruction.
        "model_config": cfg.get("model_config", None),

        "training_config": {
            "batch_size": int(cfg["batch_size"]),
            "epochs": int(cfg["epochs"]),
            "actual_epochs": int(len(history.get("epoch", []))),
            "lr": float(cfg["lr"]),
            "eval_fem_every": int(cfg["eval_fem_every"]),
            "deformation_scale": float(cfg["deformation_scale"]),
            "max_plot_cases": int(cfg["max_plot_cases"]),
            "early_stopping": {
                "enabled": bool(cfg["early_stopping"]),
                "monitor": "test_energy_loss",
                "mode": "min",
                "min_epochs": int(cfg["early_stopping_min_epochs"]),
                "patience": int(cfg["early_stopping_patience"]),
                "relative_min_delta": float(cfg["early_stopping_min_delta"]),
                "stopped_early": bool(
                    len(history.get("epoch", [])) < int(cfg["epochs"])
                ),
            },
        },

        "data_config": data_config,

        "input_schema": {
            "pc_net": "[B, Ng, 3] = [x_net, y_net, z_net], boundary point cloud",
            "query_net": "[B, Nq, 3] = [x_net, y_net, z_net]",
            "load_pc_net": "[B, Nl, 4] = [x_net, y_net, z_net, p_net]",
            "material_pc_net": "[B, Nm, 5] = [x_net, y_net, z_net, E_net, nu_net]",
            "scalar_param": "[B, 3] = [E_scalar_net, nu_scalar_net, p_scalar_net]",
            "output": "[B, Nq, 3] = [w, theta_x, theta_y]",
        },

        "manifest": _to_jsonable(manifest),
    }

    no_output_root = _get_output_root_from_method_paths(method_paths)
    out_path = no_output_root / "run_metadata.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(metadata), f, indent=2, ensure_ascii=False)

    return out_path


def run_train_holes2_um_up(
    *,
    root: Optional[str | Path] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    case_name = "holes2_um_up"
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

    # Test loader must not be shuffled, otherwise it will not match case_0000, ...
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

    metadata_path = _save_run_metadata(
        method_paths=method_paths,
        cfg=cfg,
        manifest=manifest,
        train_samples=train_samples,
        test_samples=test_samples,
        history=history,
        case_name=case_name,
        method_name=method_name,
    )

    print("=" * 80)
    print(f"[Metadata] run metadata saved: {metadata_path}")
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
        "run_metadata_path": str(metadata_path),
    }


if __name__ == "__main__":
    run_train_holes2_um_up()
