from __future__ import annotations

"""Unified training entry point for the five hole_um_up thickness cases."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from shared.prep.hole_um_up_mr import resolve_thickness_case

import torch
from torch.utils.data import DataLoader

from NO.common.core import (
    PlateSampleDataset,
    mr_plate_ginot_minimal_configs,
    plate_collate_fn,
)
from NO.common.evaluator import export_no_predictions
from NO.common.node_l2_evaluator import NodeL2FEMMetricEvaluator as FEMMetricEvaluator
from NO.common.trainer import build_energy_output_paths, train_with_fem_metrics
from shared.common.data_io import load_dataset, load_test_tags, load_train_tags
from shared.common.paths import build_paths
from shared.compare.compare import compare_cases


DEFAULT_TRAIN_CONFIG: Dict[str, Any] = {
    "batch_size": 16,
    "epochs": 500,
    # A hard guard for all MR/KL thickness studies.  Thin cases otherwise use
    # epochs=None and rely on early stopping alone.
    "max_epochs": 5000,
    "lr": 5.0e-4,
    "early_stopping": True,
    "early_stopping_min_epochs": 100,
    "early_stopping_patience": 50,
    "early_stopping_min_delta": 1.0e-3,
    "model_config": {
        "branch_args": {
            "input_channels": 3, "out_c": 64, "width": 64,
            "d_hidden": [64, 64], "n_point": 256, "n_sample": 18,
            "radius": 0.20, "num_heads": 4, "cross_attn_layers": 2,
            "self_attn_layers": 2,
        },
        "trunk_args": {
            "embed_dim": 64, "scalar_dim": 64, "num_heads": 4,
            "cross_attn_layers": 4, "out_std": [8.0e-2, 2.0e-2, 2.0e-2],
        },
    },
    "eval_fem_every": 1,
    "deformation_scale": 10.0,
    "max_plot_cases": 5,
    "method_name": None,
}


# Existing scales are 1, 1e3, and 1e6 at h=0.5, 0.05, and 0.005.  The two
# inserted thicknesses are halfway between adjacent cases on a log-thickness
# axis, so their scales are the corresponding geometric means.
LOSS_SCALE_BY_SUFFIX = {
    "base": 1.0,
    "t005": 1.0e3,
    "t0158": 10.0 ** 1.5,   # sqrt(1 * 1e3)  = 3.16227766e1
    "t00158": 10.0 ** 4.5,  # sqrt(1e3 * 1e6) = 3.16227766e4
    "t0005": 1.0e6,
}


def train_config_for_thickness(selector: str = "base") -> Tuple[str, Dict[str, Any]]:
    """Return the fully resolved energy-training configuration for a suffix."""
    suffix, _ = resolve_thickness_case(selector)
    cfg = deepcopy(DEFAULT_TRAIN_CONFIG)
    cfg["loss_scale"] = float(LOSS_SCALE_BY_SUFFIX[suffix])

    # The h=0.5 baseline uses four encoder cross-attention layers and a fixed
    # 500-epoch schedule. Thin cases retain clipping and early stopping.
    if suffix == "base":
        cfg["model_config"]["branch_args"]["cross_attn_layers"] = 4
        cfg["epochs"] = 500
        cfg["max_epochs"] = 500
        cfg["early_stopping"] = False
    else:
        cfg["epochs"] = None
        cfg["grad_clip_norm"] = 5.0
    return suffix, cfg


def run_train_hole_um_up_mr(
    selector: str = "base",
    *,
    root: Optional[str | Path] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Train one thickness case selected by ``base/t005/t0158/t00158/t0005``."""
    suffix, cfg = train_config_for_thickness(selector)
    _, definition = resolve_thickness_case(suffix)
    if config is not None:
        cfg.update(config)

    print(
        f"[train_hole_um_up_mr] suffix={suffix}, "
        f"loss_scale={cfg['loss_scale']}"
    )
    return run_train_hole_um_up(
        root=root,
        config=cfg,
        case_name=str(definition["case_name"]),
    )


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        return float(value.reshape(-1)[0]) if value.numel() == 1 else value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            return value.item() if value.shape == () else value.tolist()
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
    except Exception:
        pass
    return value


def _sample_scalar(sample: Optional[Dict[str, Any]], key: str, default=None):
    if sample is None or key not in sample:
        return default
    try:
        value = sample[key]
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu().reshape(-1)[0])
        import numpy as np
        array = np.asarray(value)
        return float(array.reshape(-1)[0]) if array.size else default
    except Exception:
        return default


def _save_run_metadata(
    *, method_paths: Dict[str, Any], cfg: Dict[str, Any], manifest: Dict[str, Any],
    train_samples, test_samples, history: Dict[str, Any], case_name: str, method_name,
) -> Path:
    reference = (
        train_samples[0] if len(train_samples) > 0
        else (test_samples[0] if len(test_samples) > 0 else None)
    )
    data_keys = (
        "E", "nu", "thickness", "pressure_min", "pressure_max", "pressure_std",
        "pressure_corr_len", "pressure_n_modes", "lx", "ly", "center_min",
        "center_max", "mean_radius", "r_min", "r_max", "roughness", "n_modes",
    )
    data_config = {key: _sample_scalar(reference, key, None) for key in data_keys}
    data_config["pressure_mean"] = _sample_scalar(
        reference, "pressure_target_mean", _sample_scalar(reference, "pressure_mean", None)
    )
    data_config.update({
        "geometry_type": str(reference.get("geometry_type", "")) if reference else "",
        "load_type": str(reference.get("load_type", "")) if reference else "",
        "material_type": str(reference.get("material_type", "uniform")) if reference else "uniform",
    })
    metadata = {
        "source_case": case_name,
        "method_name": method_name,
        "checkpoint": {
            "best_model_path": str(method_paths["best_model_path"]),
            "final_model_path": str(method_paths["final_model_path"]),
        },
        "model_config": cfg.get("model_config"),
        "training_config": {
            "batch_size": int(cfg["batch_size"]),
            "epochs": None if cfg["epochs"] is None else int(cfg["epochs"]),
            "max_epochs": None if cfg.get("max_epochs", None) is None else int(cfg["max_epochs"]),
            "actual_epochs": len(history.get("epoch", [])),
            "lr": float(cfg["lr"]), "eval_fem_every": int(cfg["eval_fem_every"]),
            "deformation_scale": float(cfg["deformation_scale"]),
            "max_plot_cases": int(cfg["max_plot_cases"]),
            "loss_scale": float(cfg.get("loss_scale", 1.0)),
            "grad_clip_norm": cfg.get("grad_clip_norm", None),
            "plate_theory": str(cfg.get("plate_theory", "Mindlin-Reissner")),
            "include_shear_energy": bool(cfg.get("include_shear_energy", True)),
            "early_stopping": {
                "enabled": bool(cfg["early_stopping"]), "monitor": "test_energy_loss",
                "mode": "min", "min_epochs": int(cfg["early_stopping_min_epochs"]),
                "patience": int(cfg["early_stopping_patience"]),
                "relative_min_delta": float(cfg["early_stopping_min_delta"]),
                "stopped_early": cfg["epochs"] is None or len(history.get("epoch", [])) < int(cfg["epochs"]),
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
        "manifest": _jsonable(manifest),
    }
    output_root = (
        Path(method_paths["no_output_root"])
        if "no_output_root" in method_paths
        else Path(method_paths["no_prediction_root"]).parent
    )
    out_path = output_root / "run_metadata.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_jsonable(metadata), indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


def run_train_hole_um_up(
    *, root: Optional[str | Path] = None, config: Optional[Dict[str, Any]] = None,
    case_name: str = "hole_um_up",
) -> Dict[str, Any]:
    """Shared energy-NO implementation for all MR thickness and mesh cases."""
    case_name = str(case_name)
    thickness_cases = {entry["case_name"] for entry in (
        resolve_thickness_case(suffix)[1]
        for suffix in ("t005", "t0158", "t00158", "t0005")
    )}
    if case_name != "hole_um_up" and case_name not in thickness_cases and not case_name.startswith("hole_um_up_mesh_"):
        raise ValueError(f"Unsupported single-hole uniform-load training case: {case_name}")
    cfg = deepcopy(DEFAULT_TRAIN_CONFIG)
    if case_name == "hole_um_up":
        _, cfg = train_config_for_thickness("base")
    if case_name in thickness_cases:
        cfg["epochs"] = None
    if config is not None:
        cfg.update(config)

    # The mesh-density and mesh-quality studies compare optimization outcomes
    # across discretizations.  They must therefore use an identical, fixed
    # schedule instead of selecting different early-stopped checkpoints.
    if case_name.startswith("hole_um_up_mesh_"):
        cfg["epochs"] = 500
        cfg["max_epochs"] = 500
        cfg["early_stopping"] = False

    paths = build_paths(case_name, root=root, create=True)
    method_name = cfg.get("method_name", None)
    method_paths = build_energy_output_paths(case_name=case_name, root=root, method_name=method_name, create=True)
    torch.manual_seed(42)
    train_samples, test_samples, manifest = load_dataset(case_name=case_name, root=root)
    test_tags = load_test_tags(case_name=case_name, root=root)
    train_tags = load_train_tags(case_name=case_name, root=root)
    padding_value = float(mr_plate_ginot_minimal_configs()["padding_value"])
    collate = lambda batch: plate_collate_fn(batch, padding_value=padding_value)
    train_loader = DataLoader(PlateSampleDataset(train_samples), batch_size=int(cfg["batch_size"]), shuffle=True, collate_fn=collate)
    test_loader = DataLoader(PlateSampleDataset(test_samples), batch_size=int(cfg["batch_size"]), shuffle=False, collate_fn=collate)
    fem_evaluator = FEMMetricEvaluator(case_name=case_name, case_tags=test_tags, root=root, padding_value=padding_value)
    operator_model, history, fem_error_history = train_with_fem_metrics(
        case_name=case_name, train_loader=train_loader, test_loader=test_loader,
        fem_evaluator=fem_evaluator, train_fem_evaluator=None, root=root,
        epochs=None if cfg["epochs"] is None else int(cfg["epochs"]), lr=float(cfg["lr"]),
        max_epochs=None if cfg.get("max_epochs", None) is None else int(cfg["max_epochs"]),
        eval_fem_every=int(cfg["eval_fem_every"]),
        grad_clip_norm=float(cfg["grad_clip_norm"]) if cfg.get("grad_clip_norm") is not None else None,
        loss_scale=float(cfg.get("loss_scale", 1.0)), model_config=cfg.get("model_config"),
        include_shear_energy=bool(cfg.get("include_shear_energy", True)),
        plate_theory=str(cfg.get("plate_theory", "Mindlin-Reissner")),
        method_name=method_name, early_stopping=bool(cfg["early_stopping"]),
        early_stopping_min_epochs=int(cfg["early_stopping_min_epochs"]),
        early_stopping_patience=int(cfg["early_stopping_patience"]),
        early_stopping_min_delta=float(cfg["early_stopping_min_delta"]),
    )
    saved_tags = export_no_predictions(operator_model=operator_model, data_loader=test_loader,
        case_name=case_name, case_tags=test_tags, root=root, padding_value=padding_value, method_name=method_name)
    compare_info = compare_cases(case_name=case_name, case_tags=saved_tags, root=root,
        deformation_scale=float(cfg["deformation_scale"]), max_plot_cases=int(cfg["max_plot_cases"]), method_name=method_name)
    metadata_path = _save_run_metadata(method_paths=method_paths, cfg=cfg, manifest=manifest,
        train_samples=train_samples, test_samples=test_samples, history=history,
        case_name=case_name, method_name=method_name)
    return {
        "case_name": case_name, "method_name": method_name,
        "paths": {key: str(value) for key, value in paths.items()},
        "method_paths": {key: str(value) for key, value in method_paths.items()},
        "manifest": manifest, "history": history, "fem_error_history": fem_error_history,
        "saved_no_tags": saved_tags, "compare_info": compare_info,
        "run_metadata_path": str(metadata_path),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one hole_um_up thickness case with the energy NO."
    )
    parser.add_argument(
        "suffix",
        nargs="?",
        default="base",
        help="Thickness suffix: base, t005, t0158, t00158, or t0005.",
    )
    parser.add_argument("--root", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_train_hole_um_up_mr(args.suffix, root=args.root)
