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
    "grad_clip_norm": 5.0,
    "loss_scale": 1.0e5,

    # Geometry is a variable-size curved annular sector. The branch sees the
    # boundary point cloud, and the material branch sees element-centroid GRF
    # material values [x, y, z, E, nu].
    "model_config": {
        "branch_args": {
            "input_channels": 3,
            "out_c": 64,
            "width": 64,
            "d_hidden": [64, 64],
            "n_point": 128,
            "n_sample": 18,
            "radius": 0.20,
            "num_heads": 4,
            "cross_attn_layers": 2,
            "self_attn_layers": 2,
        },

        "material_branch_args": {
            "input_channels": 5,
            "out_c": 96,
            "width": 64,
            "d_hidden": [64, 64],
            "n_point": 128,
            "n_sample": 36,
            "radius": 0.20,
            "num_heads": 4,
            "cross_attn_layers": 2,
            "self_attn_layers": 2,
        },

        "load_branch_args": {
            "input_channels": 4,
            "out_c": 96,
            "width": 64,
            "d_hidden": [64, 64],
            "n_point": 128,
            "n_sample": 36,
            "radius": 0.20,
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

    "eval_fem_every": 1,
    "deformation_scale": 10.0,
    "max_plot_cases": 5,
    "method_name": None,
}


def _to_jsonable(obj):
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


def _sample_text(sample: Dict[str, Any] | None, key: str, default: str = "") -> str:
    if sample is None or key not in sample:
        return default
    value = sample[key]
    if isinstance(value, str):
        return value
    try:
        import numpy as np

        arr = np.asarray(value)
        if arr.shape == ():
            return str(arr.item())
        if arr.size > 0:
            return str(arr.reshape(-1)[0])
    except Exception:
        pass
    return str(value)


def _get_output_root_from_method_paths(method_paths: Dict[str, Any]) -> Path:
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

        "pressure_value": _sample_scalar(ref_sample, "pressure_value", None),
        "pressure_min": _sample_scalar(ref_sample, "pressure_min", None),
        "pressure_max": _sample_scalar(ref_sample, "pressure_max", None),

        "line_load_mean": _sample_scalar(ref_sample, "line_load_mean", None),
        "line_load_target_mean": _sample_scalar(ref_sample, "line_load_target_mean", None),
        "line_load_min": _sample_scalar(ref_sample, "line_load_min", None),
        "line_load_max": _sample_scalar(ref_sample, "line_load_max", None),
        "line_load_log_std": _sample_scalar(ref_sample, "line_load_log_std", None),
        "line_load_corr_len": _sample_scalar(ref_sample, "line_load_corr_len", None),

        "geometry_type": _sample_text(ref_sample, "geometry_type", ""),
        "load_type": _sample_text(ref_sample, "load_type", ""),
        "material_type": _sample_text(ref_sample, "material_type", "uniform"),

        "r_inner": _sample_scalar(ref_sample, "r_inner", None),
        "r_outer": _sample_scalar(ref_sample, "r_outer", None),
        "theta_start": _sample_scalar(ref_sample, "theta_start", None),
        "theta_end": _sample_scalar(ref_sample, "theta_end", None),
        "theta_span": _sample_scalar(ref_sample, "theta_span", None),
        "theta_start_deg": _sample_scalar(ref_sample, "theta_start_deg", None),
        "theta_end_deg": _sample_scalar(ref_sample, "theta_end_deg", None),
        "theta_span_deg": _sample_scalar(ref_sample, "theta_span_deg", None),
        "mesh_lc": _sample_scalar(ref_sample, "mesh_lc", None),
        "mesh_nr": _sample_scalar(ref_sample, "mesh_nr", None),
        "mesh_nt": _sample_scalar(ref_sample, "mesh_nt", None),

        "E_mean": _sample_scalar(ref_sample, "E_mean", None),
        "E_std": _sample_scalar(ref_sample, "E_std", None),
        "E_min": _sample_scalar(ref_sample, "E_min", None),
        "E_max": _sample_scalar(ref_sample, "E_max", None),
        "E_corr_len": _sample_scalar(ref_sample, "E_corr_len", None),
        "E_n_modes": _sample_scalar(ref_sample, "E_n_modes", None),
        "nu_mean": _sample_scalar(ref_sample, "nu_mean", None),
        "nu_std": _sample_scalar(ref_sample, "nu_std", None),
        "nu_min": _sample_scalar(ref_sample, "nu_min", None),
        "nu_max": _sample_scalar(ref_sample, "nu_max", None),
        "nu_corr_len": _sample_scalar(ref_sample, "nu_corr_len", None),
        "nu_n_modes": _sample_scalar(ref_sample, "nu_n_modes", None),
    }

    metadata = {
        "source_case": case_name,
        "method_name": method_name,

        "checkpoint": {
            "best_model_path": str(method_paths["best_model_path"]),
            "final_model_path": str(method_paths["final_model_path"]),
        },

        "model_config": cfg.get("model_config", None),

        "training_config": {
            "batch_size": int(cfg["batch_size"]),
            "epochs": int(cfg["epochs"]),
            "actual_epochs": int(len(history.get("epoch", []))),
            "lr": float(cfg["lr"]),
            "grad_clip_norm": (
                float(cfg["grad_clip_norm"])
                if cfg.get("grad_clip_norm", None) is not None
                else None
            ),
            "loss_scale": float(cfg.get("loss_scale", 1.0)),
            "eval_fem_every": int(cfg["eval_fem_every"]),
            "deformation_scale": float(cfg["deformation_scale"]),
            "max_plot_cases": int(cfg["max_plot_cases"]),
        },

        "data_config": data_config,

        "input_schema": {
            "pc_net": "[B, Ng, 3] = [x_net, y_net, z_net], boundary point cloud",
            "query_net": "[B, Nq, 3] = [x_net, y_net, z_net]",
            "load_pc_net": "[B, Nq, 4] = [x_net, y_net, z_net, q_line_net]; q_line_net is zero away from the loaded edge",
            "material_pc_net": "[B, Nm, 5] = [x_net, y_net, z_net, E_net, nu_net]",
            "scalar_param": "[B, 3] = [E_scalar_net, nu_scalar_net, mean_line_load_net]",
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


def run_train_annular_sector_grfm_grfp(
    *,
    root: Optional[str | Path] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    case_name = "annular_sector_grfm_grfp"
    paths = build_paths(case_name, root=root, create=True)

    cfg = dict(DEFAULT_TRAIN_CONFIG)
    if config is not None:
        cfg.update(config)

    method_name = cfg.get("method_name", None)
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

    from shared.common.data_io import load_train_tags

    train_tags = load_train_tags(
        case_name=case_name,
        root=root,
    )

    padding_value = float(mr_plate_ginot_minimal_configs()["padding_value"])

    train_loader = DataLoader(
        PlateSampleDataset(train_samples),
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        collate_fn=lambda b: plate_collate_fn(b, padding_value=padding_value),
    )

    test_loader = DataLoader(
        PlateSampleDataset(test_samples),
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

    train_fem_evaluator = None

    operator_model, history, fem_error_history = train_with_fem_metrics(
        case_name=case_name,
        train_loader=train_loader,
        test_loader=test_loader,
        fem_evaluator=fem_evaluator,
        train_fem_evaluator=train_fem_evaluator,
        root=root,
        epochs=int(cfg["epochs"]),
        lr=float(cfg["lr"]),
        eval_fem_every=int(cfg["eval_fem_every"]),
        grad_clip_norm=float(cfg["grad_clip_norm"])
        if cfg.get("grad_clip_norm", None) is not None
        else None,
        loss_scale=float(cfg.get("loss_scale", 1.0)),
        model_config=cfg.get("model_config", None),
        method_name=method_name,
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

    print(f"[Metadata] run metadata saved: {metadata_path}")
    print("=" * 80)
    print("[Stage2][annular_sector_grfm_grfp] Finished")
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
    run_train_annular_sector_grfm_grfp()
