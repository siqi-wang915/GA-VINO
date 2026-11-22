from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import json
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[2]

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from NO.common.core import (
    MRPlateEnergyOperator,
    build_plate_operator_model,
    device,
    mr_plate_ginot_minimal_configs,
)
from NO.common.evaluator import export_no_predictions
from NO.common.node_l2_evaluator import NodeL2FEMMetricEvaluator as FEMMetricEvaluator
from NO.common.trainer import build_energy_output_paths
from NO.train.train_hole_um_up_mse import (
    PlateFEMTargetDataset,
    _append_metric_stats,
    _compute_target_stats,
    _empty_fem_error_history,
    _evaluate_mse,
    _normalized_target_mse,
    _plot_mse_loss_curve,
    _save_npz_history,
    _save_training_time_summary,
    plate_fem_target_collate_fn,
)
from shared.common.data_io import load_dataset, load_test_tags, load_train_tags
from shared.common.paths import build_paths
from shared.compare.compare import compare_cases
from shared.compare.metrics import plot_fem_error_history


DEFAULT_TRAIN_CONFIG: Dict[str, Any] = {
    "batch_size": 16,
    "epochs": 500,
    "lr": 5.0e-4,

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
    "method_name": "mse",
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
    if isinstance(obj, np.ndarray):
        if obj.shape == ():
            return obj.item()
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def _sample_scalar(sample: Dict[str, Any] | None, key: str, default=None):
    if sample is None or key not in sample:
        return default
    try:
        value = sample[key]
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu().reshape(-1)[0].item())
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
        arr = np.asarray(value)
        if arr.shape == ():
            return str(arr.item())
        if arr.size > 0:
            return str(arr.reshape(-1)[0])
    except Exception:
        pass
    return str(value)


def _save_run_metadata(
    *,
    method_paths: Dict[str, Any],
    cfg: Dict[str, Any],
    manifest: Dict[str, Any],
    train_samples,
    test_samples,
    case_name: str,
    method_name,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
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
            "lr": float(cfg["lr"]),
            "eval_fem_every": int(cfg["eval_fem_every"]),
            "deformation_scale": float(cfg["deformation_scale"]),
            "max_plot_cases": int(cfg["max_plot_cases"]),
            "objective": "supervised_fem_normalized_mse",
            "target_normalization": {
                "type": "channel_std_residual",
                "target_mean": _to_jsonable(target_mean),
                "target_std": _to_jsonable(target_std),
                "note": "Model outputs remain physical [w, theta_x, theta_y]; only residuals are divided by target_std in the loss.",
            },
        },
        "data_config": data_config,
        "input_schema": {
            "pc_net": "[B, Ng, 3] = [x_net, y_net, z_net], boundary point cloud",
            "query_net": "[B, Nq, 3] = [x_net, y_net, z_net]",
            "load_pc_net": "[B, Nq, 4] = [x_net, y_net, z_net, q_line_net]; q_line_net is zero away from the loaded edge",
            "material_pc_net": "[B, Nm, 5] = [x_net, y_net, z_net, E_net, nu_net]",
            "scalar_param": "[B, 3] = [E_scalar_net, nu_scalar_net, mean_line_load_net]",
            "target": "[B, Nq, 3] = FEM [w, theta_x, theta_y]",
            "output": "[B, Nq, 3] = [w, theta_x, theta_y]",
        },
        "manifest": _to_jsonable(manifest),
    }

    out_path = Path(method_paths["no_output_root"]) / "run_metadata.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(metadata), f, indent=2, ensure_ascii=False)
    return out_path


def run_train_annular_sector_grfm_grfp_mse(
    *,
    root: Optional[str | Path] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    case_name = "annular_sector_grfm_grfp"
    paths = build_paths(case_name, root=root, create=True)

    cfg = dict(DEFAULT_TRAIN_CONFIG)
    if config is not None:
        cfg.update(config)

    method_name = str(cfg.get("method_name", "mse"))
    method_paths = build_energy_output_paths(
        case_name=case_name,
        root=root,
        method_name=method_name,
        create=True,
    )

    torch.manual_seed(42)

    train_samples, test_samples, manifest = load_dataset(case_name=case_name, root=root)
    train_tags = load_train_tags(case_name=case_name, root=root)
    test_tags = load_test_tags(case_name=case_name, root=root)

    padding_value = float(mr_plate_ginot_minimal_configs()["padding_value"])

    train_ds = PlateFEMTargetDataset(train_samples, train_tags, case_name=case_name, root=root)
    test_ds = PlateFEMTargetDataset(test_samples, test_tags, case_name=case_name, root=root)
    target_mean, target_std = _compute_target_stats(train_ds)
    print(
        "[Target normalization] "
        f"mean=[{target_mean[0]:.6e}, {target_mean[1]:.6e}, {target_mean[2]:.6e}] | "
        f"std=[{target_std[0]:.6e}, {target_std[1]:.6e}, {target_std[2]:.6e}]"
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        collate_fn=lambda b: plate_fem_target_collate_fn(b, padding_value=padding_value),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        collate_fn=lambda b: plate_fem_target_collate_fn(b, padding_value=padding_value),
    )

    fem_evaluator = FEMMetricEvaluator(
        case_name=case_name,
        case_tags=test_tags,
        root=root,
        padding_value=padding_value,
    )

    base_model = build_plate_operator_model(
        padding_value=padding_value,
        model_config=cfg.get("model_config", None),
    ).to(device)
    operator_model = MRPlateEnergyOperator(
        model=base_model,
        padding_value=padding_value,
    ).to(device)

    optimizer = torch.optim.Adam(operator_model.parameters(), lr=float(cfg["lr"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.7,
        patience=25,
    )

    history: Dict[str, List[float]] = {
        "epoch": [],
        "train_loss": [],
        "test_loss": [],
        "lr": [],
        "train_epoch_time_sec": [],
        "epoch_total_time_sec": [],
    }
    fem_error_history = _empty_fem_error_history()

    best_test_loss = float("inf")
    best_epoch = -1
    epochs = int(cfg["epochs"])
    eval_fem_every = int(cfg["eval_fem_every"])

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    training_t0 = time.perf_counter()

    for epoch in range(1, epochs + 1):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        epoch_t0 = time.perf_counter()

        operator_model.train()
        train_t0 = time.perf_counter()

        train_total = 0.0
        n_batch = 0

        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)

            pred = operator_model.predict(batch)
            target = batch["target_fem_local"].to(device=device, dtype=pred.dtype)
            query = batch["query"].to(device)
            valid = query[:, :, 0] != padding_value

            loss = _normalized_target_mse(pred, target, valid, target_std)
            loss.backward()
            optimizer.step()

            train_total += float(loss.item())
            n_batch += 1

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        train_epoch_time_sec = float(time.perf_counter() - train_t0)

        train_loss = train_total / max(n_batch, 1)
        test_loss = _evaluate_mse(
            operator_model=operator_model,
            data_loader=test_loader,
            padding_value=padding_value,
            target_std=target_std,
        )
        scheduler.step(test_loss)

        if test_loss < best_test_loss:
            best_test_loss = float(test_loss)
            best_epoch = epoch
            torch.save(operator_model.state_dict(), method_paths["best_model_path"])

        history["epoch"].append(float(epoch))
        history["train_loss"].append(float(train_loss))
        history["test_loss"].append(float(test_loss))
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))

        do_fem_eval = (
            epoch == 1
            or epoch == epochs
            or (eval_fem_every > 0 and epoch % eval_fem_every == 0)
        )

        fem_msg = ""
        if do_fem_eval:
            test_metric_stats = fem_evaluator.evaluate_model(
                operator_model=operator_model,
                data_loader=test_loader,
            )
            fem_error_history["epoch"].append(float(epoch))
            fem_error_history["n_metric_cases"].append(float(test_metric_stats["n_metric_cases"]))
            _append_metric_stats(
                history=fem_error_history,
                stats=test_metric_stats,
                prefix="test",
                also_legacy=True,
            )
            fem_msg = (
                f" | Test_L2={test_metric_stats['l2_error_mean']:.4e}"
                f" | Test_Theta={test_metric_stats['theta_l2_error_mean']:.4e}"
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        epoch_total_time_sec = float(time.perf_counter() - epoch_t0)

        history["train_epoch_time_sec"].append(train_epoch_time_sec)
        history["epoch_total_time_sec"].append(epoch_total_time_sec)

        print(
            f"Epoch {epoch:04d}/{epochs:04d} | "
            f"lr={optimizer.param_groups[0]['lr']:.3e} | "
            f"train_mse={train_loss:.6e} | "
            f"test_mse={test_loss:.6e}"
            f"{fem_msg}"
            f" | train_time={train_epoch_time_sec:.2f}s"
            f" | epoch_time={epoch_total_time_sec:.2f}s"
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    training_total_time_sec = float(time.perf_counter() - training_t0)

    torch.save(operator_model.state_dict(), method_paths["final_model_path"])
    _save_npz_history(method_paths["training_history_path"], history)
    _plot_mse_loss_curve(history, method_paths["loss_curve_path"])

    if fem_error_history["epoch"]:
        _save_npz_history(method_paths["fem_error_history_path"], fem_error_history)
        plot_fem_error_history(
            error_history=fem_error_history,
            out_paths={
                "l2": method_paths["l2_error_epoch_plot_path"],
                "theta": method_paths["theta_error_epoch_plot_path"],
            },
        )

    n_params = sum(p.numel() for p in operator_model.parameters() if p.requires_grad)
    _save_training_time_summary(
        paths=method_paths,
        history=history,
        total_time_sec=training_total_time_sec,
        epochs=epochs,
        method_name=method_name,
        n_params=n_params,
    )

    if method_paths["best_model_path"].exists():
        operator_model.load_state_dict(torch.load(method_paths["best_model_path"], map_location=device))
        print(f"[Checkpoint] Loaded best MSE model from epoch {best_epoch}.")

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
        case_name=case_name,
        method_name=method_name,
        target_mean=target_mean,
        target_std=target_std,
    )

    print(f"[Checkpoint] final model saved: {method_paths['final_model_path']}")
    print(f"[Checkpoint] best model saved : {method_paths['best_model_path']}")
    print(f"[Checkpoint] best_epoch={best_epoch}, best_test_mse={best_test_loss:.6e}")
    print(f"[History] training history saved: {method_paths['training_history_path']}")
    print(f"[Metadata] run metadata saved: {metadata_path}")
    print("=" * 80)
    print("[Stage2][annular_sector_grfm_grfp_mse] Finished")
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
    run_train_annular_sector_grfm_grfp_mse()
