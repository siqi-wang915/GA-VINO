"""PINO strong-form and physics+data baselines for ``hole_um_up``."""

from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from NO.common.core import (
    MRPlateEnergyOperator, PlateSampleDataset, build_plate_operator_model,
    device, mr_plate_ginot_minimal_configs, plate_collate_fn,
)
from NO.common.early_stopping import RelativeEarlyStopping
from NO.common.evaluator import export_no_predictions
from NO.common.hole_strong_form import hole_um_up_strong_form_loss
from NO.common.hole_strong_form import build_hole_strong_collocation_cache
from NO.common.node_l2_evaluator import NodeL2FEMMetricEvaluator as FEMMetricEvaluator
from NO.common.trainer import _save_training_time_summary, build_energy_output_paths
from shared.common.data_io import load_dataset, load_npz_field, load_test_tags, load_train_tags
from shared.common.paths import build_paths, fem_pred_npz_path
from shared.compare.compare import compare_cases
from shared.compare.metrics import plot_fem_error_history, plot_loss_curve


BASE_CONFIG: Dict[str, Any] = {
    # Strong-form training retains a higher-order graph for the sampled
    # collocation set (interior points plus unique free-hole boundary nodes).
    # It is still substantially heavier than the discrete energy objective, so its
    # default batch remains smaller. energy_data overrides this to 16.
    "batch_size": 16,
    "epochs": 500,
    "lr": 1.0e-4,  # energy_data default; strong modes override this below.
    "early_stopping": False,
    "early_stopping_min_epochs": 100,
    "early_stopping_patience": 50,
    "early_stopping_min_delta": 1.0e-3,
    "scheduler_patience": 25,
    "scheduler_factor": 0.7,
    "data_weight": 1.0,
    # With NeRF query encoding disabled, the normalized strong residual is
    # initially O(1), so no additional global down-scaling is required.
    "strong_weight": 1.0,
    # The free-boundary terms are evaluated once per unique hole node with
    # centred-difference normals and arc-length weights.  The two rotational
    # equilibrium residuals are kept separate so that their contribution can
    # be balanced independently from transverse-force equilibrium.
    "strong_interior_weight": 1.0,
    "strong_rotation_x_weight": 100.0,
    "strong_rotation_y_weight": 100.0,
    "strong_free_shear_weight": 100.0,
    "strong_free_moment_weight": 1.0e7,
    "energy_weight": 1.0e4,
    "max_interior_points": 512, #每个样本最多随机选择 512 个板域内部节点，用于计算内部平衡方程损失
    # Retained name for configuration compatibility; it now limits unique
    # hole-boundary nodes rather than duplicated edge endpoints.
    "max_hole_edges": 128,
    "eval_fem_every": 1,
    "deformation_scale": 10.0,
    "max_plot_cases": 5,
    "model_config": {
        "branch_args": {
            "input_channels": 3,  # xyz
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
        "trunk_args": {
            "embed_dim": 64,
            "scalar_dim": 64,
            "num_heads": 4,
            "cross_attn_layers": 4,
            "out_std": [8.0e-2, 2.0e-2, 2.0e-2],
        },
    },
}


MODE_DEFAULT_BATCH_SIZE = {
    "strong": 8,
    "strong_data": 8,
    "energy_data": 16,
}


MODE_TRAINING_DEFAULTS = {
    "strong": {},
    "strong_data": {
        "data_weight": 1000.0,
    },
    "energy_data": {
        "epochs": 500,
        "early_stopping": False,
    },
}


# Strong-form objectives converge more slowly and their validation residuals
# are noisier because collocation points are resampled.  Keep their schedule
# separate from energy_data so changing the PINO experiments does not alter
# the variational baseline.
STRONG_MODE_TRAINING_DEFAULTS = {
    "epochs": 500,
    "lr": 5.0e-4,
    "early_stopping": False,
    "early_stopping_min_epochs": 300,
    "early_stopping_patience": 120,
    "early_stopping_min_delta": 1.0e-3,
    "scheduler_patience": 60,
    "scheduler_factor": 0.7,
}


class FEMTargetDataset(Dataset):
    def __init__(self, samples, tags, *, root=None):
        self.samples, self.tags = list(samples), list(tags)
        self.paths = build_paths("hole_um_up", root=root, create=True)
        if len(self.samples) != len(self.tags):
            raise ValueError("samples and FEM case tags must have the same length")

        # FEM targets are immutable throughout training.  Load them once so
        # every epoch reads tensors from memory instead of reopening hundreds
        # of compressed NPZ files in __getitem__.
        self.targets: List[torch.Tensor] = []
        for tag in self.tags:
            target = np.asarray(
                load_npz_field(fem_pred_npz_path(self.paths, tag), "fem_local"),
                dtype=np.float32,
            )
            self.targets.append(torch.from_numpy(target.copy()))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = dict(self.samples[idx])
        sample["target_fem_local"] = self.targets[idx]
        return sample


class StrongCollocationDataset(Dataset):
    """Dataset that lazily caches fixed mesh topology on CPU per sample.

    The cache holds every candidate interior node and free-hole boundary node,
    while
    ``hole_um_up_strong_form_loss`` still randomly selects a fresh subset for
    every batch.  With the current single-process DataLoader this cache lives
    for the complete training run and avoids repeated face-topology scans.
    """

    def __init__(self, samples):
        self.samples = list(samples)
        self._collocation_caches: List[Optional[Dict[str, Any]]] = [None] * len(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cache = self._collocation_caches[idx]
        if cache is None:
            cache = build_hole_strong_collocation_cache(self.samples[idx])
            self._collocation_caches[idx] = cache
        sample = dict(self.samples[idx])
        sample["strong_collocation_cache"] = cache
        return sample


def _target_collate(batch, *, padding_value):
    out = plate_collate_fn(batch, padding_value=padding_value)
    out["target_fem_local"] = pad_sequence(
        [s["target_fem_local"] for s in batch], batch_first=True, padding_value=padding_value
    )
    out["strong_collocation_cache"] = [s.get("strong_collocation_cache") for s in batch]
    return out


def _strong_collate(batch, *, padding_value):
    out = plate_collate_fn(batch, padding_value=padding_value)
    out["strong_collocation_cache"] = [s.get("strong_collocation_cache") for s in batch]
    return out


def _target_std(dataset: FEMTargetDataset) -> torch.Tensor:
    values = torch.cat(dataset.targets, dim=0)
    return torch.std(values, dim=0, unbiased=False).clamp_min(1.0e-8)


def _data_loss(pred, batch, padding_value, target_std):
    valid = batch["query"].to(device)[:, :, 0] != float(padding_value)
    # strong_data receives a sampled collocation batch. Its FEM MSE is
    # evaluated at the random interior collocation points only; the sampled
    # hole-boundary nodes are used exclusively by the natural-boundary loss.
    if "strong_interior_mask" in batch:
        valid = valid & batch["strong_interior_mask"].to(device)
    target = batch["target_fem_local"].to(device=device, dtype=pred.dtype)
    return torch.mean(((pred[valid] - target[valid]) / target_std.to(device=pred.device, dtype=pred.dtype)) ** 2)


def _metrics_history() -> Dict[str, List[float]]:
    keys = ["epoch", "n_metric_cases", "l2_error_mean", "l2_error_std", "theta_l2_error_mean", "theta_l2_error_std",
            "test_l2_error_mean", "test_l2_error_std", "test_theta_l2_error_mean", "test_theta_l2_error_std"]
    return {key: [] for key in keys}


def _append_metrics(history, stats, epoch):
    history["epoch"].append(float(epoch))
    history["n_metric_cases"].append(float(stats["n_metric_cases"]))
    for key in ("l2_error", "theta_l2_error"):
        for suffix in ("mean", "std"):
            value = float(stats[f"{key}_{suffix}"])
            history[f"{key}_{suffix}"].append(value)
            history[f"test_{key}_{suffix}"].append(value)


def _save_history(path, history):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: np.asarray(v, dtype=np.float64) for k, v in history.items()})


def _objective(operator_model, batch, *, mode, cfg, padding_value, target_std=None):
    parts: Dict[str, torch.Tensor] = {}
    if mode.startswith("strong"):
        strong, strong_parts, pred, collocation_batch = hole_um_up_strong_form_loss(
            operator_model, batch, padding_value=padding_value,
            max_interior_points=int(cfg["max_interior_points"]), max_hole_edges=int(cfg["max_hole_edges"]),
            interior_weight=float(cfg["strong_interior_weight"]),
            rotation_x_weight=float(cfg["strong_rotation_x_weight"]),
            rotation_y_weight=float(cfg["strong_rotation_y_weight"]),
            free_shear_weight=float(cfg["strong_free_shear_weight"]),
            free_moment_weight=float(cfg["strong_free_moment_weight"]),
        )
        parts.update(strong_parts)
        parts["strong"] = strong
        total = float(cfg["strong_weight"]) * strong
    else:
        energy, pred, energy_parts = operator_model(batch)
        parts["energy"] = energy
        total = float(cfg["energy_weight"]) * energy
        for key, value in energy_parts.items():
            parts[key] = pred.new_tensor(value)

    if mode.endswith("_data"):
        if target_std is None:
            raise RuntimeError("A physics+data objective requires FEM target normalization.")
        data_batch = collocation_batch if mode.startswith("strong") else batch
        data = _data_loss(pred, data_batch, padding_value, target_std)
        parts["data"] = data
        total = total + float(cfg["data_weight"]) * data
    parts["total"] = total
    return total, parts


def _evaluate(operator_model, loader, *, mode, cfg, padding_value, target_std):
    operator_model.eval()
    totals: Dict[str, float] = {}
    # Strong-form residuals differentiate predictions with respect to query
    # coordinates and therefore require autograd during evaluation.  The
    # energy_data objective uses only P1 element gradients and can avoid
    # building a parameter-gradient graph on the test set.
    grad_context = torch.enable_grad if mode.startswith("strong") else torch.no_grad
    with grad_context():
        for batch in loader:
            _, parts = _objective(operator_model, batch, mode=mode, cfg=cfg,
                                  padding_value=padding_value, target_std=target_std)
            for key, value in parts.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
    return {key: value / max(len(loader), 1) for key, value in totals.items()}


def run_train_hole_um_up_physics(
    *,
    mode: str,
    root: Optional[str | Path] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if mode not in {"strong", "strong_data", "energy_data"}:
        raise ValueError(f"Unsupported physics baseline mode: {mode}")
    cfg = deepcopy(BASE_CONFIG)
    cfg["batch_size"] = MODE_DEFAULT_BATCH_SIZE[mode]
    if mode.startswith("strong"):
        cfg.update(STRONG_MODE_TRAINING_DEFAULTS)
    cfg.update(MODE_TRAINING_DEFAULTS[mode])
    if config:
        for key, value in config.items():
            if key == "model_config" and isinstance(value, dict):
                for name, args in value.items():
                    cfg["model_config"].setdefault(name, {}).update(args)
            else:
                cfg[key] = value
    # Coordinate second derivatives enter the strong-form equilibrium
    # residual. Disable the high-frequency NeRF query encoding only for the
    # strong objectives; the energy-data baseline keeps its default encoding.
    if mode.startswith("strong"):
        cfg["model_config"]["trunk_args"]["use_nerf_query"] = False
    method_name = {
        "strong": "pino_strong",
        "strong_data": "pino_strong_data",
        "energy_data": "energy_data",
    }[mode]
    paths = build_paths("hole_um_up", root=root, create=True)
    method_paths = build_energy_output_paths(
        case_name="hole_um_up",
        root=root,
        method_name=method_name,
        create=True,
    )
    torch.manual_seed(42)
    train_samples, test_samples, manifest = load_dataset(case_name="hole_um_up", root=root)
    train_tags = load_train_tags(case_name="hole_um_up", root=root)
    test_tags = load_test_tags(case_name="hole_um_up", root=root)
    padding_value = float(mr_plate_ginot_minimal_configs()["padding_value"])
    has_data = mode.endswith("_data")
    if has_data:
        target_train_ds = FEMTargetDataset(train_samples, train_tags, root=root)
        target_test_ds = FEMTargetDataset(test_samples, test_tags, root=root)
        if mode.startswith("strong"):
            # Keep FEM targets and cached topology together for strong_data.
            train_ds = StrongCollocationDataset(target_train_ds)
            test_ds = StrongCollocationDataset(target_test_ds)
        else:
            train_ds, test_ds = target_train_ds, target_test_ds
        target_std = _target_std(target_train_ds)
        collate = lambda b: _target_collate(b, padding_value=padding_value)
    else:
        if mode.startswith("strong"):
            train_ds = StrongCollocationDataset(train_samples)
            test_ds = StrongCollocationDataset(test_samples)
            collate = lambda b: _strong_collate(b, padding_value=padding_value)
        else:
            train_ds = PlateSampleDataset(train_samples)
            test_ds = PlateSampleDataset(test_samples)
            collate = lambda b: plate_collate_fn(b, padding_value=padding_value)
        target_std = None
    train_loader = DataLoader(
        train_ds, batch_size=int(cfg["batch_size"]), shuffle=True, collate_fn=collate
    )
    test_loader = DataLoader(
        test_ds, batch_size=int(cfg["batch_size"]), shuffle=False, collate_fn=collate
    )
    evaluator = FEMMetricEvaluator(
        case_name="hole_um_up",
        case_tags=test_tags,
        root=root,
        padding_value=padding_value,
    )
    # Reset before model/optimizer creation so the reported peak includes
    # parameters, optimizer state, activations, and validation performed in
    # this training run. CUDA APIs cannot report CPU RAM, so that case is
    # represented explicitly as NaN rather than a misleading 0.0 MB.
    measure_cuda_memory = device.type == "cuda"
    if measure_cuda_memory:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    base = build_plate_operator_model(
        padding_value=padding_value,
        model_config=cfg["model_config"],
    ).to(device)
    boundary_transform = (
        "square_clamped_distance" if mode.startswith("strong") else "binary_factor"
    )
    model = MRPlateEnergyOperator(
        base,
        padding_value=padding_value,
        boundary_transform=boundary_transform,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(cfg["scheduler_factor"]),
        patience=int(cfg["scheduler_patience"]),
    )
    stopper = RelativeEarlyStopping(
        enabled=bool(cfg["early_stopping"]),
        min_epochs=int(cfg["early_stopping_min_epochs"]),
        patience=int(cfg["early_stopping_patience"]),
        min_delta=float(cfg["early_stopping_min_delta"]),
    )
    history_keys = (
        "epoch", "train_loss", "test_loss", "lr", "train_epoch_time_sec",
        "epoch_total_time_sec", "early_stopping_bad_epochs", "train_strong",
        "test_strong", "train_data", "test_data", "train_energy", "test_energy",
        "train_strong_interior", "test_strong_interior",
        "train_strong_rotation_x", "test_strong_rotation_x",
        "train_strong_rotation_y", "test_strong_rotation_y",
        "train_strong_free_shear",
        "test_strong_free_shear", "train_strong_free_moment",
        "test_strong_free_moment",
        "train_strong_interior_weighted", "test_strong_interior_weighted",
        "train_strong_rotation_x_weighted", "test_strong_rotation_x_weighted",
        "train_strong_rotation_y_weighted", "test_strong_rotation_y_weighted",
        "train_strong_free_shear_weighted", "test_strong_free_shear_weighted",
        "train_strong_free_moment_weighted", "test_strong_free_moment_weighted",
    )
    history: Dict[str, List[float]] = {key: [] for key in history_keys}
    fem_history = _metrics_history()
    best_loss, best_epoch = float("inf"), -1
    total_t0 = time.perf_counter()
    for epoch in range(1, int(cfg["epochs"]) + 1):
        epoch_t0, train_t0 = time.perf_counter(), time.perf_counter()
        model.train()
        train_parts: Dict[str, float] = {}
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss, parts = _objective(
                model,
                batch,
                mode=mode,
                cfg=cfg,
                padding_value=padding_value,
                target_std=target_std,
            )
            loss.backward()
            optimizer.step()
            for key, value in parts.items():
                train_parts[key] = train_parts.get(key, 0.0) + float(
                    value.detach().cpu()
                )
        train_time = time.perf_counter() - train_t0
        n_batches = max(len(train_loader), 1)
        train_parts = {k: v / n_batches for k, v in train_parts.items()}
        test_parts = _evaluate(
            model,
            test_loader,
            mode=mode,
            cfg=cfg,
            padding_value=padding_value,
            target_std=target_std,
        )
        test_loss = test_parts["total"]
        scheduler.step(test_loss)
        if test_loss < best_loss:
            best_loss, best_epoch = test_loss, epoch
            torch.save(model.state_dict(), method_paths["best_model_path"])
        should_stop = stopper.step(test_loss, epoch)
        history["epoch"].append(float(epoch))
        history["train_loss"].append(train_parts["total"])
        history["test_loss"].append(test_loss)
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))
        history["early_stopping_bad_epochs"].append(float(stopper.bad_epochs))
        for key in (
            "strong", "data", "energy", "strong_interior",
            "strong_rotation_x", "strong_rotation_y",
            "strong_free_shear", "strong_free_moment",
            "strong_interior_weighted", "strong_rotation_x_weighted",
            "strong_rotation_y_weighted", "strong_free_shear_weighted",
            "strong_free_moment_weighted",
        ):
            history[f"train_{key}"].append(train_parts.get(key, float("nan")))
            history[f"test_{key}"].append(test_parts.get(key, float("nan")))
        do_eval = (
            epoch == 1
            or epoch == int(cfg["epochs"])
            or should_stop
            or (
                int(cfg["eval_fem_every"]) > 0
                and epoch % int(cfg["eval_fem_every"]) == 0
            )
        )
        fem_msg = ""
        if do_eval:
            stats = evaluator.evaluate_model(operator_model=model, data_loader=test_loader)
            _append_metrics(fem_history, stats, epoch)
            fem_msg = f" | Test_L2={stats['l2_error_mean']:.4e} | Test_Theta={stats['theta_l2_error_mean']:.4e}"
        epoch_time = time.perf_counter() - epoch_t0
        history["train_epoch_time_sec"].append(train_time)
        history["epoch_total_time_sec"].append(epoch_time)
        print(
            f"[PINO:{mode}] Epoch {epoch:04d}/{int(cfg['epochs']):04d} | "
            f"train={train_parts['total']:.6e} | test={test_loss:.6e}"
            f"{fem_msg} | train_time={train_time:.2f}s"
        )
        if should_stop:
            break
    if measure_cuda_memory:
        torch.cuda.synchronize(device)
        max_memory_mb = float(torch.cuda.max_memory_allocated(device) / 1024**2)
    else:
        max_memory_mb = float("nan")
    total_time = time.perf_counter() - total_t0
    torch.save(model.state_dict(), method_paths["final_model_path"])
    _save_history(method_paths["training_history_path"], history)
    plot_loss_curve(history, method_paths["loss_curve_path"])
    _save_history(method_paths["fem_error_history_path"], fem_history)
    plot_fem_error_history(
        error_history=fem_history,
        out_paths={
            "l2": method_paths["l2_error_epoch_plot_path"],
            "theta": method_paths["theta_error_epoch_plot_path"],
        },
    )
    _save_training_time_summary(
        paths=method_paths,
        history=history,
        total_time_sec=total_time,
        epochs=len(history["epoch"]),
        method_name=method_name,
        max_memory_mb=max_memory_mb,
        n_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
    if method_paths["best_model_path"].exists():
        model.load_state_dict(
            torch.load(method_paths["best_model_path"], map_location=device)
        )
    saved = export_no_predictions(
        operator_model=model,
        data_loader=test_loader,
        case_name="hole_um_up",
        case_tags=test_tags,
        root=root,
        padding_value=padding_value,
        method_name=method_name,
    )
    compare_info = compare_cases(
        case_name="hole_um_up",
        case_tags=saved,
        root=root,
        deformation_scale=float(cfg["deformation_scale"]),
        max_plot_cases=int(cfg["max_plot_cases"]),
        method_name=method_name,
    )
    metadata = {
        "source_case": "hole_um_up",
        "method_name": method_name,
        "objective": {
            "strong_form": mode.startswith("strong"),
            "energy": mode.startswith("energy"),
            "supervised_data": has_data,
            "data_weight": float(cfg["data_weight"]),
            "strong_weight": float(cfg["strong_weight"]),
            "strong_interior_weight": float(cfg["strong_interior_weight"]),
            "strong_rotation_x_weight": float(cfg["strong_rotation_x_weight"]),
            "strong_rotation_y_weight": float(cfg["strong_rotation_y_weight"]),
            "strong_free_shear_weight": float(cfg["strong_free_shear_weight"]),
            "strong_free_moment_weight": float(cfg["strong_free_moment_weight"]),
            "essential_boundary_transform": boundary_transform,
            "energy_weight": float(cfg["energy_weight"]),
        },
        "model_config": cfg["model_config"],
        "training_config": {
            key: cfg[key]
            for key in (
                "batch_size", "epochs", "lr", "max_interior_points",
                "max_hole_edges", "scheduler_patience", "scheduler_factor",
                "early_stopping", "early_stopping_min_epochs",
                "early_stopping_patience", "early_stopping_min_delta",
            )
        },
        "manifest": manifest,
    }
    metadata_path = Path(method_paths["no_output_root"]) / "run_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    return {
        "case_name": "hole_um_up",
        "method_name": method_name,
        "method_paths": {k: str(v) for k, v in method_paths.items()},
        "history": history,
        "fem_error_history": fem_history,
        "saved_no_tags": saved,
        "compare_info": compare_info,
        "run_metadata_path": str(metadata_path),
    }


def run_train_hole_um_up_pino_strong(**kwargs):
    return run_train_hole_um_up_physics(mode="strong", **kwargs)


def run_train_hole_um_up_pino_strong_data(**kwargs):
    return run_train_hole_um_up_physics(mode="strong_data", **kwargs)


def run_train_hole_um_up_energy_data(**kwargs):
    return run_train_hole_um_up_physics(mode="energy_data", **kwargs)
