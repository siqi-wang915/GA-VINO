from __future__ import annotations

"""Compact repeated-seed study for the reviewer-requested robustness check.

The experiment covers the two representative settings used in the paper:

* Single-hole plate, uniform material and uniform load (``hole_um_up``).
* Double-hole plate, uniform material and random load (``holes2_um_grfp``).

For each case, GA-VINO (energy training) and GA-NO (supervised MSE training)
are trained with paired random seeds.  Predictions, per-sample metric tables,
figures, loss curves, and persistent checkpoints are deliberately not saved.
Only compact run-level and across-seed CSV summaries are retained.
"""

import argparse
import csv
from copy import deepcopy
from datetime import datetime
import gc
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# User-facing names are kept close to the terminology in the manuscript.
CASE_ALIASES = {
    "single-hole-um-ul": "hole_um_up",
    "double-hole-um-rl": "holes2_um_grfp",
    "hole_um_up": "hole_um_up",
    "holes2_um_grfp": "holes2_um_grfp",
}
METHOD_ALIASES = {
    "ga-vino": "ga_vino",
    "ga_vino": "ga_vino",
    "gavino": "ga_vino",
    "ga-no": "ga_no",
    "ga_no": "ga_no",
    "gano": "ga_no",
}

ERROR_METRICS = (
    "w_l2",
    "theta_l2",
    "w_h1_semi",
    "energy",
    "curvature",
    "moment_l2",
    "shear_l2",
)

RUN_COLUMNS = [
    "status",
    "case_name",
    "case_label",
    "method",
    "seed",
    "init_seed",
    "shuffle_seed",
    "fps_seed",
    "geometry_fps_mode",
    "n_train",
    "n_test",
    "batch_size",
    "learning_rate",
    "actual_epochs",
    "best_epoch",
    "stopped_early",
    "best_selection_loss",
    "final_train_loss",
    "training_wall_time_sec",
    "optimization_time_total_sec",
    "optimization_time_mean_epoch_sec",
    "optimization_time_std_epoch_sec",
    "epoch_total_time_mean_sec",
    "epoch_total_time_std_sec",
    "final_evaluation_time_sec",
    "max_gpu_memory_mb",
    "n_trainable_parameters",
    "completed_at",
] + [
    column
    for metric in ERROR_METRICS
    for column in (f"{metric}_mean_pct", f"{metric}_test_std_pct")
] + ["error_message"]

SUMMARY_BASE_COLUMNS = [
    "case_name",
    "case_label",
    "method",
    "n_runs",
    "seeds",
    "actual_epochs_mean",
    "actual_epochs_seed_std",
    "training_wall_time_mean_sec",
    "training_wall_time_seed_std_sec",
    "optimization_time_mean_sec",
    "optimization_time_seed_std_sec",
    "epoch_total_time_mean_sec",
    "epoch_total_time_seed_std_sec",
    "max_gpu_memory_mean_mb",
    "max_gpu_memory_seed_std_mb",
]

SUMMARY_COLUMNS = SUMMARY_BASE_COLUMNS + [
    column
    for metric in ERROR_METRICS
    for column in (f"{metric}_mean_pct", f"{metric}_seed_std_pct")
]

PAIRED_COLUMNS = ["case_name", "case_label", "seed"] + [
    column
    for metric in ERROR_METRICS
    for column in (
        f"{metric}_ga_vino_pct",
        f"{metric}_ga_no_pct",
        f"{metric}_reduction_pct_point",
        f"{metric}_relative_reduction_pct",
        f"{metric}_ga_vino_wins",
    )
]

PAIRED_SUMMARY_COLUMNS = [
    "case_name",
    "case_label",
    "n_pairs",
    "seeds",
] + [
    column
    for metric in ERROR_METRICS
    for column in (
        f"{metric}_reduction_mean_pct_point",
        f"{metric}_reduction_seed_std_pct_point",
        f"{metric}_relative_reduction_mean_pct",
        f"{metric}_relative_reduction_seed_std_pct",
        f"{metric}_ga_vino_win_count",
    )
]


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        return value.item() if value.numel() == 1 else value.tolist()
    if isinstance(value, np.ndarray):
        return value.item() if value.shape == () else value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _atomic_write_csv(path: Path, rows: Sequence[Dict[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})
    os.replace(tmp_path, path)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def _float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _mean_std(values: Iterable[Any], *, sample: bool = True) -> tuple[float, float]:
    arr = np.asarray([_float_or_nan(v) for v in values], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(np.mean(arr)), float(np.std(arr, ddof=1 if sample else 0))


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _set_all_seeds(seed: int, *, strict_determinism: bool) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    if strict_determinism:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)


def _swap_first_row(value: Any, selected: int) -> Any:
    if selected == 0:
        return value.clone() if isinstance(value, torch.Tensor) else np.array(value, copy=True)
    if isinstance(value, torch.Tensor):
        out = value.clone()
        first = out[0].clone()
        out[0] = out[selected]
        out[selected] = first
        return out
    out = np.array(value, copy=True)
    first = out[0].copy()
    out[0] = out[selected]
    out[selected] = first
    return out


def _with_seeded_geometry_fps_start(sample: Dict[str, Any], fps_seed: int) -> Dict[str, Any]:
    """Move a seeded random geometry point to index zero.

    The existing geometry encoder caches FPS indices by ``sample_id`` and
    therefore starts deterministic FPS from index zero.  Swapping one uniformly
    selected valid boundary point into index zero gives each run a random but
    reproducible FPS initialization while preserving the cache and avoiding
    per-epoch geometry resampling.
    """
    out = dict(sample)
    source = sample.get("pc_net", sample.get("pc", None))
    if source is None:
        return out
    n_points = int(source.shape[0])
    if n_points <= 1:
        return out

    sample_id = int(sample.get("sample_id", 0))
    mixed_seed = (int(fps_seed) + 1_000_003 * sample_id) % (2**63 - 1)
    selected = int(np.random.default_rng(mixed_seed).integers(0, n_points))
    for key in ("pc", "pc_net"):
        value = sample.get(key, None)
        if value is not None and int(value.shape[0]) == n_points:
            out[key] = _swap_first_row(value, selected)
    return out


class CompactPlateDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Dict[str, Any]],
        *,
        geometry_fps_seed: Optional[int],
        targets: Optional[Sequence[torch.Tensor]] = None,
    ) -> None:
        if geometry_fps_seed is None:
            self.samples = list(samples)
        else:
            self.samples = [
                _with_seeded_geometry_fps_start(sample, geometry_fps_seed)
                for sample in samples
            ]
        self.targets = None if targets is None else list(targets)
        if self.targets is not None and len(self.targets) != len(self.samples):
            raise ValueError("targets and samples must have equal lengths")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        if self.targets is None:
            return sample
        out = dict(sample)
        out["target_fem_local"] = self.targets[index]
        return out


def _load_supervised_targets(*, paths: Dict[str, Any], tags: Sequence[str]) -> List[torch.Tensor]:
    from shared.common.data_io import load_npz_field
    from shared.common.paths import fem_pred_npz_path

    targets: List[torch.Tensor] = []
    for tag in tags:
        target = np.asarray(
            load_npz_field(fem_pred_npz_path(paths, tag), "fem_local"),
            dtype=np.float32,
        )
        targets.append(torch.from_numpy(target.copy()))
    return targets


def _target_std(targets: Sequence[torch.Tensor]) -> torch.Tensor:
    if not targets:
        raise ValueError("Cannot compute target statistics from an empty dataset")
    joined = torch.cat(list(targets), dim=0).float()
    return torch.clamp(torch.std(joined, dim=0, unbiased=False), min=1.0e-8)


def _mse_collate(batch: List[Dict[str, Any]], *, padding_value: float) -> Dict[str, Any]:
    from NO.common.core import plate_collate_fn

    out = plate_collate_fn(batch, padding_value=padding_value)
    out["target_fem_local"] = pad_sequence(
        [sample["target_fem_local"] for sample in batch],
        batch_first=True,
        padding_value=padding_value,
    )
    return out


def _normalized_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    target_std: torch.Tensor,
) -> torch.Tensor:
    scale = target_std.to(device=pred.device, dtype=pred.dtype).view(1, 3)
    return torch.mean(((pred[valid] - target[valid]) / scale) ** 2)


@torch.no_grad()
def _evaluate_mse(
    *,
    operator_model,
    data_loader: DataLoader,
    padding_value: float,
    target_std: torch.Tensor,
    device: torch.device,
) -> float:
    operator_model.eval()
    total = 0.0
    n_batches = 0
    for batch in data_loader:
        pred = operator_model.predict(batch)
        target = batch["target_fem_local"].to(device=device, dtype=pred.dtype)
        query = batch["query"].to(device)
        valid = query[:, :, 0] != padding_value
        total += float(_normalized_mse(pred, target, valid, target_std).item())
        n_batches += 1
    return total / max(n_batches, 1)


class CompactMetricEvaluator:
    """Compute all paper metrics in memory without prediction or row files."""

    def __init__(
        self,
        *,
        case_name: str,
        case_tags: Sequence[str],
        root: Path,
        padding_value: float,
    ) -> None:
        from shared.common.data_io import load_npz_dict, load_npz_field
        from shared.common.paths import build_paths, case_npz_path, fem_pred_npz_path

        self.case_tags = list(case_tags)
        self.padding_value = float(padding_value)
        paths = build_paths(case_name, root=root, create=False)
        self.case_cache: Dict[str, Dict[str, Any]] = {}
        self.fem_cache: Dict[str, np.ndarray] = {}
        for tag in self.case_tags:
            case_path = case_npz_path(paths, tag)
            fem_path = fem_pred_npz_path(paths, tag)
            if not case_path.exists() or not fem_path.exists():
                continue
            self.case_cache[tag] = load_npz_dict(case_path)
            self.fem_cache[tag] = np.asarray(
                load_npz_field(fem_path, "fem_local"), dtype=np.float64
            )
        if not self.case_cache:
            raise RuntimeError(f"No test FEM data found for case={case_name!r}")

    @torch.no_grad()
    def evaluate(self, *, operator_model, data_loader: DataLoader, device: torch.device) -> Dict[str, float]:
        from shared.compare.metrics import (
            compute_metrics_from_arrays,
            relative_resultant_l2_errors,
        )

        operator_model.eval()
        values: Dict[str, List[float]] = {key: [] for key in ERROR_METRICS}
        global_index = 0

        for batch in data_loader:
            pred = operator_model.predict(batch)
            query = batch["query"].to(device)
            valid = query[:, :, 0] != self.padding_value
            for batch_index in range(int(query.shape[0])):
                if global_index >= len(self.case_tags):
                    break
                tag = self.case_tags[global_index]
                global_index += 1
                if tag not in self.case_cache:
                    continue

                n_query = int(valid[batch_index].sum().item())
                q_no = pred[batch_index, :n_query].detach().cpu().numpy()
                q_fem = self.fem_cache[tag]
                case = self.case_cache[tag]
                verts = np.asarray(case["verts_xy"], dtype=np.float64)
                faces = np.asarray(case["faces"], dtype=np.int64)
                E = float(np.asarray(case["E"]).reshape(-1)[0])
                nu = float(np.asarray(case["nu"]).reshape(-1)[0])
                thickness = float(np.asarray(case["thickness"]).reshape(-1)[0])
                E_elem = (
                    np.asarray(case["E_elem"], dtype=np.float64).reshape(-1)
                    if "E_elem" in case
                    else None
                )
                nu_elem = (
                    np.asarray(case["nu_elem"], dtype=np.float64).reshape(-1)
                    if "nu_elem" in case
                    else None
                )

                main = compute_metrics_from_arrays(
                    verts_xy=verts,
                    faces=faces,
                    q_no=q_no,
                    q_fem=q_fem,
                    E=E,
                    nu=nu,
                    thickness=thickness,
                    E_elem=E_elem,
                    nu_elem=nu_elem,
                )
                resultants = relative_resultant_l2_errors(
                    verts_xy=verts,
                    faces=faces,
                    q_no=q_no,
                    q_fem=q_fem,
                    E=E,
                    nu=nu,
                    thickness=thickness,
                    E_elem=E_elem,
                    nu_elem=nu_elem,
                )
                values["w_l2"].append(float(main["l2_error"]))
                values["theta_l2"].append(float(main["theta_l2_error"]))
                values["w_h1_semi"].append(float(main["h1_semi_error"]))
                values["energy"].append(float(main["energy_error"]))
                values["curvature"].append(float(main["curvature_error"]))
                values["moment_l2"].append(float(resultants["moment_l2_error"]))
                values["shear_l2"].append(float(resultants["shear_l2_error"]))

        n_cases = len(values["w_l2"])
        if n_cases == 0:
            raise RuntimeError("Final metric evaluation produced no valid test cases")

        result: Dict[str, float] = {"n_test": float(n_cases)}
        for metric, entries in values.items():
            array = np.asarray(entries, dtype=np.float64) * 100.0
            result[f"{metric}_mean_pct"] = float(np.mean(array))
            # Population SD matches the existing single-model test-set reports.
            result[f"{metric}_test_std_pct"] = float(np.std(array, ddof=0))
        return result


def _case_label(case_name: str) -> str:
    return {
        "hole_um_up": "Single-hole UM-UL",
        "holes2_um_grfp": "Double-hole UM-RL",
    }[case_name]


def _load_case_config(case_name: str, method: str) -> Dict[str, Any]:
    if case_name == "hole_um_up" and method == "ga_vino":
        from NO.train.train_hole_um_up_mr import DEFAULT_TRAIN_CONFIG
    elif case_name == "hole_um_up" and method == "ga_no":
        from NO.train.train_hole_um_up_mse import DEFAULT_TRAIN_CONFIG
    elif case_name == "holes2_um_grfp" and method == "ga_vino":
        from NO.train.train_holes2_um_grfp import DEFAULT_TRAIN_CONFIG
    elif case_name == "holes2_um_grfp" and method == "ga_no":
        from NO.train.train_holes2_um_grfp_mse import DEFAULT_TRAIN_CONFIG
    else:
        raise ValueError(f"Unsupported case/method pair: {case_name}/{method}")
    return deepcopy(DEFAULT_TRAIN_CONFIG)


def _load_best_state(path: Path, *, device: torch.device) -> Dict[str, torch.Tensor]:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def _train_one_run(
    *,
    case_name: str,
    method: str,
    seed: int,
    root: Path,
    epochs_override: Optional[int],
    strict_determinism: bool,
    random_geometry_fps: bool,
) -> Dict[str, Any]:
    from NO.common.core import (
        MRPlateEnergyOperator,
        build_plate_operator_model,
        device,
        mr_plate_ginot_minimal_configs,
        plate_collate_fn,
    )
    from NO.common.early_stopping import RelativeEarlyStopping
    from NO.common.trainer import evaluate_energy_loss
    from shared.common.data_io import load_dataset, load_test_tags, load_train_tags
    from shared.common.paths import build_paths

    cfg = _load_case_config(case_name, method)
    if epochs_override is not None:
        cfg["epochs"] = int(epochs_override)
    if cfg.get("epochs") is None:
        cfg["epochs"] = int(cfg.get("max_epochs", 5000))

    init_seed = int(seed)
    shuffle_seed = int(seed) + 10_000
    fps_seed = int(seed) + 20_000
    _set_all_seeds(init_seed, strict_determinism=strict_determinism)

    train_samples, test_samples, _ = load_dataset(case_name=case_name, root=root)
    train_tags = load_train_tags(case_name=case_name, root=root)
    test_tags = load_test_tags(case_name=case_name, root=root)
    paths = build_paths(case_name, root=root, create=False)
    padding_value = float(mr_plate_ginot_minimal_configs()["padding_value"])

    train_targets: Optional[List[torch.Tensor]] = None
    test_targets: Optional[List[torch.Tensor]] = None
    target_std: Optional[torch.Tensor] = None
    if method == "ga_no":
        train_targets = _load_supervised_targets(paths=paths, tags=train_tags)
        test_targets = _load_supervised_targets(paths=paths, tags=test_tags)
        target_std = _target_std(train_targets)

    geometry_seed = fps_seed if random_geometry_fps else None
    train_dataset = CompactPlateDataset(
        train_samples,
        geometry_fps_seed=geometry_seed,
        targets=train_targets,
    )
    test_dataset = CompactPlateDataset(
        test_samples,
        geometry_fps_seed=geometry_seed,
        targets=test_targets,
    )

    shuffle_generator = torch.Generator()
    shuffle_generator.manual_seed(shuffle_seed)
    if method == "ga_no":
        collate = lambda batch: _mse_collate(batch, padding_value=padding_value)
    else:
        collate = lambda batch: plate_collate_fn(batch, padding_value=padding_value)

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        generator=shuffle_generator,
        collate_fn=collate,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
    )

    # Parameter initialization has its own stream.  Reset the global Torch RNG
    # after model construction so stochastic FPS follows the FPS stream and
    # cannot perturb the independently seeded DataLoader shuffle order.
    torch.manual_seed(init_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(init_seed)
    base_model = build_plate_operator_model(
        padding_value=padding_value,
        model_config=cfg.get("model_config"),
    ).to(device)
    operator_model = MRPlateEnergyOperator(
        model=base_model,
        padding_value=padding_value,
    ).to(device)
    torch.manual_seed(fps_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(fps_seed)

    optimizer = torch.optim.Adam(operator_model.parameters(), lr=float(cfg["lr"]))
    if method == "ga_no":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.7,
            patience=25,
            threshold=float(cfg["early_stopping_min_delta"]),
            threshold_mode="rel",
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.7,
            patience=25,
        )

    early_stopper = RelativeEarlyStopping(
        enabled=bool(cfg["early_stopping"]),
        min_epochs=int(cfg["early_stopping_min_epochs"]),
        patience=int(cfg["early_stopping_patience"]),
        min_delta=float(cfg["early_stopping_min_delta"]),
    )

    epochs = int(cfg["epochs"])
    grad_clip_norm = cfg.get("grad_clip_norm", None)
    loss_scale = float(cfg.get("loss_scale", 1.0))
    best_loss = float("inf")
    best_epoch = -1
    final_train_loss = float("nan")
    train_epoch_times: List[float] = []
    epoch_total_times: List[float] = []
    stopped_early = False

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    _sync_cuda()
    wall_start = time.perf_counter()

    with tempfile.TemporaryDirectory(prefix="seed_variability_best_", dir="/tmp") as temp_dir:
        best_path = Path(temp_dir) / "best.pt"

        for epoch in range(1, epochs + 1):
            _sync_cuda()
            epoch_start = time.perf_counter()
            operator_model.train()
            _sync_cuda()
            train_start = time.perf_counter()
            train_total = 0.0
            n_batches = 0

            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                if method == "ga_vino":
                    raw_loss, _, _ = operator_model(batch)
                    optimization_loss = raw_loss * loss_scale
                else:
                    pred = operator_model.predict(batch)
                    target = batch["target_fem_local"].to(device=device, dtype=pred.dtype)
                    query = batch["query"].to(device)
                    valid = query[:, :, 0] != padding_value
                    raw_loss = _normalized_mse(pred, target, valid, target_std)
                    optimization_loss = raw_loss

                optimization_loss.backward()
                if grad_clip_norm is not None and float(grad_clip_norm) > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        operator_model.parameters(), float(grad_clip_norm)
                    )
                optimizer.step()
                train_total += float(raw_loss.item())
                n_batches += 1

            _sync_cuda()
            train_time = float(time.perf_counter() - train_start)
            final_train_loss = train_total / max(n_batches, 1)

            if method == "ga_vino":
                selection_loss, _ = evaluate_energy_loss(operator_model, test_loader)
            else:
                selection_loss = _evaluate_mse(
                    operator_model=operator_model,
                    data_loader=test_loader,
                    padding_value=padding_value,
                    target_std=target_std,
                    device=device,
                )
            scheduler.step(selection_loss)

            if selection_loss < best_loss:
                best_loss = float(selection_loss)
                best_epoch = int(epoch)
                torch.save(operator_model.state_dict(), best_path)

            should_stop = early_stopper.step(selection_loss, epoch)
            _sync_cuda()
            epoch_time = float(time.perf_counter() - epoch_start)
            train_epoch_times.append(train_time)
            epoch_total_times.append(epoch_time)

            print(
                f"[{_case_label(case_name)}][{method}][seed={seed}] "
                f"epoch={epoch:04d}/{epochs:04d} "
                f"train={final_train_loss:.6e} selection={selection_loss:.6e} "
                f"lr={optimizer.param_groups[0]['lr']:.3e} "
                f"train_time={train_time:.2f}s epoch_time={epoch_time:.2f}s"
            )
            if should_stop:
                stopped_early = True
                break

        _sync_cuda()
        training_wall_time = float(time.perf_counter() - wall_start)
        if not best_path.exists():
            raise RuntimeError("Training completed without producing a best checkpoint")
        operator_model.load_state_dict(_load_best_state(best_path, device=device))

    max_gpu_memory_mb = (
        float(torch.cuda.max_memory_allocated() / 1024**2)
        if torch.cuda.is_available()
        else 0.0
    )
    n_parameters = sum(p.numel() for p in operator_model.parameters() if p.requires_grad)

    metric_evaluator = CompactMetricEvaluator(
        case_name=case_name,
        case_tags=test_tags,
        root=root,
        padding_value=padding_value,
    )
    _sync_cuda()
    evaluation_start = time.perf_counter()
    metrics = metric_evaluator.evaluate(
        operator_model=operator_model,
        data_loader=test_loader,
        device=device,
    )
    _sync_cuda()
    evaluation_time = float(time.perf_counter() - evaluation_start)

    train_mean, train_std = _mean_std(train_epoch_times, sample=False)
    epoch_mean, epoch_std = _mean_std(epoch_total_times, sample=False)
    result: Dict[str, Any] = {
        "status": "complete",
        "case_name": case_name,
        "case_label": _case_label(case_name),
        "method": method,
        "seed": int(seed),
        "init_seed": init_seed,
        "shuffle_seed": shuffle_seed,
        "fps_seed": fps_seed,
        "geometry_fps_mode": (
            "seeded_random_start_cached" if random_geometry_fps else "legacy_first_point_cached"
        ),
        "n_train": len(train_dataset),
        "n_test": int(metrics.pop("n_test")),
        "batch_size": int(cfg["batch_size"]),
        "learning_rate": float(cfg["lr"]),
        "actual_epochs": len(train_epoch_times),
        "best_epoch": best_epoch,
        "stopped_early": stopped_early,
        "best_selection_loss": best_loss,
        "final_train_loss": final_train_loss,
        "training_wall_time_sec": training_wall_time,
        "optimization_time_total_sec": float(np.sum(train_epoch_times)),
        "optimization_time_mean_epoch_sec": train_mean,
        "optimization_time_std_epoch_sec": train_std,
        "epoch_total_time_mean_sec": epoch_mean,
        "epoch_total_time_std_sec": epoch_std,
        "final_evaluation_time_sec": evaluation_time,
        "max_gpu_memory_mb": max_gpu_memory_mb,
        "n_trainable_parameters": n_parameters,
        "completed_at": _now(),
        "error_message": "",
    }
    result.update(metrics)

    del metric_evaluator, operator_model, base_model, optimizer, scheduler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _summarize_runs(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[tuple[str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "complete":
            continue
        groups.setdefault((str(row["case_name"]), str(row["method"])), []).append(row)

    summaries: List[Dict[str, Any]] = []
    for (case_name, method), group in sorted(groups.items()):
        group = sorted(group, key=lambda row: int(row["seed"]))
        summary: Dict[str, Any] = {
            "case_name": case_name,
            "case_label": _case_label(case_name),
            "method": method,
            "n_runs": len(group),
            "seeds": ";".join(str(int(row["seed"])) for row in group),
        }
        mappings = {
            "actual_epochs": "actual_epochs",
            "training_wall_time": "training_wall_time_sec",
            "optimization_time": "optimization_time_total_sec",
            "epoch_total_time": "epoch_total_time_mean_sec",
            "max_gpu_memory": "max_gpu_memory_mb",
        }
        for output_prefix, source in mappings.items():
            mean, std = _mean_std((row[source] for row in group), sample=True)
            if output_prefix == "actual_epochs":
                summary["actual_epochs_mean"] = mean
                summary["actual_epochs_seed_std"] = std
            elif output_prefix == "max_gpu_memory":
                summary["max_gpu_memory_mean_mb"] = mean
                summary["max_gpu_memory_seed_std_mb"] = std
            else:
                summary[f"{output_prefix}_mean_sec"] = mean
                summary[f"{output_prefix}_seed_std_sec"] = std

        for metric in ERROR_METRICS:
            mean, std = _mean_std(
                (row[f"{metric}_mean_pct"] for row in group), sample=True
            )
            summary[f"{metric}_mean_pct"] = mean
            summary[f"{metric}_seed_std_pct"] = std
        summaries.append(summary)
    return summaries


def _paired_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    index = {
        (str(row["case_name"]), int(row["seed"]), str(row["method"])): row
        for row in rows
        if row.get("status") == "complete"
    }
    pairs: List[Dict[str, Any]] = []
    case_seed_keys = sorted({(case, seed) for case, seed, _ in index})
    for case_name, seed in case_seed_keys:
        vino = index.get((case_name, seed, "ga_vino"))
        gano = index.get((case_name, seed, "ga_no"))
        if vino is None or gano is None:
            continue
        pair: Dict[str, Any] = {
            "case_name": case_name,
            "case_label": _case_label(case_name),
            "seed": seed,
        }
        for metric in ERROR_METRICS:
            vino_value = float(vino[f"{metric}_mean_pct"])
            gano_value = float(gano[f"{metric}_mean_pct"])
            reduction = gano_value - vino_value
            pair[f"{metric}_ga_vino_pct"] = vino_value
            pair[f"{metric}_ga_no_pct"] = gano_value
            pair[f"{metric}_reduction_pct_point"] = reduction
            pair[f"{metric}_relative_reduction_pct"] = (
                100.0 * reduction / gano_value if abs(gano_value) > 1.0e-30 else float("nan")
            )
            pair[f"{metric}_ga_vino_wins"] = int(vino_value < gano_value)
        pairs.append(pair)
    return pairs


def _summarize_pairs(pairs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in pairs:
        groups.setdefault(str(row["case_name"]), []).append(row)
    summaries: List[Dict[str, Any]] = []
    for case_name, group in sorted(groups.items()):
        group = sorted(group, key=lambda row: int(row["seed"]))
        summary: Dict[str, Any] = {
            "case_name": case_name,
            "case_label": _case_label(case_name),
            "n_pairs": len(group),
            "seeds": ";".join(str(int(row["seed"])) for row in group),
        }
        for metric in ERROR_METRICS:
            reduction_mean, reduction_std = _mean_std(
                (row[f"{metric}_reduction_pct_point"] for row in group), sample=True
            )
            relative_mean, relative_std = _mean_std(
                (row[f"{metric}_relative_reduction_pct"] for row in group), sample=True
            )
            summary[f"{metric}_reduction_mean_pct_point"] = reduction_mean
            summary[f"{metric}_reduction_seed_std_pct_point"] = reduction_std
            summary[f"{metric}_relative_reduction_mean_pct"] = relative_mean
            summary[f"{metric}_relative_reduction_seed_std_pct"] = relative_std
            summary[f"{metric}_ga_vino_win_count"] = sum(
                int(row[f"{metric}_ga_vino_wins"]) for row in group
            )
        summaries.append(summary)
    return summaries


def _normalize_cases(values: Sequence[str]) -> List[str]:
    result: List[str] = []
    for value in values:
        key = str(value).strip().lower().replace("_", "-")
        # Internal names retain underscores in CASE_ALIASES.
        if value in CASE_ALIASES:
            case = CASE_ALIASES[value]
        elif key in CASE_ALIASES:
            case = CASE_ALIASES[key]
        else:
            raise ValueError(f"Unsupported case: {value}")
        if case not in result:
            result.append(case)
    return result


def _normalize_methods(values: Sequence[str]) -> List[str]:
    result: List[str] = []
    for value in values:
        key = str(value).strip().lower()
        if key not in METHOD_ALIASES:
            raise ValueError(f"Unsupported method: {value}")
        method = METHOD_ALIASES[key]
        if method not in result:
            result.append(method)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the compact five-seed GA-VINO/GA-NO robustness study without "
            "saving predictions, plots, per-sample metrics, or checkpoints."
        )
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["single-hole-um-ul", "double-hole-um-rl"],
        help="Cases to run (default: both reviewer-study cases).",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["ga-vino", "ga-no"],
        help="Methods to run (default: ga-vino ga-no).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44, 45, 46],
        help="Paired training seeds (default: 42 43 44 45 46).",
    )
    parser.add_argument("--root", type=Path, default=_PROJECT_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reviewer_seed_variability"),
        help="Compact result directory, relative to project root by default.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Training device passed through NO_DEVICE.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional epoch override, mainly for smoke tests.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Run only the first N matrix entries, mainly for smoke tests.",
    )
    parser.add_argument(
        "--strict-determinism",
        action="store_true",
        help="Request deterministic PyTorch algorithms (warn-only for unsupported operations).",
    )
    parser.add_argument(
        "--legacy-geometry-fps",
        action="store_true",
        help="Keep the current fixed first geometry point instead of seeded random FPS starts.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rerun and replace matching completed case/method/seed rows.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record a failed row and continue with the remaining runs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved experiment matrix without training or writing files.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    cases = _normalize_cases(args.cases)
    methods = _normalize_methods(args.methods)
    seeds = list(dict.fromkeys(int(seed) for seed in args.seeds))
    if not seeds:
        raise ValueError("At least one seed is required")
    if args.epochs is not None and args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.max_runs is not None and args.max_runs <= 0:
        raise ValueError("--max-runs must be positive")

    matrix = [
        (case_name, method, seed)
        for case_name in cases
        for seed in seeds
        for method in methods
    ]
    if args.max_runs is not None:
        matrix = matrix[: int(args.max_runs)]

    print("Resolved repeated-seed experiment:")
    for index, (case_name, method, seed) in enumerate(matrix, start=1):
        print(f"  {index:02d}. {_case_label(case_name)} | {method} | seed={seed}")
    print(f"Output directory: {output_dir}")
    print(
        "Geometry FPS: "
        + ("legacy fixed first point" if args.legacy_geometry_fps else "seeded random start cached per sample")
    )
    if args.dry_run:
        return

    # NO.common.core selects its device at import time, so set this before any
    # training/config module is imported by _train_one_run.
    os.environ["NO_DEVICE"] = args.device
    output_dir.mkdir(parents=True, exist_ok=True)
    run_csv = output_dir / "run_results.csv"
    summary_csv = output_dir / "summary_across_seeds.csv"
    paired_csv = output_dir / "paired_seed_comparisons.csv"
    paired_summary_csv = output_dir / "paired_summary_across_seeds.csv"
    config_path = output_dir / "experiment_config.json"

    config_payload = {
        "created_at": _now(),
        "project_root": str(root),
        "git_commit": _git_commit(root),
        "cases": cases,
        "methods": methods,
        "seeds": seeds,
        "seed_derivation": {
            "parameter_initialization": "seed",
            "dataloader_shuffle": "seed + 10000",
            "fps": "seed + 20000",
        },
        "geometry_fps_mode": (
            "legacy_first_point_cached"
            if args.legacy_geometry_fps
            else "seeded_random_start_cached"
        ),
        "fixed_data_and_split": True,
        "device_request": args.device,
        "strict_determinism": bool(args.strict_determinism),
        "epochs_override": args.epochs,
        "selection_protocol": {
            "split": "fixed test split (preserves the current paper protocol)",
            "checkpoint": "lowest selection loss; temporary checkpoint deleted after final metrics",
            "ga_vino_loss": "total potential energy",
            "ga_no_loss": "normalized supervised MSE",
        },
        "resolved_training_configs": {
            f"{case_name}/{method}": _jsonable(_load_case_config(case_name, method))
            for case_name in cases
            for method in methods
        },
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "retained_artifacts": [
            run_csv.name,
            summary_csv.name,
            paired_csv.name,
            paired_summary_csv.name,
            config_path.name,
        ],
        "not_retained": [
            "model checkpoints",
            "per-sample predictions",
            "per-sample metric tables",
            "comparison figures",
            "loss curves",
        ],
    }
    config_path.write_text(
        json.dumps(_jsonable(config_payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    rows: List[Dict[str, Any]] = list(_read_csv(run_csv))
    planned_keys = {(case, method, seed) for case, method, seed in matrix}
    if args.overwrite:
        rows = [
            row
            for row in rows
            if (str(row.get("case_name")), str(row.get("method")), int(row.get("seed", -1)))
            not in planned_keys
        ]
        _atomic_write_csv(run_csv, rows, RUN_COLUMNS)

    completed_keys = {
        (str(row.get("case_name")), str(row.get("method")), int(row.get("seed", -1)))
        for row in rows
        if row.get("status") == "complete"
    }

    for case_name, method, seed in matrix:
        key = (case_name, method, seed)
        if key in completed_keys:
            print(f"[Skip] already complete: {case_name}/{method}/seed={seed}")
            continue

        print("=" * 88)
        print(f"[Run] {_case_label(case_name)} | {method} | seed={seed}")
        print("=" * 88)
        try:
            result = _train_one_run(
                case_name=case_name,
                method=method,
                seed=seed,
                root=root,
                epochs_override=args.epochs,
                strict_determinism=bool(args.strict_determinism),
                random_geometry_fps=not bool(args.legacy_geometry_fps),
            )
        except Exception as exc:
            result = {
                "status": "failed",
                "case_name": case_name,
                "case_label": _case_label(case_name),
                "method": method,
                "seed": seed,
                "completed_at": _now(),
                "error_message": f"{type(exc).__name__}: {exc}",
            }
            rows = [
                row
                for row in rows
                if (str(row.get("case_name")), str(row.get("method")), int(row.get("seed", -1)))
                != key
            ]
            rows.append(result)
            _atomic_write_csv(run_csv, rows, RUN_COLUMNS)
            if not args.continue_on_error:
                raise
            print(f"[Failed] {result['error_message']}")
            continue

        rows = [
            row
            for row in rows
            if (str(row.get("case_name")), str(row.get("method")), int(row.get("seed", -1)))
            != key
        ]
        rows.append(result)
        rows.sort(key=lambda row: (str(row.get("case_name")), int(row.get("seed", -1)), str(row.get("method"))))
        _atomic_write_csv(run_csv, rows, RUN_COLUMNS)

        summaries = _summarize_runs(rows)
        pairs = _paired_rows(rows)
        pair_summaries = _summarize_pairs(pairs)
        _atomic_write_csv(summary_csv, summaries, SUMMARY_COLUMNS)
        _atomic_write_csv(paired_csv, pairs, PAIRED_COLUMNS)
        _atomic_write_csv(paired_summary_csv, pair_summaries, PAIRED_SUMMARY_COLUMNS)
        print(f"[Saved] compact run row -> {run_csv}")

    # Rebuild summaries once more so resume-only invocations remain consistent.
    summaries = _summarize_runs(rows)
    pairs = _paired_rows(rows)
    pair_summaries = _summarize_pairs(pairs)
    _atomic_write_csv(summary_csv, summaries, SUMMARY_COLUMNS)
    _atomic_write_csv(paired_csv, pairs, PAIRED_COLUMNS)
    _atomic_write_csv(paired_summary_csv, pair_summaries, PAIRED_SUMMARY_COLUMNS)
    print("=" * 88)
    print(f"Completed. Run-level results : {run_csv}")
    print(f"Across-seed summary          : {summary_csv}")
    print(f"Paired comparisons           : {paired_csv}")
    print(f"Paired across-seed summary   : {paired_summary_csv}")


if __name__ == "__main__":
    main()
