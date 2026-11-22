from __future__ import annotations

import time
from itertools import count

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from NO.common.core import (
    KirchhoffLovePlateEnergyOperator,
    MRPlateEnergyOperator,
    build_plate_operator_model,
    device,
    mr_plate_ginot_minimal_configs,
)
from shared.compare.metrics import plot_fem_error_history, plot_loss_curve
from shared.common.paths import build_paths


@torch.no_grad()
def evaluate_energy_loss(
    operator_model: MRPlateEnergyOperator,
    data_loader: DataLoader,
) -> tuple[float, Dict[str, float]]:
    """Evaluate the energy loss on a dataset."""
    operator_model.eval()

    total_loss = 0.0
    total_stats = {
        "Pi": 0.0,
        "U_int": 0.0,
        "U_mem": 0.0,
        "U_ben": 0.0,
        "U_shr": 0.0,
        "constraint_l2": 0.0,
        "W_ext": 0.0,
    }

    n_batches = 0

    for batch in data_loader:
        loss, _, stats = operator_model(batch)

        total_loss += float(loss.item())
        for k in total_stats:
            total_stats[k] += float(stats[k])

        n_batches += 1

    denom = max(n_batches, 1)
    return total_loss / denom, {k: v / denom for k, v in total_stats.items()}


def _print_device_info() -> None:
    print(f"[Device] torch device: {device}")

    if torch.cuda.is_available():
        try:
            print(f"[Device] CUDA device name: {torch.cuda.get_device_name(0)}")
            print(f"[Device] CUDA version: {torch.version.cuda}")
        except Exception as e:
            print(f"[Device] CUDA available, but failed to print details: {e}")
    else:
        print("[Device] CUDA is not available. Training will run on CPU.")


def _save_npz_history(path: str | Path, history: Dict[str, List[float]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {k: np.asarray(v, dtype=np.float64) for k, v in history.items()}
    np.savez_compressed(path, **payload)


def _print_ginot_config(cfg: Dict[str, Any]) -> None:
    """Print the architecture configuration used by the GINOT-like model."""
    print("[Model] GINOT-like configuration:")
    print(f"  padding_value: {cfg.get('padding_value')}")

    for name in ("branch_args", "material_branch_args", "load_branch_args", "trunk_args"):
        args = cfg.get(name, {})
        if args:
            print(f"  {name}:")
            for k, v in args.items():
                print(f"    {k}: {v}")

def _is_base_method(method_name: Optional[str]) -> bool:
    """Return True when outputs should use the original case directory."""
    if method_name is None:
        return True
    name = str(method_name).strip().lower()
    return name in {"", "base", "default", "energy", "energy_min", "energy_minimization"}


def build_energy_output_paths(
    *,
    case_name: str,
    root: Optional[str | Path] = None,
    method_name: Optional[str] = None,
    create: bool = True,
) -> Dict[str, Path]:
    """
    Build output paths for energy-based training without changing base data/FEM paths.

    Base data and FEM references still live under the original case name, e.g.:
        shared/data/square/
        shared/cases/square/
        FEM/outputs/square/

    If method_name is not a base method, NO and compare outputs are saved under:
        NO/outputs/<case_name>_<method_name>/
        compare/<case_name>_<method_name>/

    Examples:
        case_name="square", method_name="boundary_pc" -> square_boundary_pc
        case_name="square", method_name="all_pc"      -> square_all_pc
    """
    base_paths = build_paths(case_name, root=root, create=create)

    if _is_base_method(method_name):
        paths = dict(base_paths)
        no_output_root = Path(paths["no_output_root"])
        compare_root = Path(paths["compare_root"])
        paths.update(
            {
                "method_case": Path(case_name),
                "training_time_summary_path": no_output_root / "training_time_summary.npz",
                "training_time_summary_csv_path": no_output_root / "training_time_summary.csv",
                "training_time_history_csv_path": no_output_root / "training_time_history.csv",
                "theta_error_epoch_plot_path": no_output_root / "Theta_error_curve.png",
                "curvature_error_epoch_plot_path": no_output_root / "Curvature_error_curve.png",
            }
        )
        if create:
            no_output_root.mkdir(parents=True, exist_ok=True)
            Path(paths.get("no_prediction_root", no_output_root / "predictions")).mkdir(parents=True, exist_ok=True)
            compare_root.mkdir(parents=True, exist_ok=True)
        return paths

    project_root = Path(base_paths["project_root"])
    method_case = f"{case_name}_{method_name}"

    no_output_root = project_root / "NO" / "outputs" / method_case
    no_prediction_root = no_output_root / "predictions"
    compare_root = project_root / "compare" / method_case

    paths = dict(base_paths)
    paths.update(
        {
            "method_case": Path(method_case),
            "no_output_root": no_output_root,
            "no_prediction_root": no_prediction_root,
            "compare_root": compare_root,
            "best_model_path": no_output_root / "best_mr_plate_operator.pt",
            "final_model_path": no_output_root / "final_mr_plate_operator.pt",
            "training_history_path": no_output_root / "training_history.npz",
            "fem_error_history_path": no_output_root / "fem_error_history.npz",
            "loss_curve_path": no_output_root / "loss_curve.png",
            "l2_error_epoch_plot_path": no_output_root / "L2_error_curve.png",
            "h1_error_epoch_plot_path": no_output_root / "H1_error_curve.png",
            "energy_error_epoch_plot_path": no_output_root / "Energy_error_curve.png",
            "final_metrics_npz_path": compare_root / "final_metrics.npz",
            "final_metrics_csv_path": compare_root / "final_metrics.csv",
            "training_time_summary_path": no_output_root / "training_time_summary.npz",
            "training_time_summary_csv_path": no_output_root / "training_time_summary.csv",
            "training_time_history_csv_path": no_output_root / "training_time_history.csv",
            "theta_error_epoch_plot_path": no_output_root / "Theta_error_curve.png",
            "curvature_error_epoch_plot_path": no_output_root / "Curvature_error_curve.png",
        }
    )

    if create:
        no_output_root.mkdir(parents=True, exist_ok=True)
        no_prediction_root.mkdir(parents=True, exist_ok=True)
        compare_root.mkdir(parents=True, exist_ok=True)

    return paths


def _cuda_synchronize_if_needed() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _save_training_time_summary(
    *,
    paths: Dict[str, Path],
    history: Dict[str, List[float]],
    total_time_sec: float,
    epochs: int,
    method_name: Optional[str],
    max_memory_mb: float = 0.0,    # 新增
    n_params: int = 0,              # 新增
) -> Dict[str, float]:
    """Save compact training-time summary and per-epoch time history."""
    no_output_root = Path(paths["no_output_root"])
    no_output_root.mkdir(parents=True, exist_ok=True)

    train_epoch_times = np.asarray(history.get("train_epoch_time_sec", []), dtype=np.float64)
    epoch_total_times = np.asarray(history.get("epoch_total_time_sec", []), dtype=np.float64)

    def _mean(arr: np.ndarray) -> float:
        return float(np.mean(arr)) if arr.size else float("nan")

    def _std(arr: np.ndarray) -> float:
        return float(np.std(arr)) if arr.size else float("nan")

    summary = {
        "training_total_time_sec": float(total_time_sec),
        "training_total_time_min": float(total_time_sec / 60.0),
        "train_epoch_time_mean_sec": _mean(train_epoch_times),
        "train_epoch_time_std_sec": _std(train_epoch_times),
        "epoch_total_time_mean_sec": _mean(epoch_total_times),
        "epoch_total_time_std_sec": _std(epoch_total_times),
        "epochs": float(epochs),
        "max_memory_mb": float(max_memory_mb),
        "n_params": int(n_params),
    }

    np.savez_compressed(
        paths["training_time_summary_path"],
        **{k: np.asarray(v, dtype=np.float64) for k, v in summary.items()},
        method_name=np.asarray(str(method_name) if method_name is not None else "energy"),
    )

    # Human-readable summary CSV.
    with open(paths["training_time_summary_csv_path"], "w", encoding="utf-8") as f:
        f.write("metric,value\n")
        f.write(f"method_name,{str(method_name) if method_name is not None else 'energy'}\n")
        for k, v in summary.items():
            f.write(f"{k},{v}\n")

    # Per-epoch timing CSV.
    epochs_arr = np.asarray(history.get("epoch", []), dtype=np.float64)
    with open(paths["training_time_history_csv_path"], "w", encoding="utf-8") as f:
        f.write("epoch,train_epoch_time_sec,epoch_total_time_sec\n")
        n = max(len(epochs_arr), len(train_epoch_times), len(epoch_total_times))
        for i in range(n):
            ep = epochs_arr[i] if i < len(epochs_arr) else i + 1
            tt = train_epoch_times[i] if i < len(train_epoch_times) else float("nan")
            et = epoch_total_times[i] if i < len(epoch_total_times) else float("nan")
            f.write(f"{ep},{tt},{et}\n")

    return summary

def train_with_fem_metrics(
    *,
    case_name: str,
    train_loader: DataLoader,
    test_loader: DataLoader,
    fem_evaluator,
    train_fem_evaluator=None,  
    root: Optional[str | Path] = None,
    epochs: Optional[int] = 100,
    lr: float = 1.0e-3,
    eval_fem_every: int = 1,
    grad_clip_norm: Optional[float] = None,
    loss_scale: float = 1.0,
    include_shear_energy: bool = True,
    plate_theory: str = "Mindlin-Reissner",
    max_epochs: Optional[int] = None,
    model_config: Optional[Dict[str, Any]] = None,
    method_name: Optional[str] = None,
    early_stopping: bool = False,
    early_stopping_min_epochs: int = 100,
    early_stopping_patience: int = 60,
    early_stopping_min_delta: float = 1.0e-3,
) -> tuple[MRPlateEnergyOperator, Dict[str, List[float]], Dict[str, List[float]]]:
    """
    Train the MR-plate neural operator by energy minimization and evaluate FEM
    errors during training.

    Notes
    -----
    The current model is a GINOT-like encoder-decoder:
        boundary point cloud -> point-cloud geometry encoder -> geometry latent
        query points -> NeRF embedding -> cross-attention decoder

    The training objective is still VINO-style total potential energy
    minimization, not supervised MSE regression.

    When enabled, early stopping monitors the energy loss evaluated on
    ``test_loader``.  ``early_stopping_min_delta`` is a relative threshold,
    so one configuration can be used across cases with different energy
    scales.  The numerically lowest-loss checkpoint is saved independently
    from the significant-improvement test used by the patience counter.
    Set ``epochs=None`` to train without an epoch cap; this mode requires
    early stopping so that the loop still has a termination condition.
    """
    paths = build_energy_output_paths(
        case_name=case_name,
        root=root,
        method_name=method_name,
        create=True,
    )
    print(f"[Output] method_name: {method_name if method_name is not None else 'energy'}")
    print(f"[Output] method case : {paths.get('method_case', case_name)}")
    print(f"[Output] NO root     : {paths['no_output_root']}")
    theory_key = str(plate_theory).strip().lower().replace("_", "-")
    use_kirchhoff_love = theory_key in {"kirchhoff-love", "kl"}
    if not use_kirchhoff_love and not include_shear_energy:
        raise ValueError(
            "Removing transverse shear while retaining independent rotations is "
            "not a well-posed plate model. Use plate_theory='Kirchhoff-Love' "
            "for the displacement-only KL operator."
        )

    cfg = mr_plate_ginot_minimal_configs()

    if model_config is not None:
        # This mirrors the shallow update logic in build_plate_operator_model.
        for k, v in model_config.items():
            if k in ("branch_args", "material_branch_args", "load_branch_args", "trunk_args") and isinstance(v, dict):
                cfg[k].update(v)
            else:
                cfg[k] = v

    _print_device_info()
    _print_ginot_config(cfg)

    effective_model_config = model_config
    if use_kirchhoff_love:
        # The KL model predicts only w; theta=grad(w) is derived analytically.
        effective_model_config = dict(model_config or {})
        trunk_args = dict(effective_model_config.get("trunk_args", {}))
        configured_std = trunk_args.get("out_std", [8.0e-2])
        trunk_args["out_channels"] = 1
        trunk_args["out_mean"] = [0.0]
        trunk_args["out_std"] = [float(configured_std[0])]
        effective_model_config["trunk_args"] = trunk_args

    base_model = build_plate_operator_model(
        padding_value=float(cfg["padding_value"]),
        model_config=effective_model_config,
    ).to(device)

    if use_kirchhoff_love:
        operator_model = KirchhoffLovePlateEnergyOperator(
            model=base_model,
            padding_value=float(cfg["padding_value"]),
        ).to(device)
    else:
        operator_model = MRPlateEnergyOperator(
            model=base_model,
            padding_value=float(cfg["padding_value"]),
            include_shear_energy=bool(include_shear_energy),
        ).to(device)

    print(
        "[Plate theory] "
        + (
            "Kirchhoff-Love (w-only; theta=grad(w); Hessian bending energy)"
            if use_kirchhoff_love
            else "Mindlin-Reissner (bending + transverse shear)"
        )
    )

    print("[Model] parameter device:", next(operator_model.parameters()).device)

    optimizer = torch.optim.Adam(operator_model.parameters(), lr=lr)

    if grad_clip_norm is not None and float(grad_clip_norm) > 0.0:
        print(f"[Optimizer] gradient clipping max_norm: {float(grad_clip_norm):.6g}")
    else:
        grad_clip_norm = None

    loss_scale = float(loss_scale)
    if loss_scale <= 0.0:
        raise ValueError(f"loss_scale must be positive, got {loss_scale}")

    early_stopping = bool(early_stopping)
    early_stopping_min_epochs = int(early_stopping_min_epochs)
    early_stopping_patience = int(early_stopping_patience)
    early_stopping_min_delta = float(early_stopping_min_delta)

    if epochs is not None:
        epochs = int(epochs)
        if epochs <= 0:
            raise ValueError("epochs must be positive or None")
    elif not early_stopping:
        raise ValueError("epochs=None requires early_stopping=True")

    if max_epochs is not None:
        max_epochs = int(max_epochs)
        if max_epochs <= 0:
            raise ValueError("max_epochs must be positive or None")

    if early_stopping_min_epochs < 0:
        raise ValueError("early_stopping_min_epochs must be non-negative")
    if early_stopping_patience <= 0:
        raise ValueError("early_stopping_patience must be positive")
    if early_stopping_min_delta < 0.0:
        raise ValueError("early_stopping_min_delta must be non-negative")

    print(f"[Optimizer] loss_scale: {loss_scale:.6g}")
    print(
        "[Early Stopping] "
        f"enabled={early_stopping} | "
        f"min_epochs={early_stopping_min_epochs} | "
        f"patience={early_stopping_patience} | "
        f"relative_min_delta={early_stopping_min_delta:.3e}"
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.7,
        patience=25,
    )

    # For evaluation, we need a non-shuffling loader for the training set.
    train_eval_loader = DataLoader(
        train_loader.dataset,
        batch_size=train_loader.batch_size,
        shuffle=False,
        collate_fn=train_loader.collate_fn,
    )

    history: Dict[str, List[float]] = {
        "epoch": [],
        "train_loss": [],
        "test_loss": [],
        "train_loss_scaled": [],
        "test_Pi": [],
        "test_U_int": [],
        "test_U_mem": [],
        "test_U_ben": [],
        "test_U_shr": [],
        "test_constraint_l2": [],
        "test_W_ext": [],
        "early_stopping_bad_epochs": [],
        "train_epoch_time_sec": [],
        "epoch_total_time_sec": [],
    }

    fem_error_history: Dict[str, List[float]] = {
        "epoch": [],
        # Test set metrics
        "test_l2_error_mean": [],
        "test_l2_error_std": [],
        "test_theta_l2_error_mean": [],
        "test_theta_l2_error_std": [],
        # Train set metrics
        "train_l2_error_mean": [],
        "train_l2_error_std": [],
        "train_theta_l2_error_mean": [],
        "train_theta_l2_error_std": [],
        # Compatibility / legacy keys (points to test set)
        "l2_error_mean": [],
        "l2_error_std": [],
        "theta_l2_error_mean": [],
        "theta_l2_error_std": [],
        "n_metric_cases": [],
    }

    best_test_loss = float("inf")
    best_epoch = -1
    early_stopping_reference_loss = float("inf")
    epochs_without_improvement = 0
    stopped_early = False

    _cuda_synchronize_if_needed()
    training_total_t0 = time.perf_counter()

    if epochs is None:
        epoch_limit = max_epochs
    elif max_epochs is None:
        epoch_limit = epochs
    else:
        epoch_limit = min(epochs, max_epochs)
    epoch_iterator = count(1) if epoch_limit is None else range(1, epoch_limit + 1)
    epoch_limit_label = "unlimited" if epoch_limit is None else f"{epoch_limit:04d}"

    for epoch in epoch_iterator:
        _cuda_synchronize_if_needed()
        epoch_total_t0 = time.perf_counter()

        operator_model.train()

        _cuda_synchronize_if_needed()
        train_epoch_t0 = time.perf_counter()

        epoch_loss = 0.0
        n_batches = 0


        epoch_loss_scaled = 0.0

        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)

            raw_loss, _, _ = operator_model(batch)

            # Only scale the optimization objective.
            # The physical energy values in stats remain unscaled.
            opt_loss = raw_loss * loss_scale

            opt_loss.backward()

            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    operator_model.parameters(),
                    max_norm=float(grad_clip_norm),
                )

            optimizer.step()

            epoch_loss += float(raw_loss.item())
            epoch_loss_scaled += float(opt_loss.item())

            n_batches += 1


        _cuda_synchronize_if_needed()
        train_epoch_t1 = time.perf_counter()
        train_epoch_time_sec = float(train_epoch_t1 - train_epoch_t0)

        denom = max(n_batches, 1)
        train_loss = epoch_loss / denom
        train_loss_scaled = epoch_loss_scaled / denom

        test_loss, test_stats = evaluate_energy_loss(operator_model, test_loader)
        scheduler.step(test_loss)

        history["epoch"].append(float(epoch))
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["train_loss_scaled"].append(train_loss_scaled)

        for key in ("Pi", "U_int", "U_mem", "U_ben", "U_shr", "constraint_l2", "W_ext"):
            history[f"test_{key}"].append(test_stats[key])

        # Save every exact numerical best.  The early-stopping reference below
        # is separate so that tiny improvements do not continually reset the
        # patience counter.
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_epoch = epoch
            torch.save(operator_model.state_dict(), paths["best_model_path"])

        if not np.isfinite(early_stopping_reference_loss):
            significant_improvement = True
        else:
            improvement_threshold = early_stopping_min_delta * max(
                abs(early_stopping_reference_loss),
                1.0e-12,
            )
            significant_improvement = (
                test_loss
                < early_stopping_reference_loss - improvement_threshold
            )

        if significant_improvement:
            early_stopping_reference_loss = float(test_loss)
            epochs_without_improvement = 0
        elif epoch >= early_stopping_min_epochs:
            epochs_without_improvement += 1

        history["early_stopping_bad_epochs"].append(
            float(epochs_without_improvement)
        )

        should_stop = (
            early_stopping
            and epoch >= early_stopping_min_epochs
            and epochs_without_improvement >= early_stopping_patience
        )
        reached_epoch_limit = epoch_limit is not None and epoch >= epoch_limit

        do_fem_eval = (
            fem_evaluator is not None
            and (
                epoch == 1
                or reached_epoch_limit
                or should_stop
                or (eval_fem_every > 0 and epoch % eval_fem_every == 0)
            )
        )

        fem_msg = ""
        if do_fem_eval:
            # 1. Evaluate Test Set
            test_metric_stats = fem_evaluator.evaluate_model(
                operator_model=operator_model,
                data_loader=test_loader,
            )

            # 2. Evaluate Train Set (if available)
            train_metric_stats = None
            if train_fem_evaluator is not None:
                train_metric_stats = train_fem_evaluator.evaluate_model(
                    operator_model=operator_model,
                    data_loader=train_eval_loader,
                )

            fem_error_history["epoch"].append(float(epoch))

            # Save Test metrics
            for k in (
                "l2_error_mean",
                "l2_error_std",
                "theta_l2_error_mean",
                "theta_l2_error_std",
            ):
                val = float(test_metric_stats[k])
                fem_error_history[f"test_{k}"].append(val)
                fem_error_history[k].append(val)  # Legacy compatibility

            fem_error_history["n_metric_cases"].append(float(test_metric_stats["n_metric_cases"]))

            # Save Train metrics
            if train_metric_stats is not None:
                for k in (
                    "l2_error_mean",
                    "l2_error_std",
                    "theta_l2_error_mean",
                    "theta_l2_error_std",
                ):
                    fem_error_history[f"train_{k}"].append(float(train_metric_stats[k]))

            fem_msg = (
                f" | Test_L2={test_metric_stats['l2_error_mean']:.4e}±{test_metric_stats['l2_error_std']:.2e}"
                f" | Test_Theta={test_metric_stats['theta_l2_error_mean']:.4e}±{test_metric_stats['theta_l2_error_std']:.2e}"
            )
            if train_metric_stats is not None:
                fem_msg += (
                    f" | Train_L2={train_metric_stats['l2_error_mean']:.4e}±{train_metric_stats['l2_error_std']:.2e}"
                    f" | Train_Theta={train_metric_stats['theta_l2_error_mean']:.4e}±{train_metric_stats['theta_l2_error_std']:.2e}"
                )

        _cuda_synchronize_if_needed()
        epoch_total_t1 = time.perf_counter()
        epoch_total_time_sec = float(epoch_total_t1 - epoch_total_t0)

        history["train_epoch_time_sec"].append(train_epoch_time_sec)
        history["epoch_total_time_sec"].append(epoch_total_time_sec)

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:04d}/{epoch_limit_label} | "
            f"lr={lr_now:.3e} | "
            f"train_loss={train_loss:.6e} | "
            f"train_loss_scaled={train_loss_scaled:.6e} | "
            f"test_loss={test_loss:.6e} | "
            f"test_U_int={test_stats['U_int']:.6e} | "
            f"test_W_ext={test_stats['W_ext']:.6e}"
            f"{fem_msg}"
            f" | train_time={train_epoch_time_sec:.2f}s"
            f" | epoch_time={epoch_total_time_sec:.2f}s"
        )

        if should_stop or reached_epoch_limit:
            stopped_early = True
            if should_stop:
                print(
                    f"[Early Stopping] stopped_epoch={epoch} | "
                    f"best_epoch={best_epoch} | "
                    f"best_test_loss={best_test_loss:.6e} | "
                    f"patience={early_stopping_patience} | "
                    f"relative_min_delta={early_stopping_min_delta:.3e}"
                )
            else:
                print(
                    f"[Epoch Limit] stopped_epoch={epoch} | "
                    f"best_epoch={best_epoch} | "
                    f"best_test_loss={best_test_loss:.6e} | "
                    f"max_epochs={epoch_limit}"
                )
            break

    _cuda_synchronize_if_needed()
    training_total_t1 = time.perf_counter()
    training_total_time_sec = float(training_total_t1 - training_total_t0)

    max_memory_mb = 0.0
    if torch.cuda.is_available():
        max_memory_mb = torch.cuda.max_memory_allocated() / 1024**2
    
    n_params = sum(p.numel() for p in operator_model.parameters() if p.requires_grad)

    actual_epochs = len(history["epoch"])

    time_summary = _save_training_time_summary(
        paths=paths,
        history=history,
        total_time_sec=training_total_time_sec,
        epochs=actual_epochs,
        method_name=method_name,
        max_memory_mb=max_memory_mb, 
        n_params=n_params,
    )

    print(
        f"[Training Time] total={time_summary['training_total_time_sec']:.2f}s "
        f"({time_summary['training_total_time_min']:.2f} min), "
        f"Memory={max_memory_mb:.2f} MB, Params={n_params}" 
        f"mean train epoch={time_summary['train_epoch_time_mean_sec']:.2f}s, "
        f"mean total epoch={time_summary['epoch_total_time_mean_sec']:.2f}s"
    )
    print(f"[Training Time] summary saved: {paths['training_time_summary_path']}")
    print(f"[Training Time] summary csv  : {paths['training_time_summary_csv_path']}")

    torch.save(operator_model.state_dict(), paths["final_model_path"])

    _save_npz_history(paths["training_history_path"], history)
    plot_loss_curve(
        history,
        paths["loss_curve_path"],
        negative_only=use_kirchhoff_love,
    )

    _save_npz_history(paths["fem_error_history_path"], fem_error_history)
    plot_fem_error_history(
        error_history=fem_error_history,
        out_paths={
            "l2": paths["l2_error_epoch_plot_path"],
            "theta": paths["theta_error_epoch_plot_path"],
        },
    )

    print(f"[Checkpoint] final model saved: {paths['final_model_path']}")
    print(f"[Checkpoint] best model saved : {paths['best_model_path']}")
    print(f"[Checkpoint] best_epoch={best_epoch}, best_test_loss={best_test_loss:.6e}")
    print(
        f"[Early Stopping] stopped_early={stopped_early}, "
        f"actual_epochs={actual_epochs}, epoch_limit={epoch_limit}"
    )
    print(f"[History] training history saved: {paths['training_history_path']}")
    print(f"[History] FEM error history saved: {paths['fem_error_history_path']}")

    # Load best model before exporting final predictions.
    if paths["best_model_path"].exists():
        operator_model.load_state_dict(
            torch.load(paths["best_model_path"], map_location=device)
        )
        print("[Checkpoint] Loaded best model for final NO prediction export.")

    return operator_model, history, fem_error_history
