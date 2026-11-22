from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import csv
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[2]

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from NO.common.trainer import build_energy_output_paths
from shared.common.data_io import (
    load_dataset,
    load_npz_field,
    load_test_tags,
    load_train_tags,
    save_npz_dict,
)
from shared.common.paths import build_paths, fem_pred_npz_path, no_pred_npz_path
from shared.compare.compare import compare_cases
from shared.compare.metrics import plot_fem_error_history


DEFAULT_FNO_CONFIG: Dict[str, Any] = {
    "method_name": "fno",
    "batch_size": 16,
    "epochs": 500,
    "lr": 5.0e-4,
    "weight_decay": 1.0e-6,
    "modes_x": 16,
    "modes_y": 16,
    "width": 64,
    "depth": 4,
    "hidden_channels": 128,
    "grad_clip_norm": 1.0,
    "eval_fem_every": 1,
    "seed": 42,
    "deformation_scale": 10.0,
    "max_plot_cases": 5,
}


def _to_numpy(value: Any, dtype=None) -> np.ndarray:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value)
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr


def _scalar(value: Any, default: float = 0.0) -> float:
    try:
        arr = _to_numpy(value)
        return float(arr.reshape(-1)[0])
    except Exception:
        return float(default)


def _int_scalar(value: Any, default: int) -> int:
    try:
        arr = _to_numpy(value)
        return int(arr.reshape(-1)[0])
    except Exception:
        return int(default)


def _node_average_from_elements(
    *,
    n_nodes: int,
    faces: np.ndarray,
    elem_values: np.ndarray,
) -> np.ndarray:
    elem_values = np.asarray(elem_values, dtype=np.float32).reshape(-1)
    out = np.zeros((n_nodes,), dtype=np.float32)
    cnt = np.zeros((n_nodes,), dtype=np.float32)

    for e, tri in enumerate(np.asarray(faces, dtype=np.int64)):
        out[tri] += elem_values[e]
        cnt[tri] += 1.0

    cnt = np.maximum(cnt, 1.0)
    return out / cnt


def _structured_grid_indices(
    *,
    verts_xy: np.ndarray,
    lx: float,
    ly: float,
    nx: int,
    ny: int,
) -> Tuple[np.ndarray, np.ndarray]:
    x = verts_xy[:, 0]
    y = verts_xy[:, 1]
    i = np.rint(x / max(float(lx), 1.0e-12) * int(nx)).astype(np.int64)
    j = np.rint(y / max(float(ly), 1.0e-12) * int(ny)).astype(np.int64)
    i = np.clip(i, 0, int(nx))
    j = np.clip(j, 0, int(ny))
    return i, j


def _extract_nodal_q_from_grid(
    grid_chw: np.ndarray,
    *,
    node_i: np.ndarray,
    node_j: np.ndarray,
) -> np.ndarray:
    vals = np.asarray(grid_chw)[:, node_j, node_i]
    if vals.ndim != 2:
        raise ValueError(f"Expected extracted nodal values to be 2D, got {vals.shape}")
    if vals.shape[0] == 3 and vals.shape[1] == len(node_i):
        vals = vals.T
    if vals.shape != (len(node_i), 3):
        raise ValueError(f"Expected nodal q shape {(len(node_i), 3)}, got {vals.shape}")
    return vals.astype(np.float64, copy=False)


class LShapeFNOSupervisedDataset(Dataset):
    """
    Supervised FNO dataset for structured L-shape samples.

    Inputs are dense [C,H,W] tensors on the full rectangular grid. The removed
    cutout is retained as mask=0. Targets are FEM nodal values
    [w, theta_x, theta_y] placed on the same grid.
    """

    def __init__(
        self,
        *,
        case_name: str,
        samples: Sequence[Dict[str, Any]],
        tags: Sequence[str],
        root: Optional[str | Path] = None,
        target_mean: Optional[np.ndarray] = None,
        target_std: Optional[np.ndarray] = None,
    ):
        if len(samples) != len(tags):
            raise ValueError(f"samples/tags length mismatch: {len(samples)} vs {len(tags)}")

        self.case_name = case_name
        self.samples = list(samples)
        self.tags = list(tags)
        self.paths = build_paths(case_name, root=root, create=True)
        self.items: List[Dict[str, Any]] = []

        for sample, tag in zip(self.samples, self.tags):
            self.items.append(self._build_item(sample, tag))

        if target_mean is None or target_std is None:
            valid_targets = []
            for item in self.items:
                y = item["target"]
                m = item["mask"]
                vals = y[:, m[0] > 0.5].T
                valid_targets.append(vals)
            all_vals = np.concatenate(valid_targets, axis=0)
            target_mean = np.mean(all_vals, axis=0)
            target_std = np.std(all_vals, axis=0)
            target_std = np.maximum(target_std, 1.0e-8)

        self.target_mean = np.asarray(target_mean, dtype=np.float32).reshape(3)
        self.target_std = np.asarray(target_std, dtype=np.float32).reshape(3)

        for item in self.items:
            mean = self.target_mean[:, None, None]
            std = self.target_std[:, None, None]
            item["target_norm"] = (item["target"] - mean) / std

    def _build_item(self, sample: Dict[str, Any], tag: str) -> Dict[str, Any]:
        fem_path = fem_pred_npz_path(self.paths, tag)
        if not fem_path.exists():
            raise FileNotFoundError(
                f"Missing FEM result for tag={tag}: {fem_path}. "
                "Run stage1/FEM for both train and test cases first."
            )

        verts_xy = _to_numpy(sample["verts_xy"], np.float32)
        faces = _to_numpy(sample["faces"], np.int64)
        q_fem = _to_numpy(load_npz_field(fem_path, "fem_local"), np.float32)
        pressure = _to_numpy(sample["pressure"], np.float32).reshape(-1)
        bc_factor = _to_numpy(sample.get("bc_factor", np.ones((verts_xy.shape[0], 3))), np.float32)

        lx = _scalar(sample.get("lx"), float(np.max(verts_xy[:, 0])))
        ly = _scalar(sample.get("ly"), float(np.max(verts_xy[:, 1])))
        nx = _int_scalar(sample.get("mesh_nx"), 50)
        ny = _int_scalar(sample.get("mesh_ny"), 50)

        mesh_type = str(sample.get("mesh_type", "structured"))
        if mesh_type not in {"structured", "cartesian", "grid"}:
            raise ValueError(
                f"FNO baseline expects structured L-shape samples, got mesh_type={mesh_type!r}."
            )

        h = ny + 1
        w = nx + 1
        node_i, node_j = _structured_grid_indices(verts_xy=verts_xy, lx=lx, ly=ly, nx=nx, ny=ny)

        mask = np.zeros((1, h, w), dtype=np.float32)
        target = np.zeros((3, h, w), dtype=np.float32)
        pressure_grid = np.zeros((1, h, w), dtype=np.float32)
        bc_grid = np.zeros((3, h, w), dtype=np.float32)

        mask[0, node_j, node_i] = 1.0
        target[:, node_j, node_i] = q_fem.T
        pressure_grid[0, node_j, node_i] = pressure
        bc_grid[:, node_j, node_i] = bc_factor.T

        if "E_elem" in sample and "nu_elem" in sample:
            e_node = _node_average_from_elements(
                n_nodes=verts_xy.shape[0],
                faces=faces,
                elem_values=_to_numpy(sample["E_elem"], np.float32),
            )
            nu_node = _node_average_from_elements(
                n_nodes=verts_xy.shape[0],
                faces=faces,
                elem_values=_to_numpy(sample["nu_elem"], np.float32),
            )
        else:
            e_node = np.full((verts_xy.shape[0],), _scalar(sample.get("E"), 1000.0), dtype=np.float32)
            nu_node = np.full((verts_xy.shape[0],), _scalar(sample.get("nu"), 0.3), dtype=np.float32)

        e_grid = np.zeros((1, h, w), dtype=np.float32)
        nu_grid = np.zeros((1, h, w), dtype=np.float32)
        e_grid[0, node_j, node_i] = e_node
        nu_grid[0, node_j, node_i] = nu_node

        e_min = _scalar(sample.get("E_min"), float(np.min(e_node)))
        e_max = _scalar(sample.get("E_max"), float(np.max(e_node)))
        nu_min = _scalar(sample.get("nu_min"), float(np.min(nu_node)))
        nu_max = _scalar(sample.get("nu_max"), float(np.max(nu_node)))
        p_min = _scalar(sample.get("pressure_min"), float(np.min(pressure)))
        p_max = _scalar(sample.get("pressure_max"), float(np.max(pressure)))

        e_grid = 2.0 * (e_grid - e_min) / max(e_max - e_min, 1.0e-12) - 1.0
        nu_grid = 2.0 * (nu_grid - nu_min) / max(nu_max - nu_min, 1.0e-12) - 1.0
        pressure_grid = 2.0 * (pressure_grid - p_min) / max(p_max - p_min, 1.0e-12) - 1.0

        yy, xx = np.meshgrid(
            np.linspace(-1.0, 1.0, h, dtype=np.float32),
            np.linspace(-1.0, 1.0, w, dtype=np.float32),
            indexing="ij",
        )
        coord = np.stack([xx, yy], axis=0)

        x = np.concatenate(
            [
                coord,
                mask,
                pressure_grid,
                e_grid,
                nu_grid,
                bc_grid,
            ],
            axis=0,
        ).astype(np.float32)

        return {
            "tag": str(tag),
            "x": x,
            "target": target.astype(np.float32),
            "mask": mask.astype(np.float32),
            "q_fem": q_fem.astype(np.float64),
            "verts_xy": verts_xy,
            "faces": faces,
            "node_i": node_i,
            "node_j": node_j,
            "sample": sample,
        }

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        return {
            "x": torch.from_numpy(item["x"]),
            "target": torch.from_numpy(item["target_norm"]),
            "target_phys": torch.from_numpy(item["target"]),
            "mask": torch.from_numpy(item["mask"]),
            "tag": item["tag"],
            "idx": idx,
        }


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes_x: int, modes_y: int):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes_x = int(modes_x)
        self.modes_y = int(modes_y)
        scale = 1.0 / max(in_channels * out_channels, 1)
        self.weights1 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes_y, modes_x, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes_y, modes_x, dtype=torch.cfloat)
        )

    @staticmethod
    def compl_mul2d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", a, b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, h, w = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")

        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            h,
            w // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        my = min(self.modes_y, h)
        mx = min(self.modes_x, w // 2 + 1)
        out_ft[:, :, :my, :mx] = self.compl_mul2d(
            x_ft[:, :, :my, :mx],
            self.weights1[:, :, :my, :mx],
        )
        out_ft[:, :, -my:, :mx] = self.compl_mul2d(
            x_ft[:, :, -my:, :mx],
            self.weights2[:, :, :my, :mx],
        )

        return torch.fft.irfft2(out_ft, s=(h, w), norm="ortho")


class FNO2d(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int = 3,
        width: int = 64,
        modes_x: int = 16,
        modes_y: int = 16,
        depth: int = 4,
        hidden_channels: int = 128,
    ):
        super().__init__()
        self.lift = nn.Conv2d(in_channels, width, kernel_size=1)
        self.spectral = nn.ModuleList(
            [SpectralConv2d(width, width, modes_x, modes_y) for _ in range(depth)]
        )
        self.local = nn.ModuleList([nn.Conv2d(width, width, kernel_size=1) for _ in range(depth)])
        self.proj1 = nn.Conv2d(width, hidden_channels, kernel_size=1)
        self.proj2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lift(x)
        for spec, local in zip(self.spectral, self.local):
            x = F.gelu(spec(x) + local(x))
        x = F.gelu(self.proj1(x))
        return self.proj2(x)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.expand_as(pred)
    return torch.sum(weight * (pred - target) ** 2) / torch.clamp(torch.sum(weight), min=1.0)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0.0
    n = 0
    for batch in loader:
        x = batch["x"].to(device)
        target = batch["target"].to(device)
        mask = batch["mask"].to(device)
        pred = model(x)
        loss = masked_mse(pred, target, mask)
        total += float(loss.item())
        n += 1
    return total / max(n, 1)


def _save_loss_curve(history: Dict[str, List[float]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.plot(history["epoch"], history["train_loss"], label="Train loss", linewidth=2.2)
    ax.plot(history["epoch"], history["test_loss"], label="Test loss", linewidth=2.2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Masked normalized MSE")
    ax.set_title("FNO supervised loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _save_history(history: Dict[str, List[float]], out_npz: Path, out_csv: Path) -> None:
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **{k: np.asarray(v, dtype=np.float64) for k, v in history.items()})

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history.keys()))
        writer.writeheader()
        n = len(history["epoch"])
        for i in range(n):
            writer.writerow({k: history[k][i] for k in history})


def _metric_keys() -> Tuple[str, ...]:
    return (
        "l2_error",
        "theta_l2_error",
    )


def _empty_fem_error_history() -> Dict[str, List[float]]:
    history: Dict[str, List[float]] = {"epoch": [], "n_metric_cases": []}
    for prefix in ("test", "train"):
        for key in _metric_keys():
            history[f"{prefix}_{key}_mean"] = []
            history[f"{prefix}_{key}_std"] = []
    for key in _metric_keys():
        history[f"{key}_mean"] = []
        history[f"{key}_std"] = []
    return history


@torch.no_grad()
def evaluate_fno_fem_metrics(
    *,
    model: nn.Module,
    dataset: LShapeFNOSupervisedDataset,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    rows: List[Dict[str, float]] = []

    mean = torch.as_tensor(dataset.target_mean, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.as_tensor(dataset.target_std, dtype=torch.float32, device=device).view(1, 3, 1, 1)

    for batch in loader:
        x = batch["x"].to(device)
        pred_grid = model(x) * std + mean
        pred_grid_np = pred_grid.detach().cpu().numpy()

        for b, item_idx in enumerate(batch["idx"].detach().cpu().numpy().astype(np.int64)):
            item = dataset.items[int(item_idx)]
            node_i = item["node_i"]
            node_j = item["node_j"]
            q_no = _extract_nodal_q_from_grid(
                pred_grid_np[b],
                node_i=node_i,
                node_j=node_j,
            )

            q_fem = item["q_fem"]
            rows.append(
                {
                    "l2_error": _relative_node_l2_w(q_no, q_fem),
                    "theta_l2_error": _relative_node_l2_theta(q_no, q_fem),
                }
            )

    out: Dict[str, float] = {"n_metric_cases": float(len(rows))}
    if not rows:
        for key in _metric_keys():
            out[f"{key}_mean"] = float("nan")
            out[f"{key}_std"] = float("nan")
        return out

    for key in _metric_keys():
        arr = np.asarray([r[key] for r in rows], dtype=np.float64)
        out[f"{key}_mean"] = float(np.mean(arr))
        out[f"{key}_std"] = float(np.std(arr))
    return out


def _append_metric_stats(
    *,
    history: Dict[str, List[float]],
    prefix: str,
    stats: Dict[str, float],
    legacy: bool,
) -> None:
    for key in _metric_keys():
        mean_key = f"{key}_mean"
        std_key = f"{key}_std"
        history[f"{prefix}_{mean_key}"].append(float(stats[mean_key]))
        history[f"{prefix}_{std_key}"].append(float(stats[std_key]))
        if legacy:
            history[mean_key].append(float(stats[mean_key]))
            history[std_key].append(float(stats[std_key]))


def _relative_node_l2_w(q_no: np.ndarray, q_fem: np.ndarray) -> float:
    n = min(q_no.shape[0], q_fem.shape[0])
    w_no = np.asarray(q_no[:n, 0], dtype=np.float64)
    w_fem = np.asarray(q_fem[:n, 0], dtype=np.float64)
    return float(np.linalg.norm(w_no - w_fem) / max(np.linalg.norm(w_fem), 1.0e-30))


def _relative_node_l2_theta(q_no: np.ndarray, q_fem: np.ndarray) -> float:
    n = min(q_no.shape[0], q_fem.shape[0])
    th_no = np.asarray(q_no[:n, 1:3], dtype=np.float64)
    th_fem = np.asarray(q_fem[:n, 1:3], dtype=np.float64)
    return float(np.linalg.norm(th_no - th_fem) / max(np.linalg.norm(th_fem), 1.0e-30))


def _save_fem_error_outputs(
    *,
    paths: Dict[str, Path],
    fem_error_history: Dict[str, List[float]],
) -> None:
    np.savez_compressed(
        paths["fem_error_history_path"],
        **{k: np.asarray(v, dtype=np.float64) for k, v in fem_error_history.items()},
    )
    plot_fem_error_history(
        error_history=fem_error_history,
        out_paths={
            "l2": paths["l2_error_epoch_plot_path"],
            "theta": paths["theta_error_epoch_plot_path"],
        },
    )


def _save_training_time_summary(
    *,
    paths: Dict[str, Path],
    history: Dict[str, List[float]],
    total_time_sec: float,
    epochs: int,
    method_name: str,
    max_memory_mb: float,
    n_params: int,
) -> Dict[str, float]:
    no_output_root = Path(paths["no_output_root"])
    no_output_root.mkdir(parents=True, exist_ok=True)

    train_times = np.asarray(history.get("train_epoch_time_sec", []), dtype=np.float64)
    epoch_times = np.asarray(history.get("epoch_total_time_sec", []), dtype=np.float64)

    def _mean(arr: np.ndarray) -> float:
        return float(np.mean(arr)) if arr.size else float("nan")

    def _std(arr: np.ndarray) -> float:
        return float(np.std(arr)) if arr.size else float("nan")

    summary = {
        "training_total_time_sec": float(total_time_sec),
        "training_total_time_min": float(total_time_sec / 60.0),
        "train_epoch_time_mean_sec": _mean(train_times),
        "train_epoch_time_std_sec": _std(train_times),
        "epoch_total_time_mean_sec": _mean(epoch_times),
        "epoch_total_time_std_sec": _std(epoch_times),
        "epochs": float(epochs),
        "max_memory_mb": float(max_memory_mb),
        "n_params": float(n_params),
    }

    np.savez_compressed(
        paths["training_time_summary_path"],
        **{k: np.asarray(v, dtype=np.float64) for k, v in summary.items()},
        method_name=np.asarray(str(method_name)),
    )

    with open(paths["training_time_summary_csv_path"], "w", encoding="utf-8") as f:
        f.write("metric,value\n")
        f.write(f"method_name,{method_name}\n")
        for k, v in summary.items():
            f.write(f"{k},{v}\n")

    with open(paths["training_time_history_csv_path"], "w", encoding="utf-8") as f:
        f.write("epoch,train_epoch_time_sec,epoch_total_time_sec\n")
        n = len(history.get("epoch", []))
        for i in range(n):
            f.write(
                f"{history['epoch'][i]},"
                f"{history['train_epoch_time_sec'][i]},"
                f"{history['epoch_total_time_sec'][i]}\n"
            )

    return summary


@torch.no_grad()
def export_fno_predictions(
    *,
    model: nn.Module,
    dataset: LShapeFNOSupervisedDataset,
    case_name: str,
    root: Optional[str | Path],
    method_name: str,
    device: torch.device,
) -> List[str]:
    paths = build_energy_output_paths(
        case_name=case_name,
        root=root,
        method_name=method_name,
        create=True,
    )

    model.eval()
    mean = torch.as_tensor(dataset.target_mean, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.as_tensor(dataset.target_std, dtype=torch.float32, device=device).view(1, 3, 1, 1)

    saved: List[str] = []
    for idx, item in enumerate(dataset.items):
        x = torch.from_numpy(item["x"]).unsqueeze(0).to(device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        pred_norm = model(x)
        pred_grid = pred_norm * std + mean
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        pred_grid_np = pred_grid.squeeze(0).detach().cpu().numpy()
        node_i = item["node_i"]
        node_j = item["node_j"]
        pred_local = _extract_nodal_q_from_grid(
            pred_grid_np,
            node_i=node_i,
            node_j=node_j,
        )

        sample = item["sample"]
        verts_xy = item["verts_xy"]
        faces = item["faces"]
        tag = item["tag"]

        pred_global_disp = np.zeros((pred_local.shape[0], 3), dtype=np.float64)
        pred_global_disp[:, 2] = pred_local[:, 0]

        payload: Dict[str, Any] = {
            "case_tag": np.asarray(tag),
            "source_sample_id": np.asarray(_int_scalar(sample.get("sample_id"), idx), dtype=np.int64),
            "query_xyz": _to_numpy(sample["query"], np.float32),
            "verts_xy": verts_xy,
            "faces": faces,
            "pressure": _to_numpy(sample["pressure"], np.float32),
            "bc_factor": _to_numpy(sample["bc_factor"], np.float32),
            "pred_local": pred_local,
            "pred_global_disp": pred_global_disp,
            "disp_scalar": pred_local[:, 0].copy(),
            "rot_scalar": np.abs(pred_local[:, 1]) + np.abs(pred_local[:, 2]),
            "n_query_points": np.asarray(int(pred_local.shape[0]), dtype=np.int64),
            "n_geometry_points": np.asarray(int(np.prod(item["mask"].shape[-2:])), dtype=np.int64),
            "n_vertices": np.asarray(int(verts_xy.shape[0]), dtype=np.int64),
            "n_faces": np.asarray(int(faces.shape[0]), dtype=np.int64),
            "inference_time_sec": np.asarray(float(t1 - t0), dtype=np.float64),
            "inference_time_ms": np.asarray(float(1000.0 * (t1 - t0)), dtype=np.float64),
            "E": np.asarray(_scalar(sample.get("E"), 1000.0), dtype=np.float64),
            "nu": np.asarray(_scalar(sample.get("nu"), 0.3), dtype=np.float64),
            "thickness": np.asarray(_scalar(sample.get("thickness"), 0.5), dtype=np.float64),
            "lx": np.asarray(_scalar(sample.get("lx"), 1.0), dtype=np.float64),
            "ly": np.asarray(_scalar(sample.get("ly"), 1.0), dtype=np.float64),
            "bbox": _to_numpy(sample["bbox"], np.float32),
            "geometry_type": np.asarray(str(sample.get("geometry_type", "lshape"))),
            "load_type": np.asarray(str(sample.get("load_type", "uniform_pressure"))),
            "method_name": np.asarray(str(method_name)),
            "training_method": np.asarray(str(method_name)),
        }

        for key in ("E_elem", "nu_elem", "material_type"):
            if key in sample:
                payload[key] = np.asarray(sample[key]) if isinstance(sample[key], str) else _to_numpy(sample[key])

        save_npz_dict(no_pred_npz_path(paths, tag), payload, overwrite=True)
        saved.append(tag)

    print(f"[FNO Export] Saved {len(saved)} prediction files to {paths['no_prediction_root']}")
    return saved


def run_train_lshape_grfm_up_fno(
    *,
    root: Optional[str | Path] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    case_name = "lshape_grfm_up"
    cfg = dict(DEFAULT_FNO_CONFIG)
    if config is not None:
        cfg.update(config)

    method_name = str(cfg["method_name"])
    paths = build_energy_output_paths(
        case_name=case_name,
        root=root,
        method_name=method_name,
        create=True,
    )

    seed = int(cfg["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[FNO] device={device}")
    print(f"[FNO] output root={paths['no_output_root']}")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    train_samples, test_samples, manifest = load_dataset(case_name=case_name, root=root)
    train_tags = load_train_tags(case_name=case_name, root=root)
    test_tags = load_test_tags(case_name=case_name, root=root)

    train_ds = LShapeFNOSupervisedDataset(
        case_name=case_name,
        samples=train_samples,
        tags=train_tags,
        root=root,
    )
    test_ds = LShapeFNOSupervisedDataset(
        case_name=case_name,
        samples=test_samples,
        tags=test_tags,
        root=root,
        target_mean=train_ds.target_mean,
        target_std=train_ds.target_std,
    )

    train_loader = DataLoader(train_ds, batch_size=int(cfg["batch_size"]), shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=int(cfg["batch_size"]), shuffle=False)

    in_channels = int(train_ds[0]["x"].shape[0])
    model = FNO2d(
        in_channels=in_channels,
        out_channels=3,
        width=int(cfg["width"]),
        modes_x=int(cfg["modes_x"]),
        modes_y=int(cfg["modes_y"]),
        depth=int(cfg["depth"]),
        hidden_channels=int(cfg["hidden_channels"]),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
    )
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
    training_total_t0 = time.perf_counter()

    for epoch in range(1, int(cfg["epochs"]) + 1):
        epoch_total_t0 = time.perf_counter()
        model.train()
        train_total = 0.0
        n_batch = 0
        train_t0 = time.perf_counter()

        for batch in train_loader:
            x = batch["x"].to(device)
            target = batch["target"].to(device)
            mask = batch["mask"].to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = masked_mse(pred, target, mask)
            loss.backward()

            grad_clip = cfg.get("grad_clip_norm", None)
            if grad_clip is not None and float(grad_clip) > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))

            optimizer.step()
            train_total += float(loss.item())
            n_batch += 1

        train_epoch_time_sec = float(time.perf_counter() - train_t0)
        train_loss = train_total / max(n_batch, 1)
        test_loss = evaluate(model, test_loader, device)
        scheduler.step(test_loss)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_epoch = epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "target_mean": train_ds.target_mean,
                    "target_std": train_ds.target_std,
                    "config": cfg,
                },
                paths["best_model_path"],
            )

        lr_now = float(optimizer.param_groups[0]["lr"])
        history["epoch"].append(float(epoch))
        history["train_loss"].append(float(train_loss))
        history["test_loss"].append(float(test_loss))
        history["lr"].append(lr_now)

        fem_msg = ""
        do_fem_eval = (
            epoch == 1
            or epoch == int(cfg["epochs"])
            or (int(cfg.get("eval_fem_every", 0)) > 0 and epoch % int(cfg["eval_fem_every"]) == 0)
        )
        if do_fem_eval:
            test_metric_stats = evaluate_fno_fem_metrics(
                model=model,
                dataset=test_ds,
                loader=test_loader,
                device=device,
            )
            train_metric_stats = evaluate_fno_fem_metrics(
                model=model,
                dataset=train_ds,
                loader=DataLoader(train_ds, batch_size=int(cfg["batch_size"]), shuffle=False),
                device=device,
            )

            fem_error_history["epoch"].append(float(epoch))
            _append_metric_stats(
                history=fem_error_history,
                prefix="test",
                stats=test_metric_stats,
                legacy=True,
            )
            _append_metric_stats(
                history=fem_error_history,
                prefix="train",
                stats=train_metric_stats,
                legacy=False,
            )
            fem_error_history["n_metric_cases"].append(float(test_metric_stats["n_metric_cases"]))

            fem_msg = (
                f" | Test_L2={test_metric_stats['l2_error_mean']:.4e}±{test_metric_stats['l2_error_std']:.2e}"
                f" | Test_Theta={test_metric_stats['theta_l2_error_mean']:.4e}±{test_metric_stats['theta_l2_error_std']:.2e}"
                f" | Train_L2={train_metric_stats['l2_error_mean']:.4e}±{train_metric_stats['l2_error_std']:.2e}"
            )

        epoch_total_time_sec = float(time.perf_counter() - epoch_total_t0)
        history["train_epoch_time_sec"].append(train_epoch_time_sec)
        history["epoch_total_time_sec"].append(epoch_total_time_sec)

        print(
            f"Epoch {epoch:04d}/{int(cfg['epochs']):04d} | "
            f"lr={lr_now:.3e} | train_mse={train_loss:.6e} | "
            f"test_mse={test_loss:.6e}"
            f"{fem_msg}"
            f" | train_time={train_epoch_time_sec:.2f}s"
            f" | epoch_time={epoch_total_time_sec:.2f}s"
        )

    training_total_time_sec = float(time.perf_counter() - training_total_t0)

    torch.save(
        {
            "model_state": model.state_dict(),
            "target_mean": train_ds.target_mean,
            "target_std": train_ds.target_std,
            "config": cfg,
        },
        paths["final_model_path"],
    )

    _save_history(
        history,
        Path(paths["training_history_path"]),
        Path(paths["no_output_root"]) / "training_history.csv",
    )
    _save_loss_curve(history, Path(paths["loss_curve_path"]))
    if fem_error_history["epoch"]:
        _save_fem_error_outputs(paths=paths, fem_error_history=fem_error_history)

    max_memory_mb = 0.0
    if torch.cuda.is_available():
        max_memory_mb = float(torch.cuda.max_memory_allocated() / 1024**2)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    time_summary = _save_training_time_summary(
        paths=paths,
        history=history,
        total_time_sec=training_total_time_sec,
        epochs=int(cfg["epochs"]),
        method_name=method_name,
        max_memory_mb=max_memory_mb,
        n_params=n_params,
    )
    print(
        f"[Training Time] total={time_summary['training_total_time_sec']:.2f}s "
        f"({time_summary['training_total_time_min']:.2f} min), "
        f"Memory={max_memory_mb:.2f} MB, Params={n_params}, "
        f"mean epoch={time_summary['epoch_total_time_mean_sec']:.2f}s"
    )

    if Path(paths["best_model_path"]).exists():
        # PyTorch 2.6 changed torch.load's default to weights_only=True.
        # This checkpoint is produced locally by this script and also stores
        # numpy normalization arrays/config, so load the full trusted object.
        ckpt = torch.load(paths["best_model_path"], map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        print(f"[FNO] Loaded best model: epoch={best_epoch}, test_mse={best_test_loss:.6e}")

    saved_tags = export_fno_predictions(
        model=model,
        dataset=test_ds,
        case_name=case_name,
        root=root,
        method_name=method_name,
        device=device,
    )

    compare_info = compare_cases(
        case_name=case_name,
        case_tags=saved_tags,
        root=root,
        deformation_scale=float(cfg["deformation_scale"]),
        max_plot_cases=int(cfg["max_plot_cases"]),
        method_name=method_name,
    )

    print("[FNO] Finished.")
    print(f"[FNO] best_epoch={best_epoch}, best_test_mse={best_test_loss:.6e}")
    print(f"[FNO] metrics_csv={compare_info.get('metrics_csv')}")

    return {
        "case_name": case_name,
        "method_name": method_name,
        "paths": {k: str(v) for k, v in paths.items()},
        "manifest": manifest,
        "history": history,
        "fem_error_history": fem_error_history,
        "time_summary": time_summary,
        "saved_tags": saved_tags,
        "compare_info": compare_info,
    }


if __name__ == "__main__":
    run_train_lshape_grfm_up_fno()
