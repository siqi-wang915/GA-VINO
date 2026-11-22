from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[1]

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from NO.common.core import (
    MRPlateEnergyOperator,
    build_plate_operator_model,
    device,
    mr_plate_ginot_minimal_configs,
    plate_collate_fn,
)
from NO.common.node_l2_evaluator import relative_node_l2_theta, relative_node_l2_w
from NO.common.trainer import build_energy_output_paths
from NO.modules import pointnet2_utils as pnet
from shared.common.data_io import load_dataset, load_npz_dict, load_test_tags
from shared.common.paths import build_paths, fem_pred_npz_path


CASE_NAME = "hole_um_up"


def default_model_config() -> Dict[str, Any]:
    return {
        "branch_args": {
            "input_channels": 3,
            "out_c": 64,
            "width": 64,
            "d_hidden": [64, 64],
            "n_point": 64,
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
    }


class QueryDensityDataset(Dataset):
    def __init__(
        self,
        samples,
        *,
        ratio: float,
        base_ratio: float = 0.5,
        min_query_points: int = 1,
        seed: int = 1234,
    ):
        self.samples = list(samples)
        self.ratio = float(ratio)
        self.base_ratio = float(base_ratio)
        self.min_query_points = int(min_query_points)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def _random_indices(self, n: int, *, keep: int, seed: int) -> np.ndarray:
        keep = max(0, min(int(keep), int(n)))
        if keep == 0:
            return np.zeros((0,), dtype=np.int64)
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(n, size=keep, replace=False)).astype(np.int64)

    def _base_indices(self, n: int, idx: int) -> np.ndarray:
        keep = int(math.ceil(self.base_ratio * n))
        keep = max(keep, self.min_query_points)
        keep = min(keep, n)
        seed = self.seed + 1009 * idx + int(round(1000 * self.base_ratio))
        return self._random_indices(n, keep=keep, seed=seed)

    def _query_and_eval_indices(self, n: int, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        target_keep = int(math.ceil(self.ratio * n))
        target_keep = max(target_keep, self.min_query_points)
        target_keep = min(target_keep, n)

        base_ids = self._base_indices(n, idx)
        base_keep = int(base_ids.shape[0])

        if target_keep <= base_keep:
            seed = self.seed + 1009 * idx + int(round(1000 * self.ratio))
            base_pos = self._random_indices(base_keep, keep=target_keep, seed=seed)
            query_ids = base_ids[base_pos]
            eval_positions = np.arange(query_ids.shape[0], dtype=np.int64)
            eval_base_positions = base_pos
        else:
            extra_keep = target_keep - base_keep
            all_ids = np.arange(n, dtype=np.int64)
            remaining_ids = np.setdiff1d(all_ids, base_ids, assume_unique=True)
            seed = self.seed + 1009 * idx + int(round(1000 * self.ratio))
            extra_pos = self._random_indices(remaining_ids.shape[0], keep=extra_keep, seed=seed)
            query_ids = np.sort(np.concatenate([base_ids, remaining_ids[extra_pos]])).astype(np.int64)
            eval_positions = np.searchsorted(query_ids, base_ids).astype(np.int64)
            eval_base_positions = np.arange(base_keep, dtype=np.int64)

        return query_ids, eval_positions, eval_base_positions

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = dict(self.samples[idx])

        query = torch.as_tensor(sample["query"], dtype=torch.float32)
        query_ids_np, eval_positions_np, eval_base_positions_np = self._query_and_eval_indices(
            int(query.shape[0]),
            idx,
        )
        query_ids = torch.as_tensor(query_ids_np, dtype=torch.long)
        sample["query_keep_ids"] = query_ids
        sample["eval_query_positions"] = torch.as_tensor(eval_positions_np, dtype=torch.long)
        sample["eval_base_positions"] = torch.as_tensor(eval_base_positions_np, dtype=torch.long)
        sample["base_query_keep_ids"] = torch.as_tensor(self._base_indices(int(query.shape[0]), idx), dtype=torch.long)
        sample["query_source_n"] = torch.as_tensor([int(query.shape[0])], dtype=torch.long)

        for key in ("query", "query_net", "pressure", "pressure_net", "load_field", "bc_factor"):
            if key not in sample:
                continue

            arr = torch.as_tensor(sample[key], dtype=torch.float32)
            if arr.ndim == 0 or int(arr.shape[0]) != int(query.shape[0]):
                continue

            sample[key] = arr[query_ids]

        return sample

    def actual_density_mean(self) -> float:
        vals = []
        for i, sample in enumerate(self.samples):
            n = int(torch.as_tensor(sample["query"]).shape[0])
            query_ids, _, _ = self._query_and_eval_indices(n, i)
            keep = len(query_ids)
            vals.append(keep / max(n, 1))
        return float(np.mean(vals)) if vals else float("nan")

    def eval_density_mean(self) -> float:
        vals = []
        for i, sample in enumerate(self.samples):
            n = int(torch.as_tensor(sample["query"]).shape[0])
            _, eval_positions, _ = self._query_and_eval_indices(n, i)
            vals.append(len(eval_positions) / max(n, 1))
        return float(np.mean(vals)) if vals else float("nan")


def query_density_collate_fn(
    batch: List[Dict[str, Any]],
    *,
    padding_value: float,
) -> Dict[str, Any]:
    out = plate_collate_fn(batch, padding_value=padding_value)
    query_ids = [
        torch.as_tensor(s["query_keep_ids"], dtype=torch.long)
        for s in batch
    ]
    eval_positions = [
        torch.as_tensor(s["eval_query_positions"], dtype=torch.long)
        for s in batch
    ]
    eval_base_positions = [
        torch.as_tensor(s["eval_base_positions"], dtype=torch.long)
        for s in batch
    ]
    base_query_ids = [
        torch.as_tensor(s["base_query_keep_ids"], dtype=torch.long)
        for s in batch
    ]
    out["query_keep_ids"] = pad_sequence(query_ids, batch_first=True, padding_value=-1)
    out["eval_query_positions"] = pad_sequence(eval_positions, batch_first=True, padding_value=-1)
    out["eval_base_positions"] = pad_sequence(eval_base_positions, batch_first=True, padding_value=-1)
    out["base_query_keep_ids"] = pad_sequence(base_query_ids, batch_first=True, padding_value=-1)
    out["query_source_n"] = torch.stack(
        [torch.as_tensor(s["query_source_n"], dtype=torch.long).reshape(1) for s in batch],
        dim=0,
    )
    return out


def load_model_config(no_output_root: Path) -> Dict[str, Any]:
    metadata_path = no_output_root / "run_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        if metadata.get("model_config") is not None:
            return metadata["model_config"]
    return default_model_config()


def build_operator_from_checkpoint(
    *,
    root: Optional[str | Path],
    checkpoint_kind: str,
) -> tuple[MRPlateEnergyOperator, Dict[str, Path], Dict[str, Any]]:
    paths = build_energy_output_paths(
        case_name=CASE_NAME,
        root=root,
        method_name=None,
        create=True,
    )

    padding_value = float(mr_plate_ginot_minimal_configs()["padding_value"])
    model_config = load_model_config(Path(paths["no_output_root"]))

    base_model = build_plate_operator_model(
        padding_value=padding_value,
        model_config=model_config,
    ).to(device)

    operator_model = MRPlateEnergyOperator(
        model=base_model,
        padding_value=padding_value,
    ).to(device)

    checkpoint_path = (
        paths["best_model_path"]
        if checkpoint_kind == "best"
        else paths["final_model_path"]
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    operator_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    operator_model.eval()

    return operator_model, paths, model_config


@torch.no_grad()
def evaluate_density(
    *,
    operator_model: MRPlateEnergyOperator,
    data_loader: DataLoader,
    case_tags: List[str],
    root: Optional[str | Path],
    padding_value: float,
    plot_case_index: Optional[int] = None,
    plot_density_ratio: float = 1.0,
    plot_seed: int = 1234,
) -> tuple[List[Dict[str, float]], Optional[Dict[str, Any]]]:
    paths = build_paths(CASE_NAME, root=root, create=True)

    rows: List[Dict[str, float]] = []
    plot_record: Optional[Dict[str, Any]] = None
    global_index = 0

    for batch in data_loader:
        y_pred = operator_model.predict(batch)
        query = batch["query"].to(device)
        q_valid = query[:, :, 0] != padding_value

        B = query.shape[0]

        for b in range(B):
            if global_index >= len(case_tags):
                break

            tag = case_tags[global_index]
            case_index = global_index
            global_index += 1

            n_q = int(q_valid[b].sum().item())
            q_no = y_pred[b, :n_q].detach().cpu().numpy()
            query_xyz = batch["query"][b, :n_q].detach().cpu().numpy()
            if "query_keep_ids" in batch:
                query_ids = batch["query_keep_ids"][b, :n_q].detach().cpu().numpy().astype(np.int64)
            else:
                query_ids = np.arange(n_q, dtype=np.int64)

            if "eval_query_positions" in batch:
                eval_positions_all = batch["eval_query_positions"][b].detach().cpu().numpy().astype(np.int64)
                eval_positions = eval_positions_all[eval_positions_all >= 0]
            else:
                eval_positions = np.arange(n_q, dtype=np.int64)

            if "eval_base_positions" in batch:
                eval_base_positions_all = batch["eval_base_positions"][b].detach().cpu().numpy().astype(np.int64)
                eval_base_positions = eval_base_positions_all[eval_base_positions_all >= 0]
            else:
                eval_base_positions = eval_positions

            if "base_query_keep_ids" in batch:
                base_ids_all = batch["base_query_keep_ids"][b].detach().cpu().numpy().astype(np.int64)
                base_ids = base_ids_all[base_ids_all >= 0]
            else:
                base_ids = query_ids

            if "query_source_n" in batch:
                query_source_n = int(batch["query_source_n"][b].reshape(-1)[0].item())
            else:
                max_query_id = int(np.max(query_ids)) if query_ids.size else -1
                query_source_n = int(max(max_query_id + 1, n_q))

            eval_full_ids = query_ids[eval_positions]
            q_no_eval = q_no[eval_positions]
            query_xyz_eval = query_xyz[eval_positions]

            fem_data = load_npz_dict(fem_pred_npz_path(paths, tag))
            q_fem_full = np.asarray(fem_data["fem_local"], dtype=np.float64)
            if int(q_fem_full.shape[0]) == query_source_n:
                q_fem = q_fem_full[eval_full_ids]
            elif int(q_fem_full.shape[0]) == int(base_ids.shape[0]):
                q_fem = q_fem_full[eval_base_positions]
            else:
                raise ValueError(
                    f"Cannot align FEM data for {tag}: fem_local has "
                    f"{q_fem_full.shape[0]} rows, expected either full query "
                    f"count {query_source_n} or 50% base query count {base_ids.shape[0]}."
                )

            rows.append(
                {
                    "case_index": float(case_index),
                    "node_l2_w": relative_node_l2_w(q_no_eval, q_fem),
                    "node_l2_theta": relative_node_l2_theta(q_no_eval, q_fem),
                    "n_query": float(n_q),
                    "n_eval_query": float(q_no_eval.shape[0]),
                }
            )

            if plot_case_index is not None and case_index == int(plot_case_index):
                n_plot = min(q_no_eval.shape[0], q_fem.shape[0], query_xyz_eval.shape[0])
                keep = n_plot
                plot_ids = np.arange(n_plot, dtype=np.int64)

                w_no = np.asarray(q_no_eval[plot_ids, 0], dtype=np.float64)
                w_fem = np.asarray(q_fem[plot_ids, 0], dtype=np.float64)
                theta_no = np.abs(np.asarray(q_no_eval[plot_ids, 1], dtype=np.float64)) + np.abs(
                    np.asarray(q_no_eval[plot_ids, 2], dtype=np.float64)
                )
                theta_fem = np.abs(np.asarray(q_fem[plot_ids, 1], dtype=np.float64)) + np.abs(
                    np.asarray(q_fem[plot_ids, 2], dtype=np.float64)
                )
                plot_record = {
                    "case_index": int(case_index),
                    "case_tag": tag,
                    "xy": np.asarray(query_xyz_eval[plot_ids, :2], dtype=np.float64),
                    "w_no": w_no,
                    "w_fem": w_fem,
                    "w_error": w_no - w_fem,
                    "theta_no": theta_no,
                    "theta_fem": theta_fem,
                    "theta_error": theta_no - theta_fem,
                    "n_plot_points": int(keep),
                }

    return rows, plot_record


def save_outputs(
    *,
    out_root: Path,
    summary_rows: List[Dict[str, float]],
    case_rows: List[Dict[str, float]],
    density_plot_records: Optional[List[Dict[str, Any]]] = None,
) -> None:
    out_root.mkdir(parents=True, exist_ok=True)

    summary_csv = out_root / "pc_density_node_l2_summary.csv"
    case_csv = out_root / "pc_density_node_l2_cases.csv"
    npz_path = out_root / "pc_density_node_l2_results.npz"
    w_boxplot_path = out_root / "pc_density_node_l2_w_boxplot.png"
    theta_boxplot_path = out_root / "pc_density_node_l2_theta_boxplot.png"
    density_cloud_path = out_root / "w_theta_density_cloud_6x10.png"

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    with open(case_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(case_rows[0].keys()))
        writer.writeheader()
        writer.writerows(case_rows)

    np.savez_compressed(
        npz_path,
        **{k: np.asarray([r[k] for r in summary_rows], dtype=np.float64) for k in summary_rows[0]},
    )

    import matplotlib.pyplot as plt

    plot_density_l2_boxplot(
        case_rows=case_rows,
        metric_key="node_l2_w",
        color="#9ecae1",
        out_path=w_boxplot_path,
    )
    plot_density_l2_boxplot(
        case_rows=case_rows,
        metric_key="node_l2_theta",
        color="#f4a6b7",
        out_path=theta_boxplot_path,
    )

    if density_plot_records:
        plot_w_density_cloud(
            records=density_plot_records,
            out_path=density_cloud_path,
        )

    print(f"[Saved] {summary_csv}")
    print(f"[Saved] {case_csv}")
    print(f"[Saved] {npz_path}")
    print(f"[Saved] {w_boxplot_path}")
    print(f"[Saved] {theta_boxplot_path}")
    if density_plot_records:
        print(f"[Saved] {density_cloud_path}")


def plot_density_l2_boxplot(
    *,
    case_rows: List[Dict[str, float]],
    metric_key: str,
    color: str,
    out_path: str | Path,
) -> None:
    import matplotlib.pyplot as plt

    densities = sorted({float(r["target_density"]) for r in case_rows})
    labels = [f"{100.0 * d:.0f}" for d in densities]
    data = [
        np.asarray(
            [r[metric_key] for r in case_rows if math.isclose(float(r["target_density"]), d)],
            dtype=np.float64,
        )
        for d in densities
    ]

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    bp = ax.boxplot(
        data,
        tick_labels=labels,
        showmeans=True,
        patch_artist=True,
        widths=0.58,
        medianprops={
            "color": "black",
            "linewidth": 1.6,
        },
        meanprops={
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markeredgewidth": 1.2,
            "markersize": 5,
        },
        boxprops={
            "linewidth": 1.2,
            "edgecolor": "black",
        },
        whiskerprops={
            "linewidth": 1.2,
            "color": "black",
        },
        capprops={
            "linewidth": 1.2,
            "color": "black",
        },
        flierprops={
            "marker": "o",
            "markersize": 3.5,
            "markerfacecolor": "none",
            "markeredgecolor": color,
            "alpha": 0.85,
        },
    )

    for patch in bp["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.70)
        patch.set_edgecolor("black")

    ax.set_xlabel("Query point cloud density (%)", fontsize=14)
    ax.set_ylabel(r"Relative $L^2$ error", fontsize=14)
    ax.set_yscale("log")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)

    for spine in ax.spines.values():
        spine.set_linewidth(1.1)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _finite_range(values: List[np.ndarray]) -> tuple[float, float]:
    arr = np.concatenate([np.asarray(v, dtype=np.float64).reshape(-1) for v in values])
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 1.0
    vmin = float(np.min(arr))
    vmax = float(np.max(arr))
    if abs(vmax - vmin) < 1.0e-14:
        pad = max(abs(vmin), 1.0) * 1.0e-6
        return vmin - pad, vmax + pad
    return vmin, vmax


def _scatter_size(n_points: int) -> float:
    if n_points >= 10000:
        return 1.2
    if n_points >= 5000:
        return 1.8
    if n_points >= 2000:
        return 2.6
    return 4.0


def plot_w_density_cloud(
    *,
    records: List[Dict[str, Any]],
    out_path: str | Path,
) -> None:
    import matplotlib.pyplot as plt

    records = sorted(records, key=lambda r: float(r["target_density"]))
    n_cols = len(records)
    if n_cols == 0:
        return

    w_vmin, w_vmax = _finite_range(
        [r["plot"]["w_no"] for r in records] + [r["plot"]["w_fem"] for r in records]
    )
    w_err_abs = max(
        abs(_finite_range([r["plot"]["w_error"] for r in records])[0]),
        abs(_finite_range([r["plot"]["w_error"] for r in records])[1]),
    )
    if w_err_abs < 1.0e-14:
        w_err_abs = 1.0

    theta_vmin, theta_vmax = _finite_range(
        [r["plot"]["theta_no"] for r in records] + [r["plot"]["theta_fem"] for r in records]
    )
    theta_err_abs = max(
        abs(_finite_range([r["plot"]["theta_error"] for r in records])[0]),
        abs(_finite_range([r["plot"]["theta_error"] for r in records])[1]),
    )
    if theta_err_abs < 1.0e-14:
        theta_err_abs = 1.0

    fig_width = max(1.8 * n_cols, 12.0)
    fig, axes = plt.subplots(
        6,
        n_cols,
        figsize=(fig_width, 11.0),
        squeeze=False,
        constrained_layout=True,
    )

    w_no_mappable = None
    w_fem_mappable = None
    w_err_mappable = None
    theta_no_mappable = None
    theta_fem_mappable = None
    theta_err_mappable = None

    for j, rec in enumerate(records):
        plot = rec["plot"]
        xy = np.asarray(plot["xy"], dtype=np.float64)
        x = xy[:, 0]
        y = xy[:, 1]
        point_size = _scatter_size(int(xy.shape[0]))

        title = f"{100.0 * float(rec['target_density']):.0f}%"
        n_query_plot = int(plot.get("n_plot_points", xy.shape[0]))

        w_no_mappable = axes[0, j].scatter(
            x,
            y,
            c=plot["w_no"],
            s=point_size,
            cmap="coolwarm",
            vmin=w_vmin,
            vmax=w_vmax,
            linewidths=0,
        )
        axes[0, j].set_title(f"{title}\n$N_q$={n_query_plot}", fontsize=9)


        w_fem_mappable = axes[1, j].scatter(
            x,
            y,
            c=plot["w_fem"],
            s=point_size,
            cmap="coolwarm",
            vmin=w_vmin,
            vmax=w_vmax,
            linewidths=0,
        )

        w_err_mappable = axes[2, j].scatter(
            x,
            y,
            c=plot["w_error"],
            s=point_size,
            cmap="coolwarm",
            vmin=-w_err_abs,
            vmax=w_err_abs,
            linewidths=0,
        )

        theta_no_mappable = axes[3, j].scatter(
            x,
            y,
            c=plot["theta_no"],
            s=point_size,
            cmap="coolwarm",
            vmin=theta_vmin,
            vmax=theta_vmax,
            linewidths=0,
        )

        theta_fem_mappable = axes[4, j].scatter(
            x,
            y,
            c=plot["theta_fem"],
            s=point_size,
            cmap="coolwarm",
            vmin=theta_vmin,
            vmax=theta_vmax,
            linewidths=0,
        )

        theta_err_mappable = axes[5, j].scatter(
            x,
            y,
            c=plot["theta_error"],
            s=point_size,
            cmap="coolwarm",
            vmin=-theta_err_abs,
            vmax=theta_err_abs,
            linewidths=0,
        )

        for i in range(6):
            axes[i, j].set_aspect("equal", adjustable="box")
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            axes[i, j].set_frame_on(False)
            for spine in axes[i, j].spines.values():
                spine.set_visible(False)

    axes[0, 0].set_ylabel("NO w", fontsize=11)
    axes[1, 0].set_ylabel("FEM w", fontsize=11)
    axes[2, 0].set_ylabel("NO - FEM", fontsize=11)
    axes[3, 0].set_ylabel(r"NO $|\theta_x|+|\theta_y|$", fontsize=11)
    axes[4, 0].set_ylabel(r"FEM $|\theta_x|+|\theta_y|$", fontsize=11)
    axes[5, 0].set_ylabel(r"$\theta$ Error", fontsize=11)

    case_tag = records[0]["plot"]["case_tag"]
    #fig.suptitle(
    #    f"hole_um_up point-cloud density w field, case={case_tag}",
    #    fontsize=13,
    #)

    if w_no_mappable is not None:
        fig.colorbar(w_no_mappable, ax=axes[0, :], shrink=0.72, pad=0.01, fraction=0.025, label="w")
    if w_fem_mappable is not None:
        fig.colorbar(w_fem_mappable, ax=axes[1, :], shrink=0.72, pad=0.01, fraction=0.025, label="w")
    if w_err_mappable is not None:
        fig.colorbar(w_err_mappable, ax=axes[2, :], shrink=0.72, pad=0.01, fraction=0.025, label="w error")
    if theta_no_mappable is not None:
        fig.colorbar(theta_no_mappable, ax=axes[3, :], shrink=0.72, pad=0.01, fraction=0.025, label=r"$|\theta_x|+|\theta_y|$")
    if theta_fem_mappable is not None:
        fig.colorbar(theta_fem_mappable, ax=axes[4, :], shrink=0.72, pad=0.01, fraction=0.025, label=r"$|\theta_x|+|\theta_y|$")
    if theta_err_mappable is not None:
        fig.colorbar(theta_err_mappable, ax=axes[5, :], shrink=0.72, pad=0.01, fraction=0.025, label=r"$\theta$ error")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def parse_ratios(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate trained hole_um_up model under 10%-100% query point-cloud density."
    )
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--checkpoint", choices=["best", "final"], default="best")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--base-ratio", type=float, default=0.5)
    parser.add_argument("--plot-case-index", type=int, default=3)
    parser.add_argument(
        "--ratios",
        type=str,
        default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
    )
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    root = Path(args.root).resolve() if args.root is not None else _PROJECT_ROOT
    padding_value = float(mr_plate_ginot_minimal_configs()["padding_value"])

    operator_model, method_paths, model_config = build_operator_from_checkpoint(
        root=root,
        checkpoint_kind=args.checkpoint,
    )

    _, test_samples, _ = load_dataset(case_name=CASE_NAME, root=root)
    test_tags = load_test_tags(case_name=CASE_NAME, root=root)

    out_root = (
        Path(args.out_dir)
        if args.out_dir is not None
        else Path(method_paths["no_output_root"]) / "pc_density_node_l2"
    )

    summary_rows: List[Dict[str, float]] = []
    all_case_rows: List[Dict[str, float]] = []
    density_plot_records: List[Dict[str, Any]] = []

    for ratio in parse_ratios(args.ratios):
        pnet.CACHE_SAMPLE_AND_GROUP_INDECIES.clear()

        ds = QueryDensityDataset(
            test_samples,
            ratio=ratio,
            base_ratio=float(args.base_ratio),
            min_query_points=1,
            seed=int(args.seed),
        )
        loader = DataLoader(
            ds,
            batch_size=int(args.batch_size),
            shuffle=False,
            collate_fn=lambda b: query_density_collate_fn(b, padding_value=padding_value),
        )

        rows, plot_record = evaluate_density(
            operator_model=operator_model,
            data_loader=loader,
            case_tags=test_tags,
            root=root,
            padding_value=padding_value,
            plot_case_index=int(args.plot_case_index),
            plot_density_ratio=float(ratio),
            plot_seed=int(args.seed),
        )

        w = np.asarray([r["node_l2_w"] for r in rows], dtype=np.float64)
        th = np.asarray([r["node_l2_theta"] for r in rows], dtype=np.float64)

        summary_rows.append(
            {
                "target_density": float(ratio),
                "actual_density_mean": float(ds.actual_density_mean()),
                "eval_density_mean": float(ds.eval_density_mean()),
                "node_l2_w_mean": float(np.mean(w)),
                "node_l2_w_std": float(np.std(w)),
                "node_l2_theta_mean": float(np.mean(th)),
                "node_l2_theta_std": float(np.std(th)),
                "n_cases": float(len(rows)),
            }
        )

        for r in rows:
            rr = dict(r)
            rr["target_density"] = float(ratio)
            rr["actual_density_mean"] = float(ds.actual_density_mean())
            rr["eval_density_mean"] = float(ds.eval_density_mean())
            all_case_rows.append(rr)

        if plot_record is not None:
            density_plot_records.append(
                {
                    "target_density": float(ratio),
                    "actual_density_mean": float(ds.actual_density_mean()),
                    "eval_density_mean": float(ds.eval_density_mean()),
                    "plot": plot_record,
                }
            )

        print(
            f"[Density {ratio:.1f}] "
            f"actual={ds.actual_density_mean():.4f} | "
            f"eval={ds.eval_density_mean():.4f} | "
            f"w={np.mean(w):.6e} +/- {np.std(w):.2e} | "
            f"theta={np.mean(th):.6e} +/- {np.std(th):.2e}"
        )

    save_outputs(
        out_root=out_root,
        summary_rows=summary_rows,
        case_rows=all_case_rows,
        density_plot_records=density_plot_records,
    )


if __name__ == "__main__":
    main()



