from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from NO.common.core import device
from NO.common.trainer import build_energy_output_paths
from shared.common.data_io import load_npz_dict, load_npz_field, save_npz_dict
from shared.common.paths import (
    build_paths,
    case_npz_path,
    fem_pred_npz_path,
    no_pred_npz_path,
)
from shared.compare.metrics import compute_metrics_from_arrays


class FEMMetricEvaluator:
    """
    Compute FEM-based errors during NO training.

    For each epoch:
        model predicts current test samples
        evaluator compares prediction with cached FEM results
        returns mean/std of L2, H1 semi, and Energy errors
    """

    def __init__(
        self,
        *,
        case_name: str,
        case_tags: List[str],
        root: Optional[str | Path] = None,
        padding_value: float = -999.0,
    ):
        self.case_name = case_name
        self.case_tags = list(case_tags)
        self.paths = build_paths(case_name, root=root, create=True)
        self.padding_value = float(padding_value)

        self.case_cache: Dict[str, Dict[str, Any]] = {}
        # Full FEM archives may include large optional resultants.  These
        # metrics require only the nodal supervised solution.
        self.fem_cache: Dict[str, np.ndarray] = {}
        self.available_tags: List[str] = []

        self._load_cache()

    def _load_cache(self):
        for tag in self.case_tags:
            cpath = case_npz_path(self.paths, tag)
            fpath = fem_pred_npz_path(self.paths, tag)

            if not cpath.exists() or not fpath.exists():
                print(f"[FEMMetricEvaluator] Missing case/FEM for {tag}, skipped.")
                continue

            case_data = load_npz_dict(cpath)
            fem_local = np.asarray(load_npz_field(fpath, "fem_local"), dtype=np.float64)

            self.case_cache[tag] = case_data
            self.fem_cache[tag] = fem_local
            self.available_tags.append(tag)

        if not self.available_tags:
            raise RuntimeError(
                "[FEMMetricEvaluator] No available FEM results found. "
                "Please run stage1 first."
            )

        print(
            f"[FEMMetricEvaluator] Loaded {len(self.available_tags)} FEM cases "
            f"for case='{self.case_name}'."
        )

    @torch.no_grad()
    def evaluate_model(
        self,
        *,
        operator_model,
        data_loader,
    ) -> Dict[str, float]:
        operator_model.eval()

        rows: List[Dict[str, float]] = []
        global_index = 0

        for batch in data_loader:
            y_pred = operator_model.predict(batch)

            query = batch["query"].to(device)
            q_valid = query[:, :, 0] != self.padding_value

            B = query.shape[0]

            for b in range(B):
                if global_index >= len(self.case_tags):
                    break

                tag = self.case_tags[global_index]
                global_index += 1

                if tag not in self.available_tags:
                    continue

                n_q = int(q_valid[b].sum().item())
                q_no = y_pred[b, :n_q].detach().cpu().numpy()

                case_data = self.case_cache[tag]
                q_fem = self.fem_cache[tag]
                verts_xy = np.asarray(case_data["verts_xy"], dtype=np.float64)
                faces = np.asarray(case_data["faces"], dtype=np.int64)

                E = float(np.asarray(case_data["E"]).reshape(-1)[0])
                nu = float(np.asarray(case_data["nu"]).reshape(-1)[0])
                thickness = float(np.asarray(case_data["thickness"]).reshape(-1)[0])
                E_elem = np.asarray(case_data["E_elem"], dtype=np.float64).reshape(-1) if "E_elem" in case_data else None
                nu_elem = np.asarray(case_data["nu_elem"], dtype=np.float64).reshape(-1) if "nu_elem" in case_data else None

                metrics = compute_metrics_from_arrays(
                    verts_xy=verts_xy,
                    faces=faces,
                    q_no=q_no,
                    q_fem=q_fem,
                    E=E,
                    nu=nu,
                    thickness=thickness,
                    E_elem=E_elem,
                    nu_elem=nu_elem,
                )

                rows.append(metrics)

        if not rows:
            return {
                "l2_error_mean": float("nan"),
                "l2_error_std": float("nan"),
                "h1_semi_error_mean": float("nan"),
                "h1_semi_error_std": float("nan"),
                "energy_error_mean": float("nan"),
                "energy_error_std": float("nan"),
                "theta_l2_error_mean": float("nan"),
                "theta_l2_error_std": float("nan"),
                "curvature_error_mean": float("nan"),
                "curvature_error_std": float("nan"),
                "n_metric_cases": 0.0,
            }

        out: Dict[str, float] = {}

        for key in ("l2_error", "h1_semi_error", "energy_error", "theta_l2_error", "curvature_error"):
            arr = np.asarray([r[key] for r in rows], dtype=np.float64)
            out[f"{key}_mean"] = float(np.mean(arr))
            out[f"{key}_std"] = float(np.std(arr))

        out["n_metric_cases"] = float(len(rows))

        return out



@torch.no_grad()
def export_no_predictions(
    *,
    operator_model,
    data_loader,
    case_name: str,
    case_tags: List[str],
    root: Optional[str | Path] = None,
    padding_value: float = -999.0,
    method_name: Optional[str] = None,
) -> List[str]:
    """
    Export final NO predictions.

    If method_name is None / base, predictions are written to:
        NO/outputs/<case>/predictions/case_XXXX_no.npz

    If method_name is non-base, predictions are written to:
        NO/outputs/<case>_<method_name>/predictions/case_XXXX_no.npz

    The exported *_no.npz contains:
        - n_query_points: valid output/query points, usually all FEM nodes
        - n_geometry_points: valid points entering the Geometry Encoder
        - n_vertices / n_faces: underlying triangular mesh size
        - inference_time_sec / inference_time_ms: approximate per-sample forward time
    """
    paths = build_energy_output_paths(
        case_name=case_name,
        root=root,
        method_name=method_name,
        create=True,
    )
    method_label = str(method_name) if method_name is not None else "energy"

    operator_model.eval()

    saved_tags: List[str] = []
    global_index = 0

    for batch in data_loader:
        # Measure only the neural-operator forward inference time.
        # GPU kernels are asynchronous, so synchronize before and after timing.
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        y_pred = operator_model.predict(batch)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t1 = time.perf_counter()

        query = batch["query"].to(device)
        q_valid = query[:, :, 0] != padding_value


        pc_net = batch.get("pc_net", None)
        if pc_net is not None:
            pc_net = pc_net.to(device)
            pc_valid = pc_net[:, :, 0] != padding_value
        else:
            pc_valid = None

        B = query.shape[0]
        batch_inference_time_sec = float(t1 - t0)
        inference_time_sec_per_sample = batch_inference_time_sec / max(int(B), 1)

        for b in range(B):
            if global_index >= len(case_tags):
                break

            tag = case_tags[global_index]
            global_index += 1

            n_q = int(q_valid[b].sum().item())
            if pc_valid is not None:
                n_geometry_points = int(pc_valid[b].sum().item())
            else:
                n_geometry_points = -1

            pred_local = y_pred[b, :n_q].detach().cpu().numpy()

            query_xyz = batch["query"][b, :n_q].detach().cpu().numpy()
            verts_xy = batch["verts_xy"][b].detach().cpu().numpy()
            faces = batch["faces"][b].detach().cpu().numpy()
            pressure = batch["pressure"][b, :n_q].detach().cpu().numpy()
            bc_factor = batch["bc_factor"][b, :n_q].detach().cpu().numpy()

            E_elem = None
            nu_elem = None
            if "E_elem" in batch and batch["E_elem"][b] is not None:
                E_elem = batch["E_elem"][b].detach().cpu().numpy()
            if "nu_elem" in batch and batch["nu_elem"][b] is not None:
                nu_elem = batch["nu_elem"][b].detach().cpu().numpy()

            pred_global_disp = np.zeros((pred_local.shape[0], 3), dtype=np.float64)
            pred_global_disp[:, 2] = pred_local[:, 0]

            disp_scalar = pred_local[:, 0].copy()
            rot_scalar = np.abs(pred_local[:, 1]) + np.abs(pred_local[:, 2])


            payload = {
                "case_tag": np.asarray(tag),
                "source_sample_id": np.asarray(int(batch["sample_ids"][b].item()), dtype=np.int64),

                "query_xyz": query_xyz,
                "verts_xy": verts_xy,
                "faces": faces,
                "pressure": pressure,
                "bc_factor": bc_factor,

                "pred_local": pred_local,
                "pred_global_disp": pred_global_disp,
                "disp_scalar": disp_scalar,
                "rot_scalar": rot_scalar,

                # Per-sample problem size and NO inference timing.
                "n_query_points": np.asarray(int(n_q), dtype=np.int64),
                "n_geometry_points": np.asarray(int(n_geometry_points), dtype=np.int64),
                "n_vertices": np.asarray(int(verts_xy.shape[0]), dtype=np.int64),
                "n_faces": np.asarray(int(faces.shape[0]), dtype=np.int64),
                "inference_time_sec": np.asarray(float(inference_time_sec_per_sample), dtype=np.float64),
                "inference_time_ms": np.asarray(float(1000.0 * inference_time_sec_per_sample), dtype=np.float64),

                "E": np.asarray(float(batch["E"][b, 0].item()), dtype=np.float64),
                "nu": np.asarray(float(batch["nu"][b, 0].item()), dtype=np.float64),
                "thickness": np.asarray(float(batch["thickness"][b, 0].item()), dtype=np.float64),
                "lx": np.asarray(float(batch["lx"][b].item()), dtype=np.float64),
                "ly": np.asarray(float(batch["ly"][b].item()), dtype=np.float64),
                "bbox": batch["bbox"][b].detach().cpu().numpy(),

                "method_name": np.asarray(method_label),
                "training_method": np.asarray(method_label),

                "geometry_type": np.asarray(str(batch["geometry_type"][b])),
                "load_type": np.asarray(str(batch["load_type"][b])),
            }

            if E_elem is not None:
                payload["E_elem"] = E_elem
            if nu_elem is not None:
                payload["nu_elem"] = nu_elem
            if "material_type" in batch:
                payload["material_type"] = np.asarray(str(batch["material_type"][b]))

            save_npz_dict(
                no_pred_npz_path(paths, tag),
                payload,
                overwrite=True,
            )

            saved_tags.append(tag)

    print(f"[NO Export] method_name={method_label}")
    print(f"[NO Export] Saved {len(saved_tags)} NO prediction files to {paths['no_prediction_root']}")
    return saved_tags
