from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from NO.common.core import device
from shared.common.data_io import load_npz_field
from shared.common.paths import build_paths, fem_pred_npz_path


def relative_node_l2_w(q_no: np.ndarray, q_fem: np.ndarray) -> float:
    q_no = np.asarray(q_no, dtype=np.float64)
    q_fem = np.asarray(q_fem, dtype=np.float64)

    n = min(q_no.shape[0], q_fem.shape[0])
    w_no = q_no[:n, 0].reshape(-1)
    w_fem = q_fem[:n, 0].reshape(-1)

    num = np.linalg.norm(w_no - w_fem)
    den = np.linalg.norm(w_fem)
    return float(num / max(den, 1.0e-30))


def relative_node_l2_theta(q_no: np.ndarray, q_fem: np.ndarray) -> float:
    q_no = np.asarray(q_no, dtype=np.float64)
    q_fem = np.asarray(q_fem, dtype=np.float64)

    n = min(q_no.shape[0], q_fem.shape[0])
    theta_no = q_no[:n, 1:3]
    theta_fem = q_fem[:n, 1:3]

    num = np.linalg.norm(theta_no - theta_fem)
    den = np.linalg.norm(theta_fem)
    return float(num / max(den, 1.0e-30))


class NodeL2FEMMetricEvaluator:
    """
    FEM evaluator using node-wise relative L2 errors only.

    It computes:
        l2_error        = node-wise relative L2 error of w
        theta_l2_error = node-wise relative L2 error of [theta_x, theta_y]

    Mesh-dependent metrics are returned as NaN for compatibility with the
    existing trainer, but are not computed.
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

        # Node-L2 metrics only use fem_local.  Avoid retaining the optional,
        # quadrature-level FEM resultants in RAM.
        self.fem_cache: Dict[str, np.ndarray] = {}
        self.available_tags: List[str] = []

        self._load_cache()

    def _load_cache(self) -> None:
        for tag in self.case_tags:
            fpath = fem_pred_npz_path(self.paths, tag)

            if not fpath.exists():
                print(f"[NodeL2FEMMetricEvaluator] Missing FEM for {tag}, skipped.")
                continue

            self.fem_cache[tag] = np.asarray(
                load_npz_field(fpath, "fem_local"), dtype=np.float64
            )
            self.available_tags.append(tag)

        if not self.available_tags:
            raise RuntimeError(
                "[NodeL2FEMMetricEvaluator] No available FEM results found. "
                "Please run stage1 first."
            )

        print(
            f"[NodeL2FEMMetricEvaluator] Loaded {len(self.available_tags)} FEM cases "
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
                q_fem = self.fem_cache[tag]

                rows.append(
                    {
                        "l2_error": relative_node_l2_w(q_no, q_fem),
                        "theta_l2_error": relative_node_l2_theta(q_no, q_fem),
                    }
                )

        if not rows:
            return self._empty_stats()

        out: Dict[str, float] = {}

        for key in ("l2_error", "theta_l2_error"):
            arr = np.asarray([r[key] for r in rows], dtype=np.float64)
            out[f"{key}_mean"] = float(np.mean(arr))
            out[f"{key}_std"] = float(np.std(arr))

        out["h1_semi_error_mean"] = float("nan")
        out["h1_semi_error_std"] = float("nan")
        out["energy_error_mean"] = float("nan")
        out["energy_error_std"] = float("nan")
        out["curvature_error_mean"] = float("nan")
        out["curvature_error_std"] = float("nan")
        out["n_metric_cases"] = float(len(rows))

        return out

    @staticmethod
    def _empty_stats() -> Dict[str, float]:
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
