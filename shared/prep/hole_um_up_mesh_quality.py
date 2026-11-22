from __future__ import annotations

"""Prepare three fixed-size, lower-quality variants of ``hole_um_up``."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

from NO.problem.hole import save_debug_mesh_plot, save_encoder_pc_plot
from NO.problem.hole_mesh_quality import (
    QUALITY_LEVELS,
    build_quality_datasets,
    quality_case_name,
)
from shared.common.data_io import load_dataset, prepare_dataset_and_cases
from shared.common.paths import build_paths, path_dict_to_str


SOURCE_CASE_NAME = "hole_um_up"

DEFAULT_CONFIG: Dict[str, Any] = {
    "source_case": SOURCE_CASE_NAME,
    "quality_levels": list(QUALITY_LEVELS),
    "jitter_max": 0.95,
    "candidate_count": 81,
    "seed": 20260929,
    "n_debug_mesh": 5,
}


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _scalar(sample: Dict[str, Any], key: str, default=np.nan) -> float:
    if key not in sample:
        return float(default)
    array = _to_numpy(sample[key]).reshape(-1)
    return float(array[0]) if array.size else float(default)


def _mesh_rows(
    samples: Sequence[Dict[str, Any]], split: str, case_name: str
) -> list[Dict[str, Any]]:
    rows = []
    for index, sample in enumerate(samples):
        rows.append(
            {
                "case_name": case_name,
                "split": split,
                "sample_index": index,
                "sample_id": int(sample.get("sample_id", index)),
                "quality_level": int(_scalar(sample, "mesh_quality_level")),
                "jitter_amplitude": _scalar(sample, "mesh_quality_jitter_amplitude"),
                "n_vertices": int(_scalar(sample, "n_vertices")),
                "n_faces": int(_scalar(sample, "n_faces")),
                "source_n_vertices": int(_scalar(sample, "source_n_vertices")),
                "source_n_faces": int(_scalar(sample, "source_n_faces")),
                "quality_min": _scalar(sample, "quality_min"),
                "quality_p05": _scalar(sample, "quality_p05"),
                "quality_mean": _scalar(sample, "quality_mean"),
                "min_angle_min_deg": _scalar(sample, "min_angle_min_deg"),
                "min_angle_p05_deg": _scalar(sample, "min_angle_p05_deg"),
                "area_mean": _scalar(sample, "area_mean"),
                "area_cv": _scalar(sample, "area_cv"),
                "source_quality_min": _scalar(sample, "source_quality_min"),
                "source_quality_p05": _scalar(sample, "source_quality_p05"),
                "source_quality_mean": _scalar(sample, "source_quality_mean"),
            }
        )
    return rows


def _save_mesh_summary(
    *,
    data_root: Path,
    case_name: str,
    train_samples: Sequence[Dict[str, Any]],
    test_samples: Sequence[Dict[str, Any]],
) -> Dict[str, str]:
    rows = _mesh_rows(train_samples, "train", case_name) + _mesh_rows(
        test_samples, "test", case_name
    )
    csv_path = data_root / "mesh_quality_summary.csv"
    npz_path = data_root / "mesh_quality_summary.npz"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        npz_path,
        **{
            key: np.asarray(
                [row[key] for row in rows],
                dtype=str if key in ("case_name", "split") else np.float64,
            )
            for key in rows[0]
        },
    )
    return {"mesh_quality_csv": str(csv_path), "mesh_quality_npz": str(npz_path)}


def _load_source_config(root: Optional[str | Path]) -> Dict[str, Any]:
    path = build_paths(SOURCE_CASE_NAME, root=root, create=False)["dataset_config_path"]
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return dict(json.load(handle))


def prepare_hole_um_up_mesh_quality(
    *,
    root: Optional[str | Path] = None,
    overwrite: bool = True,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Prepare levels 2--4; the existing ``hole_um_up`` is level 1."""
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)
    requested_levels = tuple(int(level) for level in cfg["quality_levels"])
    if not requested_levels or any(level not in QUALITY_LEVELS for level in requested_levels):
        raise ValueError(
            f"quality_levels must be selected from {QUALITY_LEVELS}, got {requested_levels}"
        )

    train_source, test_source, source_manifest = load_dataset(
        case_name=SOURCE_CASE_NAME, root=root
    )
    train_by_level = build_quality_datasets(
        train_source,
        seed=int(cfg["seed"]),
        jitter_max=float(cfg["jitter_max"]),
        candidate_count=int(cfg["candidate_count"]),
    )
    test_by_level = build_quality_datasets(
        test_source,
        seed=int(cfg["seed"]) + 10_000_019,
        jitter_max=float(cfg["jitter_max"]),
        candidate_count=int(cfg["candidate_count"]),
    )
    source_config = _load_source_config(root)

    results: Dict[str, Any] = {}
    for level in requested_levels:
        case_name = quality_case_name(level)
        paths = build_paths(case_name, root=root, create=True)
        train_samples = train_by_level[level]
        test_samples = test_by_level[level]
        dataset_config = dict(source_config)
        dataset_config.update(
            {
                "case_name": case_name,
                "source_case": SOURCE_CASE_NAME,
                "source_manifest": source_manifest,
                "mesh_quality_level": level,
                "mesh_quality_method": "fixed_topology_boundary_locked_random_ray_degradation",
                "jitter_max": float(cfg["jitter_max"]),
                "candidate_count": int(cfg["candidate_count"]),
                "mesh_quality_seed": int(cfg["seed"]),
                "experimental_invariants": {
                    "boundary_coordinates": "identical_to_source",
                    "boundary_point_count": "identical_to_source",
                    "vertex_count": "identical_to_source_per_sample",
                    "triangle_count": "identical_to_source_per_sample",
                    "face_connectivity": "identical_to_source",
                },
            }
        )
        info = prepare_dataset_and_cases(
            case_name=case_name,
            train_samples=train_samples,
            test_samples=test_samples,
            root=root,
            config=dataset_config,
            overwrite=overwrite,
        )
        info.update(
            _save_mesh_summary(
                data_root=paths["data_root"],
                case_name=case_name,
                train_samples=train_samples,
                test_samples=test_samples,
            )
        )

        debug_dir = paths["data_root"] / "debug_mesh"
        debug_dir.mkdir(parents=True, exist_ok=True)
        for index, sample in enumerate(train_samples[: int(cfg["n_debug_mesh"])]):
            try:
                save_debug_mesh_plot(
                    sample, debug_dir / f"debug_sample_{index:02d}.png"
                )
            except OSError as exc:
                # Debug figures are diagnostic artifacts and must not abort
                # preparation/FEM/training on a problematic WSL mount.
                print(
                    f"[Mesh quality][Warning] Could not save debug mesh "
                    f"{case_name}/{index}: {exc}"
                )
            try:
                save_encoder_pc_plot(
                    sample, debug_dir / f"debug_encoder_pc_{index:02d}.png"
                )
            except OSError as exc:
                print(
                    f"[Mesh quality][Warning] Could not save encoder plot "
                    f"{case_name}/{index}: {exc}"
                )
        info["paths"] = path_dict_to_str(paths)
        results[case_name] = info
        print(
            f"[Mesh quality] prepared {case_name}: "
            f"n_train={len(train_samples)}, n_test={len(test_samples)}"
        )

    return {
        "source_case": SOURCE_CASE_NAME,
        "source_is_quality_level": 1,
        "quality_cases": results,
    }


if __name__ == "__main__":
    prepare_hole_um_up_mesh_quality()
