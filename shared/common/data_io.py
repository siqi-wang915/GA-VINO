from __future__ import annotations

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from shared.common.paths import (
    build_paths,
    case_npz_path,
    normalize_case_name,
)


# -----------------------------------------------------------------------------
# Basic conversion helpers
# -----------------------------------------------------------------------------

def to_numpy(value: Any) -> np.ndarray:
    """
    Convert torch Tensor / numpy array / scalar / list to numpy array.

    This function avoids requiring torch as a hard dependency in this file.
    """
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().numpy()

    if isinstance(value, np.ndarray):
        return value

    return np.asarray(value)


def to_numpy_float32(value: Any) -> np.ndarray:
    return np.asarray(to_numpy(value), dtype=np.float32)


def to_numpy_float64(value: Any) -> np.ndarray:
    return np.asarray(to_numpy(value), dtype=np.float64)


def to_numpy_int64(value: Any) -> np.ndarray:
    return np.asarray(to_numpy(value), dtype=np.int64)


def to_scalar_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return float(default)

    try:
        arr = np.asarray(to_numpy(value))
        if arr.shape == ():
            return float(arr.item())
        return float(arr.reshape(-1)[0])
    except Exception:
        return float(default)


def to_scalar_int(value: Any, default: int = 0) -> int:
    if value is None:
        return int(default)

    try:
        arr = np.asarray(to_numpy(value))
        if arr.shape == ():
            return int(arr.item())
        return int(arr.reshape(-1)[0])
    except Exception:
        return int(default)


def to_scalar_str(value: Any, default: str = "") -> str:
    if value is None:
        return str(default)

    try:
        arr = np.asarray(to_numpy(value))
        if arr.shape == ():
            return str(arr.item())
        if arr.size > 0:
            return str(arr.reshape(-1)[0])
    except Exception:
        pass

    return str(default)


def json_safe(obj: Any) -> Any:
    """
    Convert common non-JSON objects to JSON-safe objects.
    """
    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating,)):
        return float(obj)

    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]

    return obj


# -----------------------------------------------------------------------------
# Pickle dataset IO
# -----------------------------------------------------------------------------

def save_pickle(obj: Any, path: str | Path, overwrite: bool = True) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and not overwrite:
        raise FileExistsError(f"File already exists and overwrite=False: {path}")

    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    return path


def load_pickle(path: str | Path) -> Any:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Pickle file not found: {path}")

    with open(path, "rb") as f:
        return pickle.load(f)


def save_dataset(
    *,
    case_name: str,
    train_samples: Sequence[Dict[str, Any]],
    test_samples: Sequence[Dict[str, Any]],
    root: Optional[str | Path] = None,
    config: Optional[Dict[str, Any]] = None,
    overwrite: bool = True,
) -> Dict[str, Any]:
    """
    Save train/test samples to shared/data/<case>/.

    This is critical for the new two-stage workflow:

        Stage 1:
            generate samples -> save_dataset -> export FEM cases -> run FEM

        Stage 2:
            load_dataset -> train NO -> evaluate with existing FEM results

    By saving samples, the second run uses exactly the same train/test data
    as the first FEM run.
    """
    case = normalize_case_name(case_name)
    paths = build_paths(case, root=root, create=True)

    save_pickle(train_samples, paths["train_samples_path"], overwrite=overwrite)
    save_pickle(test_samples, paths["test_samples_path"], overwrite=overwrite)

    n_train = len(train_samples)
    n_test = len(test_samples)

    train_tags = [f"train_{i:04d}" for i in range(n_train)]
    test_tags = [f"case_{i:04d}" for i in range(n_test)]

    now = datetime.now().isoformat(timespec="seconds")

    config = {} if config is None else dict(config)

    np.savez_compressed(
        paths["dataset_manifest_path"],
        case_name=np.asarray(case),
        created_at=np.asarray(now),
        n_train=np.asarray(n_train, dtype=np.int64),
        n_test=np.asarray(n_test, dtype=np.int64),
        train_tags=np.asarray(train_tags, dtype=str),
        test_tags=np.asarray(test_tags, dtype=str),
        train_samples_path=np.asarray(str(paths["train_samples_path"])),
        test_samples_path=np.asarray(str(paths["test_samples_path"])),
        case_root=np.asarray(str(paths["case_root"])),
        no_prediction_root=np.asarray(str(paths["no_prediction_root"])),
        fem_prediction_root=np.asarray(str(paths["fem_prediction_root"])),
        compare_root=np.asarray(str(paths["compare_root"])),
    )

    with open(paths["dataset_config_path"], "w", encoding="utf-8") as f:
        json.dump(json_safe(config), f, indent=2, ensure_ascii=False)

    write_tags(paths["train_case_tags_path"], train_tags)
    write_tags(paths["test_case_tags_path"], test_tags)

    return {
        "case_name": case,
        "n_train": n_train,
        "n_test": n_test,
        "train_tags": train_tags,
        "test_tags": test_tags,
        "paths": {k: str(v) for k, v in paths.items()},
    }


def load_dataset(
    *,
    case_name: str,
    root: Optional[str | Path] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Load train/test samples and manifest from shared/data/<case>/.
    """
    case = normalize_case_name(case_name)
    paths = build_paths(case, root=root, create=True)

    train_samples = load_pickle(paths["train_samples_path"])
    test_samples = load_pickle(paths["test_samples_path"])
    manifest = load_manifest(case_name=case, root=root)

    return train_samples, test_samples, manifest


# -----------------------------------------------------------------------------
# Manifest and tag IO
# -----------------------------------------------------------------------------

def load_manifest(
    *,
    case_name: str,
    root: Optional[str | Path] = None,
) -> Dict[str, Any]:
    case = normalize_case_name(case_name)
    paths = build_paths(case, root=root, create=True)

    manifest_path = paths["dataset_manifest_path"]
    if not manifest_path.exists():
        raise FileNotFoundError(f"Dataset manifest not found: {manifest_path}")

    with np.load(manifest_path, allow_pickle=True) as data:
        manifest = {k: data[k] for k in data.files}

    # Convert commonly used arrays to Python values.
    out: Dict[str, Any] = {}
    for k, v in manifest.items():
        arr = np.asarray(v)

        if k in ("case_name", "created_at", "train_samples_path", "test_samples_path",
                 "case_root", "no_prediction_root", "fem_prediction_root", "compare_root"):
            out[k] = to_scalar_str(arr)
        elif k in ("n_train", "n_test"):
            out[k] = to_scalar_int(arr)
        elif k in ("train_tags", "test_tags"):
            out[k] = [str(x) for x in arr.tolist()]
        else:
            out[k] = arr

    return out


def write_tags(path: str | Path, tags: Iterable[str]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for tag in tags:
            f.write(f"{tag}\n")

    return path


def read_tags(path: str | Path) -> List[str]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Tag file not found: {path}")

    tags: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            tag = line.strip()
            if tag:
                tags.append(tag)

    return tags

def load_train_tags(
    *,
    case_name: str,
    root: Optional[str | Path] = None,
) -> List[str]:
    case = normalize_case_name(case_name)
    paths = build_paths(case, root=root, create=True)

    if paths["train_case_tags_path"].exists():
        return read_tags(paths["train_case_tags_path"])

    manifest = load_manifest(case_name=case, root=root)
    tags = manifest.get("train_tags", [])
    return list(tags)

def load_test_tags(
    *,
    case_name: str,
    root: Optional[str | Path] = None,
) -> List[str]:
    case = normalize_case_name(case_name)
    paths = build_paths(case, root=root, create=True)

    if paths["test_case_tags_path"].exists():
        return read_tags(paths["test_case_tags_path"])

    manifest = load_manifest(case_name=case, root=root)
    tags = manifest.get("test_tags", [])

    if not tags:
        # Fallback: scan shared/cases/<case>/case_*.npz
        tags = sorted(p.stem for p in paths["case_root"].glob("case_*.npz"))

    if not tags:
        raise RuntimeError(
            f"No test case tags found for case='{case}'. "
            f"Expected {paths['test_case_tags_path']} or case files under {paths['case_root']}."
        )

    return list(tags)


def load_train_tags(
    *,
    case_name: str,
    root: Optional[str | Path] = None,
) -> List[str]:
    case = normalize_case_name(case_name)
    paths = build_paths(case, root=root, create=True)

    if paths["train_case_tags_path"].exists():
        return read_tags(paths["train_case_tags_path"])

    manifest = load_manifest(case_name=case, root=root)
    tags = manifest.get("train_tags", [])

    if not tags:
        # For training tags, we don't have a robust fallback other than manifest.
        return []

    return list(tags)


# -----------------------------------------------------------------------------
# NPZ helpers
# -----------------------------------------------------------------------------

def save_npz_dict(path: str | Path, data: Dict[str, Any], overwrite: bool = True) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and not overwrite:
        raise FileExistsError(f"File already exists and overwrite=False: {path}")

    payload = {}
    for k, v in data.items():
        if isinstance(v, str):
            payload[k] = np.asarray(v)
        elif isinstance(v, Path):
            payload[k] = np.asarray(str(v))
        else:
            payload[k] = to_numpy(v)

    np.savez_compressed(path, **payload)
    return path


def load_npz_dict(path: str | Path) -> Dict[str, Any]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"NPZ file not found: {path}")

    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def load_npz_field(path: str | Path, key: str) -> Any:
    """Load one materialized field from a compressed NPZ archive.

    ``load_npz_dict`` intentionally materializes every member of an archive.
    FEM prediction archives also contain optional quadrature-level resultants
    that can be much larger than the displacement target used for supervised
    training.  Call this helper when only one field is needed so NumPy
    decompresses that member alone.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"NPZ file not found: {path}")

    with np.load(path, allow_pickle=True) as data:
        if key not in data.files:
            raise KeyError(f"NPZ field {key!r} not found in {path}")
        return data[key]


# -----------------------------------------------------------------------------
# FEM case export
# -----------------------------------------------------------------------------

CASE_REQUIRED_KEYS = (
    "query",
    "verts_xy",
    "faces",
    "pressure",
    "E",
    "nu",
    "thickness",
    "lx",
    "ly",
)


CASE_OPTIONAL_KEYS = (
    "bc_factor",
    "bbox",
    "geometry_type",
    "load_type",
    "pressure_value",
    "pressure_min",
    "pressure_max",
    "load_scalar",
    "pressure_net",
    "load_pc_net",
    "boundary_load_pc_net",
    "use_load_field",
    "pressure_target_mean",
    "pressure_std",
    "pressure_corr_len",
    "pressure_n_modes",
    "line_load_mean",
    "line_load_target_mean",
    "line_load_min",
    "line_load_max",
    "line_load_log_std",
    "line_load_corr_len",
    "loaded_boundary_nodes",
    "loaded_boundary_edges",

    # Hole metadata
    "outer_boundary_nodes",
    "hole_boundary_nodes",
    "hole_center",
    "hole_xy",
    "hole_radius",
    "hole_centers",
    "hole_mean_radii",
    "hole1_xy_raw",
    "hole2_xy_raw",
    "hole1_radius",
    "hole2_radius",
    "hole_merge_state",
    "n_hole_boundaries",
    "hole_union_area",

    # Elementwise material random-field metadata for L-shape and future cases.
    "E_elem",
    "nu_elem",
    "E_net",
    "nu_net",
    "material_pc_net",
    "material_type",
    "E_mean",
    "E_std",
    "E_min",
    "E_max",
    "E_corr_len",
    "E_n_modes",
    "nu_mean",
    "nu_std",
    "nu_min",
    "nu_max",
    "nu_corr_len",
    "nu_n_modes",

    # Annular-sector geometry metadata.
    "annular_center",
    "r_inner",
    "r_outer",
    "theta_start",
    "theta_end",
    "theta_span",
    "theta_start_deg",
    "theta_end_deg",
    "theta_span_deg",
    "mesh_lc",
    "mesh_nr",
    "mesh_nt",
    "boundary_nodes",
    "boundary_segments",
    "pc_fixed_flag",
    "pc_fixed_flag_boundary",

    # Mesh-convergence metadata.
    "source_vertex_ids",
    "source_n_vertices",
    "source_n_faces",
    "n_vertices",
    "n_faces",
    "mesh_density_target",
    "mesh_density_actual",
    "element_density_actual",
    "element_density_relative_error",
    "boundary_density_target",
    "boundary_density_actual",
    "boundary_density_calibrated",
    "relative_mesh_size",
    "source_n_boundary_points",
    "remesh_kind",
    "mesh_calibration_attempts",
    "outer_n_per_edge",
    "hole_n_points",
    "quality_min",
    "quality_p05",
    "quality_mean",
    "min_angle_min_deg",
    "min_angle_p05_deg",
    "area_mean",
    "area_cv",
    "source_quality_min",
    "source_quality_p05",
    "source_quality_mean",
    "source_min_angle_min_deg",
    "source_min_angle_p05_deg",
    "source_area_mean",
    "source_area_cv",
    "mesh_quality_level",
    "mesh_quality_jitter_amplitude",
    "mesh_quality_method",
)


def make_case_payload(
    *,
    sample: Dict[str, Any],
    case_tag: str,
    source_sample_id: Optional[int] = None,
    case_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convert one sample dictionary into a FEM case npz payload.

    The payload intentionally keeps fields compatible with the previous project:
        query_xyz
        verts_xy
        faces
        pressure
        bc_factor
        E, nu, thickness
        lx, ly
        bbox
        geometry_type
        load_type
    """
    for key in CASE_REQUIRED_KEYS:
        if key not in sample:
            raise KeyError(f"Sample is missing required key '{key}'.")

    query_xyz = to_numpy_float32(sample["query"])
    verts_xy = to_numpy_float32(sample["verts_xy"])
    faces = to_numpy_int64(sample["faces"])
    pressure = to_numpy_float32(sample["pressure"])

    n_node = query_xyz.shape[0]

    if "bc_factor" in sample:
        bc_factor = to_numpy_float32(sample["bc_factor"])
    else:
        bc_factor = np.ones((n_node, 3), dtype=np.float32)

    if "bbox" in sample:
        bbox = to_numpy_float32(sample["bbox"])
    else:
        bbox = np.asarray(
            [
                float(np.min(query_xyz[:, 0])),
                float(np.max(query_xyz[:, 0])),
                float(np.min(query_xyz[:, 1])),
                float(np.max(query_xyz[:, 1])),
            ],
            dtype=np.float32,
        )

    if source_sample_id is None:
        source_sample_id = to_scalar_int(sample.get("sample_id", -1), -1)

    geometry_type = to_scalar_str(
        sample.get("geometry_type", case_name if case_name is not None else ""),
        default=case_name if case_name is not None else "",
    )
    load_type = to_scalar_str(
        sample.get("load_type", "domain_pressure"),
        default="domain_pressure",
    )

    payload: Dict[str, Any] = {
        "case_tag": np.asarray(case_tag),
        "source_sample_id": np.asarray(source_sample_id, dtype=np.int64),

        "query_xyz": query_xyz,
        "verts_xy": verts_xy,
        "faces": faces,
        "pressure": pressure,
        "bc_factor": bc_factor,

        "E": np.asarray(to_scalar_float(sample["E"], 1000.0), dtype=np.float64),
        "nu": np.asarray(to_scalar_float(sample["nu"], 0.3), dtype=np.float64),
        "thickness": np.asarray(to_scalar_float(sample["thickness"], 0.05), dtype=np.float64),

        "lx": np.asarray(to_scalar_float(sample["lx"], 1.0), dtype=np.float64),
        "ly": np.asarray(to_scalar_float(sample["ly"], 1.0), dtype=np.float64),
        "bbox": bbox,

        "geometry_type": np.asarray(geometry_type),
        "load_type": np.asarray(load_type),
    }

    # Save optional metadata if present.
    for key in CASE_OPTIONAL_KEYS:
        if key not in sample:
            continue

        value = sample[key]

        if isinstance(value, str):
            payload[key] = np.asarray(value)
        else:
            payload[key] = to_numpy(value)

    return payload

def export_train_cases(
    *,
    case_name: str,
    train_samples: Sequence[Dict[str, Any]],
    root: Optional[str | Path] = None,
    overwrite: bool = True,
) -> List[str]:
    """
    将训练集样本导出为 shared/cases/<case>/train_case_XXXX.npz，
    供 FEM 求解器运行以生成训练过程中的真值。
    """
    case = normalize_case_name(case_name)
    paths = build_paths(case, root=root, create=True)

    train_tags: List[str] = []

    for i, sample in enumerate(train_samples):
        # 使用 train_ 前缀以区分测试集 (case_0000)
        tag = f"train_case_{i:04d}"
        train_tags.append(tag)

        payload = make_case_payload(
            sample=sample,
            case_tag=tag,
            source_sample_id=to_scalar_int(sample.get("sample_id", i), i),
            case_name=case,
        )

        save_npz_dict(
            case_npz_path(paths, tag),
            payload,
            overwrite=overwrite,
        )

    # 写入 train_case_tags.txt
    write_tags(paths["train_case_tags_path"], train_tags)

    return train_tags

def export_test_cases(
    *,
    case_name: str,
    test_samples: Sequence[Dict[str, Any]],
    root: Optional[str | Path] = None,
    overwrite: bool = True,
) -> List[str]:
    """
    Export test samples to shared/cases/<case>/case_XXXX.npz.

    These files are consumed by FEM solvers in Stage 1.
    """
    case = normalize_case_name(case_name)
    paths = build_paths(case, root=root, create=True)

    test_tags: List[str] = []

    for i, sample in enumerate(test_samples):
        tag = f"case_{i:04d}"
        test_tags.append(tag)

        payload = make_case_payload(
            sample=sample,
            case_tag=tag,
            source_sample_id=to_scalar_int(sample.get("sample_id", i), i),
            case_name=case,
        )

        save_npz_dict(
            case_npz_path(paths, tag),
            payload,
            overwrite=overwrite,
        )

    write_tags(paths["test_case_tags_path"], test_tags)

    return test_tags


def prepare_dataset_and_cases(
    *,
    case_name: str,
    train_samples: Sequence[Dict[str, Any]],
    test_samples: Sequence[Dict[str, Any]],
    root: Optional[str | Path] = None,
    config: Optional[Dict[str, Any]] = None,
    overwrite: bool = True,
) -> Dict[str, Any]:
    dataset_info = save_dataset(
        case_name=case_name,
        train_samples=train_samples,
        test_samples=test_samples,
        root=root,
        config=config,
        overwrite=overwrite,
    )

    # --- 新增：同时导出训练集案例 ---
    train_tags = export_train_cases(
        case_name=case_name,
        train_samples=train_samples,
        root=root,
        overwrite=overwrite,
    )

    test_tags = export_test_cases(
        case_name=case_name,
        test_samples=test_samples,
        root=root,
        overwrite=overwrite,
    )

    dataset_info["train_tags"] = train_tags
    dataset_info["test_tags"] = test_tags

    return dataset_info
