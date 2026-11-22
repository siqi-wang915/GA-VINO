from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
from scipy.spatial import cKDTree

from NO.problem.hole import save_debug_mesh_plot, save_encoder_pc_plot
from NO.problem.hole_mesh_convergence import (
    MESH_DENSITY_RATIOS,
    build_density_dataset,
    density_case_name,
)
from shared.common.data_io import (
    load_dataset,
    load_npz_dict,
    prepare_dataset_and_cases,
    save_npz_dict,
)
from shared.common.paths import build_paths, fem_pred_npz_path, path_dict_to_str


SOURCE_CASE_NAME = "hole_um_up"

DEFAULT_CONFIG: Dict[str, Any] = {
    "source_case": SOURCE_CASE_NAME,
    "density_ratios": list(MESH_DENSITY_RATIOS),
    "n_debug_mesh": 20,
    "barycentric_tolerance": 2.0e-6,
    "density_tolerance": 0.03,
    "max_calibration_steps": 5,
    "reference_kind": "linear_interpolation_of_source_100pct_fem_vertex_values",
}


def _to_numpy(value: Any, *, dtype=None) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _scalar(sample: Dict[str, Any], key: str, default=np.nan) -> float:
    if key not in sample:
        return float(default)
    return float(_to_numpy(sample[key]).reshape(-1)[0])


def _load_source_config(source_paths: Dict[str, Path]) -> Dict[str, Any]:
    path = Path(source_paths["dataset_config_path"])
    if not path.exists():
        raise FileNotFoundError(f"Source dataset config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return dict(json.load(f))


def _validate_live_sample_counts(
    train_samples: Sequence[Dict[str, Any]],
    test_samples: Sequence[Dict[str, Any]],
) -> None:
    # Import at call time so edits in the unified prep entry point are used.
    from shared.prep.hole_um_up_mr import DEFAULT_CONFIG as SOURCE_LIVE_CONFIG

    expected_train = int(SOURCE_LIVE_CONFIG["n_train"])
    expected_test = int(SOURCE_LIVE_CONFIG["n_test"])
    actual = (len(train_samples), len(test_samples))
    expected = (expected_train, expected_test)
    if actual != expected:
        raise RuntimeError(
            "The existing hole_um_up dataset does not match the current "
            f"prep/hole_um_up.py sample counts: dataset={actual}, config={expected}. "
            "Regenerate hole_um_up with stage1 before preparing convergence meshes."
        )


def _mesh_rows(samples: Sequence[Dict[str, Any]], split: str) -> list[Dict[str, Any]]:
    rows = []
    for i, sample in enumerate(samples):
        rows.append(
            {
                "split": split,
                "sample_index": i,
                "sample_id": int(sample.get("sample_id", i)),
                "target_density": _scalar(sample, "mesh_density_target"),
                "actual_vertex_density": _scalar(sample, "mesh_density_actual"),
                "actual_element_density": _scalar(sample, "element_density_actual"),
                "element_density_relative_error": _scalar(
                    sample, "element_density_relative_error"
                ),
                "target_boundary_density": _scalar(sample, "boundary_density_target"),
                "actual_boundary_density": _scalar(sample, "boundary_density_actual"),
                "calibrated_boundary_density": _scalar(
                    sample, "boundary_density_calibrated"
                ),
                "relative_mesh_size": _scalar(sample, "relative_mesh_size"),
                "mesh_lc": _scalar(sample, "mesh_lc"),
                "mesh_calibration_attempts": int(
                    _scalar(sample, "mesh_calibration_attempts")
                ),
                "n_vertices": int(_scalar(sample, "n_vertices")),
                "n_faces": int(_scalar(sample, "n_faces")),
                "n_boundary_points": int(
                    _scalar(sample, "outer_n_per_edge") * 4
                    + _scalar(sample, "hole_n_points")
                ),
                "source_n_vertices": int(_scalar(sample, "source_n_vertices")),
                "source_n_faces": int(_scalar(sample, "source_n_faces")),
                "outer_n_per_edge": int(_scalar(sample, "outer_n_per_edge")),
                "hole_n_points": int(_scalar(sample, "hole_n_points")),
                "quality_min": _scalar(sample, "quality_min"),
                "quality_p05": _scalar(sample, "quality_p05"),
                "quality_mean": _scalar(sample, "quality_mean"),
                "min_angle_min_deg": _scalar(sample, "min_angle_min_deg"),
                "min_angle_p05_deg": _scalar(sample, "min_angle_p05_deg"),
                "area_mean": _scalar(sample, "area_mean"),
                "area_cv": _scalar(sample, "area_cv"),
            }
        )
    return rows


def _save_mesh_summary(
    *,
    data_root: Path,
    train_samples: Sequence[Dict[str, Any]],
    test_samples: Sequence[Dict[str, Any]],
) -> Dict[str, str]:
    rows = _mesh_rows(train_samples, "train") + _mesh_rows(test_samples, "test")
    csv_path = data_root / "mesh_quality_summary.csv"
    npz_path = data_root / "mesh_quality_summary.npz"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        npz_path,
        **{
            key: np.asarray(
                [row[key] for row in rows],
                dtype=str if key == "split" else np.float64,
            )
            for key in rows[0]
        },
    )
    return {"mesh_quality_csv": str(csv_path), "mesh_quality_npz": str(npz_path)}


def _barycentric_coordinates(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Return barycentric weights for one point and triangle per row."""
    x, y = points[:, 0], points[:, 1]
    x0, y0 = triangles[:, 0, 0], triangles[:, 0, 1]
    x1, y1 = triangles[:, 1, 0], triangles[:, 1, 1]
    x2, y2 = triangles[:, 2, 0], triangles[:, 2, 1]
    den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if np.any(np.abs(den) < 1.0e-20):
        raise ValueError("Degenerate source triangle encountered during interpolation.")
    w0 = ((y1 - y2) * (x - x2) + (x2 - x1) * (y - y2)) / den
    w1 = ((y2 - y0) * (x - x2) + (x0 - x2) * (y - y2)) / den
    return np.column_stack((w0, w1, 1.0 - w0 - w1))


def locate_query_points_in_source_mesh(
    *,
    source_verts_xy: np.ndarray,
    source_faces: np.ndarray,
    query_xy: np.ndarray,
    barycentric_tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Locate every new query point in the original mesh without crossing the hole."""
    import matplotlib.tri as mtri

    source_verts_xy = np.asarray(source_verts_xy, dtype=np.float64)
    source_faces = np.asarray(source_faces, dtype=np.int64)
    query_xy = np.asarray(query_xy, dtype=np.float64)
    triangulation = mtri.Triangulation(
        source_verts_xy[:, 0], source_verts_xy[:, 1], triangles=source_faces
    )
    tri_ids = np.asarray(
        triangulation.get_trifinder()(query_xy[:, 0], query_xy[:, 1]), dtype=np.int64
    )

    # TriFinder can return -1 for a point that is on a boundary to roundoff.
    # Retry only nearby source triangles and accept a very small negative
    # barycentric coordinate. Genuine geometry/domain mismatch remains fatal.
    missing = np.flatnonzero(tri_ids < 0)
    if len(missing):
        triangles = source_verts_xy[source_faces]
        tree = cKDTree(triangles.mean(axis=1))
        k = min(64, len(source_faces))
        _, candidates = tree.query(query_xy[missing], k=k)
        candidates = np.asarray(candidates, dtype=np.int64).reshape(len(missing), k)
        still_missing = []
        for row, query_id in enumerate(missing):
            candidate_ids = candidates[row]
            points = np.repeat(query_xy[query_id][None, :], len(candidate_ids), axis=0)
            weights = _barycentric_coordinates(points, triangles[candidate_ids])
            score = np.min(weights, axis=1)
            best = int(np.argmax(score))
            if score[best] >= -float(barycentric_tolerance):
                tri_ids[query_id] = int(candidate_ids[best])
            else:
                still_missing.append(int(query_id))
        if still_missing:
            preview = still_missing[:8]
            raise ValueError(
                f"{len(still_missing)} target query points lie outside the original "
                f"hole_um_up FEM domain (first ids: {preview}). The independently "
                "generated boundary is not geometrically compatible with interpolation."
            )

    weights = _barycentric_coordinates(
        query_xy, source_verts_xy[source_faces[tri_ids]]
    )
    if np.min(weights) < -float(barycentric_tolerance):
        raise ValueError(
            f"Interpolation produced barycentric weight {np.min(weights):.3e} below tolerance."
        )
    # Remove boundary-level floating-point negatives while preserving partition of unity.
    weights = np.maximum(weights, 0.0)
    weights /= np.sum(weights, axis=1, keepdims=True)
    return tri_ids, weights


def _interpolated_reference_payload(
    *,
    source_fem: Dict[str, Any],
    target_sample: Dict[str, Any],
    target_case_name: str,
    tag: str,
    barycentric_tolerance: float,
) -> Dict[str, Any]:
    source_q = np.asarray(source_fem["fem_local"], dtype=np.float64)
    source_verts = np.asarray(source_fem["verts_xy"], dtype=np.float64)
    source_faces = np.asarray(source_fem["faces"], dtype=np.int64)
    if len(source_q) != len(source_verts):
        raise ValueError(
            f"Source FEM vertex field is not aligned for {tag}: "
            f"fem_local={len(source_q)}, verts={len(source_verts)}."
        )
    query_xyz = _to_numpy(target_sample["query"], dtype=np.float64)
    tri_ids, weights = locate_query_points_in_source_mesh(
        source_verts_xy=source_verts,
        source_faces=source_faces,
        query_xy=query_xyz[:, :2],
        barycentric_tolerance=barycentric_tolerance,
    )
    fem_local = np.einsum("ni,nic->nc", weights, source_q[source_faces[tri_ids]])
    verts_xy = _to_numpy(target_sample["verts_xy"], dtype=np.float64)
    faces = _to_numpy(target_sample["faces"], dtype=np.int64)

    resultant_keys = (
        "resultant_quadrature_barycentric",
        "resultant_quadrature_weights",
        "fem_moment_qp",
        "fem_shear_qp",
    )
    missing_resultants = [key for key in resultant_keys if key not in source_fem]
    if missing_resultants:
        raise KeyError(
            "The 100% source FEM result lacks native resultant fields "
            f"{missing_resultants}. Re-run hole_um_up test FEM with the updated "
            "solver before rebuilding mesh-convergence references."
        )

    resultant_bary = np.asarray(
        source_fem["resultant_quadrature_barycentric"], dtype=np.float64
    )
    resultant_weights = np.asarray(
        source_fem["resultant_quadrature_weights"], dtype=np.float64
    ).reshape(-1)
    source_moment_qp = np.asarray(source_fem["fem_moment_qp"], dtype=np.float64)
    source_shear_qp = np.asarray(source_fem["fem_shear_qp"], dtype=np.float64)
    n_qp = int(resultant_weights.shape[0])
    if resultant_bary.shape != (n_qp, 3):
        raise ValueError(
            f"Invalid source resultant quadrature shape {resultant_bary.shape}."
        )
    if source_moment_qp.shape != (source_faces.shape[0], n_qp, 3):
        raise ValueError(f"Invalid source fem_moment_qp shape {source_moment_qp.shape}.")
    if source_shear_qp.shape != (source_faces.shape[0], n_qp, 2):
        raise ValueError(f"Invalid source fem_shear_qp shape {source_shear_qp.shape}.")

    # Native CG2-gradient moments and lowest-order Nedelec shear resultants are
    # affine inside each source triangle. Recover their barycentric polynomial
    # coefficients from the seven source values, then evaluate that piecewise
    # field at every target-triangle quadrature point.
    bary_pinv = np.linalg.pinv(resultant_bary)
    moment_coeff = np.einsum("aq,eqc->eac", bary_pinv, source_moment_qp)
    shear_coeff = np.einsum("aq,eqc->eac", bary_pinv, source_shear_qp)

    target_qp_xy = np.einsum(
        "qa,eac->eqc", resultant_bary, verts_xy[faces]
    )
    flat_target_qp = target_qp_xy.reshape(-1, 2)
    source_resultant_tri_ids, source_resultant_bary = locate_query_points_in_source_mesh(
        source_verts_xy=source_verts,
        source_faces=source_faces,
        query_xy=flat_target_qp,
        barycentric_tolerance=barycentric_tolerance,
    )
    target_moment_qp = np.einsum(
        "na,nac->nc",
        source_resultant_bary,
        moment_coeff[source_resultant_tri_ids],
    ).reshape(faces.shape[0], n_qp, 3)
    target_shear_qp = np.einsum(
        "na,nac->nc",
        source_resultant_bary,
        shear_coeff[source_resultant_tri_ids],
    ).reshape(faces.shape[0], n_qp, 2)

    payload = dict(source_fem)
    payload.update(
        {
            "case_tag": np.asarray(tag),
            "case_name": np.asarray(target_case_name),
            "reference_case_name": np.asarray(SOURCE_CASE_NAME),
            "reference_kind": np.asarray(
                "linear_interpolation_of_source_100pct_fem_vertex_values"
            ),
            "reference_note": np.asarray(
                "w is the source P1 vertex field; saved theta vertex values are "
                "also interpolated linearly and therefore do not reconstruct P2 edge DOFs."
            ),
            "source_triangle_ids": tri_ids,
            "source_barycentric_weights": weights,
            "interpolation_outside_count": np.asarray(0, dtype=np.int64),
            "interpolation_min_barycentric_weight": np.asarray(
                float(np.min(weights)), dtype=np.float64
            ),
            "query_xyz": query_xyz,
            "verts_xy": verts_xy,
            "faces": faces,
            "fem_local": fem_local,
            "fem_global_disp": np.column_stack(
                (np.zeros(len(fem_local)), np.zeros(len(fem_local)), fem_local[:, 0])
            ),
            "disp_scalar": fem_local[:, 0],
            "rot_scalar": np.abs(fem_local[:, 1]) + np.abs(fem_local[:, 2]),
            "resultant_quadrature_barycentric": resultant_bary,
            "resultant_quadrature_weights": resultant_weights,
            "fem_moment_qp": target_moment_qp,
            "fem_shear_qp": target_shear_qp,
            "fem_moment_elem": np.einsum(
                "q,eqi->ei", resultant_weights, target_moment_qp
            ),
            "fem_shear_elem": np.einsum(
                "q,eqi->ei", resultant_weights, target_shear_qp
            ),
            "n_query_points": np.asarray(len(fem_local), dtype=np.int64),
            "n_vertices": np.asarray(len(verts_xy), dtype=np.int64),
            "n_faces": np.asarray(len(faces), dtype=np.int64),
            "mesh_density_target": np.asarray(
                _scalar(target_sample, "mesh_density_target"), dtype=np.float64
            ),
            "mesh_density_actual": np.asarray(
                _scalar(target_sample, "mesh_density_actual"), dtype=np.float64
            ),
            "element_density_actual": np.asarray(
                _scalar(target_sample, "element_density_actual"), dtype=np.float64
            ),
        }
    )
    return payload


def export_interpolated_fem_references(
    *,
    target_case_name: str,
    train_samples: Sequence[Dict[str, Any]],
    test_samples: Sequence[Dict[str, Any]],
    root: Optional[str | Path] = None,
    overwrite: bool = True,
    barycentric_tolerance: float = 2.0e-6,
) -> Dict[str, Any]:
    source_paths = build_paths(SOURCE_CASE_NAME, root=root, create=True)
    target_paths = build_paths(target_case_name, root=root, create=True)
    saved = []
    missing = []
    tagged_samples = [
        (f"train_case_{i:04d}", sample) for i, sample in enumerate(train_samples)
    ] + [(f"case_{i:04d}", sample) for i, sample in enumerate(test_samples)]
    for tag, sample in tagged_samples:
        source_path = fem_pred_npz_path(source_paths, tag)
        if not source_path.exists():
            missing.append(str(source_path))
            continue
        payload = _interpolated_reference_payload(
            source_fem=load_npz_dict(source_path),
            target_sample=sample,
            target_case_name=target_case_name,
            tag=tag,
            barycentric_tolerance=barycentric_tolerance,
        )
        out_path = fem_pred_npz_path(target_paths, tag)
        save_npz_dict(out_path, payload, overwrite=overwrite)
        saved.append(str(out_path))
    if missing:
        preview = "\n  ".join(missing[:5])
        raise FileNotFoundError(
            "The original hole_um_up FEM references must exist before preparing "
            f"the mesh-convergence cases. Missing {len(missing)} files, e.g.\n  {preview}"
        )
    return {
        "n_reference_files": len(saved),
        "reference_files": saved,
        "reference_kind": DEFAULT_CONFIG["reference_kind"],
    }


def prepare_hole_um_up_mesh_density(
    *,
    ratio: float,
    root: Optional[str | Path] = None,
    overwrite: bool = True,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)
    ratio = float(ratio)
    case_name = density_case_name(ratio)
    paths = build_paths(case_name, root=root, create=True)
    source_paths = build_paths(SOURCE_CASE_NAME, root=root, create=True)
    source_train, source_test, source_manifest = load_dataset(
        case_name=SOURCE_CASE_NAME, root=root
    )
    _validate_live_sample_counts(source_train, source_test)
    source_config = _load_source_config(source_paths)
    source_mesh_lc = float(source_config["mesh_lc"])

    print("=" * 80)
    print(f"[Mesh convergence] Independently remeshing {case_name}")
    print(f"[Mesh convergence] target density: {100.0 * ratio:.0f}%")
    print(
        f"[Mesh convergence] synchronized samples: "
        f"n_train={len(source_train)}, n_test={len(source_test)}"
    )
    train_samples = build_density_dataset(
        source_train,
        ratio=ratio,
        source_mesh_lc=source_mesh_lc,
        density_tolerance=float(cfg["density_tolerance"]),
        max_calibration_steps=int(cfg["max_calibration_steps"]),
    )
    test_samples = build_density_dataset(
        source_test,
        ratio=ratio,
        source_mesh_lc=source_mesh_lc,
        density_tolerance=float(cfg["density_tolerance"]),
        max_calibration_steps=int(cfg["max_calibration_steps"]),
    )

    debug_dir = paths["data_root"] / "debug_mesh"
    debug_dir.mkdir(parents=True, exist_ok=True)
    for i, sample in enumerate(train_samples[: int(cfg["n_debug_mesh"])]):
        save_debug_mesh_plot(sample, debug_dir / f"debug_sample_{i:02d}.png")
        save_encoder_pc_plot(sample, debug_dir / f"debug_encoder_pc_{i:02d}.png")

    dataset_cfg = dict(cfg)
    dataset_cfg.update(
        {
            "case_name": case_name,
            "density_ratio": ratio,
            "source_mesh_lc": source_mesh_lc,
            "source_n_train": len(source_train),
            "source_n_test": len(source_test),
            "source_manifest": source_manifest,
            "source_dataset_config": source_config,
            "reference_note": (
                "Independent Gmsh mesh of the same physical sample; the existing "
                "100% FEM vertex fields are interpolated on source triangles."
            ),
        }
    )
    info = prepare_dataset_and_cases(
        case_name=case_name,
        train_samples=train_samples,
        test_samples=test_samples,
        root=root,
        config=dataset_cfg,
        overwrite=overwrite,
    )
    info.update(
        _save_mesh_summary(
            data_root=paths["data_root"],
            train_samples=train_samples,
            test_samples=test_samples,
        )
    )
    info["reference_info"] = export_interpolated_fem_references(
        target_case_name=case_name,
        train_samples=train_samples,
        test_samples=test_samples,
        root=root,
        overwrite=overwrite,
        barycentric_tolerance=float(cfg["barycentric_tolerance"]),
    )
    info["paths"] = path_dict_to_str(paths)
    print(
        f"[Mesh convergence] prepared {case_name}: "
        f"n_train={len(train_samples)}, n_test={len(test_samples)}"
    )
    return info


def prepare_hole_um_up_mesh_convergence(
    *,
    root: Optional[str | Path] = None,
    overwrite: bool = True,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)
    results = {}
    for ratio in cfg["density_ratios"]:
        case_name = density_case_name(float(ratio))
        results[case_name] = prepare_hole_um_up_mesh_density(
            ratio=float(ratio), root=root, overwrite=overwrite, config=cfg
        )
    return {"source_case": SOURCE_CASE_NAME, "density_cases": results}
