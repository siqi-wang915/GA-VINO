from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np
import torch

from NO.problem.common import normalize_points_by_bbox, precompute_plate_face_geometry
from NO.problem.hole import (
    make_hole_boundary_point_cloud,
    make_square_clamped_bc_factor,
    triangulate_square_with_hole,
)


# These lower-density cases are generated and trained independently. The
# source hole_um_up mesh supplies the 100% data row and is retrained by the
# study driver with the same fixed schedule as the lower-density cases.
MESH_DENSITY_RATIOS = (0.05, 0.125, 0.25, 0.50, 0.75)


def density_case_name(ratio: float) -> str:
    pct = int(round(100.0 * float(ratio)))
    return f"hole_um_up_mesh_{pct:03d}"


def _to_numpy(value: Any, *, dtype=None) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def triangle_quality_values(verts_xy: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Normalized shape quality: 1 for equilateral and 0 for degenerate."""
    tri = np.asarray(verts_xy, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
    e01 = tri[:, 1] - tri[:, 0]
    e12 = tri[:, 2] - tri[:, 1]
    e20 = tri[:, 0] - tri[:, 2]
    l2_sum = (
        np.sum(e01 * e01, axis=1)
        + np.sum(e12 * e12, axis=1)
        + np.sum(e20 * e20, axis=1)
    )
    area2 = np.abs(
        e01[:, 0] * (tri[:, 2, 1] - tri[:, 0, 1])
        - e01[:, 1] * (tri[:, 2, 0] - tri[:, 0, 0])
    )
    return np.divide(
        2.0 * np.sqrt(3.0) * area2,
        l2_sum,
        out=np.zeros_like(area2),
        where=l2_sum > 0.0,
    )


def triangle_min_angles_deg(verts_xy: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = np.asarray(verts_xy, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
    lengths = np.stack(
        (
            np.linalg.norm(tri[:, 1] - tri[:, 2], axis=1),
            np.linalg.norm(tri[:, 2] - tri[:, 0], axis=1),
            np.linalg.norm(tri[:, 0] - tri[:, 1], axis=1),
        ),
        axis=1,
    )
    angles = []
    for i, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        denom = np.maximum(2.0 * lengths[:, j] * lengths[:, k], 1.0e-300)
        cosine = np.clip(
            (lengths[:, j] ** 2 + lengths[:, k] ** 2 - lengths[:, i] ** 2)
            / denom,
            -1.0,
            1.0,
        )
        angles.append(np.degrees(np.arccos(cosine)))
    return np.min(np.stack(angles, axis=1), axis=1)


def mesh_quality_statistics(verts_xy: np.ndarray, faces: np.ndarray) -> Dict[str, float]:
    q = triangle_quality_values(verts_xy, faces)
    amin = triangle_min_angles_deg(verts_xy, faces)
    tri = np.asarray(verts_xy, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
    area = 0.5 * np.abs(
        (tri[:, 1, 0] - tri[:, 0, 0]) * (tri[:, 2, 1] - tri[:, 0, 1])
        - (tri[:, 2, 0] - tri[:, 0, 0]) * (tri[:, 1, 1] - tri[:, 0, 1])
    )
    return {
        "quality_min": float(np.min(q)),
        "quality_p05": float(np.quantile(q, 0.05)),
        "quality_mean": float(np.mean(q)),
        "min_angle_min_deg": float(np.min(amin)),
        "min_angle_p05_deg": float(np.quantile(amin, 0.05)),
        "area_mean": float(np.mean(area)),
        "area_cv": float(np.std(area) / max(np.mean(area), 1.0e-300)),
    }


def _source_hole_boundary(source_sample: Dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Return the exact polygon that defines the source FEM hole.

    The source domain is the polygon through every entry of ``hole_xy``.  A
    subset of those vertices would replace several source edges by chords.
    For a non-circular rough hole, those chords can enter the source void, so
    target quadrature points no longer belong to the source FEM domain used to
    create the interpolation references.  Preserve this polygon exactly and
    vary the interior/outer-boundary resolution independently instead.
    """
    source_xy = _to_numpy(source_sample["hole_xy"], dtype=np.float64)
    if "hole_radius" in source_sample:
        source_radius = _to_numpy(
            source_sample["hole_radius"], dtype=np.float64
        ).reshape(-1)
    else:
        center = _to_numpy(source_sample["hole_center"], dtype=np.float64).reshape(2)
        source_radius = np.linalg.norm(source_xy - center[None, :], axis=1)
    if len(source_radius) != len(source_xy):
        raise ValueError("hole_radius and hole_xy have different lengths.")
    return source_xy.copy(), source_radius.copy()


def build_independent_density_sample(
    source_sample: Dict[str, Any],
    *,
    ratio: float,
    source_mesh_lc: float,
    density_tolerance: float = 0.03,
    max_calibration_steps: int = 5,
) -> Dict[str, Any]:
    """Remesh one existing physical sample independently with Gmsh."""
    ratio = float(ratio)
    if ratio not in MESH_DENSITY_RATIOS:
        raise ValueError(f"ratio must be one of {MESH_DENSITY_RATIOS}, got {ratio}")
    if bool(source_sample.get("use_load_field", False)):
        raise ValueError("This remesher currently supports hole_um_up uniform pressure only.")

    verts_source = _to_numpy(source_sample["verts_xy"], dtype=np.float64)
    faces_source = _to_numpy(source_sample["faces"], dtype=np.int64)
    hole_source = _to_numpy(source_sample["hole_xy"], dtype=np.float64)
    pc_source = _to_numpy(source_sample["pc"], dtype=np.float64)
    n_outer_source = int(len(pc_source) - len(hole_source))
    if n_outer_source <= 0 or n_outer_source % 4 != 0:
        raise ValueError("Cannot infer four equal outer boundary point sets from source pc.")

    boundary_density_target = np.sqrt(ratio)
    trial_boundary_ratio = boundary_density_target
    lx = float(source_sample["lx"])
    ly = float(source_sample["ly"])

    target_n_faces = ratio * len(faces_source)
    best_mesh = None
    calibration_attempts = 0
    for calibration_attempts in range(1, max(1, int(max_calibration_steps)) + 1):
        outer_n_per_edge = max(
            2, int(round(trial_boundary_ratio * (n_outer_source // 4)))
        )
        # The hole polygon is part of the physical geometry, rather than a
        # density-control knob.  Coarsening it changes the domain and makes a
        # source-FEM interpolation invalid in the thin band between old and
        # new hole boundaries.  Gmsh still creates a fresh interior mesh.
        hole_xy, hole_radius = _source_hole_boundary(source_sample)
        hole_n_points = len(hole_xy)
        mesh_lc = float(source_mesh_lc) / trial_boundary_ratio
        candidate_verts, candidate_faces = triangulate_square_with_hole(
            lx=lx,
            ly=ly,
            hole_xy=hole_xy,
            mesh_lc=mesh_lc,
            outer_n_per_edge=outer_n_per_edge,
        )
        actual_ratio = len(candidate_faces) / len(faces_source)
        score = abs(np.log(max(actual_ratio, 1.0e-12) / ratio))
        if best_mesh is None or score < best_mesh[0]:
            best_mesh = (
                score,
                candidate_verts,
                candidate_faces,
                mesh_lc,
                outer_n_per_edge,
                hole_n_points,
                hole_xy,
                hole_radius,
                trial_boundary_ratio,
            )
        relative_error = abs(len(candidate_faces) - target_n_faces) / target_n_faces
        if relative_error <= float(density_tolerance):
            break
        # Boundary segmentation controls the effective Gmsh size in this case.
        # Since N_element ~ h^-2 and N_boundary ~ h^-1, correct both the
        # boundary point count and mesh_lc through one boundary-density factor.
        correction = np.sqrt(ratio / max(actual_ratio, 1.0e-12))
        trial_boundary_ratio *= float(np.clip(correction, 0.75, 1.25))
        trial_boundary_ratio = float(np.clip(trial_boundary_ratio, 0.10, 1.0))
    (
        _,
        verts_xy,
        faces,
        mesh_lc,
        outer_n_per_edge,
        hole_n_points,
        hole_xy,
        hole_radius,
        selected_boundary_ratio,
    ) = best_mesh
    query = torch.zeros((len(verts_xy), 3), dtype=torch.float32)
    query[:, :2] = verts_xy
    bbox = torch.tensor([0.0, lx, 0.0, ly], dtype=torch.float32)
    pc = make_hole_boundary_point_cloud(
        lx=lx,
        ly=ly,
        outer_n_per_edge=outer_n_per_edge,
        hole_xy=hole_xy,
    )

    out = dict(source_sample)
    out.update(
        {
            "verts_xy": verts_xy.float(),
            "faces": faces.long(),
            "query": query,
            "query_net": normalize_points_by_bbox(query, bbox),
            "pc": pc.float(),
            "pc_net": normalize_points_by_bbox(pc, bbox),
            "bbox": bbox,
            "bc_factor": make_square_clamped_bc_factor(verts_xy, lx=lx, ly=ly),
            "geom_face": precompute_plate_face_geometry(verts_xy, faces),
            "hole_xy": torch.tensor(hole_xy, dtype=torch.float32),
            "hole_radius": torch.tensor(hole_radius, dtype=torch.float32),
            "pressure": torch.full(
                (len(verts_xy), 1),
                float(torch.as_tensor(source_sample["pressure"]).float().mean().item()),
                dtype=torch.float32,
            ),
            "mesh_lc": torch.tensor([mesh_lc], dtype=torch.float32),
            "mesh_calibration_attempts": torch.tensor(
                [calibration_attempts], dtype=torch.long
            ),
            "remesh_kind": "independent_gmsh_same_physical_sample",
        }
    )

    quality = mesh_quality_statistics(verts_xy.numpy(), faces.numpy())
    source_quality = mesh_quality_statistics(verts_source, faces_source)
    source_boundary_count = n_outer_source + len(hole_source)
    boundary_count = 4 * outer_n_per_edge + hole_n_points
    out.update(
        {
            "source_n_vertices": torch.tensor([len(verts_source)], dtype=torch.long),
            "source_n_faces": torch.tensor([len(faces_source)], dtype=torch.long),
            "source_n_boundary_points": torch.tensor([source_boundary_count], dtype=torch.long),
            "n_vertices": torch.tensor([len(verts_xy)], dtype=torch.long),
            "n_faces": torch.tensor([len(faces)], dtype=torch.long),
            "mesh_density_target": torch.tensor([ratio], dtype=torch.float32),
            "mesh_density_actual": torch.tensor(
                [len(verts_xy) / len(verts_source)], dtype=torch.float32
            ),
            "element_density_actual": torch.tensor(
                [len(faces) / len(faces_source)], dtype=torch.float32
            ),
            "element_density_relative_error": torch.tensor(
                [abs(len(faces) - target_n_faces) / target_n_faces], dtype=torch.float32
            ),
            "boundary_density_target": torch.tensor(
                [boundary_density_target], dtype=torch.float32
            ),
            "boundary_density_actual": torch.tensor(
                [boundary_count / source_boundary_count], dtype=torch.float32
            ),
            "boundary_density_calibrated": torch.tensor(
                [selected_boundary_ratio], dtype=torch.float32
            ),
            "relative_mesh_size": torch.tensor(
                [np.sqrt(len(faces_source) / len(faces))], dtype=torch.float32
            ),
            "outer_n_per_edge": torch.tensor([outer_n_per_edge], dtype=torch.long),
            "hole_n_points": torch.tensor([hole_n_points], dtype=torch.long),
            **{key: torch.tensor([value], dtype=torch.float32) for key, value in quality.items()},
            **{
                f"source_{key}": torch.tensor([value], dtype=torch.float32)
                for key, value in source_quality.items()
            },
        }
    )
    return out


def build_density_dataset(
    samples: Sequence[Dict[str, Any]],
    *,
    ratio: float,
    source_mesh_lc: float,
    density_tolerance: float = 0.03,
    max_calibration_steps: int = 5,
) -> list[Dict[str, Any]]:
    return [
        build_independent_density_sample(
            sample,
            ratio=ratio,
            source_mesh_lc=source_mesh_lc,
            density_tolerance=density_tolerance,
            max_calibration_steps=max_calibration_steps,
        )
        for sample in samples
    ]
