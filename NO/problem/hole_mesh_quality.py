from __future__ import annotations

"""Fixed-size mesh-quality variants for the existing ``hole_um_up`` data.

Every variant preserves the source connectivity and every boundary vertex.
Only a separated set of interior vertices is moved in random directions toward
the boundary of its local one-ring.
Consequently the physical boundary, node count, element count, pressure, and
all sample parameters remain fixed.  This controlled near-collapse produces a
wider quality range than unconstrained jitter, while an orientation-based line
search prevents inverted elements.
"""

from typing import Any, Dict, List, Sequence

import numpy as np
import torch

from NO.problem.common import normalize_points_by_bbox, precompute_plate_face_geometry
from NO.problem.hole import make_square_clamped_bc_factor
from NO.problem.hole_mesh_convergence import mesh_quality_statistics


QUALITY_LEVELS = (2, 3, 4)


def quality_case_name(level: int) -> str:
    level = int(level)
    if level not in QUALITY_LEVELS:
        raise ValueError(f"quality level must be one of {QUALITY_LEVELS}, got {level}")
    return f"hole_um_up_mesh_quality_{level:02d}"


def _to_numpy(value: Any, *, dtype=None) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def boundary_node_ids(faces: np.ndarray) -> np.ndarray:
    """Return nodes belonging to an edge owned by exactly one triangle."""
    faces = np.asarray(faces, dtype=np.int64)
    edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges = np.sort(edges, axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    return np.unique(unique[counts == 1].reshape(-1))


def _signed_double_areas(verts_xy: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = np.asarray(verts_xy, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
    ab = tri[:, 1] - tri[:, 0]
    ac = tri[:, 2] - tri[:, 0]
    return ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]


def build_vertex_collapse_field(
    verts_xy: np.ndarray,
    faces: np.ndarray,
    *,
    seed: int,
    safety: float = 0.98,
) -> tuple[np.ndarray, np.ndarray]:
    """Build heterogeneous safe displacements for separated interior nodes.

    Selected nodes never share a triangle.  Hence each affected triangle has
    only one moving vertex and its signed area varies linearly with the global
    degradation amplitude.  Directions and per-node severities are randomized;
    each displacement is shortened to stay on the original-orientation side
    of the first possible zero-area position.
    """
    if not 0.0 < float(safety) < 1.0:
        raise ValueError(f"safety must lie in (0,1), got {safety}")
    verts_xy = np.asarray(verts_xy, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    boundary_ids = boundary_node_ids(faces)
    boundary_mask = np.zeros(len(verts_xy), dtype=bool)
    boundary_mask[boundary_ids] = True

    incident_faces: List[List[int]] = [[] for _ in range(len(verts_xy))]
    neighbors: List[set[int]] = [set() for _ in range(len(verts_xy))]
    for face_id, tri in enumerate(faces):
        tri_ids = [int(value) for value in tri]
        for node in tri_ids:
            incident_faces[node].append(face_id)
            neighbors[node].update(other for other in tri_ids if other != node)

    rng = np.random.default_rng(int(seed))
    candidates = np.flatnonzero(~boundary_mask)
    rng.shuffle(candidates)
    blocked = np.zeros(len(verts_xy), dtype=bool)
    selected: List[int] = []
    for node_value in candidates:
        node = int(node_value)
        if blocked[node] or not neighbors[node]:
            continue
        selected.append(node)
        # Block the full one-ring so no triangle contains two selected nodes.
        blocked[node] = True
        blocked[np.fromiter(neighbors[node], dtype=np.int64)] = True

    if not selected:
        raise RuntimeError("No separated interior vertices were available for degradation.")

    source_area2 = _signed_double_areas(verts_xy, faces)
    displacement = np.zeros_like(verts_xy)
    for node in selected:
        neighbor_ids = np.fromiter(neighbors[node], dtype=np.int64)
        edge_lengths = np.linalg.norm(verts_xy[neighbor_ids] - verts_xy[node], axis=1)
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        raw_displacement = float(np.mean(edge_lengths)) * np.asarray(
            [np.cos(angle), np.sin(angle)], dtype=np.float64
        )
        face_ids = np.asarray(incident_faces[node], dtype=np.int64)
        area0 = source_area2[face_ids]
        target_triangles = verts_xy[faces[face_ids]].copy()
        local_positions = np.argwhere(faces[face_ids] == node)
        target_triangles[local_positions[:, 0], local_positions[:, 1]] = (
            verts_xy[node] + raw_displacement
        )
        ab = target_triangles[:, 1] - target_triangles[:, 0]
        ac = target_triangles[:, 2] - target_triangles[:, 0]
        area1 = ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]
        roots = -area0 / (area1 - area0)
        positive_roots = roots[np.isfinite(roots) & (roots > 1.0e-10)]
        if positive_roots.size == 0:
            raise RuntimeError("Could not find a local area-collapse distance.")

        # Keep enough severe vertices for P5 to reflect the requested level,
        # while assigning the rest a broad range of milder distortions.  This
        # avoids a repeated pattern of nearly identical skinny triangles.
        if float(rng.random()) < 0.65:
            node_safety = float(rng.uniform(max(0.35, safety - 0.06), safety))
        else:
            node_safety = float(rng.uniform(0.35, max(0.36, safety - 0.08)))
        displacement[node] = (
            node_safety * float(np.min(positive_roots)) * raw_displacement
        )

    displacement[boundary_mask] = 0.0
    return displacement, boundary_ids


def _valid_candidate(
    source_verts: np.ndarray,
    candidate_verts: np.ndarray,
    faces: np.ndarray,
) -> bool:
    source_area2 = _signed_double_areas(source_verts, faces)
    candidate_area2 = _signed_double_areas(candidate_verts, faces)
    if np.any(source_area2 == 0.0):
        raise ValueError("Source hole_um_up mesh contains a degenerate triangle.")
    # Preserve every triangle orientation and reject candidates too close to
    # singularity.  This keeps DOLFINx and the energy quadrature well-defined.
    same_orientation = candidate_area2 * source_area2 > 0.0
    relative_area = np.abs(candidate_area2) / np.maximum(np.abs(source_area2), 1.0e-300)
    return bool(np.all(same_orientation) and np.min(relative_area) > 1.0e-4)


def select_quality_meshes(
    verts_xy: np.ndarray,
    faces: np.ndarray,
    *,
    seed: int,
    jitter_max: float = 0.95,
    candidate_count: int = 81,
) -> List[Dict[str, Any]]:
    """Select three successively poorer meshes at evenly spaced P5 quality."""
    if jitter_max <= 0.0:
        raise ValueError("jitter_max must be positive")
    if candidate_count < len(QUALITY_LEVELS) + 1:
        raise ValueError("candidate_count is too small for three quality variants")

    source_verts = np.asarray(verts_xy, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    jitter, boundary_ids = build_vertex_collapse_field(
        source_verts, faces, seed=seed
    )
    source_stats = mesh_quality_statistics(source_verts, faces)
    source_p05 = float(source_stats["quality_p05"])

    candidates: List[Dict[str, Any]] = []
    for amplitude in np.linspace(0.0, float(jitter_max), int(candidate_count))[1:]:
        candidate_verts = source_verts + float(amplitude) * jitter
        if not _valid_candidate(source_verts, candidate_verts, faces):
            continue
        stats = mesh_quality_statistics(candidate_verts, faces)
        if float(stats["quality_p05"]) < source_p05 - 1.0e-8:
            candidates.append(
                {
                    "verts_xy": candidate_verts,
                    "jitter_amplitude": float(amplitude),
                    "quality_stats": stats,
                }
            )
    if len(candidates) < len(QUALITY_LEVELS):
        raise RuntimeError(
            "Could not generate three valid lower-quality meshes. "
            f"Only {len(candidates)} candidates degraded P5 quality."
        )

    p05 = np.asarray(
        [float(row["quality_stats"]["quality_p05"]) for row in candidates]
    )
    targets = np.linspace(source_p05, float(np.min(p05)), len(QUALITY_LEVELS) + 1)[1:]
    available = set(range(len(candidates)))
    selected: List[Dict[str, Any]] = []
    for target in targets:
        index = min(available, key=lambda i: abs(float(p05[i]) - float(target)))
        selected.append(candidates[index])
        available.remove(index)

    selected.sort(
        key=lambda row: float(row["quality_stats"]["quality_p05"]), reverse=True
    )
    # Exact boundary preservation is a central experimental invariant.
    for row in selected:
        if not np.array_equal(row["verts_xy"][boundary_ids], source_verts[boundary_ids]):
            raise RuntimeError("A mesh-quality candidate changed boundary coordinates.")
    return selected


def build_quality_variants(
    source_sample: Dict[str, Any],
    *,
    seed: int,
    jitter_max: float = 0.95,
    candidate_count: int = 81,
) -> Dict[int, Dict[str, Any]]:
    """Create levels 2--4 for one source sample without changing its size."""
    source_verts = _to_numpy(source_sample["verts_xy"], dtype=np.float64)
    source_faces = _to_numpy(source_sample["faces"], dtype=np.int64)
    selected = select_quality_meshes(
        source_verts,
        source_faces,
        seed=seed,
        jitter_max=jitter_max,
        candidate_count=candidate_count,
    )
    source_stats = mesh_quality_statistics(source_verts, source_faces)
    bbox = torch.as_tensor(source_sample["bbox"], dtype=torch.float32).reshape(4)
    lx = float(source_sample["lx"])
    ly = float(source_sample["ly"])

    variants: Dict[int, Dict[str, Any]] = {}
    for level, row in zip(QUALITY_LEVELS, selected):
        verts = torch.tensor(row["verts_xy"], dtype=torch.float32)
        faces = torch.tensor(source_faces, dtype=torch.long)
        query = torch.zeros((len(verts), 3), dtype=torch.float32)
        query[:, :2] = verts
        stats = row["quality_stats"]
        out = dict(source_sample)
        out.update(
            {
                "verts_xy": verts,
                "faces": faces,
                "query": query,
                "query_net": normalize_points_by_bbox(query, bbox),
                "bbox": bbox.clone(),
                "bc_factor": make_square_clamped_bc_factor(verts, lx=lx, ly=ly),
                "geom_face": precompute_plate_face_geometry(verts, faces),
                "mesh_quality_level": torch.tensor([level], dtype=torch.long),
                "mesh_quality_jitter_amplitude": torch.tensor(
                    [float(row["jitter_amplitude"])], dtype=torch.float32
                ),
                "mesh_quality_method": "fixed_topology_boundary_locked_random_ray_degradation",
                "source_n_vertices": torch.tensor([len(source_verts)], dtype=torch.long),
                "source_n_faces": torch.tensor([len(source_faces)], dtype=torch.long),
                "n_vertices": torch.tensor([len(verts)], dtype=torch.long),
                "n_faces": torch.tensor([len(faces)], dtype=torch.long),
                **{
                    key: torch.tensor([float(value)], dtype=torch.float32)
                    for key, value in stats.items()
                },
                **{
                    f"source_{key}": torch.tensor([float(value)], dtype=torch.float32)
                    for key, value in source_stats.items()
                },
            }
        )
        variants[level] = out
    return variants


def build_quality_datasets(
    samples: Sequence[Dict[str, Any]],
    *,
    seed: int,
    jitter_max: float = 0.95,
    candidate_count: int = 81,
) -> Dict[int, List[Dict[str, Any]]]:
    datasets = {level: [] for level in QUALITY_LEVELS}
    for index, sample in enumerate(samples):
        sample_id = int(sample.get("sample_id", index))
        variants = build_quality_variants(
            sample,
            seed=int(seed) + 104729 * sample_id,
            jitter_max=jitter_max,
            candidate_count=candidate_count,
        )
        for level in QUALITY_LEVELS:
            datasets[level].append(variants[level])
    return datasets
