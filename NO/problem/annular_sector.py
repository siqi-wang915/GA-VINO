from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from NO.problem.common import (
    normalize_points_by_bbox,
    normalize_pressure_field_for_network,
    precompute_plate_face_geometry,
)
from NO.problem.lshape import (
    generate_gaussian_random_field_on_points,
    normalize_scalar_field_for_network,
    normalize_uniform_load_for_network,
)


def _polar_to_xy(r: np.ndarray, theta: np.ndarray) -> np.ndarray:
    return np.stack([r * np.cos(theta), r * np.sin(theta)], axis=-1)


def _sample_polyline(points: np.ndarray, *, spacing: float, min_points: int = 8) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    out: List[np.ndarray] = []
    for p0, p1 in zip(points[:-1], points[1:]):
        length = float(np.linalg.norm(p1 - p0))
        n = max(int(np.ceil(length / max(float(spacing), 1.0e-12))), int(min_points))
        s = np.linspace(0.0, 1.0, n, endpoint=False, dtype=np.float64)
        out.append((1.0 - s[:, None]) * p0[None, :] + s[:, None] * p1[None, :])
    if not out:
        return points[:0].copy()
    return np.vstack(out)


def make_annular_sector_boundary_point_cloud(
    *,
    r_inner: float,
    r_outer: float,
    theta_start: float,
    theta_end: float,
    boundary_point_spacing: float = 0.10,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build boundary point cloud for the annular-sector domain.

    Boundary order:
        0. inner circular arc, free
        1. radial edge at theta_start, free
        2. outer circular arc, free
        3. radial edge at theta_end = 90 deg, fixed (x = 0)
    """
    r_inner = float(r_inner)
    r_outer = float(r_outer)
    theta_start = float(theta_start)
    theta_end = float(theta_end)
    spacing = float(boundary_point_spacing)

    n_arc = max(int(np.ceil(r_outer * abs(theta_end - theta_start) / max(spacing, 1.0e-12))), 16)
    theta_inner = np.linspace(theta_end, theta_start, n_arc, endpoint=True)
    theta_outer = np.linspace(theta_start, theta_end, n_arc, endpoint=True)

    inner_arc = _polar_to_xy(np.full_like(theta_inner, r_inner), theta_inner)
    outer_arc = _polar_to_xy(np.full_like(theta_outer, r_outer), theta_outer)

    n_radial = max(int(np.ceil((r_outer - r_inner) / max(spacing, 1.0e-12))), 8)
    r_radial = np.linspace(r_inner, r_outer, n_radial, endpoint=True)
    radial_start = _polar_to_xy(r_radial, np.full_like(r_radial, theta_start))
    radial_end = _polar_to_xy(r_radial[::-1], np.full_like(r_radial, theta_end))

    parts = [
        inner_arc[:-1],
        radial_start[:-1],
        outer_arc[:-1],
        radial_end[:-1],
    ]
    flags = [
        np.zeros((parts[0].shape[0],), dtype=np.int64),
        np.zeros((parts[1].shape[0],), dtype=np.int64),
        np.zeros((parts[2].shape[0],), dtype=np.int64),
        np.ones((parts[3].shape[0],), dtype=np.int64),
    ]

    pc_xy = np.vstack(parts).astype(np.float32)
    pc = np.zeros((pc_xy.shape[0], 3), dtype=np.float32)
    pc[:, :2] = pc_xy

    boundary_segments = np.asarray(
        [
            [inner_arc[0], inner_arc[-1]],
            [radial_start[0], radial_start[-1]],
            [outer_arc[0], outer_arc[-1]],
            [radial_end[0], radial_end[-1]],
        ],
        dtype=np.float32,
    )

    return (
        torch.tensor(pc, dtype=torch.float32),
        torch.tensor(np.concatenate(flags, axis=0), dtype=torch.long),
        torch.tensor(boundary_segments, dtype=torch.float32),
    )


def triangulate_annular_sector_structured(
    *,
    r_inner: float,
    r_outer: float,
    theta_start: float,
    theta_end: float,
    nr: int,
    nt: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a structured polar triangular mesh for an annular sector.
    """
    nr = int(nr)
    nt = int(nt)
    if nr < 1 or nt < 1:
        raise ValueError(f"Expected nr and nt >= 1, got nr={nr}, nt={nt}")

    r_vals = np.linspace(float(r_inner), float(r_outer), nr + 1, dtype=np.float64)
    theta_vals = np.linspace(float(theta_start), float(theta_end), nt + 1, dtype=np.float64)

    verts: List[List[float]] = []
    for i in range(nr + 1):
        for j in range(nt + 1):
            r = r_vals[i]
            th = theta_vals[j]
            verts.append([r * np.cos(th), r * np.sin(th)])

    def node(i: int, j: int) -> int:
        return i * (nt + 1) + j

    faces: List[List[int]] = []
    for i in range(nr):
        for j in range(nt):
            n00 = node(i, j)
            n10 = node(i + 1, j)
            n11 = node(i + 1, j + 1)
            n01 = node(i, j + 1)
            faces.append([n00, n10, n11])
            faces.append([n00, n11, n01])

    verts_xy = torch.tensor(np.asarray(verts, dtype=np.float32), dtype=torch.float32)
    faces_t = torch.tensor(np.asarray(faces, dtype=np.int64), dtype=torch.long)
    return verts_xy, faces_t


def choose_annular_sector_mesh_counts(
    *,
    r_inner: float,
    r_outer: float,
    theta_start: float,
    theta_end: float,
    mesh_lc: float,
    mesh_nr_min: int = 8,
    mesh_nt_min: int = 16,
) -> Tuple[int, int]:
    """
    Choose polar mesh counts from a target physical element size.

    Radial spacing is approximately mesh_lc. Angular spacing is chosen so the
    outer arc spacing is no larger than mesh_lc, which keeps all samples at a
    comparable physical mesh resolution while allowing different node counts.
    """
    mesh_lc = max(float(mesh_lc), 1.0e-12)
    radial_width = float(r_outer) - float(r_inner)
    theta_span = abs(float(theta_end) - float(theta_start))
    outer_arc_length = float(r_outer) * theta_span

    nr = max(int(np.ceil(radial_width / mesh_lc)), int(mesh_nr_min), 1)
    nt = max(int(np.ceil(outer_arc_length / mesh_lc)), int(mesh_nt_min), 1)
    return nr, nt


def make_annular_sector_x0_clamped_bc_factor(
    verts_xy: torch.Tensor,
    *,
    tol: float = 1.0e-5,
) -> torch.Tensor:
    """
    Boundary-condition factor for the annular-sector case.

    0 = fixed and 1 = free. The x = 0 radial edge is clamped.
    """
    x = verts_xy[:, 0]
    fixed = torch.isclose(x, torch.tensor(0.0, dtype=x.dtype, device=x.device), atol=float(tol))
    bc_factor = torch.ones((verts_xy.shape[0], 3), dtype=torch.float32)
    bc_factor[fixed, :] = 0.0
    return bc_factor


def make_annular_sector_fem_boundary_point_cloud(
    *,
    verts_xy: torch.Tensor,
    faces: torch.Tensor,
    r_inner: float,
    r_outer: float,
    theta_start: float,
    theta_end: float,
    fixed_tol: float = 1.0e-5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build the Geometry Encoder point cloud from all FEM boundary nodes.

    Boundary nodes are inferred from triangle edges that belong to exactly one
    element. The fixed flag marks the radial edge at theta_end = 90 deg, which
    is the x = 0 clamped edge for this annular-sector setup.
    """
    verts_np = verts_xy.detach().cpu().numpy().astype(np.float64)
    faces_np = faces.detach().cpu().numpy().astype(np.int64)

    edge_count: dict[tuple[int, int], int] = {}
    for tri in faces_np:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edge = (int(a), int(b)) if int(a) < int(b) else (int(b), int(a))
            edge_count[edge] = edge_count.get(edge, 0) + 1

    boundary_nodes = sorted(
        {idx for edge, count in edge_count.items() if count == 1 for idx in edge}
    )
    if not boundary_nodes:
        raise ValueError("No FEM boundary nodes were found from annular-sector faces.")

    boundary_node_ids = np.asarray(boundary_nodes, dtype=np.int64)
    pc_xy = verts_np[boundary_node_ids]
    pc = np.zeros((pc_xy.shape[0], 3), dtype=np.float32)
    pc[:, :2] = pc_xy.astype(np.float32)

    fixed = np.isclose(pc_xy[:, 0], 0.0, atol=float(fixed_tol))
    pc_fixed_flag = fixed.astype(np.int64)

    theta_inner = np.asarray([theta_end, theta_start], dtype=np.float64)
    theta_outer = np.asarray([theta_start, theta_end], dtype=np.float64)
    inner_arc = _polar_to_xy(np.full_like(theta_inner, float(r_inner)), theta_inner)
    outer_arc = _polar_to_xy(np.full_like(theta_outer, float(r_outer)), theta_outer)
    radial_start = _polar_to_xy(
        np.asarray([float(r_inner), float(r_outer)], dtype=np.float64),
        np.full((2,), float(theta_start), dtype=np.float64),
    )
    radial_end = _polar_to_xy(
        np.asarray([float(r_outer), float(r_inner)], dtype=np.float64),
        np.full((2,), float(theta_end), dtype=np.float64),
    )
    boundary_segments = np.asarray(
        [
            [inner_arc[0], inner_arc[-1]],
            [radial_start[0], radial_start[-1]],
            [outer_arc[0], outer_arc[-1]],
            [radial_end[0], radial_end[-1]],
        ],
        dtype=np.float32,
    )

    return (
        torch.tensor(pc, dtype=torch.float32),
        torch.tensor(pc_fixed_flag, dtype=torch.long),
        torch.tensor(boundary_segments, dtype=torch.float32),
        torch.tensor(boundary_node_ids, dtype=torch.long),
    )


def build_one_annular_sector_sample(
    *,
    sample_id: int,
    seed: int,
    r_inner_min: float,
    r_inner_max: float,
    r_outer_min: float,
    r_outer_max: float,
    theta_span_deg_min: float,
    theta_span_deg_max: float,
    theta_end_deg: float,
    mesh_lc: float,
    mesh_nr_min: int,
    mesh_nt_min: int,
    boundary_point_spacing: float,
    E_mean: float,
    E_std: float,
    E_min: float,
    E_max: float,
    E_corr_len: float,
    E_n_modes: int,
    nu_mean: float,
    nu_std: float,
    nu_min: float,
    nu_max: float,
    nu_corr_len: float,
    nu_n_modes: int,
    thickness: float,
    line_load_mean: float = 0.020,
    line_load_min: float = 0.0,
    line_load_max: float = 0.060,
    line_load_log_std: float = 0.85,
    line_load_corr_len: float = 0.30,
    load_type: str = "random_nonuniform_boundary_line_load",
    use_load_field_input: bool = True,
) -> Dict[str, Any]:
    rng = np.random.default_rng(int(seed) + int(sample_id))

    r_inner = float(rng.uniform(float(r_inner_min), float(r_inner_max)))
    r_outer = float(rng.uniform(float(r_outer_min), float(r_outer_max)))
    if r_outer <= r_inner:
        raise ValueError(f"Expected r_outer > r_inner, got {r_outer} <= {r_inner}")

    theta_span_deg = float(rng.uniform(float(theta_span_deg_min), float(theta_span_deg_max)))
    theta_end = np.deg2rad(float(theta_end_deg))
    theta_start = theta_end - np.deg2rad(theta_span_deg)
    mesh_nr, mesh_nt = choose_annular_sector_mesh_counts(
        r_inner=r_inner,
        r_outer=r_outer,
        theta_start=theta_start,
        theta_end=theta_end,
        mesh_lc=float(mesh_lc),
        mesh_nr_min=int(mesh_nr_min),
        mesh_nt_min=int(mesh_nt_min),
    )

    verts_xy, faces = triangulate_annular_sector_structured(
        r_inner=r_inner,
        r_outer=r_outer,
        theta_start=theta_start,
        theta_end=theta_end,
        nr=int(mesh_nr),
        nt=int(mesh_nt),
    )
    n_nodes = int(verts_xy.shape[0])

    query = torch.zeros((n_nodes, 3), dtype=torch.float32)
    query[:, :2] = verts_xy

    x_min = float(torch.min(verts_xy[:, 0]).item())
    x_max = float(torch.max(verts_xy[:, 0]).item())
    y_min = float(torch.min(verts_xy[:, 1]).item())
    y_max = float(torch.max(verts_xy[:, 1]).item())
    bbox = torch.tensor([x_min, x_max, y_min, y_max], dtype=torch.float32)

    query_net = normalize_points_by_bbox(query, bbox)

    pc, pc_fixed_flag, boundary_segments, boundary_node_ids = make_annular_sector_fem_boundary_point_cloud(
        verts_xy=verts_xy,
        faces=faces,
        r_inner=r_inner,
        r_outer=r_outer,
        theta_start=theta_start,
        theta_end=theta_end,
    )
    pc_net = normalize_points_by_bbox(pc, bbox)

    bc_factor = make_annular_sector_x0_clamped_bc_factor(verts_xy)

    load_type_key = str(load_type).lower().strip()
    supported_boundary_loads = {
        "boundary_line_load",
        "random_boundary_line_load",
        "random_nonuniform_boundary_line_load",
    }
    if load_type_key not in supported_boundary_loads:
        raise ValueError(
            f"Unsupported load_type={load_type!r}. "
            "Use 'random_nonuniform_boundary_line_load'."
        )

    # The loaded boundary is the free radial edge at theta_start (j=0 in the
    # structured polar mesh), opposite the x=0 clamped radial edge.  Store the
    # edge connectivity explicitly so the NO energy can integrate q*w over the
    # one-dimensional boundary rather than incorrectly over the plate area.
    loaded_boundary_nodes = torch.arange(
        0,
        (int(mesh_nr) + 1) * (int(mesh_nt) + 1),
        int(mesh_nt) + 1,
        dtype=torch.long,
    )
    loaded_boundary_edges = torch.stack(
        [loaded_boundary_nodes[:-1], loaded_boundary_nodes[1:]], dim=1
    )

    # Smooth lognormal random profile on normalized edge coordinate s.  The
    # profile is normalized by trapezoidal integration, giving every sample the
    # requested mean line load while allowing large pointwise variation.
    s = np.linspace(0.0, 1.0, int(mesh_nr) + 1, dtype=np.float64)
    ds = np.abs(s[:, None] - s[None, :])
    corr = max(float(line_load_corr_len), 1.0e-3)
    covariance = np.exp(-0.5 * (ds / corr) ** 2) + 1.0e-8 * np.eye(len(s))
    load_rng = np.random.default_rng(int(seed) + int(sample_id) + 1_000_000)
    z = load_rng.multivariate_normal(np.zeros(len(s)), covariance)
    z = (z - np.mean(z)) / max(float(np.std(z)), 1.0e-12)
    profile = np.exp(float(line_load_log_std) * z)
    profile /= max(float(np.trapezoid(profile, s)), 1.0e-12)

    q_edge = float(line_load_mean) * profile
    q_edge = np.clip(q_edge, float(line_load_min), float(line_load_max))
    # Re-normalize after clipping.  A final clip protects against roundoff; the
    # configured upper bound is deliberately generous for the large variation.
    q_edge *= float(line_load_mean) / max(float(np.trapezoid(q_edge, s)), 1.0e-12)
    q_edge = np.clip(q_edge, float(line_load_min), float(line_load_max))
    q_edge_t = torch.tensor(q_edge[:, None], dtype=torch.float32)

    # Compatibility/visualization field: zero throughout the domain and equal
    # to the physical line load only at nodes on the loaded boundary.
    pressure = torch.zeros((n_nodes, 1), dtype=torch.float32)
    pressure[loaded_boundary_nodes] = q_edge_t
    pressure_net = torch.zeros_like(pressure)
    pressure_net[loaded_boundary_nodes] = normalize_pressure_field_for_network(
        q_edge_t,
        pressure_min=float(line_load_min),
        pressure_max=float(line_load_max),
    )

    q_actual_mean = float(np.trapezoid(q_edge, s))
    load_scalar = torch.tensor(
        [
            normalize_uniform_load_for_network(
                q_actual_mean,
                line_load_min,
                line_load_max,
            )
        ],
        dtype=torch.float32,
    )
    # Load-function point cloud for the load-field encoder.  Keep every FEM
    # node, as in the domain-pressure cases: the loaded radial edge carries
    # q_line_net while all other nodes explicitly carry zero.  This gives FPS
    # enough distinct spatial points for the configured n_point=128.
    load_pc_net = torch.cat([query_net[:, :3], pressure_net], dim=1).float()
    load_fields = {
        "use_load_field": bool(use_load_field_input),
        "pressure_net": pressure_net.clone(),
        "load_pc_net": load_pc_net.clone(),
        "pressure_value": torch.tensor([q_actual_mean], dtype=torch.float32),
        "pressure_target_mean": torch.tensor([float(line_load_mean)], dtype=torch.float32),
        "pressure_min": torch.tensor([float(line_load_min)], dtype=torch.float32),
        "pressure_max": torch.tensor([float(line_load_max)], dtype=torch.float32),
        "line_load_mean": torch.tensor([q_actual_mean], dtype=torch.float32),
        "line_load_target_mean": torch.tensor([float(line_load_mean)], dtype=torch.float32),
        "line_load_min": torch.tensor([float(line_load_min)], dtype=torch.float32),
        "line_load_max": torch.tensor([float(line_load_max)], dtype=torch.float32),
        "line_load_log_std": torch.tensor([float(line_load_log_std)], dtype=torch.float32),
        "line_load_corr_len": torch.tensor([float(line_load_corr_len)], dtype=torch.float32),
        "loaded_boundary_nodes": loaded_boundary_nodes.clone(),
        "loaded_boundary_edges": loaded_boundary_edges.clone(),
        "load_type": "random_nonuniform_boundary_line_load",
    }

    geom_face = precompute_plate_face_geometry(verts_xy, faces)
    centroid_xy = geom_face["centroid_xy"].float()
    centroid_xyz = torch.zeros((centroid_xy.shape[0], 3), dtype=torch.float32)
    centroid_xyz[:, :2] = centroid_xy
    centroid_xyz_net = normalize_points_by_bbox(centroid_xyz, bbox)

    E_elem = generate_gaussian_random_field_on_points(
        centroid_xy,
        rng=rng,
        mean=float(E_mean),
        std=float(E_std),
        corr_len=float(E_corr_len),
        n_modes=int(E_n_modes),
        value_min=float(E_min),
        value_max=float(E_max),
    )
    nu_elem = generate_gaussian_random_field_on_points(
        centroid_xy,
        rng=rng,
        mean=float(nu_mean),
        std=float(nu_std),
        corr_len=float(nu_corr_len),
        n_modes=int(nu_n_modes),
        value_min=float(nu_min),
        value_max=float(nu_max),
    )

    E_elem_mean = float(torch.mean(E_elem).item())
    nu_elem_mean = float(torch.mean(nu_elem).item())

    E_net = normalize_scalar_field_for_network(E_elem, E_min, E_max)
    nu_net = normalize_scalar_field_for_network(nu_elem, nu_min, nu_max)
    material_pc_net = torch.cat([centroid_xyz_net[:, :3], E_net, nu_net], dim=1).float()

    E_scalar_net = normalize_scalar_field_for_network(
        torch.tensor([[E_elem_mean]], dtype=torch.float32),
        E_min,
        E_max,
    ).reshape(1)
    nu_scalar_net = normalize_scalar_field_for_network(
        torch.tensor([[nu_elem_mean]], dtype=torch.float32),
        nu_min,
        nu_max,
    ).reshape(1)
    scalar_param = torch.cat([E_scalar_net, nu_scalar_net, load_scalar.reshape(1)], dim=0).float()

    sample = {
        "pc": pc,
        "pc_net": pc_net,
        "query": query,
        "query_net": query_net,
        "verts_xy": verts_xy.clone(),
        "faces": faces.clone(),
        "geom_face": geom_face,
        "bc_factor": bc_factor.clone(),
        "pressure": pressure.clone(),
        "load_scalar": load_scalar.clone(),
        "scalar_param": scalar_param.clone(),
        "use_material_field": True,
        "E": torch.tensor([float(E_elem_mean)], dtype=torch.float32),
        "nu": torch.tensor([float(nu_elem_mean)], dtype=torch.float32),
        "thickness": torch.tensor([float(thickness)], dtype=torch.float32),
        "E_elem": E_elem.clone(),
        "nu_elem": nu_elem.clone(),
        "E_net": E_net.clone(),
        "nu_net": nu_net.clone(),
        "material_pc_net": material_pc_net.clone(),
        "material_type": "gaussian_random_field",
        "E_mean": torch.tensor([float(E_mean)], dtype=torch.float32),
        "E_std": torch.tensor([float(E_std)], dtype=torch.float32),
        "E_min": torch.tensor([float(E_min)], dtype=torch.float32),
        "E_max": torch.tensor([float(E_max)], dtype=torch.float32),
        "E_corr_len": torch.tensor([float(E_corr_len)], dtype=torch.float32),
        "E_n_modes": torch.tensor([int(E_n_modes)], dtype=torch.int64),
        "nu_mean": torch.tensor([float(nu_mean)], dtype=torch.float32),
        "nu_std": torch.tensor([float(nu_std)], dtype=torch.float32),
        "nu_min": torch.tensor([float(nu_min)], dtype=torch.float32),
        "nu_max": torch.tensor([float(nu_max)], dtype=torch.float32),
        "nu_corr_len": torch.tensor([float(nu_corr_len)], dtype=torch.float32),
        "nu_n_modes": torch.tensor([int(nu_n_modes)], dtype=torch.int64),
        "sample_id": int(sample_id),
        "lx": float(x_max - x_min),
        "ly": float(y_max - y_min),
        "bbox": bbox.clone(),
        "geometry_type": "annular_sector",
        "load_type": str(load_fields["load_type"]),
        "use_load_field": bool(use_load_field_input),
        "annular_center": torch.tensor([0.0, 0.0], dtype=torch.float32),
        "r_inner": torch.tensor([float(r_inner)], dtype=torch.float32),
        "r_outer": torch.tensor([float(r_outer)], dtype=torch.float32),
        "theta_start": torch.tensor([float(theta_start)], dtype=torch.float32),
        "theta_end": torch.tensor([float(theta_end)], dtype=torch.float32),
        "theta_span": torch.tensor([float(theta_end - theta_start)], dtype=torch.float32),
        "theta_start_deg": torch.tensor([float(np.rad2deg(theta_start))], dtype=torch.float32),
        "theta_end_deg": torch.tensor([float(theta_end_deg)], dtype=torch.float32),
        "theta_span_deg": torch.tensor([float(theta_span_deg)], dtype=torch.float32),
        "boundary_segments": boundary_segments.clone(),
        "boundary_nodes": boundary_node_ids.clone(),
        "pc_fixed_flag_boundary": pc_fixed_flag.clone(),
        "pc_fixed_flag": pc_fixed_flag.clone(),
        "mesh_type": "polar_structured",
        "mesh_lc": torch.tensor([float(mesh_lc)], dtype=torch.float32),
        "mesh_nr": torch.tensor([int(mesh_nr)], dtype=torch.int64),
        "mesh_nt": torch.tensor([int(mesh_nt)], dtype=torch.int64),
    }
    sample.update(load_fields)

    return sample


def build_annular_sector_dataset(
    *,
    n_train: int = 200,
    n_test: int = 40,
    seed: int = 42,
    **kwargs,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    train_samples: List[Dict[str, Any]] = []
    test_samples: List[Dict[str, Any]] = []

    for i in range(int(n_train)):
        train_samples.append(
            build_one_annular_sector_sample(
                sample_id=i,
                seed=int(seed),
                **kwargs,
            )
        )

    offset = int(n_train)
    for i in range(int(n_test)):
        test_samples.append(
            build_one_annular_sector_sample(
                sample_id=offset + i,
                seed=int(seed),
                **kwargs,
            )
        )

    return train_samples, test_samples


def save_debug_mesh_plot(sample: Dict[str, Any], out_path: str | Path) -> None:
    """
    Save a debug mesh figure using the same style as the other prep modules.

    The triangular mesh is black. Free boundaries are green and the fixed x=0
    radial boundary is red.
    """
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _to_np(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def _scalar(name: str) -> float:
        return float(np.asarray(_to_np(sample[name])).reshape(-1)[0])

    verts_xy = _to_np(sample["verts_xy"])
    faces = _to_np(sample["faces"]).astype(np.int64)
    r_inner = _scalar("r_inner")
    r_outer = _scalar("r_outer")
    theta_start = _scalar("theta_start")
    theta_end = _scalar("theta_end")

    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    ax.triplot(
        verts_xy[:, 0],
        verts_xy[:, 1],
        faces,
        linewidth=0.35,
        alpha=0.75,
        color="black",
    )

    n_arc = 160
    theta_free_inner = np.linspace(theta_end, theta_start, n_arc)
    theta_free_outer = np.linspace(theta_start, theta_end, n_arc)
    inner_arc = _polar_to_xy(np.full_like(theta_free_inner, r_inner), theta_free_inner)
    outer_arc = _polar_to_xy(np.full_like(theta_free_outer, r_outer), theta_free_outer)
    r_edge = np.linspace(r_inner, r_outer, 80)
    radial_start = _polar_to_xy(r_edge, np.full_like(r_edge, theta_start))
    radial_fixed = _polar_to_xy(r_edge, np.full_like(r_edge, theta_end))

    for i, curve in enumerate((inner_arc, radial_start, outer_arc)):
        ax.plot(
            curve[:, 0],
            curve[:, 1],
            linewidth=2.2,
            color="green",
            label="free boundary" if i == 0 else None,
        )

    ax.plot(
        radial_fixed[:, 0],
        radial_fixed[:, 1],
        linewidth=2.6,
        color="red",
        label="fixed boundary",
    )

    bbox = _to_np(sample["bbox"]).reshape(-1)
    x_min, x_max, y_min, y_max = [float(v) for v in bbox]
    pad = 0.03 * max(x_max - x_min, y_max - y_min, 1.0)

    ax.set_aspect("equal")
    ax.set_xlim(-0.1, 5.1)
    ax.set_ylim(-0.1, 5.1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    os.makedirs(os.path.dirname(str(out_path)), exist_ok=True)
    fig.savefig(str(out_path), dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_encoder_pc_plot(sample: Dict[str, Any], out_path: str | Path) -> None:
    """
    Plot only the active boundary point cloud entering the Geometry Encoder.
    """
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _to_np(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    pc = sample.get("pc", None)
    if pc is None:
        raise KeyError("sample['pc'] is required for encoder point-cloud plotting.")

    pc_np = _to_np(pc)
    bbox = _to_np(sample["bbox"]).reshape(-1)
    x_min, x_max, y_min, y_max = [float(v) for v in bbox]
    pad = 0.03 * max(x_max - x_min, y_max - y_min, 1.0)

    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    ax.scatter(
        pc_np[:, 0],
        pc_np[:, 1],
        s=8,
        alpha=0.85,
        color="red",
    )

    ax.set_aspect("equal")
    ax.set_xlim(-0.1, 5.1)
    ax.set_ylim(-0.1, 5.1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    os.makedirs(os.path.dirname(str(out_path)), exist_ok=True)
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
