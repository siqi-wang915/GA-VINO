from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
except ImportError as e:
    raise ImportError(
        "The two-hole case requires shapely. Please install it with: pip install shapely"
    ) from e

from NO.problem.common import (
    generate_gaussian_random_pressure_field,
    normalize_points_by_bbox,
    normalize_pressure_field_for_network,
    normalize_uniform_load_for_network,
    precompute_plate_face_geometry,
)


# -----------------------------------------------------------------------------
# Boundary and hole geometry
# -----------------------------------------------------------------------------

def generate_square_boundary_points(
    lx: float = 5.0,
    ly: float = 5.0,
    n_per_edge: int = 100,
) -> np.ndarray:
    """
    Generate ordered square boundary points.

    Order:
        bottom -> right -> top -> left

    Corners are not duplicated.
    """
    t = np.linspace(0.0, 1.0, int(n_per_edge), endpoint=False)

    bottom = np.stack([lx * t, np.zeros_like(t)], axis=1)
    right = np.stack([np.full_like(t, lx), ly * t], axis=1)
    top = np.stack([lx * (1.0 - t), np.full_like(t, ly)], axis=1)
    left = np.stack([np.zeros_like(t), ly * (1.0 - t)], axis=1)

    return np.vstack([bottom, right, top, left]).astype(np.float64)


def generate_random_hole_boundary(
    *,
    rng: np.random.Generator,
    center: Tuple[float, float],
    n_points: int = 100,
    mean_radius: float = 0.60,
    r_min: float = 0.20,
    r_max: float = 1.15,
    roughness: float = 0.4,
    n_modes: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate one arbitrary hole using the same perturbation style as the old code.

    r(theta) = mean_radius * [1 + perturb(theta)]

    perturb is a low-frequency Fourier perturbation.
    """
    theta = np.linspace(0.0, 2.0 * np.pi, int(n_points), endpoint=False)

    perturb = np.zeros_like(theta)
    for k in range(1, int(n_modes) + 1):
        a = rng.normal(0.0, roughness / k)
        b = rng.normal(0.0, roughness / k)
        perturb += a * np.cos(k * theta) + b * np.sin(k * theta)

    perturb = np.clip(perturb, -0.35, 0.35)

    radius = mean_radius * (1.0 + perturb)
    radius = np.clip(radius, r_min, r_max)

    cx, cy = center
    x = cx + radius * np.cos(theta)
    y = cy + radius * np.sin(theta)

    hole_xy = np.stack([x, y], axis=1).astype(np.float64)

    return hole_xy, radius.astype(np.float64)


# -----------------------------------------------------------------------------
# Shapely helpers
# -----------------------------------------------------------------------------

def _clean_ring_xy(xy: np.ndarray, tol: float = 1.0e-12) -> np.ndarray:
    pts = np.asarray(xy, dtype=np.float64)

    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"Expected ring shape [N,2], got {pts.shape}")

    if len(pts) >= 2 and np.linalg.norm(pts[0] - pts[-1]) < tol:
        pts = pts[:-1]

    cleaned = [pts[0]]
    for p in pts[1:]:
        if np.linalg.norm(p - cleaned[-1]) >= tol:
            cleaned.append(p)

    pts = np.asarray(cleaned, dtype=np.float64)

    if len(pts) < 3:
        raise ValueError("A ring needs at least three distinct points.")

    return pts


def _ring_signed_area(xy: np.ndarray) -> float:
    pts = _clean_ring_xy(xy)
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def _ensure_ring_orientation(xy: np.ndarray, *, ccw: bool) -> np.ndarray:
    pts = _clean_ring_xy(xy)
    area = _ring_signed_area(pts)

    if ccw and area < 0.0:
        pts = pts[::-1].copy()
    elif (not ccw) and area > 0.0:
        pts = pts[::-1].copy()

    return pts


def _make_valid_polygon(xy: np.ndarray, min_area: float = 1.0e-6) -> Polygon:
    ring = _clean_ring_xy(xy)
    poly = Polygon(ring)

    if not poly.is_valid:
        poly = poly.buffer(0.0)

    if poly.is_empty:
        raise RuntimeError("Generated hole polygon is empty.")

    if poly.geom_type != "Polygon":
        raise RuntimeError(f"Generated hole is not a single Polygon: {poly.geom_type}")

    if poly.area < min_area:
        raise RuntimeError(f"Generated hole area is too small: {poly.area}")

    return poly


def _build_domain_from_two_holes(
    *,
    lx: float,
    ly: float,
    hole1_xy: np.ndarray,
    hole2_xy: np.ndarray,
) -> Tuple[List[np.ndarray], str, float]:
    """
    Build effective hole boundaries after union of two raw holes.

    If the two holes overlap, Shapely merges them into one effective hole.
    If separated, the final domain has two interior rings.
    """
    square = Polygon([(0.0, 0.0), (lx, 0.0), (lx, ly), (0.0, ly)])

    hole1 = _make_valid_polygon(hole1_xy)
    hole2 = _make_valid_polygon(hole2_xy)

    hole_union = unary_union([hole1, hole2])
    if not hole_union.is_valid:
        hole_union = hole_union.buffer(0.0)

    if hole_union.is_empty:
        raise RuntimeError("Union of two holes is empty.")

    if not square.contains(hole_union):
        raise RuntimeError("Generated hole union is not fully inside the square.")

    domain = square.difference(hole_union)
    if not domain.is_valid:
        domain = domain.buffer(0.0)

    if domain.is_empty:
        raise RuntimeError("Generated plate domain is empty.")

    if domain.geom_type != "Polygon":
        raise RuntimeError(
            f"Expected square minus holes to be one Polygon, got {domain.geom_type}."
        )

    hole_rings: List[np.ndarray] = []
    for interior in domain.interiors:
        ring = np.asarray(interior.coords, dtype=np.float64)[:, :2]
        hole_rings.append(_clean_ring_xy(ring))

    if len(hole_rings) == 0:
        raise RuntimeError("No interior ring found after subtracting holes.")

    merge_state = "merged" if len(hole_rings) == 1 else "separated"
    hole_union_area = float(hole_union.area)

    return hole_rings, merge_state, hole_union_area


# -----------------------------------------------------------------------------
# Gmsh helpers
# -----------------------------------------------------------------------------

def _add_polyline_loop(
    gmsh,
    xy: np.ndarray,
    lc: float,
    *,
    ccw: bool,
) -> Tuple[int, List[int]]:
    """
    Add a closed polyline to gmsh and return loop tag and line tags.
    """
    pts = _ensure_ring_orientation(xy, ccw=ccw)

    point_tags: List[int] = []
    for x, y in pts:
        point_tags.append(gmsh.model.geo.addPoint(float(x), float(y), 0.0, lc))

    line_tags: List[int] = []
    n = len(point_tags)
    for i in range(n):
        line_tags.append(
            gmsh.model.geo.addLine(point_tags[i], point_tags[(i + 1) % n])
        )

    loop_tag = gmsh.model.geo.addCurveLoop(line_tags)
    return loop_tag, line_tags


def _extract_triangles_from_gmsh(gmsh) -> Tuple[np.ndarray, np.ndarray, Dict[int, int]]:
    node_tags, coords, _ = gmsh.model.mesh.getNodes()

    node_tags = np.asarray(node_tags, dtype=np.int64)
    coords = np.asarray(coords, dtype=np.float64).reshape(-1, 3)

    order = np.argsort(node_tags)
    node_tags = node_tags[order]
    coords = coords[order]

    tag_to_index = {int(tag): i for i, tag in enumerate(node_tags)}
    verts_xy = coords[:, :2].astype(np.float32)

    elem_types, _, elem_node_tags = gmsh.model.mesh.getElements(dim=2)

    tri_faces: List[np.ndarray] = []

    for etype, nodes in zip(elem_types, elem_node_tags):
        _, dim, _, num_nodes, _, _ = gmsh.model.mesh.getElementProperties(etype)

        if int(dim) == 2 and int(num_nodes) == 3:
            nodes = np.asarray(nodes, dtype=np.int64).reshape(-1, 3)
            faces = np.vectorize(lambda t: tag_to_index[int(t)])(nodes)
            tri_faces.append(faces.astype(np.int64))

    if not tri_faces:
        raise RuntimeError("No 3-node triangular elements were generated by gmsh.")

    faces_np = np.vstack(tri_faces).astype(np.int64)

    # Ensure positive orientation.
    tri = verts_xy[faces_np]
    signed_area2 = (
        (tri[:, 1, 0] - tri[:, 0, 0]) * (tri[:, 2, 1] - tri[:, 0, 1])
        - (tri[:, 2, 0] - tri[:, 0, 0]) * (tri[:, 1, 1] - tri[:, 0, 1])
    )
    flip = signed_area2 < 0.0
    if np.any(flip):
        faces_np[flip] = faces_np[flip][:, [0, 2, 1]]

    return verts_xy, faces_np, tag_to_index


def _extract_nodes_on_curves(
    gmsh,
    line_tags: List[int],
    tag_to_index: Dict[int, int],
) -> np.ndarray:
    out: List[int] = []

    for line in line_tags:
        node_tags, _, _ = gmsh.model.mesh.getNodes(
            dim=1,
            tag=line,
            includeBoundary=True,
        )

        for t in node_tags:
            t = int(t)
            if t in tag_to_index:
                out.append(tag_to_index[t])

    if len(out) == 0:
        return np.empty((0,), dtype=np.int64)

    return np.asarray(sorted(set(out)), dtype=np.int64)


# -----------------------------------------------------------------------------
# Two-hole mesh generation
# -----------------------------------------------------------------------------

def _sample_two_hole_centers(
    rng: np.random.Generator,
    *,
    center_min: float,
    center_max: float,
    center_distance_min: float,
    max_tries: int = 200,
) -> np.ndarray:
    for _ in range(max_tries):
        c1 = rng.uniform(center_min, center_max, size=2)
        c2 = rng.uniform(center_min, center_max, size=2)

        if np.linalg.norm(c1 - c2) >= center_distance_min:
            return np.stack([c1, c2], axis=0).astype(np.float64)

    raise RuntimeError("Failed to sample two valid hole centers.")


def build_two_holes_mesh(
    *,
    sample_id: int,
    seed: int = 42,
    lx: float = 5.0,
    ly: float = 5.0,
    outer_n_per_edge: int = 100,
    hole_n_points: int = 100,
    mean_radius_min: float = 0.40,
    mean_radius_max: float = 0.75,
    r_min: float = 0.20,
    r_max: float = 1.15,
    roughness: float = 0.4,
    n_modes: int = 2,
    center_min: float = 1.25,
    center_max: float = 3.75,
    center_distance_min: float = 0.25,
    mesh_lc: float = 0.225,
) -> Dict:
    try:
        import gmsh
    except Exception as e:
        raise ImportError(
            "gmsh is required for the two-hole case. Please install gmsh."
        ) from e

    rng = np.random.default_rng(seed + int(sample_id))

    outer_xy = generate_square_boundary_points(
        lx=lx,
        ly=ly,
        n_per_edge=outer_n_per_edge,
    )

    centers = _sample_two_hole_centers(
        rng,
        center_min=center_min,
        center_max=center_max,
        center_distance_min=center_distance_min,
    )

    mean_radii = rng.uniform(mean_radius_min, mean_radius_max, size=2).astype(np.float64)

    hole1_xy, radius1 = generate_random_hole_boundary(
        rng=rng,
        center=(float(centers[0, 0]), float(centers[0, 1])),
        n_points=hole_n_points,
        mean_radius=float(mean_radii[0]),
        r_min=r_min,
        r_max=r_max,
        roughness=roughness,
        n_modes=n_modes,
    )

    hole2_xy, radius2 = generate_random_hole_boundary(
        rng=rng,
        center=(float(centers[1, 0]), float(centers[1, 1])),
        n_points=hole_n_points,
        mean_radius=float(mean_radii[1]),
        r_min=r_min,
        r_max=r_max,
        roughness=roughness,
        n_modes=n_modes,
    )

    hole_rings, merge_state, hole_union_area = _build_domain_from_two_holes(
        lx=lx,
        ly=ly,
        hole1_xy=hole1_xy,
        hole2_xy=hole2_xy,
    )

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)

    try:
        gmsh.model.add(f"square_two_holes_{sample_id}")
        gmsh.option.setNumber("Mesh.Algorithm", 6)

        outer_loop, outer_lines = _add_polyline_loop(
            gmsh,
            outer_xy,
            mesh_lc,
            ccw=True,
        )

        hole_loops: List[int] = []
        all_hole_lines: List[int] = []

        for ring in hole_rings:
            hole_loop, hole_lines = _add_polyline_loop(
                gmsh,
                ring,
                mesh_lc,
                ccw=False,
            )
            hole_loops.append(hole_loop)
            all_hole_lines.extend(hole_lines)

        surface = gmsh.model.geo.addPlaneSurface([outer_loop, *hole_loops])
        gmsh.model.geo.synchronize()

        gmsh.model.addPhysicalGroup(1, outer_lines, tag=1, name="outer_boundary")
        gmsh.model.addPhysicalGroup(1, all_hole_lines, tag=2, name="hole_boundary")
        gmsh.model.addPhysicalGroup(2, [surface], tag=10, name="plate_domain")

        gmsh.model.mesh.generate(2)

        verts_xy, faces, tag_to_index = _extract_triangles_from_gmsh(gmsh)
        outer_boundary_nodes = _extract_nodes_on_curves(gmsh, outer_lines, tag_to_index)
        hole_boundary_nodes = _extract_nodes_on_curves(gmsh, all_hole_lines, tag_to_index)

    finally:
        gmsh.finalize()

    pc_parts = [outer_xy.astype(np.float32)]
    for ring in hole_rings:
        pc_parts.append(_clean_ring_xy(ring).astype(np.float32))

    pc_xy = np.vstack(pc_parts).astype(np.float32)

    return {
        "verts_xy": verts_xy.astype(np.float32),
        "faces": faces.astype(np.int64),

        "outer_boundary_nodes": outer_boundary_nodes.astype(np.int64),
        "hole_boundary_nodes": hole_boundary_nodes.astype(np.int64),

        "pc_xy": pc_xy.astype(np.float32),
        "outer_xy": outer_xy.astype(np.float32),
        "hole_rings": [r.astype(np.float32) for r in hole_rings],

        "hole_centers": centers.astype(np.float32),
        "hole_mean_radii": mean_radii.astype(np.float32),
        "hole1_xy_raw": hole1_xy.astype(np.float32),
        "hole2_xy_raw": hole2_xy.astype(np.float32),
        "hole1_radius": radius1.astype(np.float32),
        "hole2_radius": radius2.astype(np.float32),

        "hole_merge_state": str(merge_state),
        "n_hole_boundaries": int(len(hole_rings)),
        "hole_union_area": float(hole_union_area),
    }


# -----------------------------------------------------------------------------
# Sample and dataset generation
# -----------------------------------------------------------------------------

def _select_geometry_point_cloud(
    *,
    query: torch.Tensor,
    pc_boundary: torch.Tensor,
    bbox: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Fixed boundary point-cloud input for the Geometry Encoder.

    The geometry encoder always receives the outer square boundary plus the
    effective hole boundaries. All/global FEM-node geometry point-cloud modes
    are no longer generated or saved.
    """
    query_net = normalize_points_by_bbox(query, bbox)
    pc_net = normalize_points_by_bbox(pc_boundary, bbox)

    return {
        "pc": pc_boundary,
        "pc_net": pc_net,
        "query_net": query_net,
    }


def make_holes2_sample(
    sample_id: int,
    *,
    seed: int = 42,
    lx: float = 5.0,
    ly: float = 5.0,
    E: float = 1000.0,
    nu: float = 0.3,
    thickness: float = 0.5,
    pressure_value: float = 0.80,
    pressure_min: float = 0.0,
    pressure_max: float = 1.60,
    pressure_mean: Optional[float] = None,
    pressure_std: float = 0.3,
    pressure_corr_len: float = 1.5,
    pressure_n_modes: int = 64,
    outer_n_per_edge: int = 100,
    hole_n_points: int = 100,
    mean_radius_min: float = 0.40,
    mean_radius_max: float = 0.75,
    r_min: float = 0.20,
    r_max: float = 1.15,
    roughness: float = 0.4,
    n_modes: int = 2,
    center_min: float = 1.25,
    center_max: float = 3.75,
    center_distance_min: float = 0.25,
    mesh_lc: float = 0.225,
    load_type: str = "gaussian_random_field_pressure",
    use_load_field_input: bool = True,
) -> Dict:
    if pressure_mean is None:
        pressure_mean = float(pressure_value)

    mesh_data = build_two_holes_mesh(
        sample_id=sample_id,
        seed=seed,
        lx=lx,
        ly=ly,
        outer_n_per_edge=outer_n_per_edge,
        hole_n_points=hole_n_points,
        mean_radius_min=mean_radius_min,
        mean_radius_max=mean_radius_max,
        r_min=r_min,
        r_max=r_max,
        roughness=roughness,
        n_modes=n_modes,
        center_min=center_min,
        center_max=center_max,
        center_distance_min=center_distance_min,
        mesh_lc=mesh_lc,
    )

    verts_xy = torch.tensor(mesh_data["verts_xy"], dtype=torch.float32)
    faces = torch.tensor(mesh_data["faces"], dtype=torch.long)

    n_nodes = verts_xy.shape[0]

    query = torch.zeros((n_nodes, 3), dtype=torch.float32)
    query[:, :2] = verts_xy

    pc_xy = torch.tensor(mesh_data["pc_xy"], dtype=torch.float32)
    pc_boundary = torch.zeros((pc_xy.shape[0], 3), dtype=torch.float32)
    pc_boundary[:, :2] = pc_xy

    bbox = torch.tensor([0.0, float(lx), 0.0, float(ly)], dtype=torch.float32)

    pc_info = _select_geometry_point_cloud(
        query=query,
        pc_boundary=pc_boundary,
        bbox=bbox,
    )

    bc_factor = torch.ones((n_nodes, 3), dtype=torch.float32)

    outer_nodes = torch.tensor(mesh_data["outer_boundary_nodes"], dtype=torch.long)
    hole_nodes = torch.tensor(mesh_data["hole_boundary_nodes"], dtype=torch.long)

    # Outer boundary is clamped.
    bc_factor[outer_nodes, :] = 0.0

    # Hole boundaries are free.
    bc_factor[hole_nodes, :] = 1.0

    # ------------------------------------------------------------------
    # Load generation.
    #
    # Supported:
    #   1. uniform_pressure
    #      - physical pressure is uniform;
    #      - no load_pc_net is provided;
    #      - load-field encoder is disabled.
    #
    #   2. gaussian_random_field_pressure
    #      - physical pressure is a GRF nodal field;
    #      - load_pc_net = [x_net, y_net, z_net, p_net];
    #      - load-field encoder can be enabled.
    # ------------------------------------------------------------------
    load_type_key = str(load_type).lower().strip()

    if load_type_key in ("uniform", "uniform_pressure", "up"):
        pressure_uniform_value = (
            float(pressure_mean) if pressure_mean is not None else float(pressure_value)
        )

        # Physical pressure field used by FEM and energy loss.
        pressure = torch.full(
            (n_nodes, 1),
            pressure_uniform_value,
            dtype=torch.float32,
        )

        load_scalar = torch.tensor(
            [
                normalize_uniform_load_for_network(
                    pressure_uniform_value,
                    pressure_min,
                    pressure_max,
                )
            ],
            dtype=torch.float32,
        )

 
        scalar_param = torch.zeros(3, dtype=torch.float32)

        load_fields = {
            "pressure": pressure.clone(),
            "load_scalar": load_scalar.clone(),
            "scalar_param": scalar_param.clone(),
            "use_material_field": False,
            "use_load_field": False,
            "pressure_value": torch.tensor([float(pressure_uniform_value)], dtype=torch.float32),
            "pressure_min": torch.tensor([float(pressure_min)], dtype=torch.float32),
            "pressure_max": torch.tensor([float(pressure_max)], dtype=torch.float32),
            "load_type": "uniform_pressure",
        }

    elif load_type_key in ("grf", "grfp", "grf_pressure", "gaussian_random_field_pressure"):
        rng = np.random.default_rng(seed + int(sample_id) + 1_000_000)

        # Physical pressure field used by FEM and energy loss.
        pressure = generate_gaussian_random_pressure_field(
            verts_xy,
            rng=rng,
            mean=float(pressure_mean),
            std=float(pressure_std),
            corr_len=float(pressure_corr_len),
            n_modes=int(pressure_n_modes),
            pressure_min=float(pressure_min),
            pressure_max=float(pressure_max),
        )

        pressure_actual_mean = float(torch.mean(pressure).item())

        pressure_net = normalize_pressure_field_for_network(
            pressure,
            pressure_min=pressure_min,
            pressure_max=pressure_max,
        )

        load_scalar = torch.tensor(
            [
                normalize_uniform_load_for_network(
                    pressure_actual_mean,
                    pressure_min,
                    pressure_max,
                )
            ],
            dtype=torch.float32,
        )

        # Load-function point cloud for the load-field encoder:
        # [x_net, y_net, z_net, p_net].
        load_pc_net = torch.cat(
            [pc_info["query_net"][:, :3], pressure_net],
            dim=1,
        ).float()

        scalar_param = torch.cat(
            [
                torch.zeros(1, dtype=torch.float32),
                torch.zeros(1, dtype=torch.float32),
                load_scalar.reshape(1),
            ],
            dim=0,
        )

        load_fields = {
            "pressure": pressure.clone(),
            "pressure_net": pressure_net.clone(),
            "load_pc_net": load_pc_net.clone(),
            "load_scalar": load_scalar.clone(),
            "scalar_param": scalar_param.clone(),
            "use_material_field": False,
            "use_load_field": bool(use_load_field_input),
            "pressure_value": torch.tensor([float(pressure_actual_mean)], dtype=torch.float32),
            "pressure_target_mean": torch.tensor([float(pressure_mean)], dtype=torch.float32),
            "pressure_min": torch.tensor([float(pressure_min)], dtype=torch.float32),
            "pressure_max": torch.tensor([float(pressure_max)], dtype=torch.float32),
            "pressure_std": torch.tensor([float(pressure_std)], dtype=torch.float32),
            "pressure_corr_len": torch.tensor([float(pressure_corr_len)], dtype=torch.float32),
            "pressure_n_modes": torch.tensor([int(pressure_n_modes)], dtype=torch.int64),
            "load_type": "gaussian_random_field_pressure",
        }

    else:
        raise ValueError(
            f"Unsupported load_type={load_type!r}. "
            "Use 'uniform_pressure' or 'gaussian_random_field_pressure'."
        )


    geom_face = precompute_plate_face_geometry(verts_xy, faces)

    sample = {
        # Active geometry point cloud used by the Geometry Encoder.
        "pc": pc_info["pc"],
        "pc_net": pc_info["pc_net"],

        # Query points used by the Solution Decoder and energy loss.
        "query": query,
        "query_net": pc_info["query_net"],

        "verts_xy": verts_xy.clone(),
        "faces": faces.clone(),
        "geom_face": geom_face,

        "bc_factor": bc_factor.clone(),

        "E": torch.tensor([float(E)], dtype=torch.float32),
        "nu": torch.tensor([float(nu)], dtype=torch.float32),
        "thickness": torch.tensor([float(thickness)], dtype=torch.float32),

        "sample_id": int(sample_id),
        "lx": float(lx),
        "ly": float(ly),
        "bbox": bbox.clone(),

        "geometry_type": "holes2",

        "outer_boundary_nodes": outer_nodes.clone(),
        "hole_boundary_nodes": hole_nodes.clone(),

        "outer_xy": torch.tensor(mesh_data["outer_xy"], dtype=torch.float32),
        "pc_xy": torch.tensor(mesh_data["pc_xy"], dtype=torch.float32),

        "hole_rings": mesh_data["hole_rings"],

        "hole_centers": torch.tensor(mesh_data["hole_centers"], dtype=torch.float32),
        "hole_mean_radii": torch.tensor(mesh_data["hole_mean_radii"], dtype=torch.float32),
        "hole1_xy_raw": torch.tensor(mesh_data["hole1_xy_raw"], dtype=torch.float32),
        "hole2_xy_raw": torch.tensor(mesh_data["hole2_xy_raw"], dtype=torch.float32),
        "hole1_radius": torch.tensor(mesh_data["hole1_radius"], dtype=torch.float32),
        "hole2_radius": torch.tensor(mesh_data["hole2_radius"], dtype=torch.float32),

        "hole_merge_state": str(mesh_data["hole_merge_state"]),
        "n_hole_boundaries": int(mesh_data["n_hole_boundaries"]),
        "hole_union_area": float(mesh_data["hole_union_area"]),
    }

    sample.update(load_fields)

    return sample


def build_holes2_dataset(
    *,
    n_train: int = 25,
    n_test: int = 5,
    seed: int = 42,
    lx: float = 5.0,
    ly: float = 5.0,
    E: float = 1000.0,
    nu: float = 0.3,
    thickness: float = 0.5,
    pressure_value: float = 0.80,
    pressure_min: float = 0.0,
    pressure_max: float = 1.60,
    pressure_mean: Optional[float] = None,
    pressure_std: float = 0.3,
    pressure_corr_len: float = 1.5,
    pressure_n_modes: int = 64,
    outer_n_per_edge: int = 100,
    hole_n_points: int = 100,
    mean_radius_min: float = 0.40,
    mean_radius_max: float = 0.75,
    r_min: float = 0.20,
    r_max: float = 1.15,
    roughness: float = 0.4,
    n_modes: int = 2,
    center_min: float = 1.25,
    center_max: float = 3.75,
    center_distance_min: float = 0.25,
    mesh_lc: float = 0.225,
    target_merged_ratio: float = 0.50,
    balance_merge_states: bool = False,
    max_retry: int = 50,
    load_type: str = "gaussian_random_field_pressure",
    use_load_field_input: bool = True,
) -> Tuple[List[Dict], List[Dict]]:
    if pressure_mean is None:
        pressure_mean = float(pressure_value)

    def _make_samples(start_id: int, count: int) -> List[Dict]:
        samples: List[Dict] = []

        desired_merged = int(round(count * target_merged_ratio))
        desired_separated = count - desired_merged
        state_count = {"merged": 0, "separated": 0}

        attempt = 0
        relaxed = not balance_merge_states

        max_attempts_before_relax = count + max_retry * max(count, 1)
        max_total_attempts = 2 * max_attempts_before_relax

        while len(samples) < count:
            if attempt > max_total_attempts:
                raise RuntimeError(
                    f"Too many mesh-generation failures. "
                    f"Got {len(samples)}/{count}, state_count={state_count}"
                )

            if attempt > max_attempts_before_relax and not relaxed:
                print("[Dataset][holes2] Balance constraint relaxed.")
                relaxed = True

            sid = start_id + attempt
            attempt += 1

            try:
                sample = make_holes2_sample(
                    sample_id=sid,
                    seed=seed,
                    lx=lx,
                    ly=ly,
                    E=E,
                    nu=nu,
                    thickness=thickness,
                    pressure_value=pressure_value,
                    pressure_min=pressure_min,
                    pressure_max=pressure_max,
                    pressure_mean=pressure_mean,
                    pressure_std=pressure_std,
                    pressure_corr_len=pressure_corr_len,
                    pressure_n_modes=pressure_n_modes,
                    outer_n_per_edge=outer_n_per_edge,
                    hole_n_points=hole_n_points,
                    mean_radius_min=mean_radius_min,
                    mean_radius_max=mean_radius_max,
                    r_min=r_min,
                    r_max=r_max,
                    roughness=roughness,
                    n_modes=n_modes,
                    center_min=center_min,
                    center_max=center_max,
                    center_distance_min=center_distance_min,
                    mesh_lc=mesh_lc,
                    load_type=load_type,
                    use_load_field_input=use_load_field_input,
                )

                state = str(sample["hole_merge_state"])
                if state not in state_count:
                    state = "merged" if int(sample["n_hole_boundaries"]) == 1 else "separated"

                if not relaxed:
                    if state == "merged" and state_count["merged"] >= desired_merged:
                        continue
                    if state == "separated" and state_count["separated"] >= desired_separated:
                        continue

                samples.append(sample)
                state_count[state] += 1

            except Exception as e:
                print(f"[Dataset][holes2] Failed sample_id={sid}: {e}")
                continue

        print(
            f"[Dataset][holes2] Built {count} samples from start_id={start_id}. "
            f"merged={state_count['merged']}, separated={state_count['separated']}"
        )

        return samples

    train_samples = _make_samples(0, n_train)
    test_samples = _make_samples(n_train * 10, n_test)

    return train_samples, test_samples


# -----------------------------------------------------------------------------
# Debug plots
# -----------------------------------------------------------------------------

def _to_np_debug(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _draw_holes2_boundaries(ax, sample: Dict, *, draw_outer: bool = True, draw_holes: bool = True):
    if draw_holes and "hole_rings" in sample:
        for ring in sample["hole_rings"]:
            r_np = _to_np_debug(ring)
            if r_np.size > 0:
                r_closed = np.vstack([r_np, r_np[0:1]])
                ax.plot(
                    r_closed[:, 0],
                    r_closed[:, 1],
                    linewidth=1.2,
                    alpha=0.9,
                    color="green",
                )

    if draw_outer:
        lx = float(sample.get("lx", 1.0))
        ly = float(sample.get("ly", 1.0))
        outer = np.asarray(
            [
                [0.0, 0.0],
                [lx, 0.0],
                [lx, ly],
                [0.0, ly],
                [0.0, 0.0],
            ],
            dtype=float,
        )
        ax.plot(outer[:, 0], outer[:, 1], linewidth=1.8, color="red")


def save_debug_mesh_plot(sample: Dict, out_path: str):
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    verts_xy = _to_np_debug(sample["verts_xy"])
    faces = _to_np_debug(sample["faces"]).astype(np.int64)

    fig, ax = plt.subplots(figsize=(6.0, 6.0))

    ax.triplot(
        verts_xy[:, 0],
        verts_xy[:, 1],
        faces,
        linewidth=0.35,
        alpha=0.75,
        color="black",
    )

    _draw_holes2_boundaries(ax, sample, draw_outer=True, draw_holes=True)

    lx = float(sample.get("lx", 1.0))
    ly = float(sample.get("ly", 1.0))

    ax.set_aspect("equal")
    ax.set_xlim(-0.02 * lx, 1.02 * lx)
    ax.set_ylim(-0.02 * ly, 1.02 * ly)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()

    os.makedirs(os.path.dirname(str(out_path)), exist_ok=True)
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_encoder_pc_plot(sample: Dict, out_path: str):
    """
    Plot only the active boundary point cloud entering the Geometry Encoder.
    """
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pc = sample.get("pc", None)
    if pc is None:
        raise KeyError("sample['pc'] is required for encoder point-cloud plotting.")

    pc_np = _to_np_debug(pc)
    lx = float(sample.get("lx", 1.0))
    ly = float(sample.get("ly", 1.0))
    fig, ax = plt.subplots(figsize=(6.0, 6.0))

    ax.scatter(
        pc_np[:, 0],
        pc_np[:, 1],
        s=8,
        alpha=0.85,
        color="red",
    )

    ax.set_aspect("equal")
    ax.set_xlim(-0.02 * lx, 1.02 * lx)
    ax.set_ylim(-0.02 * ly, 1.02 * ly)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()

    os.makedirs(os.path.dirname(str(out_path)), exist_ok=True)
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
