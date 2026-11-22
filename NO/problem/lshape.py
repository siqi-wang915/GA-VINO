from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from NO.problem.common import (
    normalize_points_by_bbox,
    precompute_plate_face_geometry,
    generate_gaussian_random_pressure_field,
    normalize_pressure_field_for_network,
)

# -----------------------------------------------------------------------------
# Scalar normalization helpers
# -----------------------------------------------------------------------------

def normalize_uniform_load_for_network(
    pressure_value: float,
    pressure_min: float,
    pressure_max: float,
) -> float:
    """
    Normalize a scalar uniform pressure to approximately [-1, 1].

    The physical pressure field is still stored separately in ``pressure`` and is
    used by the energy functional. This normalized value is only used as an
    extra neural-operator input.
    """
    p_mid = 0.5 * (float(pressure_min) + float(pressure_max))
    p_half = max(0.5 * (float(pressure_max) - float(pressure_min)), 1.0e-12)
    return (float(pressure_value) - p_mid) / p_half


def normalize_scalar_field_for_network(
    field: torch.Tensor,
    field_min: float,
    field_max: float,
) -> torch.Tensor:
    """
    Normalize a scalar field to approximately [-1, 1] using prescribed bounds.
    """
    mid = 0.5 * (float(field_min) + float(field_max))
    half = max(0.5 * (float(field_max) - float(field_min)), 1.0e-12)
    return (field - mid) / half


# -----------------------------------------------------------------------------
# Gaussian random field helpers for elementwise material fields
# -----------------------------------------------------------------------------

def generate_gaussian_random_field_on_points(
    pts_xy: torch.Tensor,
    *,
    rng: np.random.Generator,
    mean: float,
    std: float,
    corr_len: float,
    n_modes: int,
    value_min: float,
    value_max: float,
) -> torch.Tensor:
    """
    Generate a smooth Gaussian-random-field-like scalar field on arbitrary points.

    A random Fourier feature expansion is used to avoid O(N^3) covariance
    factorization. The generated field is standardized on the current point set,
    shifted/scaled to the requested mean/std, then clipped to [value_min,
    value_max]. After clipping, it is recentered once and clipped again.
    """
    xy = pts_xy.detach().cpu().numpy().astype(np.float64)
    n_modes = max(int(n_modes), 1)
    corr_len = max(float(corr_len), 1.0e-6)

    k = rng.normal(loc=0.0, scale=1.0 / corr_len, size=(n_modes, 2))
    phase = rng.uniform(0.0, 2.0 * np.pi, size=(n_modes,))
    amp = rng.normal(loc=0.0, scale=1.0, size=(n_modes,))

    arg = xy @ k.T + phase[None, :]
    raw = np.sqrt(2.0 / n_modes) * np.sum(amp[None, :] * np.cos(arg), axis=1)

    raw = raw - float(np.mean(raw))
    raw_std = float(np.std(raw))
    if raw_std < 1.0e-12:
        raw = rng.normal(loc=0.0, scale=1.0, size=xy.shape[0])
        raw = raw - float(np.mean(raw))
        raw_std = max(float(np.std(raw)), 1.0e-12)
    raw = raw / raw_std

    field = float(mean) + float(std) * raw
    field = np.clip(field, float(value_min), float(value_max))

    field = field + (float(mean) - float(np.mean(field)))
    field = np.clip(field, float(value_min), float(value_max))

    return torch.as_tensor(field[:, None], dtype=torch.float32)


# -----------------------------------------------------------------------------
# L-shape geometry helpers
# -----------------------------------------------------------------------------

def make_lshape_vertices(
    *,
    lx: float,
    ly: float,
    cut_w: float,
    cut_h: float,
) -> np.ndarray:
    """
    Build an L-shaped polygon by cutting a rectangle from the top-right corner.

    Original rectangle:
        [0, lx] x [0, ly]

    Removed rectangle:
        [lx - cut_w, lx] x [ly - cut_h, ly]

    Remaining L-shaped domain vertices are ordered counter-clockwise:
        (0,0) -> (lx,0) -> (lx,y_cut) -> (x_cut,y_cut)
        -> (x_cut,ly) -> (0,ly)
    """
    lx = float(lx)
    ly = float(ly)
    cut_w = float(cut_w)
    cut_h = float(cut_h)

    if lx <= 0.0 or ly <= 0.0:
        raise ValueError(f"Invalid outer dimensions: lx={lx}, ly={ly}")
    if not (0.0 < cut_w < lx and 0.0 < cut_h < ly):
        raise ValueError(
            f"Invalid cutout dimensions: cut_w={cut_w}, cut_h={cut_h}, lx={lx}, ly={ly}"
        )

    x_cut = lx - cut_w
    y_cut = ly - cut_h

    verts = np.asarray(
        [
            [0.0, 0.0],
            [lx, 0.0],
            [lx, y_cut],
            [x_cut, y_cut],
            [x_cut, ly],
            [0.0, ly],
        ],
        dtype=np.float64,
    )
    return verts


def _segment_length(p0: np.ndarray, p1: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(p1, dtype=np.float64) - np.asarray(p0, dtype=np.float64)))


def _sample_segment_without_endpoint(
    p0: np.ndarray,
    p1: np.ndarray,
    *,
    spacing: float,
    min_points: int = 8,
) -> np.ndarray:
    """
    Uniformly sample a boundary segment, including p0 but excluding p1.
    This avoids duplicated corner points when all polygon segments are concatenated.
    """
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    length = _segment_length(p0, p1)
    n = max(int(np.ceil(length / max(float(spacing), 1.0e-12))), int(min_points))
    s = np.linspace(0.0, 1.0, n, endpoint=False, dtype=np.float64)
    return (1.0 - s[:, None]) * p0[None, :] + s[:, None] * p1[None, :]


def make_lshape_boundary_point_cloud(
    *,
    lshape_vertices: np.ndarray,
    boundary_point_spacing: float = 0.02,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build boundary point cloud for the L-shaped domain.

    Returns
    -------
    pc:
        [N, 3] boundary point cloud.
    pc_fixed_flag:
        [N] 1 for left fixed boundary points, 0 for free boundary points.
    boundary_segments:
        [6, 2, 2] segment endpoints. The last segment is the fixed left edge.
    """
    verts = np.asarray(lshape_vertices, dtype=np.float64)
    if verts.shape != (6, 2):
        raise ValueError(f"Expected lshape_vertices with shape [6,2], got {verts.shape}")

    parts: List[np.ndarray] = []
    flags: List[np.ndarray] = []
    segments: List[np.ndarray] = []

    n = verts.shape[0]
    for i in range(n):
        p0 = verts[i]
        p1 = verts[(i + 1) % n]
        seg_pts = _sample_segment_without_endpoint(
            p0,
            p1,
            spacing=boundary_point_spacing,
            min_points=8,
        )

        # Segment 5 is left boundary: (0, ly) -> (0, 0), fixed.
        is_fixed = (i == 5)

        parts.append(seg_pts)
        flags.append(np.full((seg_pts.shape[0],), 1 if is_fixed else 0, dtype=np.int64))
        segments.append(np.stack([p0, p1], axis=0))

    pc_xy = np.vstack(parts).astype(np.float32)
    pc = np.zeros((pc_xy.shape[0], 3), dtype=np.float32)
    pc[:, :2] = pc_xy

    pc_fixed_flag = np.concatenate(flags, axis=0).astype(np.int64)
    boundary_segments = np.stack(segments, axis=0).astype(np.float32)

    return (
        torch.tensor(pc, dtype=torch.float32),
        torch.tensor(pc_fixed_flag, dtype=torch.long),
        torch.tensor(boundary_segments, dtype=torch.float32),
    )


def make_lshape_fem_boundary_point_cloud(
    *,
    verts_xy: torch.Tensor,
    faces: torch.Tensor,
    lshape_vertices: np.ndarray,
    fixed_tol: float = 1.0e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build the Geometry Encoder point cloud from all FEM boundary nodes.

    Boundary nodes are inferred from triangle edges that belong to exactly one
    element, so this works for both structured and gmsh meshes.
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
        raise ValueError("No FEM boundary nodes were found from L-shape faces.")

    pc_xy = verts_np[np.asarray(boundary_nodes, dtype=np.int64)]
    pc = np.zeros((pc_xy.shape[0], 3), dtype=np.float32)
    pc[:, :2] = pc_xy.astype(np.float32)

    fixed = np.isclose(pc_xy[:, 0], 0.0, atol=float(fixed_tol))
    pc_fixed_flag = fixed.astype(np.int64)

    verts = np.asarray(lshape_vertices, dtype=np.float64)
    if verts.shape != (6, 2):
        raise ValueError(f"Expected lshape_vertices with shape [6,2], got {verts.shape}")
    boundary_segments = np.stack(
        [np.stack([verts[i], verts[(i + 1) % 6]], axis=0) for i in range(6)],
        axis=0,
    ).astype(np.float32)

    return (
        torch.tensor(pc, dtype=torch.float32),
        torch.tensor(pc_fixed_flag, dtype=torch.long),
        torch.tensor(boundary_segments, dtype=torch.float32),
    )


# -----------------------------------------------------------------------------
# Gmsh mesh generation
# -----------------------------------------------------------------------------

def triangulate_lshape(
    *,
    lshape_vertices: np.ndarray,
    mesh_lc: float = 0.04,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate an unstructured triangular mesh for the L-shaped plate using gmsh.
    """
    try:
        import gmsh
    except Exception as e:
        raise ImportError(
            "gmsh is required for the L-shaped plate case. Please install gmsh."
        ) from e

    verts = np.asarray(lshape_vertices, dtype=np.float64)
    if verts.shape != (6, 2):
        raise ValueError(f"Expected lshape_vertices with shape [6,2], got {verts.shape}")

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)

    try:
        gmsh.model.add("lshape_plate")
        gmsh.option.setNumber("Mesh.Algorithm", 6)

        point_tags: List[int] = []
        for x, y in verts:
            point_tags.append(gmsh.model.geo.addPoint(float(x), float(y), 0.0, float(mesh_lc)))

        line_tags: List[int] = []
        for i in range(len(point_tags)):
            line_tags.append(
                gmsh.model.geo.addLine(point_tags[i], point_tags[(i + 1) % len(point_tags)])
            )

        loop = gmsh.model.geo.addCurveLoop(line_tags)
        surface = gmsh.model.geo.addPlaneSurface([loop])

        gmsh.model.geo.synchronize()
        gmsh.model.addPhysicalGroup(1, [line_tags[-1]], tag=1, name="fixed_left_boundary")
        gmsh.model.addPhysicalGroup(1, line_tags[:-1], tag=2, name="free_boundary")
        gmsh.model.addPhysicalGroup(2, [surface], tag=10, name="plate_domain")

        gmsh.model.mesh.generate(2)

        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        node_tags = np.asarray(node_tags, dtype=np.int64)
        coords = np.asarray(coords, dtype=np.float64).reshape(-1, 3)

        # Sort nodes by gmsh tag for deterministic ordering.
        order = np.argsort(node_tags)
        node_tags = node_tags[order]
        coords = coords[order]
        tag_to_idx = {int(tag): i for i, tag in enumerate(node_tags)}

        verts_xy_np = coords[:, :2].astype(np.float64)

        elem_types, _, elem_node_tags = gmsh.model.mesh.getElements(dim=2)
        tri_faces: List[np.ndarray] = []

        for etype, nodes in zip(elem_types, elem_node_tags):
            _, dim, _, num_nodes, _, _ = gmsh.model.mesh.getElementProperties(etype)
            if int(dim) == 2 and int(num_nodes) == 3:
                nodes = np.asarray(nodes, dtype=np.int64).reshape(-1, 3)
                faces = np.vectorize(lambda t: tag_to_idx[int(t)])(nodes)
                tri_faces.append(faces.astype(np.int64))

        if not tri_faces:
            raise RuntimeError("No 3-node triangular elements were generated by gmsh.")

        faces_np = np.vstack(tri_faces).astype(np.int64)

        # Ensure positive orientation.
        tri = verts_xy_np[faces_np]
        signed_area2 = (
            (tri[:, 1, 0] - tri[:, 0, 0]) * (tri[:, 2, 1] - tri[:, 0, 1])
            - (tri[:, 2, 0] - tri[:, 0, 0]) * (tri[:, 1, 1] - tri[:, 0, 1])
        )
        flip = signed_area2 < 0.0
        if np.any(flip):
            faces_np[flip] = faces_np[flip][:, [0, 2, 1]]

        return (
            torch.tensor(verts_xy_np, dtype=torch.float32),
            torch.tensor(faces_np, dtype=torch.long),
        )

    finally:
        gmsh.finalize()

def triangulate_lshape_structured(
    *,
    lx: float,
    ly: float,
    x_cut: float,
    y_cut: float,
    nx: int = 50,
    ny: int = 50,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a structured Cartesian triangular mesh for an L-shaped plate.

    The full rectangle [0,lx] x [0,ly] is discretized by an nx-by-ny grid.
    Cells whose centers lie in the removed top-right rectangle
    (x > x_cut and y > y_cut) are skipped.

    Each active quadrilateral cell is split by the same diagonal:
        (i,j) ---- (i+1,j)
          |          |
        (i,j+1) -- (i+1,j+1)

    triangles:
        [n00, n10, n11]
        [n00, n11, n01]
    """
    lx = float(lx)
    ly = float(ly)
    x_cut = float(x_cut)
    y_cut = float(y_cut)
    nx = int(nx)
    ny = int(ny)

    if nx <= 0 or ny <= 0:
        raise ValueError(f"nx and ny must be positive, got nx={nx}, ny={ny}")

    xs = np.linspace(0.0, lx, nx + 1, dtype=np.float64)
    ys = np.linspace(0.0, ly, ny + 1, dtype=np.float64)

    used_nodes: dict[tuple[int, int], int] = {}
    verts: list[list[float]] = []
    faces: list[list[int]] = []

    def get_node(i: int, j: int) -> int:
        key = (i, j)
        if key not in used_nodes:
            used_nodes[key] = len(verts)
            verts.append([float(xs[i]), float(ys[j])])
        return used_nodes[key]

    for j in range(ny):
        for i in range(nx):
            xc = 0.5 * (xs[i] + xs[i + 1])
            yc = 0.5 * (ys[j] + ys[j + 1])

            # Skip cells inside the removed top-right cutout.
            if xc > x_cut and yc > y_cut:
                continue

            n00 = get_node(i, j)
            n10 = get_node(i + 1, j)
            n01 = get_node(i, j + 1)
            n11 = get_node(i + 1, j + 1)

            faces.append([n00, n10, n11])
            faces.append([n00, n11, n01])

    verts_xy = torch.tensor(np.asarray(verts, dtype=np.float32), dtype=torch.float32)
    faces_t = torch.tensor(np.asarray(faces, dtype=np.int64), dtype=torch.long)

    return verts_xy, faces_t

# -----------------------------------------------------------------------------
# Sample and dataset builders
# -----------------------------------------------------------------------------

def make_lshape_clamped_left_bc_factor(
    verts_xy: torch.Tensor,
    *,
    tol: float = 1.0e-8,
) -> torch.Tensor:
    """
    Boundary condition factor for a left-clamped L-shaped plate.

    DOFs:
        [w, theta_x, theta_y]

    0 = fixed, 1 = free.
    """
    x = verts_xy[:, 0]
    fixed = torch.isclose(x, torch.tensor(0.0, dtype=x.dtype, device=x.device), atol=tol)

    bc_factor = torch.ones((verts_xy.shape[0], 3), dtype=torch.float32)
    bc_factor[fixed, :] = 0.0
    return bc_factor


def make_lshape_sample(
    sample_id: int,
    *,
    seed: int = 42,
    lx_min: float = 5.0,
    lx_max: float = 5.0,
    ly_min: float = 5.0,
    ly_max: float = 5.0,
    cut_w_ratio_min: float = 0.25,
    cut_w_ratio_max: float = 0.65,
    cut_h_ratio_min: float = 0.25,
    cut_h_ratio_max: float = 0.65,
    E_mean: float = 1000.0,
    E_std: float = 200.0,
    E_min: float = 400.0,
    E_max: float = 1600.0,
    E_corr_len: float = 1.5,
    E_n_modes: int = 64,
    nu_mean: float = 0.30,
    nu_std: float = 0.04,
    nu_min: float = 0.10,
    nu_max: float = 0.45,
    nu_corr_len: float = 1.5,
    nu_n_modes: int = 64,
    thickness: float = 0.5,
    pressure_value: float = 0.8,
    pressure_min: float = 0.0,
    pressure_max: float = 1.60,
    pressure_mean: Optional[float] = None,
    pressure_std: float = 0.3,
    pressure_corr_len: float = 1.5,
    pressure_n_modes: int = 64,
    load_type: str = "uniform_pressure",
    use_load_field_input: bool = False,
    mesh_lc: float = 0.10,
    mesh_type: str = "gmsh",
    mesh_nx: int = 50,
    mesh_ny: int = 50,
    boundary_point_spacing: float = 0.05,
    sample_uniform_pressure: bool = False,
    pressure_value_min: Optional[float] = None,
    pressure_value_max: Optional[float] = None,
    use_material_field_input: bool = True,
) -> Dict:
    """
    Build one L-shaped Mindlin-Reissner plate sample.

    Boundary condition:
        left boundary clamped; all other boundaries free.

    Load:
    Controlled by load_type:
        uniform_pressure:
            uniform transverse pressure;
            no load-field encoder input.
        gaussian_random_field_pressure:
            Gaussian random field pressure defined on FEM nodes;
            load_pc_net = [x_net, y_net, z_net, p_net].

    Material:
        If use_material_field_input is true, Young's modulus and Poisson's
        ratio are elementwise Gaussian random fields defined on triangle
        centroids.  Otherwise E_mean and nu_mean are used as uniform material
        constants and no material-field point cloud is created.
    """
    rng = np.random.default_rng(int(seed) + int(sample_id))

    if pressure_mean is None:
        pressure_mean = float(pressure_value)

    lx = float(rng.uniform(lx_min, lx_max))
    ly = float(rng.uniform(ly_min, ly_max))

    cut_w = float(rng.uniform(cut_w_ratio_min, cut_w_ratio_max) * lx)
    cut_h = float(rng.uniform(cut_h_ratio_min, cut_h_ratio_max) * ly)

    x_cut = lx - cut_w
    y_cut = ly - cut_h

    lshape_vertices = make_lshape_vertices(
        lx=lx,
        ly=ly,
        cut_w=cut_w,
        cut_h=cut_h,
    )


    mesh_type_key = str(mesh_type).lower().strip()

    if mesh_type_key in ("structured", "cartesian", "grid"):
        verts_xy, faces = triangulate_lshape_structured(
            lx=lx,
            ly=ly,
            x_cut=x_cut,
            y_cut=y_cut,
            nx=mesh_nx,
            ny=mesh_ny,
        )
    elif mesh_type_key in ("gmsh", "unstructured"):
        verts_xy, faces = triangulate_lshape(
            lshape_vertices=lshape_vertices,
            mesh_lc=mesh_lc,
        )
    else:
        raise ValueError(
            f"Unsupported mesh_type={mesh_type!r}. "
            "Use 'gmsh' or 'structured'."
        )

    n_nodes = int(verts_xy.shape[0])

    query = torch.zeros((n_nodes, 3), dtype=torch.float32)
    query[:, :2] = verts_xy

    # Geometry Encoder input uses all FEM boundary nodes.
    # geometry_pc_default_mode is kept only for backward compatibility with
    # shared/prep/lshape.py and is intentionally ignored.
    pc, pc_fixed_flag, boundary_segments = make_lshape_fem_boundary_point_cloud(
        verts_xy=verts_xy,
        faces=faces,
        lshape_vertices=lshape_vertices,
    )

    bbox = torch.tensor([0.0, float(lx), 0.0, float(ly)], dtype=torch.float32)
    query_net = normalize_points_by_bbox(query, bbox)
    pc_net = normalize_points_by_bbox(pc, bbox)

    bc_factor = make_lshape_clamped_left_bc_factor(verts_xy)

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
        if bool(sample_uniform_pressure):
            pressure_uniform_value = float(
                rng.uniform(float(pressure_min), float(pressure_max))
            )
        else:
            pressure_uniform_value = (
                float(pressure_mean) if pressure_mean is not None else float(pressure_value)
            )

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
        
        load_fields = {
            "use_load_field": False,
            "pressure_value": torch.tensor([float(pressure_uniform_value)], dtype=torch.float32),
            "pressure_min": torch.tensor([float(pressure_min)], dtype=torch.float32),
            "pressure_max": torch.tensor([float(pressure_max)], dtype=torch.float32),
            "sample_uniform_pressure": torch.tensor([int(bool(sample_uniform_pressure))], dtype=torch.int64),
            "pressure_value_min": torch.tensor(
                [float(pressure_value_min) if pressure_value_min is not None else float(pressure_min)],
                dtype=torch.float32,
            ),
            "pressure_value_max": torch.tensor(
                [float(pressure_value_max) if pressure_value_max is not None else float(pressure_max)],
                dtype=torch.float32,
            ),
            "load_type": "uniform_pressure",
        }

    elif load_type_key in ("grf", "grfp", "grf_pressure", "gaussian_random_field_pressure"):
        load_rng = np.random.default_rng(int(seed) + int(sample_id) + 1_000_000)

        pressure = generate_gaussian_random_pressure_field(
            verts_xy,
            rng=load_rng,
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
        load_pc_net = torch.cat([query_net[:, :3], pressure_net], dim=1).float()

        load_fields = {
            "use_load_field": bool(use_load_field_input),
            "pressure_net": pressure_net.clone(),
            "load_pc_net": load_pc_net.clone(),
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
    material_fields: Dict[str, object]
    if bool(use_material_field_input):
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

        E_value = float(torch.mean(E_elem).item())
        nu_value = float(torch.mean(nu_elem).item())
        E_net = normalize_scalar_field_for_network(E_elem, E_min, E_max)
        nu_net = normalize_scalar_field_for_network(nu_elem, nu_min, nu_max)
        material_pc_net = torch.cat(
            [centroid_xyz_net[:, :3], E_net, nu_net], dim=1
        ).float()
        E_scalar_net = normalize_scalar_field_for_network(
            torch.tensor([[E_value]], dtype=torch.float32), E_min, E_max
        ).reshape(1)
        nu_scalar_net = normalize_scalar_field_for_network(
            torch.tensor([[nu_value]], dtype=torch.float32), nu_min, nu_max
        ).reshape(1)
        scalar_param = torch.cat(
            [E_scalar_net, nu_scalar_net, load_scalar.reshape(1)], dim=0
        ).float()
        material_fields = {
            "use_material_field": True,
            "E_elem": E_elem.clone(),
            "nu_elem": nu_elem.clone(),
            "E_net": E_net.clone(),
            "nu_net": nu_net.clone(),
            "material_pc_net": material_pc_net.clone(),
            "material_type": "gaussian_random_field",
            "E_std": torch.tensor([float(E_std)], dtype=torch.float32),
            "E_min": torch.tensor([float(E_min)], dtype=torch.float32),
            "E_max": torch.tensor([float(E_max)], dtype=torch.float32),
            "E_corr_len": torch.tensor([float(E_corr_len)], dtype=torch.float32),
            "E_n_modes": torch.tensor([int(E_n_modes)], dtype=torch.int64),
            "nu_std": torch.tensor([float(nu_std)], dtype=torch.float32),
            "nu_min": torch.tensor([float(nu_min)], dtype=torch.float32),
            "nu_max": torch.tensor([float(nu_max)], dtype=torch.float32),
            "nu_corr_len": torch.tensor([float(nu_corr_len)], dtype=torch.float32),
            "nu_n_modes": torch.tensor([int(nu_n_modes)], dtype=torch.int64),
        }
    else:
        E_value = float(E_mean)
        nu_value = float(nu_mean)
        scalar_param = torch.cat(
            [torch.zeros(2, dtype=torch.float32), load_scalar.reshape(1)], dim=0
        )
        material_fields = {
            "use_material_field": False,
            "material_type": "uniform",
        }

    sample = {
        # Boundary point cloud used by the geometry encoder.
        "pc": pc,
        "pc_net": pc_net,

        # Query points used by the solution decoder.
        "query": query,
        "query_net": query_net,

        "verts_xy": verts_xy.clone(),
        "faces": faces.clone(),
        "geom_face": geom_face,

        "bc_factor": bc_factor.clone(),
        "pressure": pressure.clone(),
        "load_scalar": load_scalar.clone(),
        "scalar_param": scalar_param.clone(),
        # Scalar material constants/representative values.
        "E": torch.tensor([E_value], dtype=torch.float32),
        "nu": torch.tensor([nu_value], dtype=torch.float32),
        "thickness": torch.tensor([float(thickness)], dtype=torch.float32),
        "E_mean": torch.tensor([float(E_mean)], dtype=torch.float32),
        "nu_mean": torch.tensor([float(nu_mean)], dtype=torch.float32),

        "sample_id": int(sample_id),
        "lx": float(lx),
        "ly": float(ly),
        "bbox": bbox.clone(),

        "geometry_type": "lshape",

        "lshape_vertices": torch.tensor(lshape_vertices, dtype=torch.float32),
        "boundary_segments": boundary_segments.clone(),
        "pc_fixed_flag_boundary": pc_fixed_flag.clone(),

        "pc_fixed_flag": pc_fixed_flag.clone(),
        "cut_w": torch.tensor([float(cut_w)], dtype=torch.float32),
        "cut_h": torch.tensor([float(cut_h)], dtype=torch.float32),
        "x_cut": torch.tensor([float(x_cut)], dtype=torch.float32),
        "y_cut": torch.tensor([float(y_cut)], dtype=torch.float32),
        "mesh_type": str(mesh_type_key),
        "mesh_nx": torch.tensor([int(mesh_nx)], dtype=torch.int64),
        "mesh_ny": torch.tensor([int(mesh_ny)], dtype=torch.int64),
    }

    sample.update(material_fields)
    sample.update(load_fields)

    return sample


def build_lshape_dataset(
    *,
    n_train: int = 200,
    n_test: int = 40,
    seed: int = 42,
    lx_min: float = 5.0,
    lx_max: float = 5.0,
    ly_min: float = 5.0,
    ly_max: float = 5.0,
    cut_w_ratio_min: float = 0.25,
    cut_w_ratio_max: float = 0.65,
    cut_h_ratio_min: float = 0.25,
    cut_h_ratio_max: float = 0.65,
    E_mean: float = 1000.0,
    E_std: float = 200.0,
    E_min: float = 400.0,
    E_max: float = 1600.0,
    E_corr_len: float = 1.5,
    E_n_modes: int = 64,
    nu_mean: float = 0.30,
    nu_std: float = 0.04,
    nu_min: float = 0.10,
    nu_max: float = 0.45,
    nu_corr_len: float = 1.5,
    nu_n_modes: int = 64,
    thickness: float = 0.5,
    pressure_value: float = 0.8,
    pressure_min: float = 0.0,
    pressure_max: float = 1.60,
    pressure_mean: Optional[float] = None,
    sample_uniform_pressure: bool = False,
    pressure_std: float = 0.3,
    pressure_corr_len: float = 1.5,
    pressure_n_modes: int = 64,
    load_type: str = "uniform_pressure",
    use_load_field_input: bool = False,
    mesh_lc: float = 0.10,
    mesh_type: str = "gmsh",
    mesh_nx: int = 50,
    mesh_ny: int = 50,
    boundary_point_spacing: float = 0.05,
    use_material_field_input: bool = True,
) -> Tuple[List[Dict], List[Dict]]:
    if pressure_mean is None:
        pressure_mean = float(pressure_value)
        
    train_samples: List[Dict] = []
    test_samples: List[Dict] = []

    for i in range(int(n_train)):
        train_samples.append(
            make_lshape_sample(
                sample_id=i,
                seed=seed,
                lx_min=lx_min,
                lx_max=lx_max,
                ly_min=ly_min,
                ly_max=ly_max,
                cut_w_ratio_min=cut_w_ratio_min,
                cut_w_ratio_max=cut_w_ratio_max,
                cut_h_ratio_min=cut_h_ratio_min,
                cut_h_ratio_max=cut_h_ratio_max,
                E_mean=E_mean,
                E_std=E_std,
                E_min=E_min,
                E_max=E_max,
                E_corr_len=E_corr_len,
                E_n_modes=E_n_modes,
                nu_mean=nu_mean,
                nu_std=nu_std,
                nu_min=nu_min,
                nu_max=nu_max,
                nu_corr_len=nu_corr_len,
                nu_n_modes=nu_n_modes,
                thickness=thickness,
                pressure_value=pressure_value,
                pressure_min=pressure_min,
                pressure_max=pressure_max,
                pressure_mean=pressure_mean,
                sample_uniform_pressure=sample_uniform_pressure,
                pressure_std=pressure_std,
                pressure_corr_len=pressure_corr_len,
                pressure_n_modes=pressure_n_modes,
                load_type=load_type,
                use_load_field_input=use_load_field_input,
                mesh_lc=mesh_lc,
                mesh_type=mesh_type,
                mesh_nx=mesh_nx,
                mesh_ny=mesh_ny,
                boundary_point_spacing=boundary_point_spacing,
                use_material_field_input=use_material_field_input,
            )
        )

    for i in range(int(n_test)):
        test_samples.append(
            make_lshape_sample(
                sample_id=int(n_train) + i,
                seed=seed,
                lx_min=lx_min,
                lx_max=lx_max,
                ly_min=ly_min,
                ly_max=ly_max,
                cut_w_ratio_min=cut_w_ratio_min,
                cut_w_ratio_max=cut_w_ratio_max,
                cut_h_ratio_min=cut_h_ratio_min,
                cut_h_ratio_max=cut_h_ratio_max,
                E_mean=E_mean,
                E_std=E_std,
                E_min=E_min,
                E_max=E_max,
                E_corr_len=E_corr_len,
                E_n_modes=E_n_modes,
                nu_mean=nu_mean,
                nu_std=nu_std,
                nu_min=nu_min,
                nu_max=nu_max,
                nu_corr_len=nu_corr_len,
                nu_n_modes=nu_n_modes,
                thickness=thickness,
                pressure_value=pressure_value,
                pressure_min=pressure_min,
                pressure_max=pressure_max,
                pressure_mean=pressure_mean,
                sample_uniform_pressure=sample_uniform_pressure,
                pressure_std=pressure_std,
                pressure_corr_len=pressure_corr_len,
                pressure_n_modes=pressure_n_modes,
                load_type=load_type,
                use_load_field_input=use_load_field_input,
                mesh_lc=mesh_lc,
                mesh_type=mesh_type,
                mesh_nx=mesh_nx,
                mesh_ny=mesh_ny,
                boundary_point_spacing=boundary_point_spacing,
                use_material_field_input=use_material_field_input,
            )
        )

    return train_samples, test_samples


# -----------------------------------------------------------------------------
# Debug plot
# -----------------------------------------------------------------------------

def save_debug_mesh_plot(sample: Dict, out_path: str):
    """
    Save a debug mesh figure for one L-shaped sample.

    Requirements:
        fixed boundary  : red
        free boundaries : green
    """
    import os
    import numpy as np
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _to_np(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    verts_xy = _to_np(sample["verts_xy"])
    faces = _to_np(sample["faces"]).astype(np.int64)
    lverts = _to_np(sample["lshape_vertices"])

    fig, ax = plt.subplots(figsize=(6.5, 6.0))

    ax.triplot(
        verts_xy[:, 0],
        verts_xy[:, 1],
        faces,
        linewidth=0.35,
        alpha=0.75,
        color="black",
    )

    # Draw free boundaries in green: segments 0-4.
    for i in range(5):
        p0 = lverts[i]
        p1 = lverts[(i + 1) % 6]
        ax.plot(
            [p0[0], p1[0]],
            [p0[1], p1[1]],
            linewidth=2.2,
            color="green",
            label="free boundary" if i == 0 else None,
        )

    # Draw fixed left boundary in red: segment 5.
    p0 = lverts[5]
    p1 = lverts[0]
    ax.plot(
        [p0[0], p1[0]],
        [p0[1], p1[1]],
        linewidth=2.6,
        color="red",
        label="fixed boundary",
    )

    lx = float(sample.get("lx", np.max(lverts[:, 0])))
    ly = float(sample.get("ly", np.max(lverts[:, 1])))

    ax.set_aspect("equal")
    ax.set_xlim(-0.03 * lx, 1.03 * lx)
    ax.set_ylim(-0.03 * ly, 1.03 * ly)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.25)
    # ax.legend(loc="upper right")

    fig.tight_layout()

    os.makedirs(os.path.dirname(str(out_path)), exist_ok=True)
    fig.savefig(str(out_path), dpi=220, bbox_inches="tight")
    plt.close(fig)


def _to_np_plot(x):
    import numpy as np
    import torch

    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def save_encoder_pc_plot(sample: Dict, out_path: str):
    """
    Plot only the active point cloud entering the Geometry Encoder.

    Plot L-shape boundary points used by the Geometry Encoder.

    No triangular mesh is plotted.
    """
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pc = sample.get("pc", None)
    if pc is None:
        raise KeyError("sample['pc'] is required for encoder point-cloud plotting.")

    pc_np = _to_np_plot(pc)
    lverts = _to_np_plot(sample["lshape_vertices"])

    lx = float(sample.get("lx", pc_np[:, 0].max() if pc_np.size else 1.0))
    ly = float(sample.get("ly", pc_np[:, 1].max() if pc_np.size else 1.0))
    fig, ax = plt.subplots(figsize=(6.5, 6.0))

    ax.scatter(
        pc_np[:, 0],
        pc_np[:, 1],
        s=8.0,
        alpha=0.85,
        color="red",
    )

    # Draw a light outline of the L-shaped boundary for visual reference.
    for i in range(6):
        p0 = lverts[i]
        p1 = lverts[(i + 1) % 6]
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], linewidth=0.8, color="black", alpha=0.35)

    ax.set_aspect("equal")
    ax.set_xlim(-0.03 * lx, 1.03 * lx)
    ax.set_ylim(-0.03 * ly, 1.03 * ly)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()

    os.makedirs(os.path.dirname(str(out_path)), exist_ok=True)
    fig.savefig(str(out_path), dpi=220, bbox_inches="tight")
    plt.close(fig)
