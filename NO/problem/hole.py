from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from NO.problem.common import (
    normalize_points_by_bbox,
    normalize_uniform_load_for_network,
    normalize_pressure_field_for_network,
    generate_gaussian_random_pressure_field,
    make_square_boundary_points,
    make_square_clamped_bc_factor,
    precompute_plate_face_geometry,
)


# -----------------------------------------------------------------------------
# Geometry generation
# -----------------------------------------------------------------------------

def generate_random_hole_boundary(
    *,
    rng: np.random.Generator,
    center: Tuple[float, float] = (0.5, 0.5),
    n_points: int = 96,
    mean_radius: float = 0.17,
    r_min: float = 0.11,
    r_max: float = 0.24,
    roughness: float = 0.18,
    n_modes: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a smooth arbitrary-shaped central hole using a radius perturbation.

    r(theta) = mean_radius * [1 + perturb(theta)]
    """
    theta = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)

    perturb = np.zeros_like(theta)
    for k in range(1, n_modes + 1):
        a = rng.normal(0.0, roughness / k)
        b = rng.normal(0.0, roughness / k)
        perturb += a * np.cos(k * theta) + b * np.sin(k * theta)

    # Keep the perturbation moderate to avoid self-intersection and poor meshes.
    perturb = np.clip(perturb, -0.60, 0.60)

    radius = mean_radius * (1.0 + perturb)
    radius = np.clip(radius, r_min, r_max)

    cx, cy = center
    x = cx + radius * np.cos(theta)
    y = cy + radius * np.sin(theta)

    hole_xy = np.stack([x, y], axis=1).astype(np.float64)

    return hole_xy, radius.astype(np.float64)


def triangulate_square_with_hole(
    *,
    lx: float,
    ly: float,
    hole_xy: np.ndarray,
    mesh_lc: float,
    outer_n_per_edge: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate an unstructured triangular mesh for a square plate with one hole
    using gmsh.

    The outer square boundary is clamped later in FEM/NO.
    The hole boundary is a natural free boundary.
    """
    try:
        import gmsh
    except Exception as e:
        raise ImportError(
            "gmsh is required for the single-hole case. "
            "Please install gmsh in the current environment."
        ) from e

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)

    try:
        gmsh.model.add("square_with_hole")

        # ---------------------------------------------------------------------
        # Outer square boundary as a polyline
        # ---------------------------------------------------------------------
        t = np.linspace(0.0, 1.0, int(outer_n_per_edge), endpoint=False)

        bottom = np.stack([lx * t, np.zeros_like(t)], axis=1)
        right = np.stack([np.full_like(t, lx), ly * t], axis=1)
        top = np.stack([lx * (1.0 - t), np.full_like(t, ly)], axis=1)
        left = np.stack([np.zeros_like(t), ly * (1.0 - t)], axis=1)

        outer_xy = np.vstack([bottom, right, top, left]).astype(np.float64)

        outer_points = []
        for x, y in outer_xy:
            outer_points.append(
                gmsh.model.geo.addPoint(float(x), float(y), 0.0, mesh_lc)
            )

        outer_lines = []
        n_outer = len(outer_points)
        for i in range(n_outer):
            outer_lines.append(
                gmsh.model.geo.addLine(outer_points[i], outer_points[(i + 1) % n_outer])
            )

        outer_loop = gmsh.model.geo.addCurveLoop(outer_lines)

        # ---------------------------------------------------------------------
        # Hole boundary
        # ---------------------------------------------------------------------
        # Reverse the hole boundary orientation to make it an inner loop.
        hole_xy_loop = np.asarray(hole_xy, dtype=np.float64)[::-1].copy()

        hole_points = []
        for x, y in hole_xy_loop:
            hole_points.append(
                gmsh.model.geo.addPoint(float(x), float(y), 0.0, mesh_lc)
            )

        hole_lines = []
        n_hole = len(hole_points)
        for i in range(n_hole):
            hole_lines.append(
                gmsh.model.geo.addLine(hole_points[i], hole_points[(i + 1) % n_hole])
            )

        hole_loop = gmsh.model.geo.addCurveLoop(hole_lines)

        surface = gmsh.model.geo.addPlaneSurface([outer_loop, hole_loop])

        gmsh.model.geo.synchronize()

        gmsh.model.addPhysicalGroup(2, [surface], 1)

        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.model.mesh.generate(2)

        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        coords = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
        node_tags = np.asarray(node_tags, dtype=np.int64)

        verts_xy_np = coords[:, :2].copy()

        tag_to_idx = {int(tag): i for i, tag in enumerate(node_tags)}

        elem_types, _, elem_node_tags = gmsh.model.mesh.getElements(dim=2)

        tri_faces: List[List[int]] = []

        for etype, enodes in zip(elem_types, elem_node_tags):
            # gmsh type 2 = 3-node triangle
            if int(etype) != 2:
                continue

            enodes = np.asarray(enodes, dtype=np.int64).reshape(-1, 3)
            for tri in enodes:
                tri_faces.append([tag_to_idx[int(t)] for t in tri])

        if len(tri_faces) == 0:
            raise RuntimeError("No triangular elements were generated by gmsh.")

        faces_np = np.asarray(tri_faces, dtype=np.int64)

        # Ensure positive orientation.
        tri = verts_xy_np[faces_np]
        signed_area2 = (
            (tri[:, 1, 0] - tri[:, 0, 0]) * (tri[:, 2, 1] - tri[:, 0, 1])
            - (tri[:, 2, 0] - tri[:, 0, 0]) * (tri[:, 1, 1] - tri[:, 0, 1])
        )
        flip = signed_area2 < 0.0
        if np.any(flip):
            faces_np[flip] = faces_np[flip][:, [0, 2, 1]]

        verts_xy = torch.tensor(verts_xy_np, dtype=torch.float32)
        faces = torch.tensor(faces_np, dtype=torch.long)

        return verts_xy, faces

    finally:
        gmsh.finalize()


def make_hole_boundary_point_cloud(
    *,
    lx: float,
    ly: float,
    outer_n_per_edge: int,
    hole_xy: np.ndarray,
) -> torch.Tensor:
    """
    Boundary point cloud = outer square boundary + hole boundary.
    """
    outer_pc = make_square_boundary_points(
        lx=lx,
        ly=ly,
        n_per_edge=outer_n_per_edge,
    )

    hole_pc = torch.zeros((hole_xy.shape[0], 3), dtype=torch.float32)
    hole_pc[:, :2] = torch.tensor(hole_xy, dtype=torch.float32)

    pc = torch.cat([outer_pc, hole_pc], dim=0)

    return pc


# -----------------------------------------------------------------------------
# Sample and dataset builders
# -----------------------------------------------------------------------------

def make_hole_sample(
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
    outer_n_per_edge: int = 64,
    hole_n_points: int = 96,
    mean_radius: float = 0.85,
    mean_radius_min: Optional[float] = None,  
    mean_radius_max: Optional[float] = None,
    r_min: float = 0.25,
    r_max: float = 1.75,
    roughness: float = 0.30,
    n_modes: int = 4,
    mesh_lc: float = 0.275,
    center_min: Optional[float] = None,
    center_max: Optional[float] = None,
    load_type: str = "gaussian_random_field_pressure",
    use_load_field_input: bool = True,
) -> Dict:
    """
    Build one square plate sample with a single arbitrary central hole.

    Boundary condition:
        outer square boundary clamped

    Hole boundary:
        free

    Load:
        Controlled by load_type:
            uniform_pressure:
                uniform transverse pressure;
                no load-field encoder input.
            gaussian_random_field_pressure:
                Gaussian random field pressure defined on FEM nodes;
                load_pc_net = [x_net, y_net, z_net, p_net].
    """
    if pressure_mean is None:
        pressure_mean = float(pressure_value)

    rng = np.random.default_rng(seed + int(sample_id))

    if mean_radius_min is not None and mean_radius_max is not None:
        mean_radius = float(rng.uniform(mean_radius_min, mean_radius_max))

    if center_min is not None and center_max is not None:
        cx = rng.uniform(center_min, center_max)
        cy = rng.uniform(center_min, center_max)
    else:
        cx, cy = 0.5 * lx, 0.5 * ly

    hole_xy, hole_radius = generate_random_hole_boundary(
        rng=rng,
        center=(cx, cy), 
        n_points=hole_n_points,
        mean_radius=mean_radius,
        r_min=r_min,
        r_max=r_max,
        roughness=roughness,
        n_modes=n_modes,
    )

    verts_xy, faces = triangulate_square_with_hole(
        lx=lx,
        ly=ly,
        hole_xy=hole_xy,
        mesh_lc=mesh_lc,
        outer_n_per_edge=outer_n_per_edge,
    )

    n_nodes = verts_xy.shape[0]

    # ------------------------------------------------------------------
    # Query points: all FEM nodes.
    # These points are used by the solution decoder and energy loss.
    # ------------------------------------------------------------------
    query = torch.zeros((n_nodes, 3), dtype=torch.float32)
    query[:, :2] = verts_xy

    # ------------------------------------------------------------------
    # Geometry point cloud: fixed to boundary points.
    # ------------------------------------------------------------------
    pc = make_hole_boundary_point_cloud(
        lx=lx,
        ly=ly,
        outer_n_per_edge=outer_n_per_edge,
        hole_xy=hole_xy,
    )

    bbox = torch.tensor([0.0, float(lx), 0.0, float(ly)], dtype=torch.float32)

    query_net = normalize_points_by_bbox(query, bbox)
    pc_net = normalize_points_by_bbox(pc, bbox)

    # Only outer square boundary is clamped. Hole boundary remains free.
    bc_factor = make_square_clamped_bc_factor(
        verts_xy,
        lx=lx,
        ly=ly,
    )

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
        pressure_uniform_value = float(pressure_mean) if pressure_mean is not None else float(pressure_value)

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

        # For the geometry-only uniform-load case, do not provide varying scalar
        # information to the model. The sample-to-sample variation should come
        # from geometry only.
        scalar_param = torch.zeros(3, dtype=torch.float32)

        load_fields = {
            "pressure": pressure,
            "load_scalar": load_scalar,
            "scalar_param": scalar_param,
            "use_material_field": False,
            "use_load_field": False,
            "pressure_value": torch.tensor([float(pressure_uniform_value)], dtype=torch.float32),
            "pressure_min": torch.tensor([float(pressure_min)], dtype=torch.float32),
            "pressure_max": torch.tensor([float(pressure_max)], dtype=torch.float32),
            "load_type": "uniform_pressure",
        }

    elif load_type_key in ("grf", "grfp", "grf_pressure", "gaussian_random_field_pressure"):
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
        load_pc_net = torch.cat([query_net[:, :3], pressure_net], dim=1).float()

        scalar_param = torch.cat(
            [
                torch.zeros(1, dtype=torch.float32),
                torch.zeros(1, dtype=torch.float32),
                load_scalar.reshape(1),
            ],
            dim=0,
        )

        load_fields = {
            "pressure": pressure,
            "pressure_net": pressure_net,
            "load_pc_net": load_pc_net,
            "load_scalar": load_scalar,
            "scalar_param": scalar_param,
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
        # Active geometry point cloud used by the geometry encoder.
        "pc": pc,
        "pc_net": pc_net,

        # Query points used by the solution decoder.
        "query": query,
        "query_net": query_net,

        "verts_xy": verts_xy.clone(),
        "faces": faces.clone(),
        "geom_face": geom_face,

        "bc_factor": bc_factor,

        "E": torch.tensor([float(E)], dtype=torch.float32),
        "nu": torch.tensor([float(nu)], dtype=torch.float32),
        "thickness": torch.tensor([float(thickness)], dtype=torch.float32),

        "sample_id": int(sample_id),
        "lx": float(lx),
        "ly": float(ly),
        "bbox": bbox.clone(),

        "geometry_type": "hole",

        # Geometry metadata for post-analysis.
        "hole_center": torch.tensor([cx, cy], dtype=torch.float32),
        "hole_xy": torch.tensor(hole_xy, dtype=torch.float32),
        "hole_radius": torch.tensor(hole_radius, dtype=torch.float32),
        "mean_radius": torch.tensor([float(mean_radius)], dtype=torch.float32),
        "r_min": torch.tensor([float(r_min)], dtype=torch.float32),
        "r_max": torch.tensor([float(r_max)], dtype=torch.float32),
        "roughness": torch.tensor([float(roughness)], dtype=torch.float32),
        "n_modes": torch.tensor([int(n_modes)], dtype=torch.int64),
    }

    sample.update(load_fields)

    return sample


def build_hole_dataset(
    *,
    n_train: int = 500,  # 500
    n_test: int = 100,  # 100
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
    outer_n_per_edge: int = 64,
    hole_n_points: int = 96,
    mean_radius: float = 0.85,
    mean_radius_min: Optional[float] = None, 
    mean_radius_max: Optional[float] = None, 
    r_min: float = 0.25,
    r_max: float = 1.75,
    roughness: float = 0.30,
    n_modes: int = 4,
    mesh_lc: float = 0.275,
    center_min: Optional[float] = None,
    center_max: Optional[float] = None,
    load_type: str = "gaussian_random_field_pressure",
    use_load_field_input: bool = True,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Build train/test datasets for a single-hole square plate.

    """
    if pressure_mean is None:
        pressure_mean = float(pressure_value)

    train_samples: List[Dict] = []
    test_samples: List[Dict] = []

    for i in range(n_train):
        train_samples.append(
            make_hole_sample(
                sample_id=i,
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
                mean_radius=mean_radius,
                mean_radius_min=mean_radius_min,
                mean_radius_max=mean_radius_max,
                r_min=r_min,
                r_max=r_max,
                roughness=roughness,
                n_modes=n_modes,
                mesh_lc=mesh_lc,
                center_min=center_min, 
                center_max=center_max,
                load_type=load_type,
                use_load_field_input=use_load_field_input,
            )
        )

    for i in range(n_test):
        test_samples.append(
            make_hole_sample(
                sample_id=n_train + i,
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
                mean_radius=mean_radius,
                mean_radius_min=mean_radius_min,
                mean_radius_max=mean_radius_max,
                r_min=r_min,
                r_max=r_max,
                roughness=roughness,
                n_modes=n_modes,
                mesh_lc=mesh_lc,
                center_min=center_min,
                center_max=center_max,
                load_type=load_type,
                use_load_field_input=use_load_field_input,
            )
        )

    return train_samples, test_samples


# -----------------------------------------------------------------------------
# Debug plotting
# -----------------------------------------------------------------------------

def _to_np(x):
    import numpy as np
    import torch

    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def save_debug_mesh_plot(sample: Dict, out_path: str):
    """
    Save a debug mesh figure for one single-hole sample.

    The figure shows:
        - triangular mesh
        - outer square boundary
        - hole boundary
    """
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    verts_xy = _to_np(sample["verts_xy"])
    faces = _to_np(sample["faces"]).astype(np.int64)

    hole_xy = _to_np(sample.get("hole_xy", np.zeros((0, 2))))

    fig, ax = plt.subplots(figsize=(6.0, 6.0))

    ax.triplot(
        verts_xy[:, 0],
        verts_xy[:, 1],
        faces,
        linewidth=0.35,
        alpha=0.75,
        color="black",
    )

    if hole_xy.size > 0:
        hole_closed = np.vstack([hole_xy, hole_xy[0:1]])
        ax.plot(
            hole_closed[:, 0],
            hole_closed[:, 1],
            linewidth=1.8,
            color="green",
        )

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

    ax.plot(
        outer[:, 0],
        outer[:, 1],
        linewidth=1.8,
        color="red",
    )

    ax.set_aspect("equal")
    ax.set_xlim(-0.02 * lx, 1.02 * lx)
    ax.set_ylim(-0.02 * ly, 1.02 * ly)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()

    out_path = str(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_encoder_pc_plot(sample: Dict, out_path: str):
    """
    Plot only the active point cloud entering the Geometry Encoder.

    No triangular mesh is plotted.
    """
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pc = sample.get("pc", None)
    if pc is None:
        raise KeyError("sample['pc'] is required for encoder point-cloud plotting.")

    pc_np = _to_np(pc)
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

    out_path = str(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
