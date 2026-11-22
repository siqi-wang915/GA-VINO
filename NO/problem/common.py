from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch


# -----------------------------------------------------------------------------
# Coordinate / scalar normalization helpers
# -----------------------------------------------------------------------------

def normalize_points_by_bbox(
    pts_xyz: torch.Tensor,
    bbox: torch.Tensor | np.ndarray | List[float],
) -> torch.Tensor:
    """
    Normalize x-y coordinates to [-1, 1] by bounding box.
    z is kept unchanged.

    Parameters
    ----------
    pts_xyz:
        Point coordinates with shape [N, 3].
    bbox:
        [x_min, x_max, y_min, y_max].

    Returns
    -------
    out:
        Tensor with the same shape as pts_xyz.
    """
    if not isinstance(bbox, torch.Tensor):
        bbox = torch.tensor(bbox, dtype=pts_xyz.dtype, device=pts_xyz.device)
    else:
        bbox = bbox.to(device=pts_xyz.device, dtype=pts_xyz.dtype)

    x_min, x_max, y_min, y_max = bbox

    out = pts_xyz.clone()
    out[:, 0] = 2.0 * (out[:, 0] - x_min) / torch.clamp(x_max - x_min, min=1.0e-12) - 1.0
    out[:, 1] = 2.0 * (out[:, 1] - y_min) / torch.clamp(y_max - y_min, min=1.0e-12) - 1.0

    return out


def normalize_uniform_load_for_network(
    pressure_value: float,
    pressure_min: float,
    pressure_max: float,
) -> float:
    """
    Normalize scalar pressure to approximately [-1, 1].

    This normalized value is used only as a neural-network input.
    The energy loss still uses the physical pressure field.
    """
    p_mid = 0.5 * (float(pressure_min) + float(pressure_max))
    p_half = max(0.5 * (float(pressure_max) - float(pressure_min)), 1.0e-12)
    return (float(pressure_value) - p_mid) / p_half


def normalize_pressure_field_for_network(
    pressure: torch.Tensor,
    pressure_min: float,
    pressure_max: float,
) -> torch.Tensor:
    """
    Normalize nodal pressure values to approximately [-1, 1].

    The normalized pressure field is used only as a neural-network input.
    The physical pressure field is still used in the energy loss and exported
    FEM cases.
    """
    p_mid = 0.5 * (float(pressure_min) + float(pressure_max))
    p_half = max(0.5 * (float(pressure_max) - float(pressure_min)), 1.0e-12)
    return (pressure - p_mid) / p_half


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
# Boundary / boundary-condition helpers
# -----------------------------------------------------------------------------

def make_square_boundary_points(
    lx: float = 5.0,
    ly: float = 5.0,
    n_per_edge: int = 64,
) -> torch.Tensor:
    """
    Generate ordered boundary point cloud on a square plate.

    The order is bottom -> right -> top -> left. Corners are not duplicated.
    Total number of points is 4 * n_per_edge.
    """
    t = torch.linspace(0.0, 1.0, int(n_per_edge) + 1, dtype=torch.float32)[:-1]

    bottom = torch.stack([lx * t, torch.zeros_like(t)], dim=1)
    right = torch.stack([torch.full_like(t, float(lx)), ly * t], dim=1)
    top = torch.stack([lx * (1.0 - t), torch.full_like(t, float(ly))], dim=1)
    left = torch.stack([torch.zeros_like(t), ly * (1.0 - t)], dim=1)

    pc_xy = torch.cat([bottom, right, top, left], dim=0)

    pc = torch.zeros((pc_xy.shape[0], 3), dtype=torch.float32)
    pc[:, :2] = pc_xy

    return pc


def make_square_clamped_bc_factor(
    verts_xy: torch.Tensor,
    *,
    lx: float = 5.0,
    ly: float = 5.0,
    tol: float = 1.0e-8,
) -> torch.Tensor:
    """
    Boundary-condition factor for a fully clamped square outer boundary.

    DOFs:
        [w, theta_x, theta_y]

    Meaning:
        0 = fixed
        1 = free
    """
    x = verts_xy[:, 0]
    y = verts_xy[:, 1]

    zero = x.new_tensor(0.0)
    lx_t = x.new_tensor(float(lx))
    ly_t = y.new_tensor(float(ly))

    fixed = (
        torch.isclose(x, zero, atol=tol)
        | torch.isclose(x, lx_t, atol=tol)
        | torch.isclose(y, zero, atol=tol)
        | torch.isclose(y, ly_t, atol=tol)
    )

    bc_factor = torch.ones((verts_xy.shape[0], 3), dtype=torch.float32)
    bc_factor[fixed, :] = 0.0

    return bc_factor


# -----------------------------------------------------------------------------
# Optional structured-square mesh helper
# -----------------------------------------------------------------------------

def make_structured_square_mesh(
    *,
    nx: int = 66,
    ny: int = 66,
    lx: float = 5.0,
    ly: float = 5.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a structured triangular mesh for a square plate.

    This is kept here as a utility. The current hole / holes2 / lshape cases do
    not need it directly, but it is useful for small tests and debugging.
    """
    xs = torch.linspace(0.0, float(lx), int(nx), dtype=torch.float32)
    ys = torch.linspace(0.0, float(ly), int(ny), dtype=torch.float32)

    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    verts_xy = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)

    faces: List[List[int]] = []

    def node_id(i: int, j: int) -> int:
        return i * int(nx) + j

    for i in range(int(ny) - 1):
        for j in range(int(nx) - 1):
            n00 = node_id(i, j)
            n10 = node_id(i, j + 1)
            n01 = node_id(i + 1, j)
            n11 = node_id(i + 1, j + 1)

            faces.append([n00, n10, n11])
            faces.append([n00, n11, n01])

    faces_t = torch.tensor(faces, dtype=torch.long)

    return verts_xy, faces_t


# -----------------------------------------------------------------------------
# Gaussian random field pressure helper
# -----------------------------------------------------------------------------

def generate_gaussian_random_pressure_field(
    verts_xy: torch.Tensor,
    *,
    rng: np.random.Generator,
    mean: float = 0.1,
    std: float = 0.02,
    corr_len: float = 0.25,
    n_modes: int = 64,
    pressure_min: float = 0.05,
    pressure_max: float = 0.15,
) -> torch.Tensor:
    """
    Generate a smooth Gaussian-random-field-like pressure on mesh nodes.

    A random Fourier feature expansion is used to avoid the O(N^3) cost of a
    full covariance decomposition. The field is standardized on the current
    mesh, shifted to the requested mean, and clipped to [pressure_min,
    pressure_max]. It is then recentered once and clipped again.
    """
    xy = verts_xy.detach().cpu().numpy().astype(np.float64)

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

    p = float(mean) + float(std) * raw
    p = np.clip(p, float(pressure_min), float(pressure_max))

    p = p + (float(mean) - float(np.mean(p)))
    p = np.clip(p, float(pressure_min), float(pressure_max))

    return torch.as_tensor(p[:, None], dtype=torch.float32)


# -----------------------------------------------------------------------------
# Triangle geometry for NO-side MR plate energy
# -----------------------------------------------------------------------------

def triangle_xy_area_and_grads(
    verts_xy: torch.Tensor,
    faces: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute area and P1 shape-function gradients for triangular elements.

    Returns
    -------
    area:
        [n_elem]
    dNdx:
        [n_elem, 3]
    dNdy:
        [n_elem, 3]
    """
    tri = verts_xy[faces]

    x1 = tri[:, 0, 0]
    y1 = tri[:, 0, 1]
    x2 = tri[:, 1, 0]
    y2 = tri[:, 1, 1]
    x3 = tri[:, 2, 0]
    y3 = tri[:, 2, 1]

    two_area_signed = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    two_area = torch.abs(two_area_signed)
    area = 0.5 * two_area

    denom = torch.where(
        torch.abs(two_area_signed) > 1.0e-14,
        two_area_signed,
        torch.full_like(two_area_signed, 1.0e-14),
    )

    dNdx = torch.stack(
        [
            (y2 - y3) / denom,
            (y3 - y1) / denom,
            (y1 - y2) / denom,
        ],
        dim=1,
    )

    dNdy = torch.stack(
        [
            (x3 - x2) / denom,
            (x1 - x3) / denom,
            (x2 - x1) / denom,
        ],
        dim=1,
    )

    return area, dNdx, dNdy


def precompute_plate_face_geometry(
    verts_xy: torch.Tensor,
    faces: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Precompute P1 triangle integration data used by the NO energy.

    Stored quantities:
        area       : [E]
        area_phys  : [E], alias for compatibility
        dNdx       : [E, 3]
        dNdy       : [E, 3]
        int_N      : [E, 3]
        mass_N     : [E, 3, 3]
        centroid_xy: [E, 2]
    """
    area, dNdx, dNdy = triangle_xy_area_and_grads(verts_xy, faces)
    tri_xy = verts_xy[faces]
    centroid_xy = tri_xy.mean(dim=1)

    int_N = (area[:, None] / 3.0) * torch.ones(
        (faces.shape[0], 3),
        dtype=torch.float32,
        device=verts_xy.device,
    )

    base_mass = torch.tensor(
        [
            [2.0, 1.0, 1.0],
            [1.0, 2.0, 1.0],
            [1.0, 1.0, 2.0],
        ],
        dtype=torch.float32,
        device=verts_xy.device,
    )

    mass_N = area[:, None, None] / 12.0 * base_mass[None, :, :]

    return {
        "area": area.float(),
        "area_phys": area.float(),
        "dNdx": dNdx.float(),
        "dNdy": dNdy.float(),
        "int_N": int_N.float(),
        "mass_N": mass_N.float(),
        "centroid_xy": centroid_xy.float(),
    }
