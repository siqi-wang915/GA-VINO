from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import csv

import numpy as np
from mpi4py import MPI

import dolfinx
import ufl
from basix.ufl import element, mixed_element
from dolfinx.fem import Function, Constant, dirichletbc, functionspace
from dolfinx.mesh import create_mesh, locate_entities_boundary
from ufl import dx, grad, inner, split, sym, tr


# -----------------------------------------------------------------------------
# Common FEM utilities for Mindlin-Reissner plate solvers
# -----------------------------------------------------------------------------
# This module contains functionality shared by hole / holes2 / lshape solvers.
# It is intentionally independent of the square case solver so that
# FEM/solvers/square.py can be removed or kept only as an unused benchmark.



def _to_float_or_nan(value: Any) -> float:
    """Convert scalar-like values to float; return nan if conversion fails."""
    try:
        return float(value)
    except Exception:
        return float("nan")


def _save_fem_timing_table(
    fem_root: str | Path,
    rows: List[Dict[str, Any]],
    *,
    csv_name: str = "fem_solve_times.csv",
    npz_name: str = "fem_solve_times.npz",
) -> Dict[str, Optional[str]]:
    """
    Save a per-case FEM timing table beside the FEM prediction .npz files.

    The table is written only by MPI rank 0. It contains both successful and
    failed/skipped cases when such rows are provided by the batch solver.
    """
    fem_root = Path(fem_root)
    csv_path = fem_root / csv_name
    npz_path = fem_root / npz_name

    if MPI.COMM_WORLD.rank != 0:
        return {"timing_csv": str(csv_path), "timing_npz": str(npz_path)}

    fem_root.mkdir(parents=True, exist_ok=True)

    if not rows:
        return {"timing_csv": None, "timing_npz": None}

    keys: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    string_keys = {"case_tag", "status", "out_path", "error", "solver_name", "fem_method"}
    payload: Dict[str, np.ndarray] = {}
    for key in keys:
        values = [row.get(key, "") for row in rows]
        if key in string_keys:
            payload[key] = np.asarray(values, dtype=str)
        else:
            numeric_values = []
            for value in values:
                numeric_values.append(_to_float_or_nan(value))
            payload[key] = np.asarray(numeric_values, dtype=np.float64)

    np.savez_compressed(npz_path, **payload)
    return {"timing_csv": str(csv_path), "timing_npz": str(npz_path)}


# -----------------------------------------------------------------------------
# Mesh construction
# -----------------------------------------------------------------------------

def _create_mesh_from_case(case_data: Dict[str, Any]) -> dolfinx.mesh.Mesh:
    """
    Create a DOLFINx 2D triangular mesh from saved case data.

    Required case fields:
        verts_xy : [N, 2]
        faces    : [E, 3]

    The computational mesh is a flat 2D mid-plane mesh. The Reissner-Mindlin
    unknowns are theta=(theta_x, theta_y) and transverse displacement w.
    """
    verts_xy = np.asarray(case_data["verts_xy"], dtype=np.float64)
    faces = np.asarray(case_data["faces"], dtype=np.int64)

    if verts_xy.ndim != 2 or verts_xy.shape[1] != 2:
        raise ValueError(f"Expected verts_xy with shape [N,2], got {verts_xy.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Expected faces with shape [E,3], got {faces.shape}")

    verts_xy = np.ascontiguousarray(verts_xy, dtype=np.float64)
    faces = np.ascontiguousarray(faces, dtype=np.int64)

    coordinate_element = element("Lagrange", "triangle", 1, shape=(2,))
    domain = ufl.Mesh(coordinate_element)

    mesh = create_mesh(
        MPI.COMM_WORLD,
        faces,
        domain,
        verts_xy,
    )

    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim - 1)

    return mesh


# -----------------------------------------------------------------------------
# Linear Reissner-Mindlin plate with Duran-Liberman/MITC shear reduction
# -----------------------------------------------------------------------------

def _inner_e(mesh: dolfinx.mesh.Mesh, x, y):
    """
    Edge inner product used by the Duran-Liberman reduction operator.

    This follows the FEniCSx-Shells demo but is written explicitly so the solver
    does not depend on projected assembly helpers.
    """
    dSp = ufl.Measure("dS", domain=mesh, metadata={"quadrature_degree": 1})
    dsp = ufl.Measure("ds", domain=mesh, metadata={"quadrature_degree": 1})

    n = ufl.FacetNormal(mesh)
    tangent = ufl.as_vector((-n[1], n[0]))

    return (
        (inner(x, tangent) * inner(y, tangent))("+") * dSp
        + (inner(x, tangent) * inner(y, tangent)) * dsp
    )


def _build_pressure_function_from_case(
    mesh: dolfinx.mesh.Mesh,
    case_data: Dict[str, Any],
) -> Function:
    """
    Build a CG1 scalar pressure function from nodal pressure values.

    The saved case pressure is ordered consistently with case_data["verts_xy"].
    DOLFINx may use a different local dof ordering, so values are assigned by
    matching dof coordinates to the saved vertex coordinates.
    """
    verts_xy = np.asarray(case_data["verts_xy"], dtype=np.float64)
    pressure_nodes = np.asarray(case_data["pressure"], dtype=np.float64).reshape(-1)

    if pressure_nodes.shape[0] != verts_xy.shape[0]:
        raise ValueError(
            f"pressure node count {pressure_nodes.shape[0]} does not match "
            f"verts_xy node count {verts_xy.shape[0]}."
        )

    P_el = element("Lagrange", mesh.basix_cell(), 1)
    P = functionspace(mesh, P_el)
    p_func = Function(P)

    dof_xy = np.asarray(P.tabulate_dof_coordinates()[:, :2], dtype=np.float64)

    # Fast path: coordinate dictionary. Rounding avoids tiny floating errors
    # introduced by DOLFINx coordinate storage.
    coord_to_value = {
        (round(float(x), 12), round(float(y), 12)): float(v)
        for (x, y), v in zip(verts_xy, pressure_nodes)
    }

    values = np.empty((dof_xy.shape[0],), dtype=np.float64)
    missing: List[int] = []

    for i, (x, y) in enumerate(dof_xy):
        key = (round(float(x), 12), round(float(y), 12))
        if key in coord_to_value:
            values[i] = coord_to_value[key]
        else:
            missing.append(i)

    # Robust fallback for any unmatched dof coordinates. This is rarely used for
    # the current structured square mesh, but keeps the function safe.
    for i in missing:
        d2 = np.sum((verts_xy - dof_xy[i][None, :]) ** 2, axis=1)
        j = int(np.argmin(d2))
        values[i] = pressure_nodes[j]

    p_func.x.array[: values.shape[0]] = values.astype(p_func.x.array.dtype, copy=False)
    p_func.x.scatter_forward()

    return p_func


def _build_problem_components(
    mesh: dolfinx.mesh.Mesh,
    case_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build the linear Reissner-Mindlin Durán-Liberman plate problem.

    Unknown order in the mixed space:
        theta    : [CG2]^2
        w        : CG1
        R_gamma  : N1curl
        p_lm     : N1curl

    The physical pressure from case_data["pressure"] is used directly as a
    spatially varying transverse load. Unlike the demo benchmark, no t^3 scaling
    is added to the external work.
    """
    E_val = float(np.asarray(case_data["E"]).reshape(-1)[0])
    nu_val = float(np.asarray(case_data["nu"]).reshape(-1)[0])
    t_val = float(np.asarray(case_data["thickness"]).reshape(-1)[0])

    pressure_nodes = np.asarray(case_data["pressure"], dtype=np.float64).reshape(-1)
    p_total = float(np.mean(pressure_nodes))

    cell = mesh.basix_cell()
    U_el = mixed_element(
        [
            element("Lagrange", cell, 2, shape=(2,)),
            element("Lagrange", cell, 1),
            element("N1curl", cell, 1),
            element("N1curl", cell, 1),
        ]
    )
    U = functionspace(mesh, U_el)

    u0 = Function(U)
    u_trial = ufl.TrialFunction(U)
    u_test = ufl.TestFunction(U)

    theta, w, R_gamma, p_lm = split(u0)

    # Material constants.
    E = Constant(mesh, dolfinx.default_scalar_type(E_val))
    nu = Constant(mesh, dolfinx.default_scalar_type(nu_val))
    t = Constant(mesh, dolfinx.default_scalar_type(t_val))
    kappa = Constant(mesh, dolfinx.default_scalar_type(5.0 / 6.0))

    # Spatial pressure field. For the old uniform-load case this function is
    # simply constant over all nodes; for the Gaussian random field case it
    # preserves the nodal variation instead of averaging it.
    p_load = _build_pressure_function_from_case(mesh, case_data)

    # Linear Reissner-Mindlin bending and shear terms.
    k = sym(grad(theta))
    D_half = (E * t**3) / (24.0 * (1.0 - nu**2))
    psi_b = D_half * ((1.0 - nu) * tr(k * k) + nu * (tr(k)) ** 2)

    psi_s = ((E * kappa * t) / (4.0 * (1.0 + nu))) * inner(R_gamma, R_gamma)

    gamma = grad(w) - theta
    Pi_R = _inner_e(mesh, gamma - R_gamma, p_lm)

    # Physical external work. Positive pressure gives positive w under Pi = U - W.
    W_ext = p_load * w * dx

    Pi = psi_b * dx + psi_s * dx + Pi_R - W_ext

    F_res = ufl.derivative(Pi, u0, u_test)
    J = ufl.derivative(F_res, u0, u_trial)

    return {
        "U": U,
        "u0": u0,
        "F_res": F_res,
        "J": J,
        "p_load": p_load,
        "p_total": p_total,
        "E": E,
        "nu": nu,
        "t": t,
        "shear_correction": kappa,
        # Legacy aliases. These make the object names familiar, although this
        # solver uses LinearProblem instead of Newton.
        "V": U,
        "q_func": u0,
    }


def triangle_quadrature_7() -> Tuple[np.ndarray, np.ndarray]:
    """Seven-point Dunavant rule in barycentric coordinates.

    The returned weights are normalized to sum to one, so a physical-triangle
    integral is ``area * sum_q weights[q] * value[q]``.
    """
    rule = [
        (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.225000000000000),
        (0.059715871789770, 0.470142064105115, 0.470142064105115, 0.132394152788506),
        (0.470142064105115, 0.059715871789770, 0.470142064105115, 0.132394152788506),
        (0.470142064105115, 0.470142064105115, 0.059715871789770, 0.132394152788506),
        (0.797426985353087, 0.101286507323456, 0.101286507323456, 0.125939180544827),
        (0.101286507323456, 0.797426985353087, 0.101286507323456, 0.125939180544827),
        (0.101286507323456, 0.101286507323456, 0.797426985353087, 0.125939180544827),
    ]
    barycentric = np.asarray([row[:3] for row in rule], dtype=np.float64)
    weights = np.asarray([row[3] for row in rule], dtype=np.float64)
    return barycentric, weights


def _map_local_cells_to_saved_faces(
    mesh: dolfinx.mesh.Mesh,
    case_data: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """Map DOLFINx local cell indices to the original saved-face order."""
    if MPI.COMM_WORLD.size != 1:
        raise RuntimeError(
            "Native FEM resultant export currently requires a serial FEM run. "
            "Run stage1 with one MPI rank so every saved face is exported once."
        )

    tdim = mesh.topology.dim
    index_map = mesh.topology.index_map(tdim)
    cells = np.arange(index_map.size_local, dtype=np.int32)
    local_centroids = np.asarray(
        dolfinx.mesh.compute_midpoints(mesh, tdim, cells)[:, :2],
        dtype=np.float64,
    )

    verts_xy = np.asarray(case_data["verts_xy"], dtype=np.float64)
    faces = np.asarray(case_data["faces"], dtype=np.int64)
    saved_centroids = verts_xy[faces].mean(axis=1)
    if cells.shape[0] != faces.shape[0]:
        raise RuntimeError(
            f"Local FEM cell count {cells.shape[0]} does not match saved face "
            f"count {faces.shape[0]}."
        )

    saved_lookup = {
        (round(float(x), 12), round(float(y), 12)): i
        for i, (x, y) in enumerate(saved_centroids)
    }
    saved_ids = np.empty((cells.shape[0],), dtype=np.int64)
    used: set[int] = set()
    for local_id, (x, y) in enumerate(local_centroids):
        key = (round(float(x), 12), round(float(y), 12))
        saved_id = saved_lookup.get(key)
        if saved_id is None:
            d2 = np.sum((saved_centroids - local_centroids[local_id]) ** 2, axis=1)
            saved_id = int(np.argmin(d2))
            if float(np.sqrt(d2[saved_id])) > 1.0e-7:
                raise RuntimeError(
                    "Could not reliably match a DOLFINx cell to the saved mesh "
                    f"(nearest-centroid distance={np.sqrt(d2[saved_id]):.3e})."
                )
        if saved_id in used:
            raise RuntimeError("DOLFINx-to-saved cell mapping is not one-to-one.")
        used.add(saved_id)
        saved_ids[local_id] = saved_id

    return cells, saved_ids


def evaluate_native_resultants_on_quadrature(
    *,
    mesh: dolfinx.mesh.Mesh,
    case_data: Dict[str, Any],
    theta_func: Function,
    R_gamma_func: Function,
    E,
    nu,
    thickness,
    shear_correction,
) -> Dict[str, np.ndarray]:
    """Evaluate native CG2 bending moments and DL shear forces at quadrature points."""
    barycentric, weights = triangle_quadrature_7()
    # On the reference triangle (0,0), (1,0), (0,1), x=(lambda_2,lambda_3).
    reference_points = np.ascontiguousarray(barycentric[:, 1:3], dtype=np.float64)
    cells, saved_ids = _map_local_cells_to_saved_faces(mesh, case_data)

    curvature = sym(grad(theta_func))
    bending_stiffness = (E * thickness**3) / (12.0 * (1.0 - nu**2))
    moment_tensor = bending_stiffness * (
        (1.0 - nu) * curvature + nu * tr(curvature) * ufl.Identity(2)
    )
    moment_vector = ufl.as_vector(
        (moment_tensor[0, 0], moment_tensor[1, 1], moment_tensor[0, 1])
    )
    shear_vector = (
        shear_correction * E * thickness / (2.0 * (1.0 + nu))
    ) * R_gamma_func

    moment_expr = dolfinx.fem.Expression(moment_vector, reference_points)
    shear_expr = dolfinx.fem.Expression(shear_vector, reference_points)
    moment_local = np.asarray(moment_expr.eval(mesh, cells), dtype=np.float64).reshape(
        cells.shape[0], reference_points.shape[0], 3
    )
    shear_local = np.asarray(shear_expr.eval(mesh, cells), dtype=np.float64).reshape(
        cells.shape[0], reference_points.shape[0], 2
    )

    n_faces = int(np.asarray(case_data["faces"]).shape[0])
    moment_qp = np.empty((n_faces, reference_points.shape[0], 3), dtype=np.float64)
    shear_qp = np.empty((n_faces, reference_points.shape[0], 2), dtype=np.float64)

    # DOLFINx may permute the three vertices of a cell relative to the saved
    # ``faces`` row.  Reorder quadrature values by matching their physical
    # coordinates so the saved barycentric coordinates also apply to the NO
    # field on that face.
    verts_xy = np.asarray(case_data["verts_xy"], dtype=np.float64)
    faces = np.asarray(case_data["faces"], dtype=np.int64)
    geometry_x = np.asarray(mesh.geometry.x[:, :2], dtype=np.float64)
    geometry_dofmap = mesh.geometry.dofmap
    for local_id, (cell, saved_id) in enumerate(zip(cells, saved_ids)):
        geometry_dofs = np.asarray(geometry_dofmap[int(cell)], dtype=np.int64).reshape(-1)
        local_vertices = geometry_x[geometry_dofs[:3]]
        saved_vertices = verts_xy[faces[int(saved_id)]]
        local_qp_xy = barycentric @ local_vertices
        saved_qp_xy = barycentric @ saved_vertices

        distance_sq = np.sum(
            (saved_qp_xy[:, None, :] - local_qp_xy[None, :, :]) ** 2,
            axis=2,
        )
        local_qp_ids = np.argmin(distance_sq, axis=1)
        if len(set(local_qp_ids.tolist())) != reference_points.shape[0]:
            raise RuntimeError("Quadrature-point mapping is not one-to-one.")
        max_distance = float(
            np.sqrt(np.max(distance_sq[np.arange(reference_points.shape[0]), local_qp_ids]))
        )
        if max_distance > 1.0e-8:
            raise RuntimeError(
                "Could not align DOLFINx and saved-face quadrature points "
                f"(maximum distance={max_distance:.3e})."
            )

        moment_qp[int(saved_id)] = moment_local[local_id, local_qp_ids]
        shear_qp[int(saved_id)] = shear_local[local_id, local_qp_ids]

    return {
        "resultant_quadrature_barycentric": barycentric,
        "resultant_quadrature_weights": weights,
        "fem_moment_qp": moment_qp,
        "fem_shear_qp": shear_qp,
        "fem_moment_elem": np.einsum("q,eqi->ei", weights, moment_qp),
        "fem_shear_elem": np.einsum("q,eqi->ei", weights, shear_qp),
    }


# -----------------------------------------------------------------------------
# Boundary conditions
# -----------------------------------------------------------------------------

def _build_boundary_conditions(
    mesh: dolfinx.mesh.Mesh,
    U,
    case_data: Dict[str, Any],
):
    """
    Fully clamped outer square boundary.

    For the mixed DL formulation, we constrain all mixed DOFs on the clamped
    boundary, following the clamped Reissner-Mindlin demo. This enforces
    theta=0 and w=0, and also constrains the auxiliary Nedelec fields on the
    clamped boundary.
    """
    fdim = mesh.topology.dim - 1

    lx = float(np.asarray(case_data["lx"]).reshape(-1)[0])
    ly = float(np.asarray(case_data["ly"]).reshape(-1)[0])
    tol = 1.0e-8

    def outer_boundary(x):
        return (
            np.isclose(x[0], 0.0, atol=tol)
            | np.isclose(x[0], lx, atol=tol)
            | np.isclose(x[1], 0.0, atol=tol)
            | np.isclose(x[1], ly, atol=tol)
        )

    boundary_facets = locate_entities_boundary(mesh, fdim, outer_boundary)

    u_boundary = Function(U)
    boundary_dofs = dolfinx.fem.locate_dofs_topological(
        U,
        fdim,
        boundary_facets,
    )

    return [dirichletbc(u_boundary, boundary_dofs)]


# -----------------------------------------------------------------------------
# Result evaluation
# -----------------------------------------------------------------------------

def _eval_on_query_points(
    mesh: dolfinx.mesh.Mesh,
    *,
    theta_func: Function,
    w_func: Function,
    query_xyz: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate theta and w on query points.

    Returns
    -------
    w_vals:
        [N]
    theta_vals:
        [N, 2]
    """
    query_xyz = np.asarray(query_xyz, dtype=np.float64)

    pts = np.zeros((query_xyz.shape[0], 3), dtype=np.float64)
    pts[:, 0] = query_xyz[:, 0]
    pts[:, 1] = query_xyz[:, 1]
    pts[:, 2] = 0.0
    pts = np.ascontiguousarray(pts, dtype=np.float64)

    bb_tree = dolfinx.geometry.bb_tree(mesh, mesh.topology.dim)
    cell_candidates = dolfinx.geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = dolfinx.geometry.compute_colliding_cells(
        mesh,
        cell_candidates,
        pts,
    )

    cells: List[int] = []
    valid_pts: List[np.ndarray] = []
    valid_ids: List[int] = []

    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) == 0:
            raise RuntimeError(
                f"Could not find containing cell for query point index {i}: {pts[i]}"
            )
        cells.append(int(links[0]))
        valid_pts.append(pts[i])
        valid_ids.append(i)

    valid_pts_arr = np.ascontiguousarray(valid_pts, dtype=np.float64)
    cells_arr = np.asarray(cells, dtype=np.int32)
    valid_ids_arr = np.asarray(valid_ids, dtype=np.int64)

    w_valid = w_func.eval(valid_pts_arr, cells_arr).reshape(-1)
    theta_valid = theta_func.eval(valid_pts_arr, cells_arr)

    w_vals = np.zeros((pts.shape[0],), dtype=np.float64)
    theta_vals = np.zeros((pts.shape[0], 2), dtype=np.float64)

    w_vals[valid_ids_arr] = w_valid
    theta_vals[valid_ids_arr, :] = theta_valid

    return w_vals, theta_vals


def _find_center_value(
    query_xyz: np.ndarray,
    disp_scalar: np.ndarray,
    lx: float,
    ly: float,
) -> float:
    xy = np.asarray(query_xyz)[:, :2]
    center = np.asarray([0.5 * lx, 0.5 * ly])
    idx = int(np.argmin(np.linalg.norm(xy - center[None, :], axis=1)))
    return float(np.asarray(disp_scalar).reshape(-1)[idx])
