from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import time

import numpy as np
from mpi4py import MPI

import dolfinx
import ufl
from basix.ufl import element, mixed_element
from dolfinx.fem import Function, Constant, dirichletbc, functionspace
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import locate_entities_boundary
from ufl import dx, grad, inner, split, sym, tr

from shared.common.data_io import load_npz_dict, save_npz_dict
from shared.common.paths import (
    build_paths,
    case_npz_path,
    fem_pred_npz_path,
)

from FEM.solvers.common import (
    _create_mesh_from_case,
    _inner_e,
    _eval_on_query_points,
    _save_fem_timing_table,
    _build_pressure_function_from_case,
    evaluate_native_resultants_on_quadrature,
)


CASE_NAME = "lshape_grfm_up"
SOLVER_NAME = "lshape_rm_dl_linear_plate_fem"




def _build_dg0_material_function(
    mesh: dolfinx.mesh.Mesh,
    elem_values: np.ndarray,
    *,
    case_data: Dict[str, Any],
    name: str,
) -> Function:
    """
    Build a DG0 scalar material field from one value per saved triangle.

    The mapping is done by matching DG0 dof coordinates, i.e. local cell
    centroids in the DOLFINx mesh, to the centroids of the saved triangles in
    case_data["faces"]. This avoids relying on DOLFINx local cell ordering.

    This is more robust than assigning values by array order and helps avoid
    possible material-cell mismatch in MPI or reordered meshes.
    """
    elem_values = np.asarray(elem_values, dtype=np.float64).reshape(-1)

    verts_xy = np.asarray(case_data["verts_xy"], dtype=np.float64)
    faces = np.asarray(case_data["faces"], dtype=np.int64)

    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Expected case_data['faces'] with shape [n_faces, 3], got {faces.shape}")

    n_faces = int(faces.shape[0])
    if elem_values.shape[0] != n_faces:
        raise ValueError(
            f"Material field {name!r} must have one value per saved face. "
            f"Got {elem_values.shape[0]}, n_faces={n_faces}."
        )

    # Centroids in the original saved mesh order.
    saved_centroids = verts_xy[faces].mean(axis=1)

    V0_el = element("DG", mesh.basix_cell(), 0)
    V0 = functionspace(mesh, V0_el)
    f = Function(V0)
    f.name = str(name)

    # For a DG0 scalar field, each dof coordinate is the cell representative point
    # and should coincide with the cell centroid for P0 on triangles.
    dof_xy = np.asarray(V0.tabulate_dof_coordinates()[:, :2], dtype=np.float64)
    n_dofs = int(dof_xy.shape[0])

    assigned = np.empty((n_dofs,), dtype=np.float64)

    # Fast path: exact/rounded centroid lookup.
    centroid_to_value: Dict[Tuple[float, float], float] = {
        (round(float(x), 12), round(float(y), 12)): float(v)
        for (x, y), v in zip(saved_centroids, elem_values)
    }

    missing: List[int] = []
    for i, (x, y) in enumerate(dof_xy):
        key = (round(float(x), 12), round(float(y), 12))
        if key in centroid_to_value:
            assigned[i] = centroid_to_value[key]
        else:
            missing.append(i)

    # Robust fallback: nearest saved centroid. This avoids adding scipy dependency.
    # For your current mesh size this O(n_dofs * n_faces) fallback is acceptable,
    # and it is usually used only for a few unmatched points.
    for i in missing:
        d2 = np.sum((saved_centroids - dof_xy[i][None, :]) ** 2, axis=1)
        j = int(np.argmin(d2))
        min_dist = float(np.sqrt(d2[j]))

        # Warn if the nearest centroid is unexpectedly far away.
        # For a correctly matched mesh this should be near machine precision.
        if min_dist > 1.0e-7:
            print(
                f"[FEM][lshape][Warning] DG0 dof {i} for {name} matched by nearest centroid "
                f"with distance {min_dist:.3e}."
            )

        assigned[i] = elem_values[j]

    if f.x.array.shape[0] != assigned.shape[0]:
        raise RuntimeError(
            f"DG0 array size mismatch for {name!r}: "
            f"f.x.array has {f.x.array.shape[0]} entries, "
            f"but tabulate_dof_coordinates returned {assigned.shape[0]} points."
        )

    f.x.array[:] = assigned.astype(f.x.array.dtype, copy=False)
    f.x.scatter_forward()

    return f


def _build_lshape_problem_components(
    mesh: dolfinx.mesh.Mesh,
    case_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build the linear Reissner-Mindlin Durán-Liberman plate problem for L-shape.

    This version supports elementwise DG0 material fields:
        E_elem  : one Young's modulus value per triangle
        nu_elem : one Poisson's ratio value per triangle

    If these fields are absent, it falls back to constant E and nu. External
    work supports either a transverse domain pressure or a prescribed
    transverse line load on the annular sector's theta_start radial edge.
    """
    E_val = float(np.asarray(case_data["E"]).reshape(-1)[0])
    nu_val = float(np.asarray(case_data["nu"]).reshape(-1)[0])
    t_val = float(np.asarray(case_data["thickness"]).reshape(-1)[0])

    pressure_nodes = np.asarray(case_data["pressure"], dtype=np.float64).reshape(-1)
    load_type = str(np.asarray(case_data.get("load_type", "")).reshape(-1)[0]).lower()
    is_boundary_line_load = "boundary_line_load" in load_type
    if is_boundary_line_load:
        p_total = float(
            np.asarray(case_data.get("line_load_mean", np.mean(pressure_nodes))).reshape(-1)[0]
        )
    else:
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

    # Material constants or DG0 material fields.
    if "E_elem" in case_data and "nu_elem" in case_data:
        E = _build_dg0_material_function(
            mesh,
            np.asarray(case_data["E_elem"], dtype=np.float64).reshape(-1),
            case_data=case_data,
            name="E_elem",
        )
        nu = _build_dg0_material_function(
            mesh,
            np.asarray(case_data["nu_elem"], dtype=np.float64).reshape(-1),
            case_data=case_data,
            name="nu_elem",
        )
    else:
        E = Constant(mesh, dolfinx.default_scalar_type(E_val))
        nu = Constant(mesh, dolfinx.default_scalar_type(nu_val))

    t = Constant(mesh, dolfinx.default_scalar_type(t_val))
    kappa = Constant(mesh, dolfinx.default_scalar_type(5.0 / 6.0))
    p_load = _build_pressure_function_from_case(mesh, case_data)

    # Linear Reissner-Mindlin bending and shear terms.
    k = sym(grad(theta))
    D_half = (E * t**3) / (24.0 * (1.0 - nu**2))
    psi_b = D_half * ((1.0 - nu) * tr(k * k) + nu * (tr(k)) ** 2)

    psi_s = ((E * kappa * t) / (4.0 * (1.0 + nu))) * inner(R_gamma, R_gamma)

    gamma = grad(w) - theta
    Pi_R = _inner_e(mesh, gamma - R_gamma, p_lm)

    if is_boundary_line_load:
        theta_start = float(np.asarray(case_data["theta_start"]).reshape(-1)[0])
        r_inner = float(np.asarray(case_data["r_inner"]).reshape(-1)[0])
        r_outer = float(np.asarray(case_data["r_outer"]).reshape(-1)[0])
        direction = np.asarray(
            [np.cos(theta_start), np.sin(theta_start)], dtype=np.float64
        )
        fdim = mesh.topology.dim - 1

        def loaded_radial_edge(x):
            cross = x[0] * direction[1] - x[1] * direction[0]
            radial = x[0] * direction[0] + x[1] * direction[1]
            return (
                np.isclose(cross, 0.0, atol=1.0e-6)
                & (radial >= r_inner - 1.0e-6)
                & (radial <= r_outer + 1.0e-6)
            )

        loaded_facets = locate_entities_boundary(mesh, fdim, loaded_radial_edge)
        if loaded_facets.size == 0:
            raise RuntimeError("No facets found on the prescribed-load radial edge.")
        order = np.argsort(loaded_facets)
        facet_tags = dolfinx.mesh.meshtags(
            mesh,
            fdim,
            loaded_facets[order],
            np.ones(loaded_facets.size, dtype=np.int32),
        )
        ds_loaded = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)
        W_ext = p_load * w * ds_loaded(1)
    else:
        W_ext = p_load * w * dx
    Pi = psi_b * dx + psi_s * dx + Pi_R - W_ext

    F_res = ufl.derivative(Pi, u0, u_test)
    J = ufl.derivative(F_res, u0, u_trial)

    return {
        "U": U,
        "u0": u0,
        "F_res": F_res,
        "J": J,
        "p_total": p_total,
        "E": E,
        "nu": nu,
        "t": t,
        "shear_correction": kappa,
        "V": U,
        "q_func": u0,
    }

def _build_left_clamped_boundary_conditions(
    mesh: dolfinx.mesh.Mesh,
    U,
    case_data: Dict[str, Any],
):
    """
    Left-clamped L-shaped plate.

    Fixed boundary:
        x = 0

    For the mixed Durán-Liberman formulation, all mixed DOFs on the clamped
    boundary are set to zero, following the clamped RM demo convention. All
    other outer edges remain free.
    """
    fdim = mesh.topology.dim - 1
    tol = 1.0e-8

    def left_boundary(x):
        return np.isclose(x[0], 0.0, atol=tol)

    boundary_facets = locate_entities_boundary(mesh, fdim, left_boundary)

    u_boundary = Function(U)
    boundary_dofs = dolfinx.fem.locate_dofs_topological(
        U,
        fdim,
        boundary_facets,
    )

    return [dirichletbc(u_boundary, boundary_dofs)]


def _find_max_abs_w(query_xyz: np.ndarray, disp_scalar: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(disp_scalar).reshape(-1))))


def solve_one_lshape_case(
    *,
    case_path: str | Path,
    out_path: str | Path,
    case_name: str = CASE_NAME,
    overwrite: bool = False,
    n_load_steps: int = 1,
) -> Dict[str, Any]:
    """
    Solve one L-shaped plate case using the linear Reissner-Mindlin
    Durán-Liberman/MITC solver.

    Boundary condition:
        left boundary clamped;
        all other boundaries free.

    n_load_steps is accepted for API compatibility and ignored.
    """
    case_path = Path(case_path)
    out_path = Path(out_path)

    if out_path.exists() and not overwrite:
        return {
            "status": "skipped_existing",
            "out_path": str(out_path),
        }

    total_t0 = time.perf_counter()

    case_data = load_npz_dict(case_path)
    mesh = _create_mesh_from_case(case_data)

    comps = _build_lshape_problem_components(mesh, case_data)
    U = comps["U"]
    F_res = comps["F_res"]
    J = comps["J"]
    p_total = comps["p_total"]

    bcs = _build_left_clamped_boundary_conditions(mesh, U, case_data)

    problem = LinearProblem(
        J,
        -F_res,
        bcs=bcs,
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
        petsc_options_prefix="lshape_rm_dl_",
    )

    solve_t0 = time.perf_counter()
    uh = problem.solve()
    uh.x.scatter_forward()
    fem_solve_time_sec = time.perf_counter() - solve_t0

    theta_sol, w_sol, R_gamma_sol, p_lm_sol = uh.split()

    native_resultants = evaluate_native_resultants_on_quadrature(
        mesh=mesh,
        case_data=case_data,
        theta_func=theta_sol,
        R_gamma_func=R_gamma_sol,
        E=comps["E"],
        nu=comps["nu"],
        thickness=comps["t"],
        shear_correction=comps["shear_correction"],
    )

    query_xyz = np.asarray(case_data["query_xyz"], dtype=np.float64)
    w_vals, theta_vals = _eval_on_query_points(
        mesh=mesh,
        theta_func=theta_sol,
        w_func=w_sol,
        query_xyz=query_xyz,
    )

    fem_local = np.column_stack(
        [
            w_vals,
            theta_vals[:, 0],
            theta_vals[:, 1],
        ]
    )

    fem_global_disp = np.zeros((query_xyz.shape[0], 3), dtype=np.float64)
    fem_global_disp[:, 2] = w_vals

    disp_scalar = w_vals.copy()
    rot_scalar = np.abs(theta_vals[:, 0]) + np.abs(theta_vals[:, 1])

    max_abs_w = _find_max_abs_w(query_xyz, disp_scalar)

    n_vertices = int(np.asarray(case_data["verts_xy"]).shape[0])
    n_faces = int(np.asarray(case_data["faces"]).shape[0])
    n_query_points = int(query_xyz.shape[0])
    fem_total_time_sec = time.perf_counter() - total_t0

    payload: Dict[str, Any] = {
        "case_tag": case_data.get("case_tag", np.asarray(case_path.stem)),
        "source_sample_id": case_data.get("source_sample_id", np.asarray(-1)),

        "query_xyz": query_xyz,
        "verts_xy": np.asarray(case_data["verts_xy"], dtype=np.float64),
        "faces": np.asarray(case_data["faces"], dtype=np.int64),

        "fem_local": fem_local,
        "fem_global_disp": fem_global_disp,
        "disp_scalar": disp_scalar,
        "rot_scalar": rot_scalar,

        **native_resultants,

        "lx": np.asarray(float(np.asarray(case_data["lx"]).reshape(-1)[0]), dtype=np.float64),
        "ly": np.asarray(float(np.asarray(case_data["ly"]).reshape(-1)[0]), dtype=np.float64),

        "case_name": np.asarray(case_name),
        "solver_name": np.asarray(SOLVER_NAME),
        "fem_method": np.asarray("linear_reissner_mindlin_duran_liberman_lshape_plate"),
        "linear_solve": np.asarray(True),
        "pressure_used": np.asarray(float(p_total), dtype=np.float64),
        "max_abs_w": np.asarray(max_abs_w, dtype=np.float64),
        "n_query_points": np.asarray(n_query_points, dtype=np.int64),
        "n_vertices": np.asarray(n_vertices, dtype=np.int64),
        "n_faces": np.asarray(n_faces, dtype=np.int64),
        "fem_solve_time_sec": np.asarray(fem_solve_time_sec, dtype=np.float64),
        "fem_solve_time_ms": np.asarray(1000.0 * fem_solve_time_sec, dtype=np.float64),
        "fem_total_time_sec": np.asarray(fem_total_time_sec, dtype=np.float64),
        "fem_total_time_ms": np.asarray(1000.0 * fem_total_time_sec, dtype=np.float64),
    }

    for key in (
        "E",
        "nu",
        "thickness",
        "pressure",
        "pressure_value",
        "pressure_min",
        "pressure_max",
        "geometry_type",
        "load_type",
        "bbox",
        "bc_factor",
        "lshape_vertices",
        "boundary_segments",
        "pc_fixed_flag",
        "cut_w",
        "cut_h",
        "x_cut",
        "y_cut",
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
        "line_load_mean",
        "line_load_target_mean",
        "line_load_min",
        "line_load_max",
        "line_load_log_std",
        "line_load_corr_len",
        "loaded_boundary_nodes",
        "loaded_boundary_edges",
    ):
        if key in case_data:
            payload[key] = case_data[key]

    save_npz_dict(out_path, payload, overwrite=True)

    return {
        "status": "solved",
        "out_path": str(out_path),
        "max_abs_w": max_abs_w,
        "n_query_points": n_query_points,
        "n_vertices": n_vertices,
        "n_faces": n_faces,
        "fem_solve_time_sec": fem_solve_time_sec,
        "fem_solve_time_ms": 1000.0 * fem_solve_time_sec,
        "fem_total_time_sec": fem_total_time_sec,
        "fem_total_time_ms": 1000.0 * fem_total_time_sec,
    }


def solve_lshape(
    *,
    case_name: str = CASE_NAME,
    case_tags: Optional[List[str]] = None,
    root: Optional[str | Path] = None,
    overwrite: bool = False,
    n_load_steps: int = 1,
    **kwargs,
) -> Dict[str, Any]:
    """
    Batch solve L-shape cases with the linear RM-DL solver.

    n_load_steps is accepted for compatibility with registry/stage1 and ignored.
    """
    paths = build_paths(case_name, root=root, create=True)

    case_root = paths["case_root"]
    fem_root = paths["fem_prediction_root"]
    fem_root.mkdir(parents=True, exist_ok=True)

    if case_tags is None:
        case_tags = sorted(p.stem for p in case_root.glob("case_*.npz"))

    solved: List[str] = []
    skipped_existing: List[str] = []
    missing_case_files: List[str] = []
    failed: List[Tuple[str, str]] = []
    timing_rows: List[Dict[str, Any]] = []

    for tag in case_tags:
        cpath = case_npz_path(paths, tag)
        fpath = fem_pred_npz_path(paths, tag)

        if not cpath.exists():
            missing_case_files.append(tag)
            timing_rows.append({
                "case_tag": tag,
                "status": "missing_case_file",
                "solver_name": SOLVER_NAME,
                "out_path": str(fpath),
                "error": "case file not found",
            })
            continue

        case_wall_t0 = time.perf_counter()

        try:
            info = solve_one_lshape_case(
                case_path=cpath,
                out_path=fpath,
                case_name=case_name,
                overwrite=overwrite,
                n_load_steps=n_load_steps,
            )

            if info["status"] == "solved":
                solved.append(tag)

                if MPI.COMM_WORLD.rank == 0:
                    print(f"[FEM][lshape][RM-DL] solved {tag} -> {fpath}")
                    print(f"[FEM][lshape][RM-DL] max |w| for {tag}: {info['max_abs_w']:.6e}")

                timing_rows.append({
                    "case_tag": tag,
                    "status": "solved",
                    "solver_name": SOLVER_NAME,
                    "out_path": str(fpath),
                    "n_query_points": info.get("n_query_points", np.nan),
                    "n_vertices": info.get("n_vertices", np.nan),
                    "n_faces": info.get("n_faces", np.nan),
                    "fem_solve_time_sec": info.get("fem_solve_time_sec", np.nan),
                    "fem_solve_time_ms": info.get("fem_solve_time_ms", np.nan),
                    "fem_total_time_sec": info.get("fem_total_time_sec", np.nan),
                    "fem_total_time_ms": info.get("fem_total_time_ms", np.nan),
                    "case_wall_time_sec": time.perf_counter() - case_wall_t0,
                    "case_wall_time_ms": 1000.0 * (time.perf_counter() - case_wall_t0),
                    "max_abs_w": info.get("max_abs_w", np.nan),
                    "error": "",
                })

            elif info["status"] == "skipped_existing":
                skipped_existing.append(tag)

                if MPI.COMM_WORLD.rank == 0:
                    print(f"[FEM][lshape][RM-DL] skipped existing {tag}")

                timing_rows.append({
                    "case_tag": tag,
                    "status": "skipped_existing",
                    "solver_name": SOLVER_NAME,
                    "out_path": str(fpath),
                    "case_wall_time_sec": time.perf_counter() - case_wall_t0,
                    "case_wall_time_ms": 1000.0 * (time.perf_counter() - case_wall_t0),
                    "error": "",
                })

        except Exception as e:
            failed.append((tag, str(e)))

            if MPI.COMM_WORLD.rank == 0:
                print(f"[FEM][lshape][RM-DL] failed {tag}: {e}")

            timing_rows.append({
                "case_tag": tag,
                "status": "failed",
                "solver_name": SOLVER_NAME,
                "out_path": str(fpath),
                "case_wall_time_sec": time.perf_counter() - case_wall_t0,
                "case_wall_time_ms": 1000.0 * (time.perf_counter() - case_wall_t0),
                "error": str(e),
            })

    MPI.COMM_WORLD.Barrier()

    timing_paths = _save_fem_timing_table(fem_root, timing_rows)

    status = "done" if len(failed) == 0 else "partial_failed"

    return {
        "status": status,
        "case_name": case_name,
        "solver_name": SOLVER_NAME,
        "solved": solved,
        "skipped_existing": skipped_existing,
        "missing_case_files": missing_case_files,
        "failed": failed,
        "fem_prediction_root": str(fem_root),
        "timing_csv": timing_paths.get("timing_csv"),
        "timing_npz": timing_paths.get("timing_npz"),
    }


if __name__ == "__main__":
    info = solve_lshape(
        overwrite=False,
        n_load_steps=1,
    )

    if MPI.COMM_WORLD.rank == 0:
        print(info)
