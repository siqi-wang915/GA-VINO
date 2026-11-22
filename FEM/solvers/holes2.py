from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import time

import numpy as np
from mpi4py import MPI
from dolfinx.fem.petsc import LinearProblem

from shared.common.data_io import load_npz_dict, save_npz_dict
from shared.common.paths import (
    build_paths,
    case_npz_path,
    fem_pred_npz_path,
)

from FEM.solvers.common import (
    _create_mesh_from_case,
    _build_problem_components,
    _build_boundary_conditions,
    _eval_on_query_points,
    _find_center_value,
    _save_fem_timing_table,
    evaluate_native_resultants_on_quadrature,
)


CASE_NAME = "holes2_um_grfp"
SOLVER_NAME = "holes2_rm_dl_linear_plate_fem"


def solve_one_holes2_case(
    *,
    case_path: str | Path,
    out_path: str | Path,
    case_name: str = CASE_NAME,
    overwrite: bool = False,
    n_load_steps: int = 1,
) -> Dict[str, Any]:
    """
    Solve one two-hole flat plate case using the linear Reissner-Mindlin
    Durán-Liberman/MITC solver.

    n_load_steps is accepted for API compatibility with registry/stage1, but is
    ignored because the linear system is solved in one step.

    Boundary condition:
        outer square boundary clamped;
        hole boundaries free.

    Load:
        the spatial pressure field saved in case_data["pressure"] is used by
        the shared square _build_problem_components() function. Therefore, for
        Gaussian-random-field loading, FEM solves with p(x,y), not mean(p).
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

    comps = _build_problem_components(mesh, case_data)
    U = comps["U"]
    F_res = comps["F_res"]
    J = comps["J"]
    p_total = comps["p_total"]

    # Reuse the square outer-boundary clamped BC. The inner hole boundaries are not
    # selected by this predicate and remain free.
    bcs = _build_boundary_conditions(mesh, U, case_data)

    problem = LinearProblem(
        J,
        -F_res,
        bcs=bcs,
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
        petsc_options_prefix="holes2_rm_dl_",
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

    # Linear RM convention is the same as the NO convention:
    #   fem_local = [w, theta_x, theta_y]
    #   gamma = grad(w) - theta
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

    lx = float(np.asarray(case_data["lx"]).reshape(-1)[0])
    ly = float(np.asarray(case_data["ly"]).reshape(-1)[0])
    center_uz = _find_center_value(query_xyz, disp_scalar, lx, ly)
    max_abs_w = float(np.max(np.abs(disp_scalar)))

    n_vertices = int(np.asarray(case_data["verts_xy"]).shape[0])
    n_faces = int(np.asarray(case_data["faces"]).shape[0])
    n_query_points = int(query_xyz.shape[0])
    fem_total_time_sec = time.perf_counter() - total_t0

    pressure_array = np.asarray(case_data["pressure"], dtype=np.float64)

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

        "lx": np.asarray(lx, dtype=np.float64),
        "ly": np.asarray(ly, dtype=np.float64),

        "case_name": np.asarray(case_name),
        "solver_name": np.asarray(SOLVER_NAME),
        "fem_method": np.asarray("linear_reissner_mindlin_duran_liberman_plate_with_two_free_holes"),
        "linear_solve": np.asarray(True),
        "pressure_used": np.asarray(float(p_total), dtype=np.float64),
        "pressure_mean_used": np.asarray(float(np.mean(pressure_array)), dtype=np.float64),
        "pressure_min_used": np.asarray(float(np.min(pressure_array)), dtype=np.float64),
        "pressure_max_used": np.asarray(float(np.max(pressure_array)), dtype=np.float64),
        "max_abs_w": np.asarray(max_abs_w, dtype=np.float64),
        "center_uz": np.asarray(center_uz, dtype=np.float64),
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
        "pressure_net",
        "pressure_value",
        "pressure_target_mean",
        "pressure_min",
        "pressure_max",
        "pressure_std",
        "pressure_corr_len",
        "pressure_n_modes",
        "geometry_type",
        "load_type",
        "bbox",
        "bc_factor",
        "outer_boundary_nodes",
        "hole_boundary_nodes",
        "outer_xy",
        "pc_xy",
        "hole_rings",
        "hole_centers",
        "hole_mean_radii",
        "hole1_xy_raw",
        "hole2_xy_raw",
        "hole1_radius",
        "hole2_radius",
        "hole_merge_state",
        "n_hole_boundaries",
        "hole_union_area",
    ):
        if key in case_data:
            payload[key] = case_data[key]

    save_npz_dict(out_path, payload, overwrite=True)

    return {
        "status": "solved",
        "out_path": str(out_path),
        "center_uz": center_uz,
        "max_abs_w": max_abs_w,
        "n_query_points": n_query_points,
        "n_vertices": n_vertices,
        "n_faces": n_faces,
        "fem_solve_time_sec": fem_solve_time_sec,
        "fem_solve_time_ms": 1000.0 * fem_solve_time_sec,
        "fem_total_time_sec": fem_total_time_sec,
        "fem_total_time_ms": 1000.0 * fem_total_time_sec,
    }


def solve_holes2(
    *,
    case_name: str = CASE_NAME,
    case_tags: Optional[List[str]] = None,
    root: Optional[str | Path] = None,
    overwrite: bool = False,
    n_load_steps: int = 1,
    **kwargs,
) -> Dict[str, Any]:
    """
    Batch solve hole cases with the linear RM-DL solver.

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
            info = solve_one_holes2_case(
                case_path=cpath,
                out_path=fpath,
                case_name=case_name,
                overwrite=overwrite,
                n_load_steps=n_load_steps,
            )

            if info["status"] == "solved":
                solved.append(tag)

                if MPI.COMM_WORLD.rank == 0:
                    print(f"[FEM][holes2][RM-DL] solved {tag} -> {fpath}")
                    print(f"[FEM][holes2][RM-DL] center uz for {tag}: {info['center_uz']:.6e}")
                    print(f"[FEM][holes2][RM-DL] max |w| for {tag}: {info['max_abs_w']:.6e}")

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
                    "center_uz": info.get("center_uz", np.nan),
                    "max_abs_w": info.get("max_abs_w", np.nan),
                    "error": "",
                })

            elif info["status"] == "skipped_existing":
                skipped_existing.append(tag)

                if MPI.COMM_WORLD.rank == 0:
                    print(f"[FEM][holes2][RM-DL] skipped existing {tag}")

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
                print(f"[FEM][holes2][RM-DL] failed {tag}: {e}")

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
    info = solve_holes2(
        overwrite=False,
        n_load_steps=1,
    )

    if MPI.COMM_WORLD.rank == 0:
        print(info)
