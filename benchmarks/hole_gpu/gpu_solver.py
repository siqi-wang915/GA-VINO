from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

# Keep direct calls to this module consistent with run.py.  PETSc reads the
# GPU-aware-MPI setting during initialization, so configure it before import.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
if "-use_gpu_aware_mpi" not in os.environ.get("PETSC_OPTIONS", ""):
    existing_options = os.environ.get("PETSC_OPTIONS", "").strip()
    os.environ["PETSC_OPTIONS"] = (
        f"{existing_options} -use_gpu_aware_mpi 0".strip()
    )

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx.fem.petsc import LinearProblem

from FEM.solvers.common import (
    _build_boundary_conditions,
    _build_problem_components,
    _create_mesh_from_case,
    _eval_on_query_points,
    _find_center_value,
)
from shared.common.data_io import load_npz_dict, save_npz_dict


SOLVER_NAME_SUFFIX = "rm_dl_linear_plate_fem_gpu_strumpack_cuda"

SQUARE_HOLE_CASES = (
    "hole_um_up",
    "hole_um_grfp",
    "holes2_um_up",
    "holes2_um_grfp",
)
VARIABLE_MATERIAL_CASES = (
    "lshape_grfm_up",
    "lshape_grfm_grfp",
    "lshape_grfm_up_065_085",
    "annular_sector_grfm_grfp",
)
SUPPORTED_GPU_CASES = SQUARE_HOLE_CASES + VARIABLE_MATERIAL_CASES


def require_petsc_strumpack() -> None:
    """Fail early unless PETSc provides CUDA-enabled STRUMPACK support."""
    if not PETSc.Sys.hasExternalPackage("cuda"):
        raise RuntimeError(
            "This PETSc build cannot run the GPU benchmark: CUDA support is "
            "missing. The benchmark requires a PETSc "
            "build configured with CUDA/cuSPARSE; installing a CUDA toolkit or "
            "detecting an NVIDIA GPU is not sufficient. Verify with: "
            "python -c \"from petsc4py import PETSc; "
            "print(PETSc.Sys.hasExternalPackage('cuda'))\""
        )
    if not PETSc.Sys.hasExternalPackage("strumpack"):
        raise RuntimeError(
            "This mixed FEM system is an indefinite saddle-point problem and "
            "requires PETSc built with CUDA-enabled STRUMPACK for a robust "
            "GPU-accelerated direct solve. Rebuild PETSc with +strumpack and "
            "STRUMPACK with +cuda, then verify with: python -c "
            "\"from petsc4py import PETSc; "
            "print(PETSc.Sys.hasExternalPackage('strumpack'))\""
        )


def _build_case_components_and_bcs(
    *, mesh, case_data: dict[str, Any], case_name: str
):
    if case_name in SQUARE_HOLE_CASES:
        comps = _build_problem_components(mesh, case_data)
        bcs = _build_boundary_conditions(mesh, comps["U"], case_data)
        return comps, bcs

    if case_name in VARIABLE_MATERIAL_CASES:
        # The annular-sector CPU solver intentionally reuses the L-shape
        # formulation: both clamp the x=0 edge and support DG0 material fields.
        from FEM.solvers.lshape import (
            _build_left_clamped_boundary_conditions,
            _build_lshape_problem_components,
        )

        comps = _build_lshape_problem_components(mesh, case_data)
        bcs = _build_left_clamped_boundary_conditions(
            mesh, comps["U"], case_data
        )
        return comps, bcs

    supported = ", ".join(SUPPORTED_GPU_CASES)
    raise ValueError(
        f"Unsupported GPU FEM case_name={case_name!r}. Supported: {supported}"
    )


def solve_one_gpu_case(
    *,
    case_path: str | Path,
    out_path: str | Path,
    case_name: str = "hole_um_up",
) -> dict[str, Any]:
    """Solve one case with CPU assembly and a CUDA-enabled STRUMPACK solve."""
    if MPI.COMM_WORLD.size != 1:
        raise RuntimeError("The GPU solver requires exactly one MPI rank.")
    require_petsc_strumpack()

    case_path = Path(case_path)
    out_path = Path(out_path)
    total_start = time.perf_counter()

    case_data = load_npz_dict(case_path)
    mesh = _create_mesh_from_case(case_data)
    comps, bcs = _build_case_components_and_bcs(
        mesh=mesh, case_data=case_data, case_name=case_name
    )
    solver_name = f"{case_name}_{SOLVER_NAME_SUFFIX}"

    problem = LinearProblem(
        comps["J"],
        -comps["F_res"],
        bcs=bcs,
        petsc_options={
            # STRUMPACK accepts the assembled host AIJ matrix and offloads its
            # computationally intensive sparse factorization to CUDA.
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "strumpack",
            "ksp_error_if_not_converged": True,
        },
        petsc_options_prefix=f"{case_name}_rm_dl_gpu_",
    )

    # STRUMPACK imports PETSc's host CSR matrix and manages CUDA transfers
    # internally. Converting A to AIJCUSPARSE would prevent PETSc from finding
    # the STRUMPACK factorization registered for AIJ matrices.
    problem.solver.setOperators(problem.A, problem.A)

    matrix_type = problem.A.getType()
    rhs_type = problem.b.getType()
    solution_type = problem.x.getType()
    print(f"[FEM][GPU] PETSc matrix type: {matrix_type}")
    print(f"[FEM][GPU] PETSc RHS vector type: {rhs_type}")
    print(f"[FEM][GPU] PETSc solution vector type: {solution_type}")

    mesh.comm.Barrier()
    solve_start = time.perf_counter()
    uh = problem.solve()
    uh.x.scatter_forward()
    mesh.comm.Barrier()
    assembly_and_solve_time = time.perf_counter() - solve_start

    ksp = problem.solver
    reason = int(ksp.getConvergedReason())
    iterations = int(ksp.getIterationNumber())
    residual = float(ksp.getResidualNorm())
    if reason <= 0:
        raise RuntimeError(
            "PETSc linear solver did not converge: "
            f"reason={reason}, iterations={iterations}, residual={residual:.6e}"
        )

    theta_sol, w_sol, _, _ = uh.split()
    query_xyz = np.asarray(case_data["query_xyz"], dtype=np.float64)
    w_vals, theta_vals = _eval_on_query_points(
        mesh=mesh,
        theta_func=theta_sol,
        w_func=w_sol,
        query_xyz=query_xyz,
    )
    fem_local = np.column_stack((w_vals, theta_vals[:, 0], theta_vals[:, 1]))
    fem_global_disp = np.zeros((query_xyz.shape[0], 3), dtype=np.float64)
    fem_global_disp[:, 2] = w_vals

    lx = float(np.asarray(case_data["lx"]).reshape(-1)[0])
    ly = float(np.asarray(case_data["ly"]).reshape(-1)[0])
    center_uz = float("nan")
    if case_name in SQUARE_HOLE_CASES:
        center_uz = _find_center_value(query_xyz, w_vals, lx, ly)
    max_abs_w = float(np.max(np.abs(w_vals)))
    n_vertices = int(np.asarray(case_data["verts_xy"]).shape[0])
    n_faces = int(np.asarray(case_data["faces"]).shape[0])
    total_time = time.perf_counter() - total_start

    payload: dict[str, Any] = {
        "case_tag": case_data.get("case_tag", np.asarray(case_path.stem)),
        "source_sample_id": case_data.get("source_sample_id", np.asarray(-1)),
        "case_name": np.asarray(case_name),
        "solver_name": np.asarray(solver_name),
        "compute_device": np.asarray("GPU"),
        "linear_solver": np.asarray("LU+STRUMPACK (CUDA build)"),
        "query_xyz": query_xyz,
        "verts_xy": np.asarray(case_data["verts_xy"], dtype=np.float64),
        "faces": np.asarray(case_data["faces"], dtype=np.int64),
        "fem_local": fem_local,
        "fem_global_disp": fem_global_disp,
        "disp_scalar": w_vals,
        "rot_scalar": np.abs(theta_vals[:, 0]) + np.abs(theta_vals[:, 1]),
        "petsc_matrix_type": np.asarray(matrix_type),
        "petsc_rhs_vector_type": np.asarray(rhs_type),
        "petsc_solution_vector_type": np.asarray(solution_type),
        "ksp_converged_reason": np.asarray(reason, dtype=np.int64),
        "ksp_iterations": np.asarray(iterations, dtype=np.int64),
        "ksp_residual_norm": np.asarray(residual, dtype=np.float64),
        "max_abs_w": np.asarray(max_abs_w, dtype=np.float64),
        "n_query_points": np.asarray(query_xyz.shape[0], dtype=np.int64),
        "n_vertices": np.asarray(n_vertices, dtype=np.int64),
        "n_faces": np.asarray(n_faces, dtype=np.int64),
        "fem_solve_time_sec": np.asarray(assembly_and_solve_time),
        "fem_total_time_sec": np.asarray(total_time),
    }
    if case_name in SQUARE_HOLE_CASES:
        payload["center_uz"] = np.asarray(center_uz, dtype=np.float64)
    # Preserve every case-specific geometry, material and load field. Existing
    # result keys above take precedence over same-named source keys.
    for key, value in case_data.items():
        payload.setdefault(key, value)
    save_npz_dict(out_path, payload, overwrite=True)

    print(
        f"[FEM][GPU] iterations={iterations}, residual={residual:.6e}, "
        f"assembly+solve={1000.0 * assembly_and_solve_time:.3f} ms"
    )
    return {
        "status": "solved",
        "out_path": str(out_path),
        "solver_name": solver_name,
        "compute_device": "GPU",
        "ksp_converged_reason": reason,
        "ksp_iterations": iterations,
        "ksp_residual_norm": residual,
        "center_uz": center_uz,
        "max_abs_w": max_abs_w,
        "n_query_points": int(query_xyz.shape[0]),
        "n_vertices": n_vertices,
        "n_faces": n_faces,
        "fem_solve_time_sec": assembly_and_solve_time,
        "fem_total_time_sec": total_time,
    }


# Backward-compatible name for callers outside this benchmark directory.
solve_one_hole_gpu_case = solve_one_gpu_case
