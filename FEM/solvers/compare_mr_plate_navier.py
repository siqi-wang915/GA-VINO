from __future__ import annotations

"""
Compare triangular Reissner-Mindlin FEM solutions with Navier analytical
solutions for a simply supported square plate.

Default benchmark cases (run sequentially on the same meshes):
    domain:     [0, 5] x [0, 5]
    material:   E = 1000, nu = 0.3
    case 1:     h = 0.5, p(x,y) = 0.8 sin(pi x/a) sin(pi y/b)
    case 2:     h = 0.005, uniform pressure p(x,y) = 8e-7
    mesh:       n x n squares, each split into two right isosceles triangles
    n values:   20, 40, 60, 80, 100, 120

Unknown convention:
    nodal DOFs = [w, theta_x, theta_y]
    gamma = grad(w) - theta

The FEM solve uses the same DOLFINx Durán-Liberman mixed Reissner-Mindlin
formulation used by the project plate solvers:
    theta    : [CG2]^2
    w        : CG1
    R_gamma  : N1curl
    p_lm     : N1curl

The convergence-error post-processing intentionally remains P1-style on the
mesh vertices so the existing CSV fields and convergence plots are preserved:
    bending strain: kappa = [theta_x,x, theta_y,y, theta_x,y + theta_y,x]
    shear strain:   gamma = [w,x - theta_x, w,y - theta_y]
"""

from dataclasses import dataclass
from pathlib import Path
import argparse
import math
import sys
import time
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix


_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[2]

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import dolfinx
import ufl
from basix.ufl import element, mixed_element
from dolfinx.fem import Constant, Function, dirichletbc, functionspace, locate_dofs_topological
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import create_mesh, locate_entities_boundary
from mpi4py import MPI
from ufl import dx, grad, inner, split, sym, tr

from FEM.solvers.common import _eval_on_query_points, _inner_e
from shared.compare.metrics import compute_metrics_from_arrays

@dataclass(frozen=True)
class PlateParams:
    a: float = 5.0
    b: float = 5.0
    thickness: float = 0.5
    E: float = 1000.0
    nu: float = 0.3
    p0: float = 0.8
    load_type: str = "sinusoidal"
    navier_terms: int = 31
    shear_correction: float = 5.0 / 6.0

    @property
    def bending_D(self) -> np.ndarray:
        h = self.thickness
        coef = self.E * h**3 / (12.0 * (1.0 - self.nu**2))
        return coef * np.array(
            [
                [1.0, self.nu, 0.0],
                [self.nu, 1.0, 0.0],
                [0.0, 0.0, 0.5 * (1.0 - self.nu)],
            ],
            dtype=np.float64,
        )

    @property
    def shear_D(self) -> np.ndarray:
        h = self.thickness
        G = self.E / (2.0 * (1.0 + self.nu))
        Ks = self.shear_correction * G * h
        return Ks * np.eye(2, dtype=np.float64)


@dataclass(frozen=True)
class NavierSolution:
    params: PlateParams
    m: np.ndarray
    n: np.ndarray
    W: np.ndarray
    Tx: np.ndarray
    Ty: np.ndarray

    @property
    def alpha(self) -> np.ndarray:
        return self.m * math.pi / self.params.a

    @property
    def beta(self) -> np.ndarray:
        return self.n * math.pi / self.params.b

    def pressure(self, x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray | float:
        if self.params.load_type == "uniform":
            return np.asarray(x) * 0.0 + np.asarray(y) * 0.0 + self.params.p0
        return self.params.p0 * np.sin(math.pi * np.asarray(x) / self.params.a) * np.sin(
            math.pi * np.asarray(y) / self.params.b
        )

    def evaluate(self, x: float, y: float) -> dict[str, np.ndarray | float]:
        alpha = self.alpha
        beta = self.beta

        sx = np.sin(alpha * x)
        cx = np.cos(alpha * x)
        sy = np.sin(beta * y)
        cy = np.cos(beta * y)

        w = np.sum(self.W * sx * sy)
        theta_x = np.sum(self.Tx * cx * sy)
        theta_y = np.sum(self.Ty * sx * cy)

        grad_w = np.array(
            [
                np.sum(alpha * self.W * cx * sy),
                np.sum(beta * self.W * sx * cy),
            ],
            dtype=np.float64,
        )

        kappa = np.array(
            [
                np.sum(-alpha * self.Tx * sx * sy),
                np.sum(-beta * self.Ty * sx * sy),
                np.sum((beta * self.Tx + alpha * self.Ty) * cx * cy),
            ],
            dtype=np.float64,
        )

        gamma = np.array(
            [
                grad_w[0] - theta_x,
                grad_w[1] - theta_y,
            ],
            dtype=np.float64,
        )

        return {
            "w": float(w),
            "theta": np.array([theta_x, theta_y], dtype=np.float64),
            "grad_w": grad_w,
            "kappa": kappa,
            "gamma": gamma,
        }

    def evaluate_nodes(self, nodes: np.ndarray) -> np.ndarray:
        """Vectorized nodal values [w, theta_x, theta_y] for all Navier modes."""
        x = nodes[:, 0:1]
        y = nodes[:, 1:2]
        sx = np.sin(x * self.alpha[None, :])
        cx = np.cos(x * self.alpha[None, :])
        sy = np.sin(y * self.beta[None, :])
        cy = np.cos(y * self.beta[None, :])
        return np.column_stack(
            [
                (sx * sy) @ self.W,
                (cx * sy) @ self.Tx,
                (sx * cy) @ self.Ty,
            ]
        )


def build_navier_solution(params: PlateParams) -> NavierSolution:
    """Build the Navier solution for sinusoidal or uniformly distributed load."""
    if params.load_type == "sinusoidal":
        mode_loads = [(1, 1, params.p0)]
    elif params.load_type == "uniform":
        if params.navier_terms < 1:
            raise ValueError("navier_terms must be at least 1.")
        odd_modes = range(1, params.navier_terms + 1, 2)
        mode_loads = [
            (m, n, 16.0 * params.p0 / (math.pi**2 * m * n))
            for m in odd_modes
            for n in odd_modes
        ]
    else:
        raise ValueError(f"Unsupported load_type: {params.load_type!r}")

    Db = params.bending_D
    Ks = params.shear_D[0, 0]
    coefficients: list[tuple[int, int, float, float, float]] = []

    for m, n, modal_pressure in mode_loads:
        alpha = m * math.pi / params.a
        beta = n * math.pi / params.b
        A = np.zeros((3, 3), dtype=np.float64)
        rhs = np.zeros(3, dtype=np.float64)
        A[0, 0] = Ks * (alpha**2 + beta**2)
        A[0, 1] = -Ks * alpha
        A[0, 2] = -Ks * beta
        rhs[0] = modal_pressure
        A[1, 0] = -Ks * alpha
        A[1, 1] = Ks + Db[0, 0] * alpha**2 + Db[2, 2] * beta**2
        A[1, 2] = (Db[0, 1] + Db[2, 2]) * alpha * beta
        A[2, 0] = -Ks * beta
        A[2, 1] = (Db[0, 1] + Db[2, 2]) * alpha * beta
        A[2, 2] = Ks + Db[1, 1] * beta**2 + Db[2, 2] * alpha**2
        W, Tx, Ty = np.linalg.solve(A, rhs)
        coefficients.append((m, n, W, Tx, Ty))

    values = np.asarray(coefficients, dtype=np.float64)
    return NavierSolution(
        params=params,
        m=values[:, 0],
        n=values[:, 1],
        W=values[:, 2],
        Tx=values[:, 3],
        Ty=values[:, 4],
    )


def make_square_tri_mesh(n: int, a: float, b: float) -> tuple[np.ndarray, np.ndarray]:
    """Create an n x n square grid and split every square by the same diagonal."""
    xs = np.linspace(0.0, a, n + 1)
    ys = np.linspace(0.0, b, n + 1)

    nodes = np.array([(x, y) for y in ys for x in xs], dtype=np.float64)

    def node_id(i: int, j: int) -> int:
        return j * (n + 1) + i

    elems: list[tuple[int, int, int]] = []
    for j in range(n):
        for i in range(n):
            bl = node_id(i, j)
            br = node_id(i + 1, j)
            tl = node_id(i, j + 1)
            tr = node_id(i + 1, j + 1)

            # Diagonal from bottom-left to top-right.
            elems.append((bl, br, tr))
            elems.append((bl, tr, tl))

    return nodes, np.asarray(elems, dtype=np.int64)


def create_dolfinx_mesh_from_arrays(nodes: np.ndarray, elems: np.ndarray) -> dolfinx.mesh.Mesh:
    """Create a DOLFINx triangular mesh from the structured arrays used here."""
    nodes = np.ascontiguousarray(nodes, dtype=np.float64)
    elems = np.ascontiguousarray(elems, dtype=np.int64)

    coordinate_element = element("Lagrange", "triangle", 1, shape=(2,))

    # Different DOLFINx versions use different create_mesh signatures.
    # Newer versions: create_mesh(comm, cells, x, element)
    # Older versions: create_mesh(comm, cells, domain, x)
    try:
        mesh = create_mesh(MPI.COMM_WORLD, elems, nodes, coordinate_element)
    except (TypeError, AttributeError, IndexError):
        domain = ufl.Mesh(coordinate_element)
        mesh = create_mesh(MPI.COMM_WORLD, elems, domain, nodes)

    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim - 1)
    return mesh

def build_dl_problem_components(mesh: dolfinx.mesh.Mesh, params: PlateParams) -> dict[str, object]:
    """
    Build the same Durán-Liberman mixed Reissner-Mindlin formulation used by
    the project FEM solvers, specialized to the Navier square benchmark.
    """
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

    E = Constant(mesh, dolfinx.default_scalar_type(float(params.E)))
    nu = Constant(mesh, dolfinx.default_scalar_type(float(params.nu)))
    t = Constant(mesh, dolfinx.default_scalar_type(float(params.thickness)))
    shear_correction = Constant(mesh, dolfinx.default_scalar_type(float(params.shear_correction)))

    x = ufl.SpatialCoordinate(mesh)
    if params.load_type == "uniform":
        p_load = float(params.p0)
    elif params.load_type == "sinusoidal":
        p_load = (
            float(params.p0)
            * ufl.sin(math.pi * x[0] / float(params.a))
            * ufl.sin(math.pi * x[1] / float(params.b))
        )
    else:
        raise ValueError(f"Unsupported load_type: {params.load_type!r}")

    k = sym(grad(theta))
    D_half = (E * t**3) / (24.0 * (1.0 - nu**2))
    psi_b = D_half * ((1.0 - nu) * tr(k * k) + nu * (tr(k)) ** 2)
    psi_s = ((E * shear_correction * t) / (4.0 * (1.0 + nu))) * inner(R_gamma, R_gamma)

    gamma = grad(w) - theta
    Pi_R = _inner_e(mesh, gamma - R_gamma, p_lm)
    W_ext = p_load * w * dx
    Pi = psi_b * dx + psi_s * dx + Pi_R - W_ext

    F_res = ufl.derivative(Pi, u0, u_test)
    J = ufl.derivative(F_res, u0, u_trial)

    return {
        "U": U,
        "F_res": F_res,
        "J": J,
    }


def build_simply_supported_bcs(mesh: dolfinx.mesh.Mesh, U, params: PlateParams):
    """
    Navier simply-supported square boundary conditions:
        all boundaries: w = 0
        x = 0,a     : theta_y = 0
        y = 0,b     : theta_x = 0
    """
    fdim = mesh.topology.dim - 1
    tol = 1.0e-10

    def all_boundary(x):
        return (
            np.isclose(x[0], 0.0, atol=tol)
            | np.isclose(x[0], float(params.a), atol=tol)
            | np.isclose(x[1], 0.0, atol=tol)
            | np.isclose(x[1], float(params.b), atol=tol)
        )

    def x_boundary(x):
        return np.isclose(x[0], 0.0, atol=tol) | np.isclose(x[0], float(params.a), atol=tol)

    def y_boundary(x):
        return np.isclose(x[1], 0.0, atol=tol) | np.isclose(x[1], float(params.b), atol=tol)

    facets_all = locate_entities_boundary(mesh, fdim, all_boundary)
    facets_x = locate_entities_boundary(mesh, fdim, x_boundary)
    facets_y = locate_entities_boundary(mesh, fdim, y_boundary)

    bcs = []

    W_sub = U.sub(1)
    W0, _ = W_sub.collapse()
    zero_w = Function(W0)
    dofs_w = locate_dofs_topological((W_sub, W0), fdim, facets_all)
    bcs.append(dirichletbc(zero_w, dofs_w, W_sub))

    theta_x_sub = U.sub(0).sub(0)
    Tx0, _ = theta_x_sub.collapse()
    zero_tx = Function(Tx0)
    dofs_tx = locate_dofs_topological((theta_x_sub, Tx0), fdim, facets_y)
    bcs.append(dirichletbc(zero_tx, dofs_tx, theta_x_sub))

    theta_y_sub = U.sub(0).sub(1)
    Ty0, _ = theta_y_sub.collapse()
    zero_ty = Function(Ty0)
    dofs_ty = locate_dofs_topological((theta_y_sub, Ty0), fdim, facets_x)
    bcs.append(dirichletbc(zero_ty, dofs_ty, theta_y_sub))

    return bcs


def solve_fem_dl_on_nodes(
    nodes: np.ndarray,
    elems: np.ndarray,
    params: PlateParams,
) -> np.ndarray:
    """Solve the Navier benchmark with the project DL formulation and return [w, tx, ty] at mesh nodes."""
    mesh = create_dolfinx_mesh_from_arrays(nodes, elems)
    comps = build_dl_problem_components(mesh, params)
    U = comps["U"]

    problem = LinearProblem(
        comps["J"],
        -comps["F_res"],
        bcs=build_simply_supported_bcs(mesh, U, params),
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
        petsc_options_prefix="navier_rm_dl_",
    )

    uh = problem.solve()
    uh.x.scatter_forward()

    theta_sol, w_sol, _, _ = uh.split()

    query_xyz = np.zeros((nodes.shape[0], 3), dtype=np.float64)
    query_xyz[:, :2] = nodes
    w_vals, theta_vals = _eval_on_query_points(
        mesh=mesh,
        theta_func=theta_sol,
        w_func=w_sol,
        query_xyz=query_xyz,
    )

    return np.column_stack([w_vals, theta_vals[:, 0], theta_vals[:, 1]]).astype(np.float64)


def triangle_geometry(coords: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    x1, y1 = coords[0]
    x2, y2 = coords[1]
    x3, y3 = coords[2]

    signed_double_area = (
        (x2 - x1) * (y3 - y1)
        - (x3 - x1) * (y2 - y1)
    )
    area = 0.5 * signed_double_area
    if area <= 0.0:
        raise ValueError("Triangle node ordering must be counter-clockwise.")

    dNdx = np.array([y2 - y3, y3 - y1, y1 - y2], dtype=np.float64) / (2.0 * area)
    dNdy = np.array([x3 - x2, x1 - x3, x2 - x1], dtype=np.float64) / (2.0 * area)
    return area, dNdx, dNdy


def bending_B(dNdx: np.ndarray, dNdy: np.ndarray) -> np.ndarray:
    Bb = np.zeros((3, 9), dtype=np.float64)
    for i in range(3):
        c = 3 * i
        # theta_x column
        Bb[0, c + 1] = dNdx[i]
        Bb[2, c + 1] = dNdy[i]
        # theta_y column
        Bb[1, c + 2] = dNdy[i]
        Bb[2, c + 2] = dNdx[i]
    return Bb


def shear_B(dNdx: np.ndarray, dNdy: np.ndarray, N: np.ndarray) -> np.ndarray:
    Bs = np.zeros((2, 9), dtype=np.float64)
    for i in range(3):
        c = 3 * i
        # w column
        Bs[0, c] = dNdx[i]
        Bs[1, c] = dNdy[i]
        # theta_x and theta_y columns, gamma = grad(w) - theta
        Bs[0, c + 1] = -N[i]
        Bs[1, c + 2] = -N[i]
    return Bs


def triangle_quadrature_7() -> tuple[np.ndarray, np.ndarray]:
    """Seven-point Dunavant rule; barycentric weights sum to one."""
    q = [
        (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.225000000000000),
        (0.059715871789770, 0.470142064105115, 0.470142064105115, 0.132394152788506),
        (0.470142064105115, 0.059715871789770, 0.470142064105115, 0.132394152788506),
        (0.470142064105115, 0.470142064105115, 0.059715871789770, 0.132394152788506),
        (0.797426985353087, 0.101286507323456, 0.101286507323456, 0.125939180544827),
        (0.101286507323456, 0.797426985353087, 0.101286507323456, 0.125939180544827),
        (0.101286507323456, 0.101286507323456, 0.797426985353087, 0.125939180544827),
    ]
    bary = np.array([row[:3] for row in q], dtype=np.float64)
    weights = np.array([row[3] for row in q], dtype=np.float64)
    return bary, weights


def local_load_vector(coords: np.ndarray, area: float, navier: NavierSolution) -> np.ndarray:
    fe = np.zeros(9, dtype=np.float64)
    bary, weights = triangle_quadrature_7()
    for N, wq in zip(bary, weights):
        x, y = N @ coords
        p = navier.pressure(x, y)
        for i in range(3):
            fe[3 * i] += area * wq * N[i] * p
    return fe


def assemble_system(
    nodes: np.ndarray,
    elems: np.ndarray,
    params: PlateParams,
    navier: NavierSolution,
) -> tuple[coo_matrix, np.ndarray]:
    ndofs = 3 * nodes.shape[0]
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    rhs = np.zeros(ndofs, dtype=np.float64)

    Db = params.bending_D
    Ds = params.shear_D
    N_centroid = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float64)

    for elem in elems:
        coords = nodes[elem]
        area, dNdx, dNdy = triangle_geometry(coords)

        Bb = bending_B(dNdx, dNdy)
        Bs = shear_B(dNdx, dNdy, N_centroid)

        ke = area * (Bb.T @ Db @ Bb + Bs.T @ Ds @ Bs)
        fe = local_load_vector(coords, area, navier)

        dofs = np.empty(9, dtype=np.int64)
        for local_i, node in enumerate(elem):
            dofs[3 * local_i : 3 * local_i + 3] = [3 * node, 3 * node + 1, 3 * node + 2]

        rhs[dofs] += fe
        rr, cc = np.meshgrid(dofs, dofs, indexing="ij")
        rows.extend(rr.ravel().tolist())
        cols.extend(cc.ravel().tolist())
        data.extend(ke.ravel().tolist())

    K = coo_matrix((data, (rows, cols)), shape=(ndofs, ndofs))
    return K, rhs


def simply_supported_w_dofs(nodes: np.ndarray, params: PlateParams, tol: float = 1.0e-12) -> np.ndarray:
    x = nodes[:, 0]
    y = nodes[:, 1]
    
    # 查找各边界的节点 ID
    on_x0 = np.isclose(x, 0.0, atol=tol)
    on_xa = np.isclose(x, params.a, atol=tol)
    on_y0 = np.isclose(y, 0.0, atol=tol)
    on_yb = np.isclose(y, params.b, atol=tol)
    
    fixed_dofs = []

    # 1. 所有边界上的 w = 0 (位移)
    on_boundary = on_x0 | on_xa | on_y0 | on_yb
    fixed_dofs.extend(3 * np.nonzero(on_boundary)[0])

    # 2. x = constant 边界上，切向旋转 theta_y = 0
    fixed_dofs.extend(3 * np.nonzero(on_x0 | on_xa)[0] + 2)

    # 3. y = constant 边界上，切向旋转 theta_x = 0
    fixed_dofs.extend(3 * np.nonzero(on_y0 | on_yb)[0] + 1)

    return np.unique(fixed_dofs)


def solve_fem(nodes: np.ndarray, elems: np.ndarray, params: PlateParams, navier: NavierSolution) -> np.ndarray:
    # Use the same DOLFINx Durán-Liberman mixed formulation as the project
    # FEM solvers. The flat [w, theta_x, theta_y] vector is kept only so the
    # existing convergence/error/plotting code can remain unchanged.
    q_local = solve_fem_dl_on_nodes(nodes, elems, params)
    u = np.empty(3 * nodes.shape[0], dtype=np.float64)
    u[0::3] = q_local[:, 0]
    u[1::3] = q_local[:, 1]
    u[2::3] = q_local[:, 2]
    return u


def evaluate_fem_on_element(
    ue: np.ndarray,
    dNdx: np.ndarray,
    dNdy: np.ndarray,
    N: np.ndarray,
) -> dict[str, np.ndarray | float]:
    w_nodes = ue[0::3]
    tx_nodes = ue[1::3]
    ty_nodes = ue[2::3]

    w = float(N @ w_nodes)
    theta = np.array([N @ tx_nodes, N @ ty_nodes], dtype=np.float64)
    grad_w = np.array([dNdx @ w_nodes, dNdy @ w_nodes], dtype=np.float64)

    kappa = np.array(
        [
            dNdx @ tx_nodes,
            dNdy @ ty_nodes,
            dNdy @ tx_nodes + dNdx @ ty_nodes,
        ],
        dtype=np.float64,
    )
    gamma = grad_w - theta

    return {
        "w": w,
        "theta": theta,
        "grad_w": grad_w,
        "kappa": kappa,
        "gamma": gamma,
    }


def element_mass_matrix(area: float) -> np.ndarray:
    """线性三角形单元的 3x3 质量矩阵矩阵 (基于 P1 形状函数)."""
    # 积分 (Ni * Nj) dA 的结果
    base = np.array([
        [2.0, 1.0, 1.0],
        [1.0, 2.0, 1.0],
        [1.0, 1.0, 2.0]
    ], dtype=np.float64)
    return (area / 12.0) * base

def build_exact_q_on_nodes(nodes: np.ndarray, navier: NavierSolution) -> np.ndarray:
    return navier.evaluate_nodes(nodes)

def compute_relative_errors(
    nodes: np.ndarray,
    elems: np.ndarray,
    u: np.ndarray,
    params: PlateParams,
    navier: NavierSolution,
) -> dict[str, float]:
    q_fem = np.zeros((nodes.shape[0], 3), dtype=np.float64)
    q_fem[:, 0] = u[0::3]
    q_fem[:, 1] = u[1::3]
    q_fem[:, 2] = u[2::3]

    q_exact = build_exact_q_on_nodes(nodes, navier)

    metrics = compute_metrics_from_arrays(
        verts_xy=nodes,
        faces=elems,
        q_no=q_fem,
        q_fem=q_exact,
        E=float(params.E),
        nu=float(params.nu),
        thickness=float(params.thickness),
    )

    w_err = q_fem[:, 0] - q_exact[:, 0]

    return {
        "L2_w": float(metrics["l2_error"]),
        "H1_w": float(metrics["h1_semi_error"]),
        "energy_error": float(metrics["energy_error"]),
        "theta_l2_error": float(metrics["theta_l2_error"]),
        "curvature_error": float(metrics["curvature_error"]),
        "abs_L2_w": float("nan"),
        "abs_H1_w": float("nan"),
        "abs_energy_error": float("nan"),
        "max_abs_w_error": float(np.max(np.abs(w_err))),
    }

def convergence_rates(errors: Iterable[float]) -> list[float]:
    values = list(errors)
    rates = [float("nan")]
    for prev, curr in zip(values[:-1], values[1:]):
        if prev > 0.0 and curr > 0.0:
            rates.append(math.log(prev / curr) / math.log(2.0))
        else:
            rates.append(float("nan"))
    return rates


def run_study(n_values: list[int], params: PlateParams, out_dir: Path) -> pd.DataFrame:
    navier = build_navier_solution(params)

    rows: list[dict[str, float | int]] = []
    for n in n_values:
        t0 = time.perf_counter()
        nodes, elems = make_square_tri_mesh(n, params.a, params.b)
        u = solve_fem(nodes, elems, params, navier)
        errors = compute_relative_errors(nodes, elems, u, params, navier)
        elapsed = time.perf_counter() - t0

        # 为每个 n 生成四联图
        out_comp = out_dir / f"compare_fields_n{n:03d}.png"
        save_comparison_plots(nodes, elems, u, navier, params, out_comp)

        row: dict[str, float | int] = {
            "thickness": params.thickness,
            "load_type": params.load_type,
            "p0": params.p0,
            "navier_mode_count": int(navier.W.size),
            "n": n,
            "h_mesh": params.a / n,
            "num_nodes": int(nodes.shape[0]),
            "num_elements": int(elems.shape[0]),
            "num_dofs": int(3 * nodes.shape[0]),
            "solve_total_time_sec": elapsed,
            **errors,
        }
        rows.append(row)

        print(
            f"n={n:4d} | nodes={nodes.shape[0]:7d} | elems={elems.shape[0]:7d} | "
            f"L2_w={errors['L2_w']:.6e} | H1_w={errors['H1_w']:.6e} | "
            f"energy={errors['energy_error']:.6e} | time={elapsed:.2f}s"
            f"theta={errors['theta_l2_error']:.6e} | "
            f"curvature={errors['curvature_error']:.6e} | "
            f"time={elapsed:.2f}s"
        )

    df = pd.DataFrame(rows)
    df["rate_L2_w"] = convergence_rates(df["L2_w"].to_list())
    df["rate_H1_w"] = convergence_rates(df["H1_w"].to_list())
    df["rate_energy"] = convergence_rates(df["energy_error"].to_list())
    df["rate_theta_l2"] = convergence_rates(df["theta_l2_error"].to_list())
    df["rate_curvature"] = convergence_rates(df["curvature_error"].to_list())
    return df


def save_plot(df: pd.DataFrame, out_png: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for plotting. Install it with "
            "`pip install matplotlib`, or keep the Excel/CSV results and plot them elsewhere."
        ) from exc

    fig, ax = plt.subplots(figsize=(5.8, 4.8), dpi=180)

    x_pos = df["num_elements"].to_numpy(dtype=float)
    x_labels = [str(int(n)) for n in df["num_elements"].to_numpy()]

    percent = 100.0
    ax.plot(x_pos, percent * df["L2_w"], marker="o", linewidth=2.0, label=r"$w$ $L^2$ norm")
    ax.plot(
        x_pos,
        percent * df["theta_l2_error"],
        marker="D",
        linewidth=2.0,
        label=r"$\theta$ $L^2$ norm",
    )
    ax.plot(x_pos, percent * df["H1_w"], marker="s", linewidth=2.0, label=r"$w$ $H^1$ seminorm")
    ax.plot(x_pos, percent * df["energy_error"], marker="^", linewidth=2.0, label="Energy norm")
    #ax.plot(x_pos, df["curvature_error"], marker="v", linewidth=2.0, label=r"$\kappa$ $L^2$ norm")

    ax.set_xlabel("Number of triangular elements")
    ax.set_ylabel("Error (%)")
    ax.set_yscale("log")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def save_comparison_plots(
    nodes: np.ndarray,
    elems: np.ndarray,
    u: np.ndarray,
    navier: NavierSolution,
    params: PlateParams,
    out_png: Path,
) -> None:
    """生成四联图：3D变形、FEM云图、解析解云图、误差云图。"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    w_fem = u[0::3]
    w_exact = navier.evaluate_nodes(nodes)[:, 0]
    w_err = w_fem - w_exact

    vmin = min(w_fem.min(), w_exact.min())
    vmax = max(w_fem.max(), w_exact.max())
    emax = max(abs(w_err.min()), abs(w_err.max()), 1e-15)

    # 减小高度比例以便布局更紧凑
    fig = plt.figure(figsize=(20, 4.8), dpi=150)
    x, y = nodes[:, 0], nodes[:, 1]
    
    # 子图通用设置
    def setup_2d_ax(ax, title):
        ax.set_title(title, fontsize=12)
        ax.set_xlim(0, params.a)
        ax.set_ylim(0, params.b)
        ax.set_aspect("equal")

    # 1. 3D 位位移图
    ax1 = fig.add_subplot(1, 4, 1, projection="3d")
    ax1.plot_trisurf(x, y, w_fem, triangles=elems, cmap="coolwarm", 
                     linewidth=0, antialiased=True, shade=False, alpha=0.9)
    ax1.set_xlim(0, params.a)
    ax1.set_ylim(0, params.b)
    ax1.set_title("3D FEM Shape", fontsize=12)
    ax1.view_init(elev=22, azim=-55)

    # 2. 2D FEM 云图
    ax2 = fig.add_subplot(1, 4, 2)
    m2 = ax2.tripcolor(x, y, elems, w_fem, shading="flat", cmap="coolwarm", vmin=vmin, vmax=vmax)
    setup_2d_ax(ax2, "FEM Solution")
    fig.colorbar(m2, ax=ax2, fraction=0.046, pad=0.04)

    # 3. 2D 解析解云图
    ax3 = fig.add_subplot(1, 4, 3)
    m3 = ax3.tripcolor(x, y, elems, w_exact, shading="flat", cmap="coolwarm", vmin=vmin, vmax=vmax)
    setup_2d_ax(ax3, "Navier Exact")
    fig.colorbar(m3, ax=ax3, fraction=0.046, pad=0.04)

    # 4. 2D 误差云图
    ax4 = fig.add_subplot(1, 4, 4)
    m4 = ax4.tripcolor(x, y, elems, w_err, shading="flat", cmap="coolwarm", vmin=-emax, vmax=emax)
    setup_2d_ax(ax4, "Error (Abs)")
    fig.colorbar(m4, ax=ax4, fraction=0.046, pad=0.04)

    n = int(math.sqrt(elems.shape[0] / 2))
    fig.suptitle(f"Plate Deflection Comparison (n={n})", y=0.98, fontsize=14)
    
    fig.tight_layout(pad=1.2)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def parse_n_values(text: str) -> list[int]:
    values = [int(v.strip()) for v in text.split(",") if v.strip()]
    if not values:
        raise ValueError("At least one mesh density must be provided.")
    return values


def thickness_folder_name(thickness: float) -> str:
    return f"thickness_{thickness:.12g}"


def save_study_outputs(df: pd.DataFrame, out_dir: Path) -> None:
    df.to_excel(out_dir / "fem_vs_navier_errors.xlsx", index=False)
    df.to_csv(out_dir / "fem_vs_navier_errors.csv", index=False)
    save_plot(df, out_dir / "fem_vs_navier_error_curves.png")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Two-case FEM/Navier mesh-convergence study for a simply supported Mindlin-Reissner plate."
    )
    parser.add_argument("--n-values", type=str, default="20,40,60,80,100,120")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "navier_mr_plate_results")
    parser.add_argument("--a", type=float, default=5.0)
    parser.add_argument("--b", type=float, default=5.0)
    parser.add_argument("--thickness", type=float, default=0.5)
    parser.add_argument("--E", type=float, default=1000.0)
    parser.add_argument("--nu", type=float, default=0.3)
    parser.add_argument("--p0", type=float, default=0.8)
    parser.add_argument("--thin-thickness", type=float, default=0.005)
    parser.add_argument("--thin-p0", type=float, default=8.0e-7)
    parser.add_argument(
        "--uniform-navier-terms",
        type=int,
        default=31,
        help="Largest odd mode index in each direction for the uniform-load Navier series.",
    )
    args = parser.parse_args()

    n_values = parse_n_values(args.n_values)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cases = [
        PlateParams(
            a=args.a,
            b=args.b,
            thickness=args.thickness,
            E=args.E,
            nu=args.nu,
            p0=args.p0,
            load_type="sinusoidal",
        ),
        PlateParams(
            a=args.a,
            b=args.b,
            thickness=args.thin_thickness,
            E=args.E,
            nu=args.nu,
            p0=args.thin_p0,
            load_type="uniform",
            navier_terms=args.uniform_navier_terms,
        ),
    ]

    for case_index, params in enumerate(cases, start=1):
        case_out_dir = args.out_dir / thickness_folder_name(params.thickness)
        case_out_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"\n=== Case {case_index}/{len(cases)}: thickness={params.thickness:g}, "
            f"load={params.load_type}, p0={params.p0:g} ==="
        )
        print(f"Output directory: {case_out_dir}")
        df = run_study(n_values, params, case_out_dir)
        save_study_outputs(df, case_out_dir)

        navier = build_navier_solution(params)
        print(f"Navier modes: {navier.W.size}")
        print(
            f"First-mode coefficients: W={navier.W[0]:.12e}, "
            f"Tx={navier.Tx[0]:.12e}, Ty={navier.Ty[0]:.12e}"
        )

    print("\nAll studies completed. Saved under:")
    for params in cases:
        print(f"  {args.out_dir / thickness_folder_name(params.thickness)}")


if __name__ == "__main__":
    main()
