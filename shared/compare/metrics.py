from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import csv
import shutil
import tempfile

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize


# -----------------------------------------------------------------------------
# IO and figure save
# -----------------------------------------------------------------------------

def load_npz_dict(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def safe_savefig(fig, out_path: str | Path, **kwargs):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        fig.savefig(str(out_path), **kwargs)
        return
    except OSError:
        pass

    with tempfile.NamedTemporaryFile(suffix=out_path.suffix, delete=False, dir="/tmp") as tmp:
        tmp_path = Path(tmp.name)

    try:
        fig.savefig(str(tmp_path), **kwargs)
        shutil.copy2(str(tmp_path), str(out_path))
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


def to_float(value, default: float = 0.0) -> float:
    try:
        arr = np.asarray(value)
        if arr.shape == ():
            return float(arr.item())
        return float(arr.reshape(-1)[0])
    except Exception:
        return float(default)


def to_str(value, default: str = "") -> str:
    try:
        arr = np.asarray(value)
        if arr.shape == ():
            return str(arr.item())
        if arr.size > 0:
            return str(arr.reshape(-1)[0])
    except Exception:
        pass
    return default


# -----------------------------------------------------------------------------
# Triangle integration
# -----------------------------------------------------------------------------

def triangle_area_and_grads(
    verts_xy: np.ndarray,
    faces: np.ndarray,
):
    verts_xy = np.asarray(verts_xy, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)

    tri = verts_xy[faces]

    x1 = tri[:, 0, 0]
    y1 = tri[:, 0, 1]
    x2 = tri[:, 1, 0]
    y2 = tri[:, 1, 1]
    x3 = tri[:, 2, 0]
    y3 = tri[:, 2, 1]

    two_area_signed = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    area = 0.5 * np.abs(two_area_signed)

    denom = np.where(np.abs(two_area_signed) > 1.0e-14, two_area_signed, 1.0e-14)

    dNdx = np.stack(
        [
            (y2 - y3) / denom,
            (y3 - y1) / denom,
            (y1 - y2) / denom,
        ],
        axis=1,
    )

    dNdy = np.stack(
        [
            (x3 - x2) / denom,
            (x1 - x3) / denom,
            (x2 - x1) / denom,
        ],
        axis=1,
    )

    return area, dNdx, dNdy


def triangle_mass_matrices(area: np.ndarray) -> np.ndarray:
    base = np.asarray(
        [
            [2.0, 1.0, 1.0],
            [1.0, 2.0, 1.0],
            [1.0, 1.0, 2.0],
        ],
        dtype=np.float64,
    )
    return np.asarray(area, dtype=np.float64)[:, None, None] / 12.0 * base[None, :, :]


# -----------------------------------------------------------------------------
# Error metrics
# -----------------------------------------------------------------------------

def relative_l2_error_w(
    verts_xy: np.ndarray,
    faces: np.ndarray,
    w_no: np.ndarray,
    w_fem: np.ndarray,
) -> float:
    area, _, _ = triangle_area_and_grads(verts_xy, faces)
    mass_N = triangle_mass_matrices(area)

    w_no = np.asarray(w_no, dtype=np.float64).reshape(-1)
    w_fem = np.asarray(w_fem, dtype=np.float64).reshape(-1)
    err = w_no - w_fem

    err_nodes = err[faces]
    ref_nodes = w_fem[faces]

    num = np.einsum("eij,ei,ej->", mass_N, err_nodes, err_nodes)
    den = np.einsum("eij,ei,ej->", mass_N, ref_nodes, ref_nodes)

    return float(np.sqrt(num / max(den, 1.0e-30)))


def relative_h1_semi_error_w(
    verts_xy: np.ndarray,
    faces: np.ndarray,
    w_no: np.ndarray,
    w_fem: np.ndarray,
) -> float:
    area, dNdx, dNdy = triangle_area_and_grads(verts_xy, faces)

    w_no = np.asarray(w_no, dtype=np.float64).reshape(-1)
    w_fem = np.asarray(w_fem, dtype=np.float64).reshape(-1)

    err = w_no - w_fem

    err_nodes = err[faces]
    ref_nodes = w_fem[faces]

    derr_dx = np.einsum("ei,ei->e", dNdx, err_nodes)
    derr_dy = np.einsum("ei,ei->e", dNdy, err_nodes)

    dref_dx = np.einsum("ei,ei->e", dNdx, ref_nodes)
    dref_dy = np.einsum("ei,ei->e", dNdy, ref_nodes)

    num = np.sum(area * (derr_dx * derr_dx + derr_dy * derr_dy))
    den = np.sum(area * (dref_dx * dref_dx + dref_dy * dref_dy))

    return float(np.sqrt(num / max(den, 1.0e-30)))

def relative_l2_error_theta(
    verts_xy: np.ndarray,
    faces: np.ndarray,
    q_no: np.ndarray,
    q_fem: np.ndarray,
) -> float:
    area, _, _ = triangle_area_and_grads(verts_xy, faces)
    mass_N = triangle_mass_matrices(area)

    q_no = np.asarray(q_no, dtype=np.float64)
    q_fem = np.asarray(q_fem, dtype=np.float64)

    theta_no = q_no[:, 1:3]
    theta_fem = q_fem[:, 1:3]
    theta_err = theta_no - theta_fem

    err_nodes = theta_err[faces]
    ref_nodes = theta_fem[faces]

    num = np.einsum("eij,eia,eja->", mass_N, err_nodes, err_nodes)
    den = np.einsum("eij,eia,eja->", mass_N, ref_nodes, ref_nodes)

    return float(np.sqrt(num / max(den, 1.0e-30)))


def relative_curvature_error(
    verts_xy: np.ndarray,
    faces: np.ndarray,
    q_no: np.ndarray,
    q_fem: np.ndarray,
) -> float:
    area, dNdx, dNdy = triangle_area_and_grads(verts_xy, faces)

    q_no = np.asarray(q_no, dtype=np.float64)
    q_fem = np.asarray(q_fem, dtype=np.float64)

    def _curvature(q_local: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        tri_q = q_local[faces]
        thx_nodes = tri_q[:, :, 1]
        thy_nodes = tri_q[:, :, 2]

        dthx_dx = np.einsum("ei,ei->e", dNdx, thx_nodes)
        dthx_dy = np.einsum("ei,ei->e", dNdy, thx_nodes)
        dthy_dx = np.einsum("ei,ei->e", dNdx, thy_nodes)
        dthy_dy = np.einsum("ei,ei->e", dNdy, thy_nodes)

        kxx = dthx_dx
        kyy = dthy_dy
        kxy = 0.5 * (dthx_dy + dthy_dx)
        return kxx, kyy, kxy

    kxx_no, kyy_no, kxy_no = _curvature(q_no)
    kxx_fem, kyy_fem, kxy_fem = _curvature(q_fem)

    dkxx = kxx_no - kxx_fem
    dkyy = kyy_no - kyy_fem
    dkxy = kxy_no - kxy_fem

    num = np.sum(area * (dkxx * dkxx + dkyy * dkyy + 2.0 * dkxy * dkxy))
    den = np.sum(area * (kxx_fem * kxx_fem + kyy_fem * kyy_fem + 2.0 * kxy_fem * kxy_fem))

    return float(np.sqrt(num / max(den, 1.0e-30)))


def mr_resultants_p1(
    *,
    verts_xy: np.ndarray,
    faces: np.ndarray,
    q_local: np.ndarray,
    E: float,
    nu: float,
    thickness: float,
    E_elem: Optional[np.ndarray] = None,
    nu_elem: Optional[np.ndarray] = None,
    shear_correction: float = 5.0 / 6.0,
) -> Dict[str, np.ndarray]:
    """Reconstruct Mindlin--Reissner resultants from nodal P1 fields.

    ``q_local`` is ordered as ``[w, theta_x, theta_y]``.  Bending moments are
    constant on each triangle.  Transverse shear forces are retained at the
    three local triangle nodes, so their piecewise-linear L2 norm can be
    integrated exactly with the triangle mass matrix.  ``shear_elem`` is the
    centroid value used only for plotting.

    The same reconstruction is applied to both the neural-operator output and
    the FEM solution sampled at the saved mesh vertices.  This deliberately
    compares both sides in the same reduced P1 space.
    """
    faces = np.asarray(faces, dtype=np.int64)
    q_local = np.asarray(q_local, dtype=np.float64)
    n_vertices = int(np.asarray(verts_xy).shape[0])

    if q_local.shape != (n_vertices, 3):
        raise ValueError(
            f"q_local must have shape ({n_vertices}, 3)=[w,theta_x,theta_y], "
            f"got {q_local.shape}."
        )

    _, dNdx, dNdy = triangle_area_and_grads(verts_xy, faces)
    tri_q = q_local[faces]
    w_nodes = tri_q[:, :, 0]
    thx_nodes = tri_q[:, :, 1]
    thy_nodes = tri_q[:, :, 2]

    dw_dx = np.einsum("ei,ei->e", dNdx, w_nodes)
    dw_dy = np.einsum("ei,ei->e", dNdy, w_nodes)
    dthx_dx = np.einsum("ei,ei->e", dNdx, thx_nodes)
    dthx_dy = np.einsum("ei,ei->e", dNdy, thx_nodes)
    dthy_dx = np.einsum("ei,ei->e", dNdx, thy_nodes)
    dthy_dy = np.einsum("ei,ei->e", dNdy, thy_nodes)

    E_eff, nu_eff = _make_element_material(
        faces=faces,
        E=E,
        nu=nu,
        E_elem=E_elem,
        nu_elem=nu_elem,
    )
    t = float(thickness)
    bending_stiffness = E_eff * t**3 / (12.0 * (1.0 - nu_eff**2))
    shear_stiffness = (
        float(shear_correction) * E_eff * t / (2.0 * (1.0 + nu_eff))
    )

    moment_elem = np.column_stack(
        [
            bending_stiffness * (dthx_dx + nu_eff * dthy_dy),
            bending_stiffness * (dthy_dy + nu_eff * dthx_dx),
            0.5
            * bending_stiffness
            * (1.0 - nu_eff)
            * (dthx_dy + dthy_dx),
        ]
    )

    # gamma = grad(w) - theta, evaluated at all three local P1 nodes.
    shear_elem_nodes = np.empty((faces.shape[0], 3, 2), dtype=np.float64)
    shear_elem_nodes[:, :, 0] = shear_stiffness[:, None] * (
        dw_dx[:, None] - thx_nodes
    )
    shear_elem_nodes[:, :, 1] = shear_stiffness[:, None] * (
        dw_dy[:, None] - thy_nodes
    )

    return {
        "moment_elem": moment_elem,
        "shear_elem_nodes": shear_elem_nodes,
        "shear_elem": np.mean(shear_elem_nodes, axis=1),
    }


def relative_resultant_l2_errors(
    *,
    verts_xy: np.ndarray,
    faces: np.ndarray,
    q_no: np.ndarray,
    q_fem: np.ndarray,
    E: float,
    nu: float,
    thickness: float,
    E_elem: Optional[np.ndarray] = None,
    nu_elem: Optional[np.ndarray] = None,
    shear_correction: float = 5.0 / 6.0,
    moment_no_elem: Optional[np.ndarray] = None,
    shear_no_elem_nodes: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Return P1-to-P1 relative L2 errors of moment and shear resultants.

    Both the neural-operator prediction and the FEM reference are represented
    only by ``[w, theta_x, theta_y]`` at the saved triangle vertices.  The FEM
    vertex values are therefore deliberately reduced to the same P1 space as
    the neural-operator output before either resultant is reconstructed.

    The bending moment is elementwise constant.  The shear vector is P1 on
    each element, so its squared L2 norm is integrated exactly with the
    consistent triangle mass matrix.  Native CG2/Nedelec FEM resultants are
    intentionally not used here.
    """
    area, _, _ = triangle_area_and_grads(verts_xy, faces)
    mass_N = triangle_mass_matrices(area)
    if (moment_no_elem is None) != (shear_no_elem_nodes is None):
        raise ValueError("Direct moment and shear predictions must be supplied together.")
    if moment_no_elem is None:
        resultants_no = mr_resultants_p1(
            verts_xy=verts_xy,
            faces=faces,
            q_local=q_no,
            E=E,
            nu=nu,
            thickness=thickness,
            E_elem=E_elem,
            nu_elem=nu_elem,
            shear_correction=shear_correction,
        )
    else:
        resultants_no = {
            "moment_elem": np.asarray(moment_no_elem, dtype=np.float64),
            "shear_elem_nodes": np.asarray(shear_no_elem_nodes, dtype=np.float64),
        }
    resultants_fem = mr_resultants_p1(
        verts_xy=verts_xy,
        faces=faces,
        q_local=q_fem,
        E=E,
        nu=nu,
        thickness=thickness,
        E_elem=E_elem,
        nu_elem=nu_elem,
        shear_correction=shear_correction,
    )

    moment_no = resultants_no["moment_elem"]
    moment_fem = resultants_fem["moment_elem"]
    dm = moment_no - moment_fem

    # Frobenius norm of the symmetric 2x2 moment tensor: xx^2+yy^2+2xy^2.
    moment_num = np.sum(
        area * (dm[:, 0] ** 2 + dm[:, 1] ** 2 + 2.0 * dm[:, 2] ** 2)
    )
    moment_den = np.sum(
        area
        * (moment_fem[:, 0] ** 2 + moment_fem[:, 1] ** 2 + 2.0 * moment_fem[:, 2] ** 2)
    )

    shear_fem_nodes = resultants_fem["shear_elem_nodes"]
    ds_nodes = resultants_no["shear_elem_nodes"] - shear_fem_nodes
    shear_num = np.einsum("eij,eia,eja->", mass_N, ds_nodes, ds_nodes)
    shear_den = np.einsum(
        "eij,eia,eja->", mass_N, shear_fem_nodes, shear_fem_nodes
    )

    return {
        "moment_l2_error": float(np.sqrt(moment_num / max(moment_den, 1.0e-30))),
        "shear_l2_error": float(np.sqrt(shear_num / max(shear_den, 1.0e-30))),
    }

def plane_stress_lame(E, nu):
    E = np.asarray(E, dtype=np.float64)
    nu = np.asarray(nu, dtype=np.float64)
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / (1.0 - nu * nu)
    return lam, mu


def _make_element_material(
    *,
    faces: np.ndarray,
    E: float,
    nu: float,
    E_elem: Optional[np.ndarray] = None,
    nu_elem: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    n_faces = int(np.asarray(faces).shape[0])

    if E_elem is not None and nu_elem is not None:
        E_eff = np.asarray(E_elem, dtype=np.float64).reshape(-1)
        nu_eff = np.asarray(nu_elem, dtype=np.float64).reshape(-1)
        if E_eff.shape[0] != n_faces or nu_eff.shape[0] != n_faces:
            raise ValueError(
                f"E_elem/nu_elem must have one value per face. "
                f"Got E_elem={E_eff.shape[0]}, nu_elem={nu_eff.shape[0]}, n_faces={n_faces}."
            )
        return E_eff, nu_eff

    return (
        np.full((n_faces,), float(E), dtype=np.float64),
        np.full((n_faces,), float(nu), dtype=np.float64),
    )


def mr_internal_energy_p1(
    *,
    verts_xy: np.ndarray,
    faces: np.ndarray,
    q_local: np.ndarray,
    E: float,
    nu: float,
    thickness: float,
    E_elem: Optional[np.ndarray] = None,
    nu_elem: Optional[np.ndarray] = None,
    shear_correction: float = 5.0 / 6.0,
) -> Dict[str, float]:
    area, dNdx, dNdy = triangle_area_and_grads(verts_xy, faces)
    mass_N = triangle_mass_matrices(area)

    q_local = np.asarray(q_local, dtype=np.float64)

    tri_q = q_local[faces]

    w_nodes = tri_q[:, :, 0]
    thx_nodes = tri_q[:, :, 1]
    thy_nodes = tri_q[:, :, 2]
    theta_nodes = tri_q[:, :, 1:3]

    dw_dx = np.einsum("ei,ei->e", dNdx, w_nodes)
    dw_dy = np.einsum("ei,ei->e", dNdy, w_nodes)

    dthx_dx = np.einsum("ei,ei->e", dNdx, thx_nodes)
    dthx_dy = np.einsum("ei,ei->e", dNdy, thx_nodes)
    dthy_dx = np.einsum("ei,ei->e", dNdx, thy_nodes)
    dthy_dy = np.einsum("ei,ei->e", dNdy, thy_nodes)

    kxx = dthx_dx
    kyy = dthy_dy
    kxy = 0.5 * (dthx_dy + dthy_dx)

    tr_kappa = kxx + kyy
    kappa_contract = kxx * kxx + kyy * kyy + 2.0 * kxy * kxy

    E_eff, nu_eff = _make_element_material(
        faces=faces,
        E=E,
        nu=nu,
        E_elem=E_elem,
        nu_elem=nu_elem,
    )
    lam, mu = plane_stress_lame(E_eff, nu_eff)
    t = float(thickness)

    U_ben_e = 0.5 * (t**3 / 12.0) * (
        lam * area * tr_kappa * tr_kappa
        + 2.0 * mu * area * kappa_contract
    )

    grad_w_sq = dw_dx * dw_dx + dw_dy * dw_dy

    int_N = area[:, None] / 3.0
    theta_int = np.einsum("ei,eia->ea", int_N, theta_nodes)

    grad_w_theta_int = dw_dx * theta_int[:, 0] + dw_dy * theta_int[:, 1]

    theta_quad = np.einsum(
        "eij,eia,eja->e",
        mass_N,
        theta_nodes,
        theta_nodes,
    )

    U_shr_e = 0.5 * shear_correction * mu * t * (
        area * grad_w_sq - 2.0 * grad_w_theta_int + theta_quad
    )

    U_ben = float(np.sum(U_ben_e))
    U_shr = float(np.sum(U_shr_e))
    U_int = U_ben + U_shr

    return {
        "U_int": U_int,
        "U_ben": U_ben,
        "U_shr": U_shr,
    }


def relative_energy_error(
    *,
    verts_xy: np.ndarray,
    faces: np.ndarray,
    q_no: np.ndarray,
    q_fem: np.ndarray,
    E: float,
    nu: float,
    thickness: float,
    E_elem: Optional[np.ndarray] = None,
    nu_elem: Optional[np.ndarray] = None,
):
    q_no = np.asarray(q_no, dtype=np.float64)
    q_fem = np.asarray(q_fem, dtype=np.float64)
    q_err = q_no - q_fem

    err_energy = mr_internal_energy_p1(
        verts_xy=verts_xy,
        faces=faces,
        q_local=q_err,
        E=E,
        nu=nu,
        thickness=thickness,
        E_elem=E_elem,
        nu_elem=nu_elem,
    )["U_int"]

    ref_energy = mr_internal_energy_p1(
        verts_xy=verts_xy,
        faces=faces,
        q_local=q_fem,
        E=E,
        nu=nu,
        thickness=thickness,
        E_elem=E_elem,
        nu_elem=nu_elem,
    )["U_int"]

    return float(np.sqrt(max(err_energy, 0.0) / max(ref_energy, 1.0e-30)))


def compute_metrics_from_arrays(
    *,
    verts_xy: np.ndarray,
    faces: np.ndarray,
    q_no: np.ndarray,
    q_fem: np.ndarray,
    E: float,
    nu: float,
    thickness: float,
    E_elem: Optional[np.ndarray] = None,
    nu_elem: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    w_no = np.asarray(q_no)[:, 0]
    w_fem = np.asarray(q_fem)[:, 0]

    l2 = relative_l2_error_w(verts_xy, faces, w_no, w_fem)
    h1 = relative_h1_semi_error_w(verts_xy, faces, w_no, w_fem)
    energy = relative_energy_error(
        verts_xy=verts_xy,
        faces=faces,
        q_no=q_no,
        q_fem=q_fem,
        E=E,
        nu=nu,
        thickness=thickness,
        E_elem=E_elem,
        nu_elem=nu_elem,
    )
    theta_l2 = relative_l2_error_theta(verts_xy, faces, q_no, q_fem)
    curvature = relative_curvature_error(verts_xy, faces, q_no, q_fem)

    return {
        "l2_error": float(l2),
        "h1_semi_error": float(h1),
        "energy_error": float(energy),
        "theta_l2_error": float(theta_l2),
        "curvature_error": float(curvature),
    }


def _read_element_field(
    *,
    key: str,
    case_data: Dict[str, Any],
    no_data: Dict[str, Any],
    fem_data: Optional[Dict[str, Any]] = None,
) -> Optional[np.ndarray]:
    for data in (case_data, no_data, fem_data or {}):
        if key in data:
            return np.asarray(data[key], dtype=np.float64).reshape(-1)
    return None


def compute_case_error_metrics(
    *,
    case_tag: str,
    case_data: Dict[str, Any],
    no_data: Dict[str, Any],
    fem_data: Dict[str, Any],
) -> Dict[str, Any]:
    verts_xy = np.asarray(case_data["verts_xy"], dtype=np.float64)
    faces = np.asarray(case_data["faces"], dtype=np.int64)

    q_no = np.asarray(no_data["pred_local"], dtype=np.float64)
    q_fem = np.asarray(fem_data["fem_local"], dtype=np.float64)

    E = to_float(case_data.get("E", no_data.get("E", fem_data.get("E", 1000.0))), 1000.0)
    nu = to_float(case_data.get("nu", no_data.get("nu", fem_data.get("nu", 0.3))), 0.3)
    thickness = to_float(
        case_data.get("thickness", no_data.get("thickness", fem_data.get("thickness", 0.05))),
        0.05,
    )

    E_elem = _read_element_field(key="E_elem", case_data=case_data, no_data=no_data, fem_data=fem_data)
    nu_elem = _read_element_field(key="nu_elem", case_data=case_data, no_data=no_data, fem_data=fem_data)

    metrics = compute_metrics_from_arrays(
        verts_xy=verts_xy,
        faces=faces,
        q_no=q_no,
        q_fem=q_fem,
        E=E,
        nu=nu,
        thickness=thickness,
        E_elem=E_elem,
        nu_elem=nu_elem,
    )
    direct_moment = no_data.get("pred_moment_elem", None)
    direct_shear = no_data.get("pred_shear_elem_nodes", None)
    resultant_metrics = relative_resultant_l2_errors(
        verts_xy=verts_xy,
        faces=faces,
        q_no=q_no,
        q_fem=q_fem,
        E=E,
        nu=nu,
        thickness=thickness,
        E_elem=E_elem,
        nu_elem=nu_elem,
        moment_no_elem=direct_moment,
        shear_no_elem_nodes=direct_shear,
    )

    w_no = q_no[:, 0]
    w_fem = q_fem[:, 0]

    row: Dict[str, Any] = {
        "case_tag": case_tag,
        "l2_error": metrics["l2_error"],
        "h1_semi_error": metrics["h1_semi_error"],
        "theta_l2_error": metrics["theta_l2_error"],
        "curvature_error": metrics["curvature_error"],
        "energy_error": metrics["energy_error"],
        "moment_l2_error": resultant_metrics["moment_l2_error"],
        "shear_l2_error": resultant_metrics["shear_l2_error"],
        "resultant_comparison": (
            "direct_resultants_vs_p1_fem" if direct_moment is not None
            else "p1_vertex_reconstruction"
        ),
        "max_abs_error_w": float(np.max(np.abs(w_no - w_fem))),
        "max_abs_fem_w": float(np.max(np.abs(w_fem))),
        "E": E,
        "nu": nu,
        "thickness": thickness,
        "geometry_type": to_str(case_data.get("geometry_type", ""), ""),
        "load_type": to_str(case_data.get("load_type", ""), ""),
    }

    if E_elem is not None:
        row["E_elem_mean"] = float(np.mean(E_elem))
        row["E_elem_min"] = float(np.min(E_elem))
        row["E_elem_max"] = float(np.max(E_elem))
    if nu_elem is not None:
        row["nu_elem_mean"] = float(np.mean(nu_elem))
        row["nu_elem_min"] = float(np.min(nu_elem))
        row["nu_elem_max"] = float(np.max(nu_elem))

    def _read_int_field(data: Dict[str, Any], key: str, default: int) -> int:
        try:
            if key in data:
                return int(np.asarray(data[key]).reshape(-1)[0])
        except Exception:
            pass
        return int(default)

    if "query_xyz" in no_data:
        default_n_query = int(np.asarray(no_data["query_xyz"]).shape[0])
    elif "query_xyz" in case_data:
        default_n_query = int(np.asarray(case_data["query_xyz"]).shape[0])
    else:
        default_n_query = int(q_no.shape[0])

    default_n_faces = int(faces.shape[0])

    default_n_geometry_points = -1

    row.update(
        {
            "n_query_points": _read_int_field(no_data, "n_query_points", default_n_query),
            "n_geometry_points": _read_int_field(no_data, "n_geometry_points", default_n_geometry_points),
            "n_faces": _read_int_field(no_data, "n_faces", default_n_faces),
            "inference_time_ms": to_float(no_data.get("inference_time_ms", np.nan), np.nan),
        }
    )

    return row


# -----------------------------------------------------------------------------
# Table summaries
# -----------------------------------------------------------------------------

def summarize_metric_rows(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    if not rows:
        return {}

    out: Dict[str, float] = {}

    for key in (
        "l2_error",
        "h1_semi_error",
        "energy_error",
        "theta_l2_error",
        "curvature_error",
        "moment_l2_error",
        "shear_l2_error",
    ):
        values = np.asarray([float(r[key]) for r in rows], dtype=np.float64)
        out[f"{key}_mean"] = float(np.mean(values))
        out[f"{key}_std"] = float(np.std(values))
        out[f"{key}_min"] = float(np.min(values))
        out[f"{key}_max"] = float(np.max(values))

    return out


def save_metrics_table(
    *,
    rows: List[Dict[str, Any]],
    npz_path: str | Path,
    csv_path: str | Path,
):
    npz_path = Path(npz_path)
    csv_path = Path(csv_path)

    npz_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    keys: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)

    payload = {}
    for k in keys:
        values = [r.get(k, "") for r in rows]
        if k in ("case_tag", "geometry_type", "load_type"):
            payload[k] = np.asarray(values, dtype=str)
        else:
            try:
                payload[k] = np.asarray(values, dtype=np.float64)
            except Exception:
                payload[k] = np.asarray(values, dtype=str)

    np.savez_compressed(npz_path, **payload)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# -----------------------------------------------------------------------------
# Training-history plots
# -----------------------------------------------------------------------------

PLOT_TITLE_FONTSIZE = 16
PLOT_LABEL_FONTSIZE = 14
PLOT_TICK_FONTSIZE = 12
PLOT_LEGEND_FONTSIZE = 12
PLOT_LINEWIDTH = 2.4
PLOT_FIGSIZE = (7.0, 5.0)


def _style_epoch_axis(
    ax,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    scientific_y: bool = False,
):
    ax.set_xlabel(xlabel, fontsize=PLOT_LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=PLOT_LABEL_FONTSIZE)
    ax.set_title(title, fontsize=PLOT_TITLE_FONTSIZE)
    ax.tick_params(axis="both", labelsize=PLOT_TICK_FONTSIZE)

    if scientific_y:
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
        ax.yaxis.get_offset_text().set_fontsize(PLOT_TICK_FONTSIZE)

    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=PLOT_LEGEND_FONTSIZE)


def plot_loss_curve(
    history: Dict[str, List[float]],
    out_path: str | Path,
    *,
    negative_only: bool = False,
):
    """Plot train/test losses, optionally focusing on their negative range.

    KL total potential is often initially positive and several orders of
    magnitude larger than its converged negative value.  Keeping that positive
    transient in the same y-range hides convergence of the physical minimum.
    """
    train_loss = np.asarray(history["train_loss"], dtype=float)
    test_loss = np.asarray(history["test_loss"], dtype=float)
    epochs = np.arange(1, len(train_loss) + 1)

    plotted_train = train_loss
    plotted_test = test_loss
    negative_range_available = False
    if negative_only:
        plotted_train = np.where(train_loss < 0.0, train_loss, np.nan)
        plotted_test = np.where(test_loss < 0.0, test_loss, np.nan)
        negative_range_available = bool(
            np.isfinite(plotted_train).any() or np.isfinite(plotted_test).any()
        )

        # Preserve a useful plot if a short KL run has not reached Pi < 0 yet.
        if not negative_range_available:
            plotted_train = train_loss
            plotted_test = test_loss

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
    ax.plot(epochs, plotted_train, label="Train loss", linewidth=PLOT_LINEWIDTH)
    ax.plot(epochs, plotted_test, label="Test loss", linewidth=PLOT_LINEWIDTH)

    title = "Loss Function"
    ylabel = "Loss Function"
    if negative_only and negative_range_available:
        negative_values = np.concatenate((
            plotted_train[np.isfinite(plotted_train)],
            plotted_test[np.isfinite(plotted_test)],
        ))
        lower = float(np.min(negative_values))
        padding = max(abs(lower) * 0.05, np.finfo(float).eps)
        ax.set_ylim(lower - padding, 0.0)
        title = "Loss Function (negative range)"
        ylabel = "Loss Function (< 0)"

    _style_epoch_axis(ax, xlabel="Epoch", ylabel=ylabel, title=title, scientific_y=True)

    fig.tight_layout()
    safe_savefig(fig, out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

def plot_fem_error_history(
    *,
    error_history: Dict[str, List[float]],
    out_paths: Dict[str, Path],
):
    epochs = np.asarray(error_history["epoch"], dtype=float)
    n_epochs = epochs.size
    
    # 定义需要绘制的指标映射
    metrics_map = {
        "l2": (r"Relative $L^2$ error", "l2_error"),
        "h1": (r"Relative $H^1$ semi-norm error", "h1_semi_error"),
        "energy": ("Relative energy norm error", "energy_error"),
        "theta": (r"Relative $L^2$ rotation error", "theta_l2_error"),
        "curvature": (r"Relative $L^2$ curvature error", "curvature_error"),
    }

    for key, (ylabel, base_name) in metrics_map.items():
        if key not in out_paths:
            continue

        train_pair = _epoch_aligned_error_pair(
            error_history,
            mean_key=f"train_{base_name}_mean",
            std_key=f"train_{base_name}_std",
            n_epochs=n_epochs,
        )
        test_pair = _epoch_aligned_error_pair(
            error_history,
            mean_key=f"test_{base_name}_mean",
            std_key=f"test_{base_name}_std",
            n_epochs=n_epochs,
        )
        legacy_pair = _epoch_aligned_error_pair(
            error_history,
            mean_key=f"{base_name}_mean",
            std_key=f"{base_name}_std",
            n_epochs=n_epochs,
        )

        if train_pair is not None and test_pair is not None:
            # 绘制双曲线
            plot_two_errors_vs_epoch(
                epochs=epochs,
                train_mean=train_pair[0],
                train_std=train_pair[1],
                test_mean=test_pair[0],
                test_std=test_pair[1],
                ylabel=ylabel,
                title=ylabel,
                out_path=out_paths[key],
            )
        elif test_pair is not None:
            plot_one_error_vs_epoch(
                epochs=epochs,
                mean=test_pair[0],
                std=test_pair[1],
                ylabel=ylabel,
                title=ylabel,
                out_path=out_paths[key],
            )
        elif train_pair is not None:
            plot_one_error_vs_epoch(
                epochs=epochs,
                mean=train_pair[0],
                std=train_pair[1],
                ylabel=ylabel,
                title=ylabel,
                out_path=out_paths[key],
            )
        elif legacy_pair is not None:
            plot_one_error_vs_epoch(
                epochs=epochs,
                mean=legacy_pair[0],
                std=legacy_pair[1],
                ylabel=ylabel,
                title=ylabel,
                out_path=out_paths[key],
            )
        else:
            continue


def _epoch_aligned_error_pair(
    error_history: Dict[str, List[float]],
    *,
    mean_key: str,
    std_key: str,
    n_epochs: int,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    if mean_key not in error_history or std_key not in error_history:
        return None

    mean = np.asarray(error_history[mean_key], dtype=float).reshape(-1)
    std = np.asarray(error_history[std_key], dtype=float).reshape(-1)
    if mean.size != n_epochs or std.size != n_epochs:
        return None

    return mean, std

def plot_two_errors_vs_epoch(
    *,
    epochs: np.ndarray,
    train_mean: np.ndarray,
    train_std: np.ndarray,
    test_mean: np.ndarray,
    test_std: np.ndarray,
    ylabel: str,
    title: str,
    out_path: str | Path,
):
    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)

    # 绘制训练集曲线 (通常设为蓝色 C0)
    ax.plot(epochs, train_mean, linewidth=PLOT_LINEWIDTH, label="Train ", color="C0")
    ax.fill_between(epochs, train_mean - train_std, train_mean + train_std, alpha=0.15, color="C0")

    # 绘制测试集曲线 (通常设为橙色 C1)
    ax.plot(epochs, test_mean, linewidth=PLOT_LINEWIDTH, label="Test ", color="C1")
    ax.fill_between(epochs, test_mean - test_std, test_mean + test_std, alpha=0.15, color="C1")

    _style_epoch_axis(ax, xlabel="Epoch", ylabel=ylabel, title=title, scientific_y=False)

    fig.tight_layout()
    safe_savefig(fig, out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

def plot_one_error_vs_epoch(
    *,
    epochs: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    ylabel: str,
    title: str,
    out_path: str | Path,
):
    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)

    ax.plot(epochs, mean, linewidth=PLOT_LINEWIDTH, label="mean")
    ax.fill_between(epochs, mean - std, mean + std, alpha=0.20, label="mean ± std")

    _style_epoch_axis(ax, xlabel="Epoch", ylabel=ylabel, title=title, scientific_y=False)

    fig.tight_layout()
    safe_savefig(fig, out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Compare figures
# -----------------------------------------------------------------------------

def _face_mean_scalar(nodal_scalar: np.ndarray, faces: np.ndarray) -> np.ndarray:
    nodal_scalar = np.asarray(nodal_scalar).reshape(-1)
    faces = np.asarray(faces, dtype=np.int64)
    return nodal_scalar[faces].mean(axis=1)


def _shared_range(a, b):
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    vmin = float(min(a.min(), b.min()))
    vmax = float(max(a.max(), b.max()))
    if abs(vmax - vmin) < 1.0e-14:
        vmax = vmin + 1.0e-14
    return vmin, vmax


def _finite_range(a):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    if abs(vmax - vmin) < 1.0e-14:
        vmax = vmin + 1.0e-14
    return vmin, vmax

def _field_is_nonuniform(a, *, rtol: float = 1.0e-8, atol: float = 1.0e-12) -> bool:
    """
    Return True if a scalar field has meaningful spatial variation.
    Used to distinguish a real random field from a uniform scalar field
    saved on all nodes/elements.
    """
    if a is None:
        return False

    arr = np.asarray(a, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]

    if arr.size <= 1:
        return False

    vmin = float(np.min(arr))
    vmax = float(np.max(arr))
    scale = max(1.0, abs(vmin), abs(vmax))

    return (vmax - vmin) > (atol + rtol * scale)


def _read_nodal_field(
    *,
    key: str,
    case_data: Dict[str, Any],
    no_data: Dict[str, Any],
    fem_data: Dict[str, Any],
) -> Optional[np.ndarray]:
    for data in (case_data, no_data, fem_data):
        if key in data:
            return np.asarray(data[key], dtype=np.float64).reshape(-1)
    return None


def _has_grf_load(
    *,
    case_data: Dict[str, Any],
    no_data: Dict[str, Any],
    fem_data: Dict[str, Any],
) -> bool:
    """
    Decide whether to draw the pressure random-field panel.

    A uniform pressure field is usually saved as case_data["pressure"],
    but should not be plotted as an extra random-field panel.
    """
    load_type = to_str(
        case_data.get("load_type", no_data.get("load_type", fem_data.get("load_type", ""))),
        "",
    ).lower()

    pressure = _read_nodal_field(
        key="pressure",
        case_data=case_data,
        no_data=no_data,
        fem_data=fem_data,
    )

    if pressure is None:
        return False

    if any(k in load_type for k in ("grf", "random", "gaussian")):
        return True

    return _field_is_nonuniform(pressure)


def _has_grf_material(
    *,
    case_data: Dict[str, Any],
    no_data: Dict[str, Any],
    fem_data: Dict[str, Any],
) -> bool:
    """
    Decide whether to draw material random-field panels.
    """
    material_type = to_str(
        case_data.get(
            "material_type",
            no_data.get("material_type", fem_data.get("material_type", "")),
        ),
        "",
    ).lower()

    E_elem = _read_element_field(key="E_elem", case_data=case_data, no_data=no_data, fem_data=fem_data)
    nu_elem = _read_element_field(key="nu_elem", case_data=case_data, no_data=no_data, fem_data=fem_data)

    if E_elem is None or nu_elem is None:
        return False

    if any(k in material_type for k in ("grf", "random", "gaussian")):
        return True

    return _field_is_nonuniform(E_elem) or _field_is_nonuniform(nu_elem)


def _plot_2d_scalar_panel(
    ax,
    query_xyz,
    faces,
    nodal_scalar,
    lx,
    ly,
    title,
    cmap="viridis",
    vmin=None,
    vmax=None,
    cbar_label=None,
):
    q = np.asarray(query_xyz)
    f = np.asarray(faces, dtype=np.int64)

    x_plot = q[:, 0]
    y_plot = q[:, 1]

    face_scalar = _face_mean_scalar(nodal_scalar, f)

    m = ax.tripcolor(
        x_plot,
        y_plot,
        f,
        facecolors=face_scalar,
        shading="flat",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.set_xlim(0.0, float(lx))
    ax.set_ylim(0.0, float(ly))

    cbar = plt.colorbar(m, ax=ax, shrink=0.82, pad=0.025)
    if cbar_label is not None:
        cbar.set_label(cbar_label)


def _plot_boundary_line_load_panel(
    ax,
    query_xyz,
    faces,
    nodal_load,
    loaded_boundary_edges,
    lx,
    ly,
):
    """Plot zero load in the domain and color only the prescribed-load edge."""
    q = np.asarray(query_xyz, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    values = np.asarray(nodal_load, dtype=np.float64).reshape(-1)
    edges = np.asarray(loaded_boundary_edges, dtype=np.int64).reshape(-1, 2)

    vmax = max(float(np.max(values[edges])) if edges.size else 0.0, 1.0e-14)
    norm = Normalize(vmin=0.0, vmax=vmax)
    # Match the red-blue colour scale used by the other scalar-field panels.
    cmap = plt.get_cmap("coolwarm")

    # The plate interior is exactly zero load.  Using constant face colors
    # avoids smearing the nonzero boundary nodal values into adjacent elements.
    ax.tripcolor(
        q[:, 0],
        q[:, 1],
        f,
        facecolors=np.zeros(f.shape[0], dtype=np.float64),
        shading="flat",
        cmap=cmap,
        norm=norm,
    )
    if edges.size:
        segments = q[edges, :2]
        edge_values = 0.5 * (values[edges[:, 0]] + values[edges[:, 1]])
        collection = LineCollection(
            segments,
            cmap=cmap,
            norm=norm,
            linewidths=5.0,
            zorder=4,
        )
        collection.set_array(edge_values)
        ax.add_collection(collection)
        node_ids = np.unique(edges.reshape(-1))
        ax.scatter(
            q[node_ids, 0],
            q[node_ids, 1],
            c=values[node_ids],
            cmap=cmap,
            norm=norm,
            s=11,
            linewidths=0.0,
            zorder=5,
        )

    ax.set_title("Boundary load")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.set_xlim(0.0, float(lx))
    ax.set_ylim(0.0, float(ly))
    scalar_mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_mappable.set_array([])
    cbar = plt.colorbar(scalar_mappable, ax=ax, shrink=0.82, pad=0.025)
    cbar.set_label("Boundary load")


def _plot_2d_element_scalar_panel(
    ax,
    query_xyz,
    faces,
    elem_scalar,
    lx,
    ly,
    title,
    cmap="viridis",
    vmin=None,
    vmax=None,
    cbar_label=None,
):
    q = np.asarray(query_xyz)
    f = np.asarray(faces, dtype=np.int64)
    elem_scalar = np.asarray(elem_scalar, dtype=np.float64).reshape(-1)

    if elem_scalar.shape[0] != f.shape[0]:
        raise ValueError(
            f"Element scalar length {elem_scalar.shape[0]} does not match number of faces {f.shape[0]}."
        )

    m = ax.tripcolor(
        q[:, 0],
        q[:, 1],
        f,
        facecolors=elem_scalar,
        shading="flat",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.set_xlim(0.0, float(lx))
    ax.set_ylim(0.0, float(ly))

    cbar = plt.colorbar(m, ax=ax, shrink=0.82, pad=0.025)
    if cbar_label is not None:
        cbar.set_label(cbar_label)


def make_resultants_compare_figure(
    *,
    query_xyz,
    verts_xy,
    faces,
    q_no,
    q_fem,
    E,
    nu,
    thickness,
    E_elem,
    nu_elem,
    lx,
    ly,
    out_path,
    moment_no_elem=None,
    shear_no_elem_nodes=None,
):
    """Plot direct or P1-reconstructed NO/FEM resultants."""
    if (moment_no_elem is None) != (shear_no_elem_nodes is None):
        raise ValueError("Direct moment and shear predictions must be supplied together.")
    if moment_no_elem is None:
        resultants_no = mr_resultants_p1(
            verts_xy=verts_xy,
            faces=faces,
            q_local=q_no,
            E=E,
            nu=nu,
            thickness=thickness,
            E_elem=E_elem,
            nu_elem=nu_elem,
        )
        title_prefix = "P1-reconstructed"
    else:
        local_shear = np.asarray(shear_no_elem_nodes, dtype=np.float64)
        resultants_no = {
            "moment_elem": np.asarray(moment_no_elem, dtype=np.float64),
            "shear_elem": np.mean(local_shear, axis=1),
        }
        title_prefix = "Direct H-R"
    resultants_fem = mr_resultants_p1(
        verts_xy=verts_xy,
        faces=faces,
        q_local=q_fem,
        E=E,
        nu=nu,
        thickness=thickness,
        E_elem=E_elem,
        nu_elem=nu_elem,
    )
    moment_no_elem = resultants_no["moment_elem"]
    moment_fem_elem = resultants_fem["moment_elem"]
    shear_no_elem = resultants_no["shear_elem"]
    shear_fem_elem = resultants_fem["shear_elem"]

    fields = (
        (r"$M_{xx}$", moment_no_elem[:, 0], moment_fem_elem[:, 0]),
        (r"$M_{yy}$", moment_no_elem[:, 1], moment_fem_elem[:, 1]),
        (r"$M_{xy}$", moment_no_elem[:, 2], moment_fem_elem[:, 2]),
        (r"$Q_x$", shear_no_elem[:, 0], shear_fem_elem[:, 0]),
        (r"$Q_y$", shear_no_elem[:, 1], shear_fem_elem[:, 1]),
    )

    fig, axes = plt.subplots(5, 3, figsize=(18.0, 22.0), squeeze=False)
    column_titles = ("NO", "FEM", "NO - FEM")

    for row_id, (symbol, no_values, fem_values) in enumerate(fields):
        error_values = no_values - fem_values
        vmin, vmax = _shared_range(no_values, fem_values)
        emax = float(np.max(np.abs(error_values)))
        if emax < 1.0e-14:
            emax = 1.0e-14

        row_values = (no_values, fem_values, error_values)
        row_limits = ((vmin, vmax), (vmin, vmax), (-emax, emax))
        for col_id, (values, limits) in enumerate(zip(row_values, row_limits)):
            title = column_titles[col_id] if row_id == 0 else ""
            _plot_2d_element_scalar_panel(
                axes[row_id, col_id],
                query_xyz,
                faces,
                values,
                lx,
                ly,
                title,
                cmap="coolwarm",
                vmin=limits[0],
                vmax=limits[1],
                cbar_label=symbol,
            )
            if col_id == 0:
                axes[row_id, col_id].set_ylabel(f"{symbol}\ny")

    fig.suptitle(
        f"{title_prefix} bending moments and transverse shear forces",
        y=0.995,
        fontsize=18,
    )
    plt.subplots_adjust(
        left=0.050,
        right=0.990,
        bottom=0.030,
        top=0.970,
        hspace=0.16,
        wspace=0.03,
    )
    safe_savefig(fig, out_path, dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _plot_3d_shape_panel(
    ax,
    query_xyz,
    faces,
    global_disp,
    lx,
    ly,
    deformation_scale,
    title,
):
    q = np.asarray(query_xyz)
    f = np.asarray(faces, dtype=np.int64)
    g = np.asarray(global_disp)

    x = q[:, 0]
    y = q[:, 1]
    z = q[:, 2] + float(deformation_scale) * g[:, 2]

    surf = ax.plot_trisurf(
        x,
        y,
        z,
        triangles=f,
        cmap="coolwarm",
        linewidth=0.2,
        antialiased=True,
        shade=False,
        alpha=0.90,
    )
    surf.set_array(z[f].mean(axis=1))
    surf.autoscale()

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim(0.0, float(lx))
    ax.set_ylim(0.0, float(ly))

    try:
        ax.set_box_aspect((1.0, 1.0, 0.35))
    except Exception:
        pass

    def true_w_formatter(val, pos):
        true_val = val / float(deformation_scale)
        return f"{true_val:g}"

    ax.zaxis.set_major_formatter(ticker.FuncFormatter(true_w_formatter))

    ax.view_init(elev=22, azim=-55)


def make_scalar_four_panel_compare(
    *,
    query_xyz,
    faces,
    no_global_disp,
    no_scalar,
    fem_scalar,
    lx,
    ly,
    out_path,
    fig_title,
    no_label,
    fem_label,
    err_label,
    deformation_scale=10.0,
):
    no_scalar = np.asarray(no_scalar).reshape(-1)
    fem_scalar = np.asarray(fem_scalar).reshape(-1)
    err_scalar = no_scalar - fem_scalar

    vmin, vmax = _shared_range(no_scalar, fem_scalar)
    emax = float(np.max(np.abs(err_scalar)))
    if emax < 1.0e-14:
        emax = 1.0e-14

    fig = plt.figure(figsize=(24, 5.6))

    ax1 = fig.add_subplot(1, 4, 1, projection="3d")
    _plot_3d_shape_panel(ax1, query_xyz, faces, no_global_disp, lx, ly, deformation_scale, "3D deformed plate shape")

    ax2 = fig.add_subplot(1, 4, 2)
    _plot_2d_scalar_panel(ax2, query_xyz, faces, no_scalar, lx, ly, "NO", cmap="coolwarm", vmin=vmin, vmax=vmax, cbar_label=no_label)

    ax3 = fig.add_subplot(1, 4, 3)
    _plot_2d_scalar_panel(ax3, query_xyz, faces, fem_scalar, lx, ly, "FEM", cmap="coolwarm", vmin=vmin, vmax=vmax, cbar_label=fem_label)

    ax4 = fig.add_subplot(1, 4, 4)
    _plot_2d_scalar_panel(ax4, query_xyz, faces, err_scalar, lx, ly, "Error", cmap="coolwarm", vmin=-emax, vmax=emax, cbar_label=err_label)

    fig.suptitle(fig_title, y=0.98)
    plt.subplots_adjust(left=0.03, right=0.99, bottom=0.10, top=0.87, wspace=0.35)

    safe_savefig(fig, out_path, dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def make_scalar_five_panel_compare(
    *,
    query_xyz,
    faces,
    no_global_disp,
    pressure_scalar,
    no_scalar,
    fem_scalar,
    lx,
    ly,
    out_path,
    fig_title,
    no_label,
    fem_label,
    err_label,
    pressure_label="Pressure",
    deformation_scale=10.0,
):
    pressure_scalar = np.asarray(pressure_scalar).reshape(-1)
    no_scalar = np.asarray(no_scalar).reshape(-1)
    fem_scalar = np.asarray(fem_scalar).reshape(-1)
    err_scalar = no_scalar - fem_scalar

    vmin, vmax = _shared_range(no_scalar, fem_scalar)
    emax = float(np.max(np.abs(err_scalar)))
    if emax < 1.0e-14:
        emax = 1.0e-14

    pmin, pmax = _finite_range(pressure_scalar)

    fig = plt.figure(figsize=(30, 5.6))

    ax1 = fig.add_subplot(1, 5, 1, projection="3d")
    _plot_3d_shape_panel(ax1, query_xyz, faces, no_global_disp, lx, ly, deformation_scale, "3D deformed plate shape")

    ax2 = fig.add_subplot(1, 5, 2)
    _plot_2d_scalar_panel(ax2, query_xyz, faces, pressure_scalar, lx, ly, "Pressure", cmap="coolwarm", vmin=pmin, vmax=pmax, cbar_label=pressure_label)

    ax3 = fig.add_subplot(1, 5, 3)
    _plot_2d_scalar_panel(ax3, query_xyz, faces, no_scalar, lx, ly, "NO", cmap="coolwarm", vmin=vmin, vmax=vmax, cbar_label=no_label)

    ax4 = fig.add_subplot(1, 5, 4)
    _plot_2d_scalar_panel(ax4, query_xyz, faces, fem_scalar, lx, ly, "FEM", cmap="coolwarm", vmin=vmin, vmax=vmax, cbar_label=fem_label)

    ax5 = fig.add_subplot(1, 5, 5)
    _plot_2d_scalar_panel(ax5, query_xyz, faces, err_scalar, lx, ly, "Error", cmap="coolwarm", vmin=-emax, vmax=emax, cbar_label=err_label)

    fig.suptitle(fig_title, y=0.98)
    plt.subplots_adjust(left=0.025, right=0.995, bottom=0.10, top=0.87, wspace=0.35)

    safe_savefig(fig, out_path, dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def make_scalar_six_panel_material_compare(
    *,
    query_xyz,
    faces,
    no_global_disp,
    E_elem,
    nu_elem,
    no_scalar,
    fem_scalar,
    lx,
    ly,
    out_path,
    fig_title,
    no_label,
    fem_label,
    err_label,
    deformation_scale=10.0,
):
    E_elem = np.asarray(E_elem, dtype=np.float64).reshape(-1)
    nu_elem = np.asarray(nu_elem, dtype=np.float64).reshape(-1)
    no_scalar = np.asarray(no_scalar).reshape(-1)
    fem_scalar = np.asarray(fem_scalar).reshape(-1)
    err_scalar = no_scalar - fem_scalar

    vmin, vmax = _shared_range(no_scalar, fem_scalar)
    emax = float(np.max(np.abs(err_scalar)))
    if emax < 1.0e-14:
        emax = 1.0e-14

    Emin, Emax = _finite_range(E_elem)
    nmin, nmax = _finite_range(nu_elem)

    fig = plt.figure(figsize=(36, 5.6))

    ax1 = fig.add_subplot(1, 6, 1, projection="3d")
    _plot_3d_shape_panel(ax1, query_xyz, faces, no_global_disp, lx, ly, deformation_scale, "3D deformed plate shape")

    ax2 = fig.add_subplot(1, 6, 2)
    _plot_2d_element_scalar_panel(ax2, query_xyz, faces, E_elem, lx, ly, "Young's modulus", cmap="coolwarm", vmin=Emin, vmax=Emax, cbar_label="E")

    ax3 = fig.add_subplot(1, 6, 3)
    _plot_2d_element_scalar_panel(ax3, query_xyz, faces, nu_elem, lx, ly, "Poisson's ratio", cmap="coolwarm", vmin=nmin, vmax=nmax, cbar_label=r"$\nu$")

    ax4 = fig.add_subplot(1, 6, 4)
    _plot_2d_scalar_panel(ax4, query_xyz, faces, no_scalar, lx, ly, "NO", cmap="coolwarm", vmin=vmin, vmax=vmax, cbar_label=no_label)

    ax5 = fig.add_subplot(1, 6, 5)
    _plot_2d_scalar_panel(ax5, query_xyz, faces, fem_scalar, lx, ly, "FEM", cmap="coolwarm", vmin=vmin, vmax=vmax, cbar_label=fem_label)

    ax6 = fig.add_subplot(1, 6, 6)
    _plot_2d_scalar_panel(ax6, query_xyz, faces, err_scalar, lx, ly, "Error", cmap="coolwarm", vmin=-emax, vmax=emax, cbar_label=err_label)

    fig.suptitle(fig_title, y=0.98)
    plt.subplots_adjust(left=0.02, right=0.995, bottom=0.10, top=0.87, wspace=0.35)

    safe_savefig(fig, out_path, dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

def make_scalar_dynamic_panel_compare(
    *,
    query_xyz,
    faces,
    no_global_disp,
    no_scalar,
    fem_scalar,
    lx,
    ly,
    out_path,
    fig_title,
    no_label,
    fem_label,
    err_label,
    pressure_scalar=None,
    E_elem=None,
    nu_elem=None,
    show_pressure: bool = False,
    show_material: bool = False,
    loaded_boundary_edges=None,
    deformation_scale=10.0,
):
    no_scalar = np.asarray(no_scalar).reshape(-1)
    fem_scalar = np.asarray(fem_scalar).reshape(-1)
    err_scalar = no_scalar - fem_scalar

    vmin, vmax = _shared_range(no_scalar, fem_scalar)
    emax = float(np.max(np.abs(err_scalar)))
    if emax < 1.0e-14:
        emax = 1.0e-14

    panels = []

    # Always draw 3D shape first.
    panels.append(("3d", None))

    # Optional random-field panels.
    if show_material:
        if E_elem is None or nu_elem is None:
            raise KeyError("show_material=True, but E_elem or nu_elem is missing.")
        panels.append(("E", np.asarray(E_elem, dtype=np.float64).reshape(-1)))
        panels.append(("nu", np.asarray(nu_elem, dtype=np.float64).reshape(-1)))

    if show_pressure:
        if pressure_scalar is None:
            raise KeyError("show_pressure=True, but pressure_scalar is missing.")
        panels.append(("pressure", np.asarray(pressure_scalar, dtype=np.float64).reshape(-1)))

    # Always draw NO/FEM/Error at the end.
    panels.append(("NO", no_scalar))
    panels.append(("FEM", fem_scalar))
    panels.append(("Error", err_scalar))

    ncols = len(panels)

    # Keep each panel close to the old width.
    fig_width = max(6.0 * ncols, 24.0)
    fig = plt.figure(figsize=(fig_width, 5.6))

    for i, (kind, values) in enumerate(panels, start=1):
        if kind == "3d":
            ax = fig.add_subplot(1, ncols, i, projection="3d")
            _plot_3d_shape_panel(
                ax,
                query_xyz,
                faces,
                no_global_disp,
                lx,
                ly,
                deformation_scale,
                "3D deformed plate shape",
            )

        elif kind == "E":
            ax = fig.add_subplot(1, ncols, i)
            Emin, Emax = _finite_range(values)
            _plot_2d_element_scalar_panel(
                ax,
                query_xyz,
                faces,
                values,
                lx,
                ly,
                "Young's modulus",
                cmap="coolwarm",
                vmin=Emin,
                vmax=Emax,
                cbar_label="E",
            )

        elif kind == "nu":
            ax = fig.add_subplot(1, ncols, i)
            nmin, nmax = _finite_range(values)
            _plot_2d_element_scalar_panel(
                ax,
                query_xyz,
                faces,
                values,
                lx,
                ly,
                "Poisson's ratio",
                cmap="coolwarm",
                vmin=nmin,
                vmax=nmax,
                cbar_label=r"$\nu$",
            )

        elif kind == "pressure":
            ax = fig.add_subplot(1, ncols, i)
            if loaded_boundary_edges is not None:
                _plot_boundary_line_load_panel(
                    ax,
                    query_xyz,
                    faces,
                    values,
                    loaded_boundary_edges,
                    lx,
                    ly,
                )
            else:
                pmin, pmax = _finite_range(values)
                _plot_2d_scalar_panel(
                    ax,
                    query_xyz,
                    faces,
                    values,
                    lx,
                    ly,
                    "Pressure",
                    cmap="coolwarm",
                    vmin=pmin,
                    vmax=pmax,
                    cbar_label="Pressure",
                )

        elif kind == "NO":
            ax = fig.add_subplot(1, ncols, i)
            _plot_2d_scalar_panel(
                ax,
                query_xyz,
                faces,
                values,
                lx,
                ly,
                "NO",
                cmap="coolwarm",
                vmin=vmin,
                vmax=vmax,
                cbar_label=no_label,
            )

        elif kind == "FEM":
            ax = fig.add_subplot(1, ncols, i)
            _plot_2d_scalar_panel(
                ax,
                query_xyz,
                faces,
                values,
                lx,
                ly,
                "FEM",
                cmap="coolwarm",
                vmin=vmin,
                vmax=vmax,
                cbar_label=fem_label,
            )

        elif kind == "Error":
            ax = fig.add_subplot(1, ncols, i)
            _plot_2d_scalar_panel(
                ax,
                query_xyz,
                faces,
                values,
                lx,
                ly,
                "Error",
                cmap="coolwarm",
                vmin=-emax,
                vmax=emax,
                cbar_label=err_label,
            )

    fig.suptitle(fig_title, y=0.98)
    plt.subplots_adjust(
        left=0.02,
        right=0.995,
        bottom=0.10,
        top=0.87,
        wspace=0.35,
    )

    safe_savefig(fig, out_path, dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def generate_case_compare_figures(
    *,
    case_data,
    no_data,
    fem_data,
    displacement_out_path,
    rotation_out_path,
    resultants_out_path,
    deformation_scale=10.0,
):
    query_xyz = case_data["query_xyz"]
    faces = case_data["faces"]
    
    geometry_type = to_str(case_data.get("geometry_type", ""), "").lower()
    if geometry_type == "annular_sector":
        lx = 5.0
        ly = 5.0
    else:
        lx = float(np.asarray(case_data["lx"]).reshape(-1)[0])
        ly = float(np.asarray(case_data["ly"]).reshape(-1)[0])

    no_global_disp = no_data["pred_global_disp"]

    show_material = _has_grf_material(
        case_data=case_data,
        no_data=no_data,
        fem_data=fem_data,
    )
    show_pressure = _has_grf_load(
        case_data=case_data,
        no_data=no_data,
        fem_data=fem_data,
    )

    E_elem = None
    nu_elem = None
    if show_material:
        E_elem = _read_element_field(
            key="E_elem",
            case_data=case_data,
            no_data=no_data,
            fem_data=fem_data,
        )
        nu_elem = _read_element_field(
            key="nu_elem",
            case_data=case_data,
            no_data=no_data,
            fem_data=fem_data,
        )

    pressure_scalar = None
    if show_pressure:
        pressure_scalar = _read_nodal_field(
            key="pressure",
            case_data=case_data,
            no_data=no_data,
            fem_data=fem_data,
        )
    loaded_boundary_edges = case_data.get("loaded_boundary_edges", None)

    make_scalar_dynamic_panel_compare(
        query_xyz=query_xyz,
        faces=faces,
        no_global_disp=no_global_disp,
        E_elem=E_elem,
        nu_elem=nu_elem,
        pressure_scalar=pressure_scalar,
        show_material=show_material,
        show_pressure=show_pressure,
        loaded_boundary_edges=loaded_boundary_edges,
        no_scalar=no_data["disp_scalar"],
        fem_scalar=fem_data["disp_scalar"],
        lx=lx,
        ly=ly,
        out_path=displacement_out_path,
        fig_title="Displacement comparison",
        no_label="NO w",
        fem_label="FEM w",
        err_label="Error",
        deformation_scale=deformation_scale,
    )

    q_no = np.asarray(no_data["pred_local"], dtype=np.float64)
    E = to_float(case_data.get("E", no_data.get("E", fem_data.get("E", 1000.0))), 1000.0)
    nu = to_float(case_data.get("nu", no_data.get("nu", fem_data.get("nu", 0.3))), 0.3)
    thickness = to_float(
        case_data.get("thickness", no_data.get("thickness", fem_data.get("thickness", 0.05))),
        0.05,
    )
    result_E_elem = _read_element_field(
        key="E_elem", case_data=case_data, no_data=no_data, fem_data=fem_data
    )
    result_nu_elem = _read_element_field(
        key="nu_elem", case_data=case_data, no_data=no_data, fem_data=fem_data
    )
    make_resultants_compare_figure(
        query_xyz=query_xyz,
        verts_xy=case_data["verts_xy"],
        faces=faces,
        q_no=q_no,
        q_fem=np.asarray(fem_data["fem_local"], dtype=np.float64),
        E=E,
        nu=nu,
        thickness=thickness,
        E_elem=result_E_elem,
        nu_elem=result_nu_elem,
        lx=lx,
        ly=ly,
        out_path=resultants_out_path,
        moment_no_elem=no_data.get("pred_moment_elem", None),
        shear_no_elem_nodes=no_data.get("pred_shear_elem_nodes", None),
    )

    make_scalar_dynamic_panel_compare(
        query_xyz=query_xyz,
        faces=faces,
        no_global_disp=no_global_disp,
        E_elem=E_elem,
        nu_elem=nu_elem,
        pressure_scalar=pressure_scalar,
        show_material=show_material,
        show_pressure=show_pressure,
        loaded_boundary_edges=loaded_boundary_edges,
        no_scalar=no_data["rot_scalar"],
        fem_scalar=fem_data["rot_scalar"],
        lx=lx,
        ly=ly,
        out_path=rotation_out_path,
        fig_title="Rotation comparison",
        no_label=r"NO $|\theta_x|+|\theta_y|$",
        fem_label=r"FEM $|\theta_x|+|\theta_y|$",
        err_label="Error",
        deformation_scale=deformation_scale,
    )
