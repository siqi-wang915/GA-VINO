from __future__ import annotations

"""
Study the influence of triangular-mesh shape quality on the FEM solution.

The physical case is fixed to the existing thick-plate Navier benchmark:
    domain:       [0, 5] x [0, 5]
    thickness:    0.5
    E, nu:        1000, 0.3
    pressure:     0.8 sin(pi*x/a) sin(pi*y/b)
    point density: n = 100 -> 10,201 nodes and exactly 20,000 triangles

The boundary is fixed. Interior points are first randomized and then
Laplacian-relaxed before Delaunay triangulation, producing a high-quality
unstructured reference mesh without visible row/column alignment. Independent
point jitter then generates lower-quality candidates. Five meshes are selected
whose P5 quality values are as evenly spaced as possible. Element shape quality
is measured with the normalized triangular mean-ratio metric

    q_e = 4*sqrt(3)*A_e / (l1^2 + l2^2 + l3^2),  0 < q_e <= 1.

The primary whole-mesh indicator is P5(q_e), the fifth percentile of all
element qualities. Higher values mean better mesh shape quality.
"""

import argparse
import math
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from FEM.solvers import compare_mr_plate_navier as benchmark


DEFAULT_N = 100
DEFAULT_JITTER_MIN = 0.0
DEFAULT_JITTER_MAX = 0.80
DEFAULT_CANDIDATE_COUNT = 81
N_QUALITY_GROUPS = 5
REFERENCE_RANDOMIZATION = 0.80
REFERENCE_RELAXATION_STEPS = 15
REFERENCE_RELAXATION_WEIGHT = 0.60


def signed_double_areas(nodes: np.ndarray, elems: np.ndarray) -> np.ndarray:
    tri = nodes[elems]
    ab = tri[:, 1] - tri[:, 0]
    ac = tri[:, 2] - tri[:, 0]
    return ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]


def triangle_mean_ratio(nodes: np.ndarray, elems: np.ndarray) -> np.ndarray:
    """Return normalized shape quality for every linear triangular element."""
    tri = nodes[elems]
    edge01_sq = np.sum((tri[:, 1] - tri[:, 0]) ** 2, axis=1)
    edge12_sq = np.sum((tri[:, 2] - tri[:, 1]) ** 2, axis=1)
    edge20_sq = np.sum((tri[:, 0] - tri[:, 2]) ** 2, axis=1)
    double_area = signed_double_areas(nodes, elems)
    denominator = edge01_sq + edge12_sq + edge20_sq
    return 2.0 * math.sqrt(3.0) * double_area / np.maximum(denominator, 1.0e-300)


def triangle_minimum_angles_deg(nodes: np.ndarray, elems: np.ndarray) -> np.ndarray:
    tri = nodes[elems]
    edge_sq = np.column_stack(
        [
            np.sum((tri[:, 1] - tri[:, 2]) ** 2, axis=1),
            np.sum((tri[:, 0] - tri[:, 2]) ** 2, axis=1),
            np.sum((tri[:, 0] - tri[:, 1]) ** 2, axis=1),
        ]
    )
    lengths = np.sqrt(np.maximum(edge_sq, 1.0e-300))
    angles = []
    for i, j, k in ((0, 1, 2), (1, 0, 2), (2, 0, 1)):
        cosine = (edge_sq[:, j] + edge_sq[:, k] - edge_sq[:, i]) / (
            2.0 * lengths[:, j] * lengths[:, k]
        )
        angles.append(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    return np.min(np.column_stack(angles), axis=1)


def mesh_quality_statistics(nodes: np.ndarray, elems: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    quality = triangle_mean_ratio(nodes, elems)
    if np.any(quality <= 0.0):
        raise ValueError("Mesh contains a degenerate or inverted triangular element.")
    min_angles = triangle_minimum_angles_deg(nodes, elems)
    stats = {
        "quality_min": float(np.min(quality)),
        "quality_p05": float(np.percentile(quality, 5.0)),
        "quality_p10": float(np.percentile(quality, 10.0)),
        "quality_mean": float(np.mean(quality)),
        "quality_max": float(np.max(quality)),
        "minimum_angle_deg": float(np.min(min_angles)),
        "angle_p05_deg": float(np.percentile(min_angles, 5.0)),
    }
    return quality, stats


def build_random_jitter_field(
    base_nodes: np.ndarray,
    a: float,
    b: float,
    mesh_spacing: float,
    seed: int,
) -> np.ndarray:
    """Return independent 2-D jitter with a boundary-safe amplitude taper."""
    rng = np.random.default_rng(seed)
    boundary = (
        np.isclose(base_nodes[:, 0], 0.0)
        | np.isclose(base_nodes[:, 0], a)
        | np.isclose(base_nodes[:, 1], 0.0)
        | np.isclose(base_nodes[:, 1], b)
    )
    jitter = np.zeros_like(base_nodes, dtype=np.float64)
    jitter[~boundary] = rng.uniform(-1.0, 1.0, size=(int(np.sum(~boundary)), 2))
    distance_to_boundary = np.minimum.reduce(
        [
            base_nodes[:, 0],
            a - base_nodes[:, 0],
            base_nodes[:, 1],
            b - base_nodes[:, 1],
        ]
    )
    taper = np.clip(distance_to_boundary / mesh_spacing, 0.0, 1.0)
    jitter *= taper[:, None]
    return jitter


def delaunay_elements(nodes: np.ndarray) -> np.ndarray:
    """Delaunay-triangulate points and orient all triangles counter-clockwise."""
    try:
        from scipy.spatial import Delaunay
    except ImportError as exc:
        raise ImportError(
            "scipy is required to generate the unstructured quality-study meshes."
        ) from exc

    elems = np.asarray(Delaunay(nodes).simplices, dtype=np.int64)

    # DOLFINx and the in-house error integration both expect CCW triangles.
    inverted = signed_double_areas(nodes, elems) < 0.0
    if np.any(inverted):
        elems[inverted] = elems[inverted][:, [0, 2, 1]]
    if np.any(signed_double_areas(nodes, elems) <= 0.0):
        raise ValueError("Delaunay generated a degenerate triangular element.")
    return elems


def build_relaxed_reference_points(
    lattice_nodes: np.ndarray,
    a: float,
    b: float,
    mesh_spacing: float,
    seed: int,
    randomization: float = REFERENCE_RANDOMIZATION,
    relaxation_steps: int = REFERENCE_RELAXATION_STEPS,
    relaxation_weight: float = REFERENCE_RELAXATION_WEIGHT,
) -> np.ndarray:
    """Create irregular, quasi-uniform interior points by Laplacian relaxation."""
    if not 0.0 < randomization < 1.0:
        raise ValueError("reference randomization must lie between 0 and 1")
    if relaxation_steps < 0 or not 0.0 <= relaxation_weight <= 1.0:
        raise ValueError("invalid reference relaxation parameters")

    nodes = np.asarray(lattice_nodes, dtype=np.float64).copy()
    boundary = (
        np.isclose(nodes[:, 0], 0.0)
        | np.isclose(nodes[:, 0], a)
        | np.isclose(nodes[:, 1], 0.0)
        | np.isclose(nodes[:, 1], b)
    )
    rng = np.random.default_rng(seed)
    nodes[~boundary] += (
        randomization
        * mesh_spacing
        * rng.uniform(-1.0, 1.0, size=(int(np.sum(~boundary)), 2))
    )

    for _ in range(relaxation_steps):
        elems = delaunay_elements(nodes)
        edges = np.vstack(
            [elems[:, [0, 1]], elems[:, [1, 2]], elems[:, [2, 0]]]
        )
        edges = np.unique(np.sort(edges, axis=1), axis=0)
        neighbor_sum = np.zeros_like(nodes)
        neighbor_count = np.zeros(nodes.shape[0], dtype=np.float64)
        np.add.at(neighbor_sum, edges[:, 0], nodes[edges[:, 1]])
        np.add.at(neighbor_sum, edges[:, 1], nodes[edges[:, 0]])
        np.add.at(neighbor_count, edges[:, 0], 1.0)
        np.add.at(neighbor_count, edges[:, 1], 1.0)
        neighbor_mean = neighbor_sum / np.maximum(neighbor_count[:, None], 1.0)
        nodes[~boundary] = (
            (1.0 - relaxation_weight) * nodes[~boundary]
            + relaxation_weight * neighbor_mean[~boundary]
        )

    return nodes


def make_unstructured_mesh(
    base_nodes: np.ndarray,
    jitter_field: np.ndarray,
    jitter_amplitude: float,
    mesh_spacing: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Jitter relaxed points and recompute an unstructured Delaunay mesh."""
    if jitter_amplitude < 0.0:
        raise ValueError("jitter_amplitude must be non-negative")

    nodes = np.asarray(base_nodes, dtype=np.float64).copy()
    nodes += float(jitter_amplitude) * float(mesh_spacing) * jitter_field
    elems = delaunay_elements(nodes)
    return nodes, elems


def select_evenly_spaced_quality_meshes(
    base_nodes: np.ndarray,
    jitter_field: np.ndarray,
    mesh_spacing: float,
    jitter_min: float,
    jitter_max: float,
    candidate_count: int,
    n_groups: int = N_QUALITY_GROUPS,
) -> list[dict[str, object]]:
    """Select meshes closest to evenly spaced targets in P5-quality space."""
    if jitter_min < 0.0 or jitter_max <= jitter_min or jitter_max >= 1.0:
        raise ValueError("Require 0 <= jitter_min < jitter_max < 1 mesh spacing")
    if candidate_count < n_groups:
        raise ValueError("candidate_count must be at least the number of quality groups")

    candidates: list[dict[str, float]] = []
    for amplitude in np.linspace(jitter_min, jitter_max, candidate_count):
        nodes, elems = make_unstructured_mesh(
            base_nodes, jitter_field, float(amplitude), mesh_spacing
        )
        _, stats = mesh_quality_statistics(nodes, elems)
        candidates.append(
            {"jitter_amplitude": float(amplitude), "quality_p05": stats["quality_p05"]}
        )

    p5_values = np.asarray([row["quality_p05"] for row in candidates])
    targets = np.linspace(float(np.max(p5_values)), float(np.min(p5_values)), n_groups)
    available = set(range(len(candidates)))
    selected_indices: list[int] = []
    for target in targets:
        index = min(available, key=lambda i: abs(p5_values[i] - target))
        selected_indices.append(index)
        available.remove(index)

    selected: list[dict[str, object]] = []
    for target, index in zip(targets, selected_indices):
        amplitude = candidates[index]["jitter_amplitude"]
        nodes, elems = make_unstructured_mesh(
            base_nodes, jitter_field, amplitude, mesh_spacing
        )
        quality, stats = mesh_quality_statistics(nodes, elems)
        selected.append(
            {
                "nodes": nodes,
                "elems": elems,
                "quality": quality,
                "quality_stats": stats,
                "jitter_amplitude": amplitude,
                "target_quality_p05": float(target),
            }
        )

    # Return from highest to lowest P5 quality.
    selected.sort(
        key=lambda row: float(row["quality_stats"]["quality_p05"]), reverse=True
    )
    return selected


def save_mesh_plot(
    nodes: np.ndarray,
    elems: np.ndarray,
    quality: np.ndarray,
    stats: dict[str, float],
    out_png: Path,
    params: benchmark.PlateParams,
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    triangulation = mtri.Triangulation(nodes[:, 0], nodes[:, 1], elems)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2), dpi=180)

    full = axes[0].tripcolor(
        triangulation,
        facecolors=quality,
        shading="flat",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        # A grayscale string is compatible with older Matplotlib versions;
        # older tripcolor() implementations call .lower() on this argument.
        edgecolors="0.25",
        linewidth=0.10,
    )
    axes[0].set_title("Full mesh")
    axes[0].set_xlim(0.0, params.a)
    axes[0].set_ylim(0.0, params.b)

    axes[1].tripcolor(
        triangulation,
        facecolors=quality,
        shading="flat",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        edgecolors="0.10",
        linewidth=0.35,
    )
    half_width = 0.30
    axes[1].set_xlim(0.5 * params.a - half_width, 0.5 * params.a + half_width)
    axes[1].set_ylim(0.5 * params.b - half_width, 0.5 * params.b + half_width)
    axes[1].set_title("Center detail")

    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")

    colorbar_ax = fig.add_axes([0.92, 0.16, 0.018, 0.68])
    fig.colorbar(full, cax=colorbar_ax, label="Triangle mean-ratio quality")
    fig.suptitle(
        f"{elems.shape[0]:,}-element mesh: P5={stats['quality_p05']:.4f}, "
        f"min={stats['quality_min']:.4f}, mean={stats['quality_mean']:.4f}",
        y=0.98,
    )
    fig.subplots_adjust(left=0.07, right=0.86, bottom=0.10, top=0.88, wspace=0.20)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def save_quality_error_plot(df: pd.DataFrame, out_png: Path) -> None:
    import matplotlib.pyplot as plt

    ordered = df.sort_values("quality_p05")
    x = ordered["quality_p05"].to_numpy(dtype=float)
    percent = 100.0

    fig, ax = plt.subplots(figsize=(6.2, 4.9), dpi=180)
    ax.plot(x, percent * ordered["L2_w"], marker="o", linewidth=2.0, label=r"$w$ $L^2$ norm")
    ax.plot(
        x,
        percent * ordered["theta_l2_error"],
        marker="D",
        linewidth=2.0,
        label=r"$\theta$ $L^2$ norm",
    )
    ax.plot(x, percent * ordered["H1_w"], marker="s", linewidth=2.0, label=r"$w$ $H^1$ seminorm")
    ax.plot(x, percent * ordered["energy_error"], marker="^", linewidth=2.0, label="Energy norm")
    ax.set_xlabel(r"Mesh quality $q_{5\%}$")
    ax.set_ylabel("Error (%)")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{value:.3f}" for value in x])
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def run_quality_study(
    out_dir: Path,
    seed: int,
    jitter_min: float = DEFAULT_JITTER_MIN,
    jitter_max: float = DEFAULT_JITTER_MAX,
    candidate_count: int = DEFAULT_CANDIDATE_COUNT,
) -> pd.DataFrame:
    params = benchmark.PlateParams(
        a=5.0,
        b=5.0,
        thickness=0.5,
        E=1000.0,
        nu=0.3,
        p0=0.8,
        load_type="sinusoidal",
    )
    n = DEFAULT_N
    lattice_nodes, lattice_elems = benchmark.make_square_tri_mesh(n, params.a, params.b)
    if int(lattice_elems.shape[0]) != 20_000:
        raise RuntimeError(f"Expected exactly 20,000 triangles, got {lattice_elems.shape[0]}.")
    mesh_spacing = params.a / n
    base_nodes = build_relaxed_reference_points(
        lattice_nodes=lattice_nodes,
        a=params.a,
        b=params.b,
        mesh_spacing=mesh_spacing,
        seed=seed,
    )
    jitter_field = build_random_jitter_field(
        base_nodes,
        params.a,
        params.b,
        mesh_spacing,
        seed + 1,
    )
    random_meshes = select_evenly_spaced_quality_meshes(
        base_nodes=base_nodes,
        jitter_field=jitter_field,
        mesh_spacing=mesh_spacing,
        jitter_min=jitter_min,
        jitter_max=jitter_max,
        candidate_count=candidate_count,
    )
    regular_quality, regular_stats = mesh_quality_statistics(lattice_nodes, lattice_elems)
    regular_mesh = {
        "nodes": lattice_nodes,
        "elems": lattice_elems,
        "quality": regular_quality,
        "quality_stats": regular_stats,
        "jitter_amplitude": 0.0,
        "target_quality_p05": regular_stats["quality_p05"],
        "mesh_generation": "structured_right_isosceles",
        "reference_randomization": 0.0,
        "reference_relaxation_steps": 0,
        "reference_relaxation_weight": 0.0,
    }
    # Preserve the existing randomized construction and its four lower-quality
    # levels, while forcing group 1 to be the regular right-isosceles mesh.
    selected_meshes = [regular_mesh, *random_meshes[1:]]
    navier = benchmark.build_navier_solution(params)
    rows: list[dict[str, float | int | str]] = []

    out_dir.mkdir(parents=True, exist_ok=True)
    for group_index, mesh_data in enumerate(selected_meshes, start=1):
        nodes = np.asarray(mesh_data["nodes"], dtype=np.float64)
        elems = np.asarray(mesh_data["elems"], dtype=np.int64)
        quality = np.asarray(mesh_data["quality"], dtype=np.float64)
        quality_stats = dict(mesh_data["quality_stats"])
        jitter_amplitude = float(mesh_data["jitter_amplitude"])
        target_quality_p05 = float(mesh_data["target_quality_p05"])
        mesh_generation = str(mesh_data.get("mesh_generation", "relaxed_random_points_delaunay"))
        reference_randomization = float(
            mesh_data.get("reference_randomization", REFERENCE_RANDOMIZATION)
        )
        reference_relaxation_steps = int(
            mesh_data.get("reference_relaxation_steps", REFERENCE_RELAXATION_STEPS)
        )
        reference_relaxation_weight = float(
            mesh_data.get("reference_relaxation_weight", REFERENCE_RELAXATION_WEIGHT)
        )
        if int(elems.shape[0]) != 20_000:
            raise RuntimeError(f"Expected exactly 20,000 triangles, got {elems.shape[0]}.")

        group_name = f"quality_{group_index:02d}_p5_{quality_stats['quality_p05']:.4f}"
        group_dir = out_dir / group_name
        group_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"\n=== Quality group {group_index}/{N_QUALITY_GROUPS}: "
            f"jitter={jitter_amplitude:.4f} cells, target P5={target_quality_p05:.4f}, "
            f"actual P5={quality_stats['quality_p05']:.4f} ==="
        )
        np.savez_compressed(
            group_dir / "mesh.npz",
            nodes=nodes,
            elems=elems,
            element_mean_ratio_quality=quality,
            mesh_generation=np.asarray(mesh_generation),
            reference_randomization=np.asarray(reference_randomization),
            reference_relaxation_steps=np.asarray(reference_relaxation_steps),
            reference_relaxation_weight=np.asarray(reference_relaxation_weight),
            jitter_amplitude=np.asarray(jitter_amplitude),
            target_quality_p05=np.asarray(target_quality_p05),
            random_seed=np.asarray(seed),
        )
        save_mesh_plot(nodes, elems, quality, quality_stats, group_dir / "mesh_quality.png", params)

        start = time.perf_counter()
        solution = benchmark.solve_fem(nodes, elems, params, navier)
        errors = benchmark.compute_relative_errors(nodes, elems, solution, params, navier)
        elapsed = time.perf_counter() - start
        benchmark.save_comparison_plots(
            nodes,
            elems,
            solution,
            navier,
            params,
            group_dir / "compare_fields.png",
        )

        row: dict[str, float | int | str] = {
            "quality_group": group_index,
            "group_name": group_name,
            "num_nodes": int(nodes.shape[0]),
            "num_elements": int(elems.shape[0]),
            "mesh_generation": mesh_generation,
            "reference_randomization": reference_randomization,
            "reference_relaxation_steps": reference_relaxation_steps,
            "reference_relaxation_weight": reference_relaxation_weight,
            "jitter_amplitude": jitter_amplitude,
            "target_quality_p05": target_quality_p05,
            "quality_p05_target_error": quality_stats["quality_p05"] - target_quality_p05,
            "random_seed": seed,
            "solve_total_time_sec": elapsed,
            **quality_stats,
            **errors,
        }
        rows.append(row)
        pd.DataFrame([row]).to_csv(group_dir / "metrics.csv", index=False)
        print(
            f"L2_w={errors['L2_w']:.6e}, theta_L2={errors['theta_l2_error']:.6e}, "
            f"H1_w={errors['H1_w']:.6e}, energy={errors['energy_error']:.6e}"
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "fem_mesh_quality_errors.csv", index=False)
    df.to_excel(out_dir / "fem_mesh_quality_errors.xlsx", index=False)
    save_quality_error_plot(df, out_dir / "fem_mesh_quality_error_curves.png")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Five-level unstructured-mesh quality study for the 20,000-element "
            "thick square plate."
        )
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).parent / "navier_mr_plate_mesh_quality_results" / "thickness_0.5",
    )
    parser.add_argument(
        "--jitter-min",
        type=float,
        default=DEFAULT_JITTER_MIN,
        help="Smallest candidate interior-point jitter, in mesh spacings.",
    )
    parser.add_argument(
        "--jitter-max",
        type=float,
        default=DEFAULT_JITTER_MAX,
        help="Largest candidate interior-point jitter, in mesh spacings.",
    )
    parser.add_argument(
        "--candidate-count",
        type=int,
        default=DEFAULT_CANDIDATE_COUNT,
        help="Number of candidate meshes scanned before selecting five evenly spaced P5 values.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260909,
        help="Point-jitter seed; use a different seed to create another mesh realization.",
    )
    args = parser.parse_args()

    results = run_quality_study(
        args.out_dir,
        args.seed,
        jitter_min=args.jitter_min,
        jitter_max=args.jitter_max,
        candidate_count=args.candidate_count,
    )
    print("\nMesh-quality study completed:")
    print(results[["quality_group", "quality_p05", "quality_min", "quality_mean", "L2_w"]].to_string(index=False))
    print(f"Results saved under: {args.out_dir}")


if __name__ == "__main__":
    main()
