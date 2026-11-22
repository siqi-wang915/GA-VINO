from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from shared.common.data_io import prepare_dataset_and_cases
from shared.common.paths import build_paths, path_dict_to_str
from NO.problem.annular_sector import (
    build_annular_sector_dataset,
    save_debug_mesh_plot,
    save_encoder_pc_plot,
)


DEFAULT_CONFIG: Dict[str, Any] = {
    # Dataset size
    "n_train": 1000,
    "n_test": 200,

    # Annular-sector geometry.
    # theta_end is fixed at 90 deg, theta_start = theta_end - theta_span.
    "r_inner_min": 2.0,
    "r_inner_max": 3.0,
    "r_outer_min": 4.0,
    "r_outer_max": 5.0,
    "theta_span_deg_min": 45.0,
    "theta_span_deg_max": 90.0,
    "theta_end_deg": 90.0,

    # Structured polar triangular mesh.
    # mesh_lc controls the target physical element size. Each sample computes
    # its own mesh_nr/mesh_nt from its radial thickness and outer arc length.
    # The geometry encoder uses all FEM boundary nodes as its boundary point
    # cloud; boundary_point_spacing is kept only for old config compatibility.
    "mesh_lc": 0.05,
    "mesh_nr_min": 8,
    "mesh_nt_min": 16,
    "boundary_point_spacing": 0.10,

    # Material: elementwise Gaussian random fields.
    "E_mean": 1000.0,
    "E_std": 400.0,
    "E_min": 200.0,
    "E_max": 1800.0,
    "E_corr_len": 1.5,
    "E_n_modes": 64,

    "nu_mean": 0.30,
    "nu_std": 0.08,
    "nu_min": 0.10,
    "nu_max": 0.45,
    "nu_corr_len": 1.5,
    "nu_n_modes": 64,

    "thickness": 0.5,

    # Loading: smooth random nonuniform transverse line load on the free radial
    # edge at theta_start.  The domain has no transverse pressure.  A mean of
    # 0.020 gives approximately the same deflection scale as the old p=0.015
    # domain-pressure case, while log_std=0.85 creates strong spatial variation.
    "line_load_mean": 0.020,
    "line_load_min": 0.0,
    "line_load_max": 0.060,
    "line_load_log_std": 0.85,
    "line_load_corr_len": 0.30,

    "load_type": "random_nonuniform_boundary_line_load",
    "use_load_field_input": True,

    # Random seed
    "seed": 42,

    # Debug figures
    "n_debug_mesh": 20,
}


def prepare_annular_sector_grfm_grfp(
    *,
    root: Optional[str | Path] = None,
    overwrite: bool = True,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Stage-1 preparation for the annular-sector plate case with GRF material
    and a random nonuniform transverse line load.

    Geometry:
        center = (0, 0)
        r_inner in [2, 3]
        r_outer in [4, 5]
        theta_end = 90 deg
        theta_start = 90 deg - theta_span
        theta_span in [45 deg, 90 deg]

    Boundary:
        x = 0 radial edge clamped; the opposite radial edge carries the
        prescribed transverse line load; both circular arcs are free.
    """
    cfg = dict(DEFAULT_CONFIG)
    if config is not None:
        cfg.update(config)

    case_name = "annular_sector_grfm_grfp"
    paths = build_paths(case_name, root=root, create=True)

    print("=" * 80)
    print("[Prepare][annular_sector_grfm_grfp] Building annular-sector plate dataset")
    print(f"[Prepare][annular_sector_grfm_grfp] Project root : {paths['project_root']}")
    print(f"[Prepare][annular_sector_grfm_grfp] Data root    : {paths['data_root']}")
    print(f"[Prepare][annular_sector_grfm_grfp] Case root    : {paths['case_root']}")
    print("[Prepare][annular_sector_grfm_grfp] Config:")
    for k, v in cfg.items():
        print(f"  {k}: {v}")
    print("=" * 80)

    train_samples, test_samples = build_annular_sector_dataset(
        n_train=int(cfg["n_train"]),
        n_test=int(cfg["n_test"]),
        seed=int(cfg["seed"]),
        r_inner_min=float(cfg["r_inner_min"]),
        r_inner_max=float(cfg["r_inner_max"]),
        r_outer_min=float(cfg["r_outer_min"]),
        r_outer_max=float(cfg["r_outer_max"]),
        theta_span_deg_min=float(cfg["theta_span_deg_min"]),
        theta_span_deg_max=float(cfg["theta_span_deg_max"]),
        theta_end_deg=float(cfg["theta_end_deg"]),
        mesh_lc=float(cfg["mesh_lc"]),
        mesh_nr_min=int(cfg["mesh_nr_min"]),
        mesh_nt_min=int(cfg["mesh_nt_min"]),
        boundary_point_spacing=float(cfg["boundary_point_spacing"]),
        E_mean=float(cfg["E_mean"]),
        E_std=float(cfg["E_std"]),
        E_min=float(cfg["E_min"]),
        E_max=float(cfg["E_max"]),
        E_corr_len=float(cfg["E_corr_len"]),
        E_n_modes=int(cfg["E_n_modes"]),
        nu_mean=float(cfg["nu_mean"]),
        nu_std=float(cfg["nu_std"]),
        nu_min=float(cfg["nu_min"]),
        nu_max=float(cfg["nu_max"]),
        nu_corr_len=float(cfg["nu_corr_len"]),
        nu_n_modes=int(cfg["nu_n_modes"]),
        thickness=float(cfg["thickness"]),
        line_load_mean=float(cfg["line_load_mean"]),
        line_load_min=float(cfg["line_load_min"]),
        line_load_max=float(cfg["line_load_max"]),
        line_load_log_std=float(cfg["line_load_log_std"]),
        line_load_corr_len=float(cfg["line_load_corr_len"]),
        load_type=str(cfg["load_type"]),
        use_load_field_input=bool(cfg["use_load_field_input"]),
    )

    debug_dir = paths["data_root"] / "debug_mesh"
    debug_dir.mkdir(parents=True, exist_ok=True)

    all_for_debug = train_samples if train_samples else test_samples
    print(f"[Prepare][annular_sector_grfm_grfp] Saving debug mesh figures to: {debug_dir}")
    for i in range(min(int(cfg["n_debug_mesh"]), len(all_for_debug))):
        try:
            out_path = debug_dir / f"debug_sample_{i:02d}.png"
            save_debug_mesh_plot(all_for_debug[i], out_path)
            print(f"[Prepare][annular_sector_grfm_grfp] saved debug mesh: {out_path}")
        except Exception as e:
            print(f"[Prepare][annular_sector_grfm_grfp][Warning] Failed to save debug mesh {i}: {e}")

        try:
            out_path = debug_dir / f"debug_encoder_pc_{i:02d}.png"
            save_encoder_pc_plot(all_for_debug[i], out_path)
            print(f"[Prepare][annular_sector_grfm_grfp] saved encoder pc plot: {out_path}")
        except Exception as e:
            print(f"[Prepare][annular_sector_grfm_grfp][Warning] Failed to save encoder pc plot {i}: {e}")

    info = prepare_dataset_and_cases(
        case_name=case_name,
        train_samples=train_samples,
        test_samples=test_samples,
        root=root,
        config=cfg,
        overwrite=overwrite,
    )

    print("=" * 80)
    print("[Prepare][annular_sector_grfm_grfp] Dataset and FEM cases exported")
    print(f"[Prepare][annular_sector_grfm_grfp] n_train = {info['n_train']}")
    print(f"[Prepare][annular_sector_grfm_grfp] n_test  = {info['n_test']}")
    print(f"[Prepare][annular_sector_grfm_grfp] train samples: {paths['train_samples_path']}")
    print(f"[Prepare][annular_sector_grfm_grfp] test samples : {paths['test_samples_path']}")
    print(f"[Prepare][annular_sector_grfm_grfp] manifest     : {paths['dataset_manifest_path']}")
    print(f"[Prepare][annular_sector_grfm_grfp] cases        : {paths['case_root']}")
    print("=" * 80)

    info["paths"] = path_dict_to_str(paths)
    return info


if __name__ == "__main__":
    prepare_annular_sector_grfm_grfp()
