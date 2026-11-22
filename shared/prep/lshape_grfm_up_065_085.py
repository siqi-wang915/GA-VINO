from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from NO.problem.lshape import (
    build_lshape_dataset,
    save_debug_mesh_plot,
    save_encoder_pc_plot,
)
from shared.common.data_io import prepare_dataset_and_cases
from shared.common.paths import build_paths, path_dict_to_str


DEFAULT_CONFIG: Dict[str, Any] = {
    # Dataset size
    "n_train": 0,
    "n_test": 100,

    # Outer rectangle size
    "lx_min": 5.0,
    "lx_max": 5.0,
    "ly_min": 5.0,
    "ly_max": 5.0,

    # Top-right cutout ratio
    "cut_w_ratio_min": 0.65,
    "cut_w_ratio_max": 0.85,
    "cut_h_ratio_min": 0.65,
    "cut_h_ratio_max": 0.85,

    # Mesh and boundary point cloud
    "mesh_lc": 0.1,
    "mesh_type": "structured",
    "mesh_nx": 100,
    "mesh_ny": 100,
    "boundary_point_spacing": 0.1,
    # Material: elementwise Gaussian random fields.
    # E and nu below are representative mean values retained only for
    # compatibility with old code and metadata.
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

    # Loading: sample-wise variable uniform transverse pressure.
    # Each sample uses one constant pressure value sampled from [0.01, 0.02].
    "pressure_value": 0.015,
    "pressure_min": 0.01,
    "pressure_max": 0.02,
    "sample_uniform_pressure": True,

    # Random seed
    "seed": 42,

    # Debug figures
    "n_debug_mesh": 20,

    "load_type": "uniform_pressure",
    "use_load_field_input": False,
}


def prepare_lshape_grfm_up_065_085(
    *,
    root: Optional[str | Path] = None,
    overwrite: bool = True,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Stage-1 preparation for the L-shaped plate case.

    This function:
        1. generates train/test samples;
        2. saves samples to shared/data/lshape_grfm_up/;
        3. exports test FEM cases to shared/cases/lshape/;
        4. saves dataset_manifest and tag files.

    It does NOT run FEM. FEM is called by scripts/stage1.py.
    """
    cfg = dict(DEFAULT_CONFIG)
    if config is not None:
        cfg.update(config)


    case_name = "lshape_grfm_up_065_085"
    paths = build_paths(case_name, root=root, create=True)

    print("=" * 80)
    print("[Prepare][lshape_grfm_up] Building L-shaped plate dataset")
    print(f"[Prepare][lshape_grfm_up] Project root : {paths['project_root']}")
    print(f"[Prepare][lshape_grfm_up] Data root    : {paths['data_root']}")
    print(f"[Prepare][lshape_grfm_up] Case root    : {paths['case_root']}")
    print("[Prepare][lshape_grfm_up] Config:")
    for k, v in cfg.items():
        print(f"  {k}: {v}")
    print("=" * 80)

    train_samples, test_samples = build_lshape_dataset(
        n_train=int(cfg["n_train"]),
        n_test=int(cfg["n_test"]),
        seed=int(cfg["seed"]),
        lx_min=float(cfg["lx_min"]),
        lx_max=float(cfg["lx_max"]),
        ly_min=float(cfg["ly_min"]),
        ly_max=float(cfg["ly_max"]),
        cut_w_ratio_min=float(cfg["cut_w_ratio_min"]),
        cut_w_ratio_max=float(cfg["cut_w_ratio_max"]),
        cut_h_ratio_min=float(cfg["cut_h_ratio_min"]),
        cut_h_ratio_max=float(cfg["cut_h_ratio_max"]),
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
        pressure_value=float(cfg["pressure_value"]),
        pressure_min=float(cfg["pressure_min"]),
        pressure_max=float(cfg["pressure_max"]),
        sample_uniform_pressure=bool(cfg["sample_uniform_pressure"]),
        mesh_lc=float(cfg["mesh_lc"]),
        mesh_type=str(cfg["mesh_type"]),
        mesh_nx=int(cfg["mesh_nx"]),
        mesh_ny=int(cfg["mesh_ny"]),
        boundary_point_spacing=float(cfg["boundary_point_spacing"]),
        load_type=str(cfg["load_type"]),
        use_load_field_input=bool(cfg["use_load_field_input"]),
    )

    debug_dir = paths["data_root"] / "debug_mesh"
    debug_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Prepare][lshape_grfm_up] Saving debug mesh figures to: {debug_dir}")
    for i in range(min(int(cfg["n_debug_mesh"]), len(train_samples))):
        try:
            out_path = debug_dir / f"debug_sample_{i:02d}.png"
            save_debug_mesh_plot(train_samples[i], str(out_path))
            print(f"[Prepare][lshape_grfm_up] saved debug mesh: {out_path}")
        except Exception as e:
            print(f"[Prepare][lshape_grfm_up][Warning] Failed to save debug mesh {i}: {e}")

        try:
            out_path = debug_dir / f"debug_encoder_pc_{i:02d}.png"
            save_encoder_pc_plot(train_samples[i], str(out_path))
            print(f"[Prepare][lshape_grfm_up] saved encoder pc plot: {out_path}")
        except Exception as e:
            print(f"[Prepare][lshape_grfm_up][Warning] Failed to save encoder pc plot {i}: {e}")

    info = prepare_dataset_and_cases(
        case_name=case_name,
        train_samples=train_samples,
        test_samples=test_samples,
        root=root,
        config=cfg,
        overwrite=overwrite,
    )

    print("=" * 80)
    print("[Prepare][lshape_grfm_up] Dataset and FEM cases exported")
    print(f"[Prepare][lshape_grfm_up] n_train = {info['n_train']}")
    print(f"[Prepare][lshape_grfm_up] n_test  = {info['n_test']}")
    print(f"[Prepare][lshape_grfm_up] train samples: {paths['train_samples_path']}")
    print(f"[Prepare][lshape_grfm_up] test samples : {paths['test_samples_path']}")
    print(f"[Prepare][lshape_grfm_up] manifest     : {paths['dataset_manifest_path']}")
    print(f"[Prepare][lshape_grfm_up] cases        : {paths['case_root']}")
    print("=" * 80)

    info["paths"] = path_dict_to_str(paths)
    return info
