from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from NO.problem.hole import (
    build_hole_dataset,
    save_debug_mesh_plot,
    save_encoder_pc_plot,
)
from shared.common.data_io import prepare_dataset_and_cases
from shared.common.paths import build_paths, path_dict_to_str


DEFAULT_CONFIG: Dict[str, Any] = {
    # Dataset size
    "n_train": 500,  # 500
    "n_test": 100,  # 100

    # Geometry
    "lx": 5.0,
    "ly": 5.0,
    "outer_n_per_edge": 100, #64
    "hole_n_points": 128, #64

    # Hole shape, scaled from the original L=1 case by a factor of 5.
    "mean_radius_min": 0.40,
    "mean_radius_max": 0.75,
    "r_min": 0.2,
    "r_max": 1.15,
    "roughness": 0.40,  # dimensionless perturbation amplitude
    "n_modes": 2,

    "center_min": 1.25,     
    "center_max": 3.75,       
    

    # Mesh, scaled from 0.055 by a factor of 5.
    "mesh_lc": 0.175,

    # Material
    "E": 1000.0,
    "nu": 0.3,
    "thickness": 0.5,

    # Loading: Gaussian random field pressure.
    # pressure_mean is the target spatial mean. pressure_min / pressure_max are
    # clipping bounds and normalization bounds for the network.
    "pressure_min": 0.0,
    "pressure_max": 1.60,
    "pressure_mean": 0.80,
    "pressure_std": 0.3,
    "pressure_corr_len": 1.5,
    "pressure_n_modes": 64,

    # Random seed
    "seed": 42,

    # Debug figures
    "n_debug_mesh": 20,

    "load_type": "gaussian_random_field_pressure",
    "use_load_field_input": True,
}


def prepare_hole_um_grfp(
    *,
    root: Optional[str | Path] = None,
    overwrite: bool = True,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Stage-1 preparation for the single-hole plate case.

    This function:
        1. generates train/test samples;
        2. saves samples to shared/data/hole_um_grfp/;
        3. exports test FEM cases to shared/cases/hole_um_grfp/;
        4. saves dataset_manifest and tag files.

    It does NOT run FEM. FEM is called by scripts/stage1.py.
    """
    cfg = dict(DEFAULT_CONFIG)
    if config is not None:
        cfg.update(config)


    case_name = "hole_um_grfp"
    paths = build_paths(case_name, root=root, create=True)

    print("=" * 80)
    print("[Prepare][hole] Building single-hole plate dataset")
    print(f"[Prepare][hole] Project root : {paths['project_root']}")
    print(f"[Prepare][hole] Data root    : {paths['data_root']}")
    print(f"[Prepare][hole] Case root    : {paths['case_root']}")
    print("[Prepare][hole] Config:")
    for k, v in cfg.items():
        print(f"  {k}: {v}")
    print("=" * 80)

    train_samples, test_samples = build_hole_dataset(
        n_train=int(cfg["n_train"]),
        n_test=int(cfg["n_test"]),
        seed=int(cfg["seed"]),
        lx=float(cfg["lx"]),
        ly=float(cfg["ly"]),
        E=float(cfg["E"]),
        nu=float(cfg["nu"]),
        thickness=float(cfg["thickness"]),
        pressure_min=float(cfg["pressure_min"]),
        pressure_max=float(cfg["pressure_max"]),
        pressure_mean=float(cfg["pressure_mean"]),
        pressure_std=float(cfg["pressure_std"]),
        pressure_corr_len=float(cfg["pressure_corr_len"]),
        pressure_n_modes=int(cfg["pressure_n_modes"]),
        outer_n_per_edge=int(cfg["outer_n_per_edge"]),
        hole_n_points=int(cfg["hole_n_points"]),
        mean_radius_min=float(cfg["mean_radius_min"]),
        mean_radius_max=float(cfg["mean_radius_max"]),
        r_min=float(cfg["r_min"]),
        r_max=float(cfg["r_max"]),
        roughness=float(cfg["roughness"]),
        n_modes=int(cfg["n_modes"]),
        mesh_lc=float(cfg["mesh_lc"]),
        center_min=float(cfg["center_min"]),
        center_max=float(cfg["center_max"]),
        load_type=str(cfg["load_type"]),
        use_load_field_input=bool(cfg["use_load_field_input"]),
    )

    debug_dir = paths["data_root"] / "debug_mesh"
    debug_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Prepare][hole] Saving debug mesh figures to: {debug_dir}")

    for i in range(min(int(cfg["n_debug_mesh"]), len(train_samples))):
        try:
            out_path = debug_dir / f"debug_sample_{i:02d}.png"
            save_debug_mesh_plot(train_samples[i], str(out_path))
            print(f"[Prepare][hole] saved debug mesh: {out_path}")
        except Exception as e:
            print(f"[Prepare][hole][Warning] Failed to save debug mesh {i}: {e}")

        try:
            out_path = debug_dir / f"debug_encoder_pc_{i:02d}.png"
            save_encoder_pc_plot(train_samples[i], str(out_path))
            print(f"[Prepare][hole] saved encoder pc plot: {out_path}")
        except Exception as e:
            print(f"[Prepare][hole][Warning] Failed to save encoder pc plot {i}: {e}")

    info = prepare_dataset_and_cases(
        case_name=case_name,
        train_samples=train_samples,
        test_samples=test_samples,
        root=root,
        config=cfg,
        overwrite=overwrite,
    )

    print("=" * 80)
    print("[Prepare][hole] Dataset and FEM cases exported")
    print(f"[Prepare][hole] n_train = {info['n_train']}")
    print(f"[Prepare][hole] n_test  = {info['n_test']}")
    print(f"[Prepare][hole] train samples: {paths['train_samples_path']}")
    print(f"[Prepare][hole] test samples : {paths['test_samples_path']}")
    print(f"[Prepare][hole] manifest     : {paths['dataset_manifest_path']}")
    print(f"[Prepare][hole] cases        : {paths['case_root']}")
    print("=" * 80)

    info["paths"] = path_dict_to_str(paths)

    return info
