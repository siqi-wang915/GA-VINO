from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from NO.problem.holes2 import (
    build_holes2_dataset,
    save_debug_mesh_plot,
    save_encoder_pc_plot,
)
from shared.common.data_io import prepare_dataset_and_cases
from shared.common.paths import build_paths, path_dict_to_str


DEFAULT_CONFIG: Dict[str, Any] = {
    "n_train": 500,
    "n_test": 100,

    # Geometry, scaled from L=1 to L=5.
    "lx": 5.0,
    "ly": 5.0,

    "outer_n_per_edge": 100,
    "hole_n_points": 96,

    # Two-hole geometry, scaled by 5 from the original L=1 setup.
    "mean_radius_min": 0.40,
    "mean_radius_max": 0.75,
    "r_min": 0.20,
    "r_max": 1.15,
    "roughness": 0.4,
    "n_modes": 2,

    "center_min": 1.25,
    "center_max": 3.75,
    "center_distance_min": 0.25,

    "mesh_lc": 0.175,

    "E": 1000.0,
    "nu": 0.3,
    "thickness": 0.5,

    # Loading: Gaussian random field pressure.
    "pressure_min": 0.0,
    "pressure_max": 1.60,
    "pressure_mean": 0.80,
    "pressure_std": 0.3,
    "pressure_corr_len": 1.5,
    "pressure_n_modes": 64,

    "target_merged_ratio": 0.50,
    "balance_merge_states": False,
    "max_retry": 50,

    "seed": 42,

    # Debug figures
    "n_debug_mesh": 20,
    "load_type": "gaussian_random_field_pressure",
    "use_load_field_input": True,
}

def prepare_holes2_um_grfp(
    *,
    root: Optional[str | Path] = None,
    overwrite: bool = True,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    if config is not None:
        cfg.update(config)


    case_name = "holes2_um_grfp"
    paths = build_paths(case_name, root=root, create=True)

    print("=" * 80)
    print("[Prepare][holes2_um_grfp] Building two-hole plate dataset")
    print(f"[Prepare][holes2_um_grfp] Project root : {paths['project_root']}")
    print(f"[Prepare][holes2_um_grfp] Data root    : {paths['data_root']}")
    print(f"[Prepare][holes2_um_grfp] Case root    : {paths['case_root']}")
    print("[Prepare][holes2_um_grfp] Config:")
    for k, v in cfg.items():
        print(f"  {k}: {v}")
    print("=" * 80)

    train_samples, test_samples = build_holes2_dataset(
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
        center_min=float(cfg["center_min"]),
        center_max=float(cfg["center_max"]),
        center_distance_min=float(cfg["center_distance_min"]),
        mesh_lc=float(cfg["mesh_lc"]),
        target_merged_ratio=float(cfg["target_merged_ratio"]),
        balance_merge_states=bool(cfg["balance_merge_states"]),
        max_retry=int(cfg["max_retry"]),
        load_type=str(cfg["load_type"]),
        use_load_field_input=bool(cfg["use_load_field_input"]),
    )

    debug_dir = paths["data_root"] / "debug_mesh"
    debug_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Prepare][holes2] Saving debug mesh figures to: {debug_dir}")
    for i in range(min(int(cfg["n_debug_mesh"]), len(train_samples))):
        state = train_samples[i].get("hole_merge_state", "unknown")

        try:
            out_path = debug_dir / f"debug_sample_{i:02d}_{state}.png"
            save_debug_mesh_plot(train_samples[i], str(out_path))
            print(f"[Prepare][holes2] saved debug mesh: {out_path}")
        except Exception as e:
            print(f"[Prepare][holes2][Warning] Failed debug mesh {i}: {e}")

        try:
            out_path = debug_dir / f"debug_encoder_pc_{i:02d}_{state}.png"
            save_encoder_pc_plot(train_samples[i], str(out_path))
            print(f"[Prepare][holes2] saved encoder pc plot: {out_path}")
        except Exception as e:
            print(f"[Prepare][holes2][Warning] Failed encoder pc plot {i}: {e}")

    info = prepare_dataset_and_cases(
        case_name=case_name,
        train_samples=train_samples,
        test_samples=test_samples,
        root=root,
        config=cfg,
        overwrite=overwrite,
    )

    print("=" * 80)
    print("[Prepare][holes2] Dataset and FEM cases exported")
    print(f"[Prepare][holes2] n_train = {info['n_train']}")
    print(f"[Prepare][holes2] n_test  = {info['n_test']}")
    print(f"[Prepare][holes2] train samples: {paths['train_samples_path']}")
    print(f"[Prepare][holes2] test samples : {paths['test_samples_path']}")
    print(f"[Prepare][holes2] manifest     : {paths['dataset_manifest_path']}")
    print(f"[Prepare][holes2] cases        : {paths['case_root']}")
    print("=" * 80)

    info["paths"] = path_dict_to_str(paths)
    return info
