from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import sys

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from NO.problem.lshape import (
    build_lshape_dataset,
    save_debug_mesh_plot,
    save_encoder_pc_plot,
)
from shared.common.data_io import prepare_dataset_and_cases
from shared.common.paths import build_paths, path_dict_to_str
from shared.prep.lshape_grfm_grfp import DEFAULT_CONFIG as GRFM_GRFP_CONFIG


CASE_NAME = "lshape_um_grfp"

# Paired material-ablation case for lshape_grfm_grfp. Geometry, mesh,
# pressure-field statistics, sample counts, and seed remain identical.
DEFAULT_CONFIG: Dict[str, Any] = dict(GRFM_GRFP_CONFIG)
for _key in (
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
):
    DEFAULT_CONFIG.pop(_key, None)
DEFAULT_CONFIG.update(
    {
        "E": 1000.0,
        "nu": 0.3,
        "material_type": "uniform",
        "use_material_field_input": False,
    }
)


def prepare_lshape_um_grfp(
    *,
    root: Optional[str | Path] = None,
    overwrite: bool = True,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Prepare a uniform-material L-shaped plate with GRF pressure.

    With the default configuration, its geometry and pressure fields are
    paired sample-by-sample with ``lshape_grfm_grfp``. FEM cases contain only
    scalar E and nu, and the neural operator skips its material-field encoder.
    """
    cfg = dict(DEFAULT_CONFIG)
    if config is not None:
        cfg.update(config)

    paths = build_paths(CASE_NAME, root=root, create=True)

    print("=" * 80)
    print(f"[Prepare][{CASE_NAME}] Building uniform-material L-shaped dataset")
    print(f"[Prepare][{CASE_NAME}] Project root : {paths['project_root']}")
    print(f"[Prepare][{CASE_NAME}] Data root    : {paths['data_root']}")
    print(f"[Prepare][{CASE_NAME}] Case root    : {paths['case_root']}")
    print(f"[Prepare][{CASE_NAME}] Config:")
    for key, value in cfg.items():
        print(f"  {key}: {value}")
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
        E_mean=float(cfg["E"]),
        nu_mean=float(cfg["nu"]),
        thickness=float(cfg["thickness"]),
        pressure_value=float(cfg["pressure_mean"]),
        pressure_min=float(cfg["pressure_min"]),
        pressure_max=float(cfg["pressure_max"]),
        pressure_mean=float(cfg["pressure_mean"]),
        pressure_std=float(cfg["pressure_std"]),
        pressure_corr_len=float(cfg["pressure_corr_len"]),
        pressure_n_modes=int(cfg["pressure_n_modes"]),
        load_type=str(cfg["load_type"]),
        use_load_field_input=bool(cfg["use_load_field_input"]),
        use_material_field_input=False,
        mesh_lc=float(cfg["mesh_lc"]),
        mesh_type=str(cfg["mesh_type"]),
        mesh_nx=int(cfg["mesh_nx"]),
        mesh_ny=int(cfg["mesh_ny"]),
        boundary_point_spacing=float(cfg["boundary_point_spacing"]),
    )

    debug_dir = paths["data_root"] / "debug_mesh"
    debug_dir.mkdir(parents=True, exist_ok=True)
    for i in range(min(int(cfg["n_debug_mesh"]), len(train_samples))):
        try:
            out_path = debug_dir / f"debug_sample_{i:02d}.png"
            save_debug_mesh_plot(train_samples[i], str(out_path))
            print(f"[Prepare][{CASE_NAME}] saved debug mesh: {out_path}")
        except Exception as exc:
            print(f"[Prepare][{CASE_NAME}][Warning] debug mesh {i} failed: {exc}")

        try:
            out_path = debug_dir / f"debug_encoder_pc_{i:02d}.png"
            save_encoder_pc_plot(train_samples[i], str(out_path))
            print(f"[Prepare][{CASE_NAME}] saved encoder point cloud: {out_path}")
        except Exception as exc:
            print(f"[Prepare][{CASE_NAME}][Warning] encoder plot {i} failed: {exc}")

    info = prepare_dataset_and_cases(
        case_name=CASE_NAME,
        train_samples=train_samples,
        test_samples=test_samples,
        root=root,
        config=cfg,
        overwrite=overwrite,
    )
    info["paths"] = path_dict_to_str(paths)

    print("=" * 80)
    print(f"[Prepare][{CASE_NAME}] Dataset and FEM cases exported")
    print(f"[Prepare][{CASE_NAME}] n_train = {info['n_train']}")
    print(f"[Prepare][{CASE_NAME}] n_test  = {info['n_test']}")
    print("=" * 80)
    return info


if __name__ == "__main__":
    prepare_lshape_um_grfp()
