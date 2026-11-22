from __future__ import annotations

"""Unified preparation entry point for the five hole-plate thickness studies.

The selected case name is deliberately kept equal to the historical case name.
Consequently, selecting a thickness never changes the data, FEM, NO, or
comparison output-directory convention.
"""

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from NO.problem.hole import (
    build_hole_dataset,
    save_debug_mesh_plot,
    save_encoder_pc_plot,
)
from shared.common.data_io import prepare_dataset_and_cases
from shared.common.paths import build_paths, path_dict_to_str


DEFAULT_CONFIG: Dict[str, Any] = {
    "n_train": 500,
    "n_test": 100,
    "lx": 5.0,
    "ly": 5.0,
    "outer_n_per_edge": 100,
    "hole_n_points": 128,
    "mean_radius_min": 0.40,
    "mean_radius_max": 0.75,
    "r_min": 0.2,
    "r_max": 1.15,
    "roughness": 0.40,
    "n_modes": 2,
    "center_min": 1.25,
    "center_max": 3.75,
    "mesh_lc": 0.175,
    "E": 1000.0,
    "nu": 0.3,
    "thickness": 0.5,
    "pressure_min": 0.0,
    "pressure_max": 1.60,
    "pressure_mean": 0.80,
    "pressure_std": 0.3,
    "pressure_corr_len": 1.5,
    "pressure_n_modes": 64,
    "seed": 42,
    "n_debug_mesh": 20,
    "load_type": "uniform_pressure",
    "use_load_field_input": False,
}


# ``pressure_mean`` is the uniform distributed load used by both FEM and the
# energy loss.  pressure_max remains twice the mean, matching the old cases.
THICKNESS_CASES: Dict[str, Dict[str, Any]] = {
    "base": {
        "case_name": "hole_um_up",
        "thickness": 0.5,
        "pressure_mean": 8.0e-1,
    },
    "t005": {
        "case_name": "hole_um_up_t005",
        "thickness": 5.0e-2,
        "pressure_mean": 8.0e-4,
    },
    "t0158": {
        "case_name": "hole_um_up_t0158",
        "thickness": 1.58e-1,
        "pressure_mean": 2.53e-2,
    },
    "t00158": {
        "case_name": "hole_um_up_t00158",
        "thickness": 1.58e-2,
        "pressure_mean": 2.53e-5,
    },
    "t0005": {
        "case_name": "hole_um_up_t0005",
        "thickness": 5.0e-3,
        "pressure_mean": 8.0e-7,
    },
}

_SELECTOR_ALIASES = {
    "": "base",
    "base": "base",
    "t05": "base",
    "0.5": "base",
    "5e_1": "base",
    "t005": "t005",
    "0.05": "t005",
    "5e_2": "t005",
    "t0158": "t0158",
    "0.158": "t0158",
    "1.58e_1": "t0158",
    "t00158": "t00158",
    "0.0158": "t00158",
    "1.58e_2": "t00158",
    "t0005": "t0005",
    "0.005": "t0005",
    "5e_3": "t0005",
}


def resolve_thickness_case(selector: str = "base") -> Tuple[str, Dict[str, Any]]:
    """Return the canonical suffix and immutable thickness-case definition."""
    key = str(selector).strip().lower().replace("-", "_")
    if key == "hole_um_up":
        key = "base"
    elif key.startswith("hole_um_up_"):
        # Keep compatibility with the Python 3.8 environments used by this
        # project (str.removeprefix was added only in Python 3.9).
        key = key[len("hole_um_up_") :]

    suffix = _SELECTOR_ALIASES.get(key)
    if suffix is None:
        supported = ", ".join(THICKNESS_CASES)
        raise ValueError(
            f"Unsupported hole_um_up_mr thickness selector '{selector}'. "
            f"Use one of: {supported}."
        )
    return suffix, dict(THICKNESS_CASES[suffix])


def prep_config_for_thickness(selector: str = "base") -> Tuple[str, Dict[str, Any]]:
    """Build the old per-case preparation configuration for one suffix."""
    suffix, definition = resolve_thickness_case(selector)
    pressure_mean = float(definition["pressure_mean"])
    cfg = deepcopy(DEFAULT_CONFIG)
    cfg.update(
        {
            "thickness": float(definition["thickness"]),
            "pressure_min": 0.0,
            "pressure_mean": pressure_mean,
            "pressure_max": 2.0 * pressure_mean,
        }
    )
    return suffix, cfg


def prepare_hole_um_up(
    *,
    root: Optional[str | Path] = None,
    overwrite: bool = True,
    config: Optional[Dict[str, Any]] = None,
    case_name: str = "hole_um_up",
) -> Dict[str, Any]:
    """Generate one single-hole dataset and export its FEM case files."""
    cfg = deepcopy(DEFAULT_CONFIG)
    if config is not None:
        cfg.update(config)

    case_name = str(case_name)
    paths = build_paths(case_name, root=root, create=True)
    print("=" * 80)
    print(f"[Prepare][{case_name}] Building single-hole plate dataset")
    print(f"[Prepare][{case_name}] Project root : {paths['project_root']}")
    print(f"[Prepare][{case_name}] Data root    : {paths['data_root']}")
    print(f"[Prepare][{case_name}] Case root    : {paths['case_root']}")
    print(f"[Prepare][{case_name}] Config:")
    for key, value in cfg.items():
        print(f"  {key}: {value}")
    print("=" * 80)

    train_samples, test_samples = build_hole_dataset(
        n_train=int(cfg["n_train"]), n_test=int(cfg["n_test"]),
        seed=int(cfg["seed"]), lx=float(cfg["lx"]), ly=float(cfg["ly"]),
        E=float(cfg["E"]), nu=float(cfg["nu"]), thickness=float(cfg["thickness"]),
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
        r_min=float(cfg["r_min"]), r_max=float(cfg["r_max"]),
        roughness=float(cfg["roughness"]), n_modes=int(cfg["n_modes"]),
        mesh_lc=float(cfg["mesh_lc"]), center_min=float(cfg["center_min"]),
        center_max=float(cfg["center_max"]), load_type=str(cfg["load_type"]),
        use_load_field_input=bool(cfg["use_load_field_input"]),
    )

    debug_dir = paths["data_root"] / "debug_mesh"
    debug_dir.mkdir(parents=True, exist_ok=True)
    for index in range(min(int(cfg["n_debug_mesh"]), len(train_samples))):
        try:
            out_path = debug_dir / f"debug_sample_{index:02d}.png"
            save_debug_mesh_plot(train_samples[index], str(out_path))
        except Exception as exc:
            print(f"[Prepare][hole][Warning] Failed to save debug mesh {index}: {exc}")
        try:
            out_path = debug_dir / f"debug_encoder_pc_{index:02d}.png"
            save_encoder_pc_plot(train_samples[index], str(out_path))
        except Exception as exc:
            print(f"[Prepare][hole][Warning] Failed to save encoder pc plot {index}: {exc}")

    info = prepare_dataset_and_cases(
        case_name=case_name, train_samples=train_samples, test_samples=test_samples,
        root=root, config=cfg, overwrite=overwrite,
    )
    info["paths"] = path_dict_to_str(paths)
    return info


def prepare_hole_um_up_mr(
    selector: str = "base",
    *,
    root: Optional[str | Path] = None,
    overwrite: bool = True,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Prepare one of the five thickness cases selected by its CLI suffix."""
    suffix, cfg = prep_config_for_thickness(selector)
    _, definition = resolve_thickness_case(suffix)
    if config is not None:
        cfg.update(config)

    print(
        f"[hole_um_up_mr] suffix={suffix}, h={cfg['thickness']}, "
        f"uniform pressure={cfg['pressure_mean']}"
    )
    return prepare_hole_um_up(
        root=root,
        overwrite=overwrite,
        config=cfg,
        case_name=str(definition["case_name"]),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare one hole_um_up thickness case."
    )
    parser.add_argument(
        "suffix",
        nargs="?",
        default="base",
        help="Thickness suffix: base, t005, t0158, t00158, or t0005.",
    )
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--no-overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    prepare_hole_um_up_mr(
        args.suffix,
        root=args.root,
        overwrite=not args.no_overwrite,
    )
