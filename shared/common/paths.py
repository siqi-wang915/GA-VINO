from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional


# -----------------------------------------------------------------------------
# Short case names used in the new project
# -----------------------------------------------------------------------------
# square = 完整方板 + 均布压力
# hole   = 单孔方板 + 固定均布压力
# holes2 = 双孔方板 + 固定均布压力
# -----------------------------------------------------------------------------

CASE_DESCRIPTIONS = {
    # Experiment cases
    "hole_um_up": "single-hole plate, uniform material, uniform pressure",
    "hole_um_up_t005": "single-hole plate, h=0.05, uniform material, p=8e-4",
    "hole_um_up_t0158": "single-hole plate, h=0.158, uniform material, p=2.53e-2",
    "hole_um_up_t00158": "single-hole plate, h=0.0158, uniform material, p=2.53e-5",
    "hole_um_up_t0005": "single-hole plate, h=0.005, uniform material, p=8e-7",
    "hole_um_grfp": "single-hole plate, uniform material, GRF pressure",
    "holes2_um_up": "two-hole plate, uniform material, uniform pressure",
    "holes2_um_grfp": "two-hole plate, uniform material, GRF pressure",
    "lshape_grfm_up": "L-shaped plate, GRF material, uniform pressure",
    "lshape_grfm_grfp": "L-shaped plate, GRF material, GRF pressure",
    "lshape_um_grfp": "L-shaped plate, uniform material, GRF pressure",
    "lshape_grfm_up_065_085": "L-shaped plate, GRF material, uniform pressure, large cutout OOD",
    "annular_sector_grfm_grfp": "annular-sector plate, GRF material, random nonuniform boundary line load",
    "hole_um_up_mesh_005": "single-hole plate mesh-convergence study, 5% retained vertices",
    "hole_um_up_mesh_012": "single-hole plate mesh-convergence study, 12.5% retained vertices",
    "hole_um_up_mesh_025": "single-hole plate mesh-convergence study, 25% retained vertices",
    "hole_um_up_mesh_050": "single-hole plate mesh-convergence study, 50% retained vertices",
    "hole_um_up_mesh_075": "single-hole plate mesh-convergence study, 75% retained vertices",
    "hole_um_up_mesh_quality_02": "single-hole plate mesh-quality study, quality level 2",
    "hole_um_up_mesh_quality_03": "single-hole plate mesh-quality study, quality level 3",
    "hole_um_up_mesh_quality_04": "single-hole plate mesh-quality study, quality level 4",
}


CASE_ALIASES = {

    # Experiment cases
    "hole_um_up": "hole_um_up",
    "hole_uniform_pressure": "hole_um_up",
    "hole_uniform_load": "hole_um_up",
    "hole_uniform_material_uniform_pressure": "hole_um_up",
    "hole_um_up_t005": "hole_um_up_t005",
    "hole_um_up_t0158": "hole_um_up_t0158",
    "hole_um_up_t00158": "hole_um_up_t00158",
    "hole_um_up_t0005": "hole_um_up_t0005",

    "hole_um_grfp": "hole_um_grfp",
    "hole_grfp": "hole_um_grfp",
    "hole_grf_pressure": "hole_um_grfp",
    "hole_uniform_material_grf_pressure": "hole_um_grfp",

    "holes2_um_up": "holes2_um_up",
    "holes2_um_grfp": "holes2_um_grfp",
    "holes2_grfp": "holes2_um_grfp",
    "twoholes_um_grfp": "holes2_um_grfp",
    "two_holes_um_grfp": "holes2_um_grfp",

    "lshape_grfm_up": "lshape_grfm_up",
    "lshape_grf_material_uniform_pressure": "lshape_grfm_up",

    "lshape_grfm_grfp": "lshape_grfm_grfp",
    "lshape_grf_material_grf_pressure": "lshape_grfm_grfp",

    "lshape_um_grfp": "lshape_um_grfp",
    "lshape_uniform_material_grf_pressure": "lshape_um_grfp",

    "lshape_up_ood": "lshape_grfm_up_065_085",
    "lshape_grfm_up_cut065_085": "lshape_grfm_up_065_085",

    "annular_sector": "annular_sector_grfm_grfp",
    "annular_sector_grfm_grfp": "annular_sector_grfm_grfp",
    "sector_grfm_grfp": "annular_sector_grfm_grfp",
    "annular_sector_grfp": "annular_sector_grfm_grfp",

    "hole_um_up_mesh_005": "hole_um_up_mesh_005",
    "hole_um_up_mesh_012": "hole_um_up_mesh_012",
    "hole_um_up_mesh_025": "hole_um_up_mesh_025",
    "hole_um_up_mesh_050": "hole_um_up_mesh_050",
    "hole_um_up_mesh_075": "hole_um_up_mesh_075",
    "hole_um_up_mesh_quality_02": "hole_um_up_mesh_quality_02",
    "hole_um_up_mesh_quality_03": "hole_um_up_mesh_quality_03",
    "hole_um_up_mesh_quality_04": "hole_um_up_mesh_quality_04",
}

def project_root() -> Path:
    """
    Infer project root from this file location.

    Expected file path:
        <project_root>/shared/common/paths.py
    """
    return Path(__file__).resolve().parents[2]


def normalize_case_name(case_name: str) -> str:
    """
    Validate and return the canonical case name.
    """
    key = str(case_name).strip().lower().replace("-", "_")

    if key in CASE_ALIASES:
        return CASE_ALIASES[key]

    if key in CASE_DESCRIPTIONS:
        return key

    supported = ", ".join(sorted(CASE_DESCRIPTIONS.keys()))
    raise ValueError(
        f"Unsupported case_name='{case_name}'. "
        f"Supported case names are: {supported}"
    )


def supported_cases() -> list[str]:
    return sorted(CASE_DESCRIPTIONS.keys())


def case_description(case_name: str) -> str:
    case_name = normalize_case_name(case_name)
    return CASE_DESCRIPTIONS[case_name]


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_paths(
    case_name: str,
    root: Optional[str | Path] = None,
    create: bool = True,
) -> Dict[str, Path]:
    """
    Build all standard paths for one case.

    Project layout:

        <root>/
        ├─ NO/
        │  └─ outputs/<case>/
        │     └─ predictions/
        │
        ├─ FEM/
        │  └─ outputs/<case>/
        │     └─ predictions/
        │
        ├─ shared/
        │  ├─ data/<case>/
        │  └─ cases/<case>/
        │
        └─ compare/<case>/
    """
    case = normalize_case_name(case_name)

    root_path = Path(root).resolve() if root is not None else project_root()

    no_root = root_path / "NO"
    fem_root = root_path / "FEM"
    shared_root = root_path / "shared"
    compare_root_all = root_path / "compare"

    data_root = shared_root / "data" / case
    case_root = shared_root / "cases" / case

    no_output_root = no_root / "outputs" / case
    no_prediction_root = no_output_root / "predictions"

    fem_output_root = fem_root / "outputs" / case
    fem_prediction_root = fem_output_root / "predictions"

    compare_root = compare_root_all / case

    paths: Dict[str, Path] = {
        "project_root": root_path,

        "no_root": no_root,
        "fem_root": fem_root,
        "shared_root": shared_root,

        "data_root": data_root,
        "case_root": case_root,

        "no_output_root": no_output_root,
        "no_prediction_root": no_prediction_root,

        "fem_output_root": fem_output_root,
        "fem_prediction_root": fem_prediction_root,

        "compare_root": compare_root,

        # Dataset files
        "train_samples_path": data_root / "train_samples.pkl",
        "test_samples_path": data_root / "test_samples.pkl",
        "dataset_manifest_path": data_root / "dataset_manifest.npz",
        "dataset_config_path": data_root / "dataset_config.json",
        "test_case_tags_path": data_root / "test_case_tags.txt",
        "train_case_tags_path": data_root / "train_case_tags.txt",

        # NO outputs
        "best_model_path": no_output_root / "best_mr_plate_operator.pt",
        "final_model_path": no_output_root / "final_mr_plate_operator.pt",
        "training_history_path": no_output_root / "training_history.npz",
        "loss_curve_path": no_output_root / "loss_curve.png",

        # FEM-error history during NO training
        "fem_error_history_path": no_output_root / "fem_error_history.npz",
        "l2_error_epoch_plot_path": no_output_root / "L2_error_curve.png",
        "h1_error_epoch_plot_path": no_output_root / "H1_error_curve.png",
        "energy_error_epoch_plot_path": no_output_root / "Energy_error_curve.png",
        "theta_error_epoch_plot_path": no_output_root / "Theta_error_curve.png",
        "curvature_error_epoch_plot_path": no_output_root / "Curvature_error_curve.png",

        # Final comparison metrics
        "final_metrics_npz_path": compare_root / "final_test_error_metrics.npz",
        "final_metrics_csv_path": compare_root / "final_test_error_metrics.csv",
    }

    if create:
        for key in (
            "data_root",
            "case_root",
            "no_output_root",
            "no_prediction_root",
            "fem_output_root",
            "fem_prediction_root",
            "compare_root",
        ):
            ensure_dir(paths[key])

    return paths


def case_npz_path(paths: Dict[str, Path], case_tag: str) -> Path:
    return paths["case_root"] / f"{case_tag}.npz"


def no_pred_npz_path(paths: Dict[str, Path], case_tag: str) -> Path:
    return paths["no_prediction_root"] / f"{case_tag}_no.npz"


def fem_pred_npz_path(paths: Dict[str, Path], case_tag: str) -> Path:
    return paths["fem_prediction_root"] / f"{case_tag}_fem.npz"


def path_dict_to_str(paths: Dict[str, Path]) -> Dict[str, str]:
    return {k: str(v) for k, v in paths.items()}
