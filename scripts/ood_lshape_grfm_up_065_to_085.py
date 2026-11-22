from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import argparse
import json
import sys

import torch
from torch.utils.data import DataLoader

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[1]

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from NO.common.core import (
    PlateSampleDataset,
    MRPlateEnergyOperator,
    build_plate_operator_model,
    device,
    mr_plate_ginot_minimal_configs,
    plate_collate_fn,
)
from NO.common.evaluator import export_no_predictions
from NO.common.trainer import build_energy_output_paths
from shared.common.data_io import load_dataset, load_test_tags
from shared.compare.compare import compare_cases


def _load_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON metadata file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _to_jsonable(obj):
    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, torch.Tensor):
        arr = obj.detach().cpu()
        if arr.numel() == 1:
            return float(arr.reshape(-1)[0].item())
        return arr.tolist()

    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]

    try:
        import numpy as np

        if isinstance(obj, np.ndarray):
            if obj.shape == ():
                return obj.item()
            return obj.tolist()

        if isinstance(obj, (np.integer,)):
            return int(obj)

        if isinstance(obj, (np.floating,)):
            return float(obj)
    except Exception:
        pass

    return obj


def run_ood_lshape_grfm_up_025065_to_065085(
    *,
    root: Optional[str | Path] = None,
    source_case: str = "lshape_grfm_up",
    target_case: str = "lshape_grfm_up_065_085",
    source_method_name: Optional[str] = None,
    target_method_name: str = "ood_from_lshape_grfm_up_025065",
    checkpoint_name: str = "best",
    batch_size: Optional[int] = None,
    deformation_scale: float = 10.0,
    max_plot_cases: int = 5,
) -> Dict[str, Any]:
    """
    Zero-shot OOD evaluation for L-shaped plate geometry extrapolation.

    Train/source distribution:
        lshape_grfm_up
        cut_w/lx, cut_h/ly in [0.25, 0.65]

    Target/OOD distribution:
        lshape_grfm_up_065_085
        cut_w/lx, cut_h/ly in [0.65, 0.85]

    This script:
        1. reads source run_metadata.json;
        2. rebuilds the exact source model architecture;
        3. loads the source checkpoint;
        4. predicts on target test samples;
        5. compares with existing target FEM results.

    Required before running:
        - source case has completed stage2 training:
            NO/outputs/lshape_grfm_up/run_metadata.json
            NO/outputs/lshape_grfm_up/best_mr_plate_operator.pt

        - target case has completed stage1 data + FEM:
            shared/data/lshape_grfm_up_065_085/
            shared/cases/lshape_grfm_up_065_085/
            FEM/outputs/lshape_grfm_up_065_085/predictions/
    """
    root_path = Path(root).resolve() if root is not None else _PROJECT_ROOT

    source_paths = build_energy_output_paths(
        case_name=source_case,
        root=root_path,
        method_name=source_method_name,
        create=False,
    )

    target_paths = build_energy_output_paths(
        case_name=target_case,
        root=root_path,
        method_name=target_method_name,
        create=True,
    )

    metadata_path = Path(source_paths["no_output_root"]) / "run_metadata.json"
    metadata = _load_json(metadata_path)

    model_config = metadata.get("model_config", None)
    if model_config is None:
        raise RuntimeError(
            f"model_config is missing in {metadata_path}. "
            f"Please rerun Stage2 training for source_case='{source_case}' "
            "after enabling run_metadata saving."
        )

    checkpoint_key = str(checkpoint_name).lower()
    if checkpoint_key == "best":
        ckpt_path = Path(source_paths["best_model_path"])
    elif checkpoint_key == "final":
        ckpt_path = Path(source_paths["final_model_path"])
    else:
        ckpt_path = Path(checkpoint_name)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    padding_value = float(mr_plate_ginot_minimal_configs()["padding_value"])

    print("=" * 80)
    print("[OOD] L-shape GRFM-UP 0.25~0.65 -> 0.65~0.85 zero-shot evaluation")
    print(f"[OOD] project root      : {root_path}")
    print(f"[OOD] source case       : {source_case}")
    print(f"[OOD] target case       : {target_case}")
    print(f"[OOD] source method     : {source_method_name}")
    print(f"[OOD] target method     : {target_method_name}")
    print(f"[OOD] source metadata   : {metadata_path}")
    print(f"[OOD] source checkpoint : {ckpt_path}")
    print(f"[OOD] target NO outputs : {target_paths['no_prediction_root']}")
    print(f"[OOD] target compare dir: {target_paths['compare_root']}")
    print("=" * 80)

    # 1. Rebuild model using the exact source architecture.
    base_model = build_plate_operator_model(
        padding_value=padding_value,
        model_config=model_config,
    ).to(device)

    operator_model = MRPlateEnergyOperator(
        model=base_model,
        padding_value=padding_value,
    ).to(device)

    # 2. Load source trained weights.
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    operator_model.load_state_dict(state)
    operator_model.eval()

    # 3. Load target OOD test samples.
    _, target_test_samples, target_manifest = load_dataset(
        case_name=target_case,
        root=root_path,
    )

    target_test_tags = load_test_tags(
        case_name=target_case,
        root=root_path,
    )

    if batch_size is None:
        batch_size = int(metadata.get("training_config", {}).get("batch_size", 1))

    target_test_ds = PlateSampleDataset(target_test_samples)
    target_test_loader = DataLoader(
        target_test_ds,
        batch_size=int(batch_size),
        shuffle=False,
        collate_fn=lambda b: plate_collate_fn(b, padding_value=padding_value),
    )

    # 4. Export OOD NO predictions into:
    #    NO/outputs/<target_case>_<target_method_name>/predictions/
    saved_tags = export_no_predictions(
        operator_model=operator_model,
        data_loader=target_test_loader,
        case_name=target_case,
        case_tags=target_test_tags,
        root=root_path,
        padding_value=padding_value,
        method_name=target_method_name,
    )

    # 5. Compare against existing target FEM results:
    #    FEM/outputs/<target_case>/predictions/
    compare_info = compare_cases(
        case_name=target_case,
        case_tags=saved_tags,
        root=root_path,
        deformation_scale=float(deformation_scale),
        max_plot_cases=int(max_plot_cases),
        method_name=target_method_name,
    )

    # 6. Save OOD metadata.
    ood_metadata = {
        "ood_type": "zero_shot_lshape_geometry_extrapolation",
        "source_case": source_case,
        "target_case": target_case,
        "source_distribution": {
            "cut_w_ratio": [0.25, 0.65],
            "cut_h_ratio": [0.25, 0.65],
        },
        "target_distribution": {
            "cut_w_ratio": [0.65, 0.85],
            "cut_h_ratio": [0.65, 0.85],
        },
        "source_method_name": source_method_name,
        "target_method_name": target_method_name,
        "source_run_metadata": str(metadata_path),
        "source_checkpoint": str(ckpt_path),
        "target_no_prediction_root": str(target_paths["no_prediction_root"]),
        "target_compare_root": str(target_paths["compare_root"]),
        "target_manifest": _to_jsonable(target_manifest),
        "saved_tags": saved_tags,
        "compare_info": _to_jsonable(compare_info),
    }

    ood_metadata_path = Path(target_paths["no_output_root"]) / "ood_metadata.json"
    with open(ood_metadata_path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(ood_metadata), f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print("[OOD] Finished")
    print(f"[OOD] saved NO predictions: {target_paths['no_prediction_root']}")
    print(f"[OOD] compare results     : {target_paths['compare_root']}")
    print(f"[OOD] metadata            : {ood_metadata_path}")
    print("=" * 80)

    return {
        "status": "done",
        "source_case": source_case,
        "target_case": target_case,
        "source_checkpoint": str(ckpt_path),
        "target_method_name": target_method_name,
        "saved_tags": saved_tags,
        "compare_info": compare_info,
        "ood_metadata_path": str(ood_metadata_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OOD evaluation: lshape_grfm_up 0.25~0.65 -> 0.65~0.85."
    )

    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Project root. Defaults to the parent directory of scripts/.",
    )
    parser.add_argument(
        "--source-case",
        type=str,
        default="lshape_grfm_up",
        help="Source training case.",
    )
    parser.add_argument(
        "--target-case",
        type=str,
        default="lshape_grfm_up_065_085",
        help="Target OOD case.",
    )
    parser.add_argument(
        "--source-method-name",
        type=str,
        default=None,
        help="Optional source method name. Use None/base for NO/outputs/<source_case>/.",
    )
    parser.add_argument(
        "--target-method-name",
        type=str,
        default="ood_from_lshape_grfm_up_025065",
        help="Method suffix for OOD output folder.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best",
        choices=["best", "final"],
        help="Which checkpoint to use.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Prediction batch size. Defaults to source training batch_size from metadata.",
    )
    parser.add_argument(
        "--deformation-scale",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--max-plot-cases",
        type=int,
        default=5,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    source_method_name = args.source_method_name
    if source_method_name is not None and source_method_name.lower() in {"none", "null", "base", "default", ""}:
        source_method_name = None

    run_ood_lshape_grfm_up_025065_to_065085(
        root=args.root,
        source_case=args.source_case,
        target_case=args.target_case,
        source_method_name=source_method_name,
        target_method_name=args.target_method_name,
        checkpoint_name=args.checkpoint,
        batch_size=args.batch_size,
        deformation_scale=args.deformation_scale,
        max_plot_cases=args.max_plot_cases,
    )


if __name__ == "__main__":
    main()
