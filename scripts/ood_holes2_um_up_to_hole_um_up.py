from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import argparse
import json
import sys

import torch
from torch.utils.data import DataLoader

_THIS_FILE = Path(__file__).resolve()

# When this file is placed under scripts/, project root is its parent directory.
# Fallback to current working directory for ad-hoc execution.
_PROJECT_ROOT = _THIS_FILE.parents[1] if _THIS_FILE.parent.name == "scripts" else Path.cwd()

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
    return obj


def run_ood_cross_case(
    *,
    root: Optional[str | Path] = None,
    source_case: str = "holes2_um_up",
    target_case: str = "hole_um_up",
    source_method_name: Optional[str] = None,
    target_method_name: str = "ood_from_holes2_um_up",
    checkpoint_name: str = "best",
    batch_size: Optional[int] = None,
    deformation_scale: float = 10.0,
    max_plot_cases: int = 5,
) -> Dict[str, Any]:
    """
    Zero-shot cross-geometry OOD evaluation.

    Source model:
        NO/outputs/<source_case>/best_mr_plate_operator.pt or final_mr_plate_operator.pt

    Target prediction folder:
        NO/outputs/<target_case>_<target_method_name>/predictions/

    Target comparison folder:
        compare/<target_case>_<target_method_name>/

    The target case must already have:
        shared/data/<target_case>/
        shared/cases/<target_case>/
        FEM/outputs/<target_case>/predictions/
    """
    root_path = Path(root).resolve() if root is not None else _PROJECT_ROOT.resolve()

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
            f"Please rerun stage2 training for source_case={source_case!r}."
        )

    ckpt_key = checkpoint_name.lower()
    if ckpt_key == "best":
        ckpt_path = Path(source_paths["best_model_path"])
    elif ckpt_key == "final":
        ckpt_path = Path(source_paths["final_model_path"])
    else:
        ckpt_path = Path(checkpoint_name)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    padding_value = float(mr_plate_ginot_minimal_configs()["padding_value"])

    print("=" * 80)
    print("[OOD] zero-shot cross-geometry evaluation")
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

    # 1. Rebuild exactly the source training model architecture.
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

    # 3. Load target test samples.
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

    # 4. Export OOD predictions to method-specific target folder.
    saved_tags = export_no_predictions(
        operator_model=operator_model,
        data_loader=target_test_loader,
        case_name=target_case,
        case_tags=target_test_tags,
        root=root_path,
        padding_value=padding_value,
        method_name=target_method_name,
    )

    # 5. Compare against target FEM results.
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
        "ood_type": "zero_shot_cross_geometry",
        "source_case": source_case,
        "target_case": target_case,
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


def run_ood_holes2_um_up_to_hole_um_up(
    *,
    root: Optional[str | Path] = None,
    source_method_name: Optional[str] = None,
    target_method_name: str = "ood_from_holes2_um_up",
    checkpoint_name: str = "best",
    batch_size: Optional[int] = None,
    deformation_scale: float = 10.0,
    max_plot_cases: int = 5,
) -> Dict[str, Any]:
    return run_ood_cross_case(
        root=root,
        source_case="holes2_um_up",
        target_case="hole_um_up",
        source_method_name=source_method_name,
        target_method_name=target_method_name,
        checkpoint_name=checkpoint_name,
        batch_size=batch_size,
        deformation_scale=deformation_scale,
        max_plot_cases=max_plot_cases,
    )


def main():
    parser = argparse.ArgumentParser(description="OOD: source holes2_um_up model predicts target hole_um_up test samples.")
    parser.add_argument("--root", type=str, default=None, help="Project root. Default: inferred from scripts/ location.")
    parser.add_argument("--source-method-name", type=str, default=None, help="Source method name. Usually None/base.")
    parser.add_argument("--target-method-name", type=str, default="ood_from_holes2_um_up", help="Method name suffix for target OOD output.")
    parser.add_argument("--checkpoint", type=str, default="best", choices=["best", "final"], help="Source checkpoint to load.")
    parser.add_argument("--batch-size", type=int, default=None, help="Target prediction batch size. Default: source training batch size.")
    parser.add_argument("--deformation-scale", type=float, default=10.0)
    parser.add_argument("--max-plot-cases", type=int, default=5)
    args = parser.parse_args()

    run_ood_holes2_um_up_to_hole_um_up(
        root=args.root,
        source_method_name=args.source_method_name,
        target_method_name=args.target_method_name,
        checkpoint_name=args.checkpoint,
        batch_size=args.batch_size,
        deformation_scale=args.deformation_scale,
        max_plot_cases=args.max_plot_cases,
    )


if __name__ == "__main__":
    main()
