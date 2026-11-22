from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import argparse
import json
import sys

import numpy as np
import torch

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[1] if _THIS_FILE.parent.name == "scripts" else Path.cwd()

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from NO.common.trainer import build_energy_output_paths
from NO.train.train_lshape_grfm_up_fno import (
    FNO2d,
    LShapeFNOSupervisedDataset,
    export_fno_predictions,
)
from shared.common.data_io import load_dataset, load_test_tags
from shared.compare.compare import compare_cases


def _to_jsonable(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.Tensor):
        arr = obj.detach().cpu()
        if arr.numel() == 1:
            return float(arr.reshape(-1)[0].item())
        return arr.tolist()
    if isinstance(obj, np.ndarray):
        if obj.shape == ():
            return obj.item()
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def run_ood_dd_lshape_to_lshape_ood(
    *,
    root: Optional[str | Path] = None,
    source_case: str = "lshape_grfm_up",
    target_case: str = "lshape_grfm_up_065_085",
    source_method_name: str = "fno",
    target_method_name: str = "ood_dd_from_lshape_grfm_up_fno_025065",
    checkpoint_name: str = "best",
    deformation_scale: float = 10.0,
    max_plot_cases: int = 5,
) -> Dict[str, Any]:
    """
    Pure data-driven L-shape OOD evaluation with the supervised FNO baseline.

    Default experiment:
        source model : lshape_grfm_up_fno
        target data  : lshape_grfm_up_065_085

    Required source files by default:
        NO/outputs/lshape_grfm_up_fno/best_mr_plate_operator.pt
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

    ckpt_key = str(checkpoint_name).lower()
    if ckpt_key == "best":
        ckpt_path = Path(source_paths["best_model_path"])
    elif ckpt_key == "final":
        ckpt_path = Path(source_paths["final_model_path"])
    else:
        ckpt_path = Path(checkpoint_name)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if not isinstance(ckpt, dict) or "model_state" not in ckpt:
        raise RuntimeError(
            f"Expected an FNO checkpoint with key 'model_state', got: {ckpt_path}"
        )

    cfg = dict(ckpt.get("config", {}))
    target_mean = np.asarray(ckpt["target_mean"], dtype=np.float32).reshape(3)
    target_std = np.asarray(ckpt["target_std"], dtype=np.float32).reshape(3)

    _, target_test_samples, target_manifest = load_dataset(
        case_name=target_case,
        root=root_path,
    )
    target_test_tags = load_test_tags(case_name=target_case, root=root_path)
    target_ds = LShapeFNOSupervisedDataset(
        case_name=target_case,
        samples=target_test_samples,
        tags=target_test_tags,
        root=root_path,
        target_mean=target_mean,
        target_std=target_std,
    )

    in_channels = int(target_ds[0]["x"].shape[0])
    model = FNO2d(
        in_channels=in_channels,
        out_channels=3,
        width=int(cfg.get("width", 64)),
        modes_x=int(cfg.get("modes_x", 16)),
        modes_y=int(cfg.get("modes_y", 16)),
        depth=int(cfg.get("depth", 4)),
        hidden_channels=int(cfg.get("hidden_channels", 128)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print("=" * 80)
    print("[OOD-DD] L-shape FNO -> L-shape OOD zero-shot evaluation")
    print(f"[OOD-DD] project root      : {root_path}")
    print(f"[OOD-DD] source case       : {source_case}")
    print(f"[OOD-DD] target case       : {target_case}")
    print(f"[OOD-DD] source method     : {source_method_name}")
    print(f"[OOD-DD] target method     : {target_method_name}")
    print(f"[OOD-DD] source checkpoint : {ckpt_path}")
    print(f"[OOD-DD] target NO outputs : {target_paths['no_prediction_root']}")
    print(f"[OOD-DD] target compare dir: {target_paths['compare_root']}")
    print("=" * 80)

    saved_tags = export_fno_predictions(
        model=model,
        dataset=target_ds,
        case_name=target_case,
        root=root_path,
        method_name=target_method_name,
        device=device,
    )

    compare_info = compare_cases(
        case_name=target_case,
        case_tags=saved_tags,
        root=root_path,
        deformation_scale=float(deformation_scale),
        max_plot_cases=int(max_plot_cases),
        method_name=target_method_name,
    )

    ood_metadata = {
        "ood_type": "zero_shot_lshape_geometry_extrapolation",
        "training_type": "pure_data_driven_supervised_fno",
        "source_case": source_case,
        "target_case": target_case,
        "source_method_name": source_method_name,
        "target_method_name": target_method_name,
        "source_checkpoint": str(ckpt_path),
        "source_config": _to_jsonable(cfg),
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
    print("[OOD-DD] Finished")
    print(f"[OOD-DD] saved NO predictions: {target_paths['no_prediction_root']}")
    print(f"[OOD-DD] compare results     : {target_paths['compare_root']}")
    print(f"[OOD-DD] metadata            : {ood_metadata_path}")
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


def main():
    parser = argparse.ArgumentParser(
        description="Pure data-driven OOD: L-shape FNO model predicts L-shape OOD case."
    )
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--source-case", type=str, default="lshape_grfm_up")
    parser.add_argument("--target-case", type=str, default="lshape_grfm_up_065_085")
    parser.add_argument("--source-method-name", type=str, default="fno")
    parser.add_argument(
        "--target-method-name",
        type=str,
        default="ood_dd_from_lshape_grfm_up_fno_025065",
    )
    parser.add_argument("--checkpoint", type=str, default="best", choices=["best", "final"])
    parser.add_argument("--deformation-scale", type=float, default=10.0)
    parser.add_argument("--max-plot-cases", type=int, default=5)
    args = parser.parse_args()

    run_ood_dd_lshape_to_lshape_ood(
        root=args.root,
        source_case=args.source_case,
        target_case=args.target_case,
        source_method_name=args.source_method_name,
        target_method_name=args.target_method_name,
        checkpoint_name=args.checkpoint,
        deformation_scale=args.deformation_scale,
        max_plot_cases=args.max_plot_cases,
    )


if __name__ == "__main__":
    main()
