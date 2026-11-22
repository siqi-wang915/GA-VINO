from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# These settings must be present before petsc4py initializes PETSc.  This
# benchmark is deliberately single-rank and single-GPU.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
if "-use_gpu_aware_mpi" not in os.environ.get("PETSC_OPTIONS", ""):
    existing_options = os.environ.get("PETSC_OPTIONS", "").strip()
    os.environ["PETSC_OPTIONS"] = (
        f"{existing_options} -use_gpu_aware_mpi 0".strip()
    )

from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
import petsc4py

from benchmarks.hole_gpu.gpu_solver import (
    SUPPORTED_GPU_CASES,
    require_petsc_strumpack,
    solve_one_gpu_case,
)
from shared.common.paths import build_paths, normalize_case_name


SUPPORTED_CASES = SUPPORTED_GPU_CASES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone plate FEM GPU benchmark."
    )
    parser.add_argument(
        "--case-name",
        choices=SUPPORTED_CASES,
        default="hole_um_up",
        help="Input case family (default: hole_um_up).",
    )
    parser.add_argument(
        "--case-tags",
        nargs="+",
        default=None,
        help="Case tags without .npz. If omitted, case_*.npz files are used.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Limit the number of selected cases.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of unreported warm-up solves using the first case.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="Measured repetitions per case.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=THIS_FILE.parent / "outputs",
        help="Independent benchmark output directory.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if MPI.COMM_WORLD.size != 1:
        raise RuntimeError("This GPU benchmark requires exactly one MPI rank.")
    require_petsc_strumpack()
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")
    if args.repeat < 1:
        raise ValueError("--repeat must be >= 1")
    if args.max_cases is not None and args.max_cases < 1:
        raise ValueError("--max-cases must be >= 1")



def gpu_name() -> str:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        return "; ".join(names) if names else "unknown"
    except (FileNotFoundError, subprocess.SubprocessError):
        return "unavailable"


def petsc_configure_options() -> str:
    """Best-effort PETSc configuration discovery across petsc4py versions."""
    getter = getattr(PETSc.Sys, "getConfigureOptions", None)
    if getter is not None:
        try:
            return str(getter())
        except Exception:
            pass

    try:
        config = petsc4py.get_config()
        if config:
            return json.dumps(config, sort_keys=True)
    except Exception:
        pass

    return "unavailable"


def environment_info(case_name: str) -> dict[str, Any]:
    return {
        "case_name": case_name,
        "python": sys.version.split()[0],
        "dolfinx": dolfinx.__version__,
        "petsc": ".".join(str(x) for x in PETSc.Sys.getVersion()),
        "petsc_configure_options": petsc_configure_options(),
        "gpu": gpu_name(),
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "mpi_ranks": MPI.COMM_WORLD.size,
    }


def selected_cases(args: argparse.Namespace) -> list[tuple[str, Path]]:
    case_name = normalize_case_name(args.case_name)
    paths = build_paths(case_name, root=PROJECT_ROOT, create=False)
    case_root = paths["case_root"]

    if args.case_tags is None:
        tags = sorted(path.stem for path in case_root.glob("case_*.npz"))
    else:
        tags = [tag[:-4] if tag.endswith(".npz") else tag for tag in args.case_tags]

    if args.max_cases is not None:
        tags = tags[: args.max_cases]
    if not tags:
        raise RuntimeError(f"No cases selected under {case_root}")

    selected = [(tag, case_root / f"{tag}.npz") for tag in tags]
    missing = [str(path) for _, path in selected if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing case files:\n" + "\n".join(missing))
    return selected


def solve_once(
    *, case_name: str, tag: str, case_path: Path, out_path: Path
) -> dict[str, Any]:
    wall_start = time.perf_counter()
    info = solve_one_gpu_case(
        case_path=case_path,
        out_path=out_path,
        case_name=case_name,
    )
    wall_time = time.perf_counter() - wall_start
    return {
        "case_name": case_name,
        "case_tag": tag,
        "wall_time_sec": wall_time,
        "fem_total_time_sec": info["fem_total_time_sec"],
        # LinearProblem.solve() includes assembly/BC work as well as KSPSolve.
        "assembly_and_solve_time_sec": info["fem_solve_time_sec"],
        "ksp_iterations": info["ksp_iterations"],
        "ksp_residual_norm": info["ksp_residual_norm"],
        "ksp_converged_reason": info["ksp_converged_reason"],
        "n_vertices": info["n_vertices"],
        "n_faces": info["n_faces"],
        "n_query_points": info["n_query_points"],
        "max_abs_w": info["max_abs_w"],
        "center_uz": info.get("center_uz", float("nan")),
        "output_path": str(out_path),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    groups = dict.fromkeys((row["case_name"], row["case_tag"]) for row in rows)
    for case_name, tag in groups:
        group = [
            row
            for row in rows
            if row["case_name"] == case_name and row["case_tag"] == tag
        ]
        for field in (
            "wall_time_sec",
            "fem_total_time_sec",
            "assembly_and_solve_time_sec",
        ):
            values = [float(row[field]) for row in group]
            summaries.append(
                {
                    "case_name": case_name,
                    "case_tag": tag,
                    "metric": field,
                    "count": len(values),
                    "min_sec": min(values),
                    "mean_sec": statistics.fmean(values),
                    "median_sec": statistics.median(values),
                }
            )
    return summaries


def main() -> None:
    args = parse_args()
    validate_args(args)
    cases = selected_cases(args)
    output_dir = args.output_dir.resolve()
    prediction_dir = output_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)

    env = environment_info(args.case_name)
    environment_path = output_dir / f"environment_{args.case_name}.json"
    runs_path = output_dir / f"benchmark_runs_{args.case_name}.csv"
    summary_path = output_dir / f"benchmark_summary_{args.case_name}.csv"
    environment_path.write_text(
        json.dumps(env, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"[Benchmark] case_name : {args.case_name}")
    print(f"[Benchmark] cases     : {len(cases)}")
    print(f"[Benchmark] warmup    : {args.warmup}")
    print(f"[Benchmark] repeat    : {args.repeat}")
    print(f"[Benchmark] GPU list  : {env['gpu']}")
    print(f"[Benchmark] GPU used  : CUDA_VISIBLE_DEVICES={env['cuda_visible_devices']}")
    print(f"[Benchmark] outputs   : {output_dir}")

    warmup_tag, warmup_path = cases[0]
    for index in range(args.warmup):
        print(f"[Benchmark] warmup {index + 1}/{args.warmup}: {warmup_tag}")
        solve_once(
            case_name=args.case_name,
            tag=warmup_tag,
            case_path=warmup_path,
            out_path=(
                prediction_dir
                / f"{args.case_name}_{warmup_tag}_gpu_warmup.npz"
            ),
        )

    rows: list[dict[str, Any]] = []
    for tag, case_path in cases:
        out_path = prediction_dir / f"{args.case_name}_{tag}_gpu.npz"
        for repetition in range(1, args.repeat + 1):
            print(f"[Benchmark] measured {tag}: {repetition}/{args.repeat}")
            row = solve_once(
                case_name=args.case_name,
                tag=tag,
                case_path=case_path,
                out_path=out_path,
            )
            row = {"repetition": repetition, **row}
            rows.append(row)

    write_csv(runs_path, rows)
    summaries = summary_rows(rows)
    write_csv(summary_path, summaries)

    print("[Benchmark] completed")
    print(f"[Benchmark] environment: {environment_path}")
    print(f"[Benchmark] runs      : {runs_path}")
    print(f"[Benchmark] summary   : {summary_path}")


if __name__ == "__main__":
    main()
