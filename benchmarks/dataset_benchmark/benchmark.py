#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.common.scenario_utils import (
    RAYSPACE_DIR,
    canonical_cube_pair_paths,
    canonical_microns_aggregated_paths,
    canonical_nn_pair_paths,
    canonical_nu_pair_paths,
    create_benchmark_run_layout,
    ensure_cube_pair_dataset,
    get_shared_data_dirs,
    write_json,
)
from benchmarks.mesh_overlap.adapters.raytracer_adapter import RaytracerAdapter
from benchmarks.mesh_overlap.adapters.base import run_command_streaming


@dataclass(frozen=True)
class DatasetRow:
    dataset_id: str
    source_path: Path
    grid_cell_size: float


def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


def _human_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def _grid_token(grid_cell_size: float) -> str:
    return str(grid_cell_size).replace(".", "_")


def _run_preprocess(
    *,
    dataset: DatasetRow,
    preprocessed_dir: Path,
    timings_dir: Path,
    logs_dir: Path,
) -> Dict[str, float | int | str]:
    _log(f"[preprocess:{dataset.dataset_id}] START source={dataset.source_path} grid={dataset.grid_cell_size}")
    preprocess_exec = RAYSPACE_DIR / "preprocess" / "build" / "bin" / "preprocess_dataset"
    if not preprocess_exec.exists():
        raise FileNotFoundError(f"Preprocess executable not found: {preprocess_exec}")

    mode = "dt" if dataset.source_path.suffix == ".dt" else "mesh"
    pre_path = preprocessed_dir / f"{dataset.source_path.stem}_g{_grid_token(dataset.grid_cell_size)}.pre"
    timing_path = timings_dir / f"{dataset.source_path.stem}_g{_grid_token(dataset.grid_cell_size)}_timing.json"
    log_path = logs_dir / f"preprocess_{dataset.dataset_id}.log"

    cmd = [
        str(preprocess_exec),
        "--mode", mode,
        "--dataset", str(dataset.source_path),
        "--output-geometry", str(pre_path),
        "--output-timing", str(timing_path),
        "--generate-grid",
        "--grid-cell-size", str(dataset.grid_cell_size),
    ]
    _log(f"[preprocess:{dataset.dataset_id}] command: {' '.join(cmd)}")

    t0 = time.time()
    try:
        stdout_text, stderr_text = run_command_streaming(
            cmd,
            timeout=None,
            log_path=str(log_path),
            prefix=f"[preprocess:{dataset.dataset_id}]",
        )
    except subprocess.CalledProcessError as exc:
        elapsed = time.time() - t0
        combined_err = (exc.output or "") + (exc.stderr or "")
        _log(
            f"[preprocess:{dataset.dataset_id}] failed return_code={exc.returncode} "
            f"elapsed={elapsed:.2f}s log={log_path}"
        )
        raise RuntimeError(f"Preprocess failed for {dataset.dataset_id}: {combined_err}") from exc

    elapsed = time.time() - t0
    combined = (stdout_text or "") + (stderr_text or "")
    _log(
        f"[preprocess:{dataset.dataset_id}] finished return_code=0 "
        f"elapsed={elapsed:.2f}s log={log_path}"
    )

    objects = None
    triangles = None

    m_obj = re.search(r"Loaded tile with\s+(\d+)\s+objects\.", combined)
    if m_obj:
        objects = int(m_obj.group(1))
    else:
        m_obj2 = re.search(r"Successfully loaded\s+(\d+)\s+object\(s\)", combined)
        if m_obj2:
            objects = int(m_obj2.group(1))

    m_tri = re.search(r"Total triangles:\s*(\d+)", combined)
    if m_tri:
        triangles = int(m_tri.group(1))

    if objects is None:
        raise RuntimeError(f"Could not parse object count for {dataset.dataset_id}")
    if triangles is None:
        raise RuntimeError(f"Could not parse triangle count for {dataset.dataset_id}")

    if not timing_path.exists():
        raise RuntimeError(f"Timing json missing for {dataset.dataset_id}: {timing_path}")

    timing_payload = json.loads(timing_path.read_text(encoding="utf-8"))
    preprocess_ms = float(timing_payload.get("total", {}).get("duration_ms", 0.0))
    _log(
        f"[preprocess:{dataset.dataset_id}] parsed objects={objects} triangles={triangles} "
        f"preprocess_ms={preprocess_ms:.2f}"
    )

    return {
        "objects": objects,
        "triangles": triangles,
        "preprocess_ms": preprocess_ms,
        "preprocessed_path": str(pre_path),
        "timing_path": str(timing_path),
        "log_path": str(log_path),
    }


def _run_join_and_parse_loading(
    *,
    case_name: str,
    mesh1: Path,
    mesh2: Path,
    grid_cell_size: float,
    preprocessed_dir: Path,
    timings_dir: Path,
    logs_dir: Path,
) -> Tuple[float, float, str]:
    _log(f"[join:{case_name}] START mesh1={mesh1.name} mesh2={mesh2.name} grid={grid_cell_size}")
    case_log_dir = logs_dir / case_name
    case_log_dir.mkdir(parents=True, exist_ok=True)

    attempted_errors: List[str] = []
    for mode in ["direct_estimation", "exact", "estimated"]:
        _log(f"[join:{case_name}] trying mode={mode}")
        adapter = RaytracerAdapter(
            str(RAYSPACE_DIR),
            mode=mode,
            preprocessed_dir=str(preprocessed_dir),
            timings_dir=str(timings_dir),
            grid_cell_size=grid_cell_size,
            warmup_runs=1,
        )

        t0 = time.time()
        result = adapter.run_overlap(
            str(mesh1),
            str(mesh2),
            num_runs=1,
            timeout=1200.0,
            log_dir=str(case_log_dir),
            query_direction="both",
        )
        elapsed = time.time() - t0
        _log(f"[join:{case_name}] mode={mode} finished elapsed={elapsed:.2f}s")
        if "error" in result:
            attempted_errors.append(f"{mode}: {result['error']}")
            _log(f"[join:{case_name}] mode={mode} failed: {result['error']}")
            continue

        run_log = case_log_dir / f"Raytracer_{mode}" / "run_000.log"
        if not run_log.exists():
            attempted_errors.append(f"{mode}: missing log {run_log}")
            _log(f"[join:{case_name}] mode={mode} failed: missing log {run_log}")
            continue

        text = run_log.read_text(encoding="utf-8", errors="replace")
        m1 = re.search(r"Upload Mesh1:\s+\d+\s+microseconds\s+\(([0-9.]+)\s+ms\)", text)
        m2 = re.search(r"Upload Mesh2:\s+\d+\s+microseconds\s+\(([0-9.]+)\s+ms\)", text)
        if not m1 or not m2:
            attempted_errors.append(f"{mode}: could not parse upload timings from {run_log}")
            _log(f"[join:{case_name}] mode={mode} failed: upload timing parse failed")
            continue

        _log(
            f"[join:{case_name}] SUCCESS mode={mode} upload_mesh1_ms={float(m1.group(1)):.2f} "
            f"upload_mesh2_ms={float(m2.group(1)):.2f} log={run_log}"
        )
        return float(m1.group(1)), float(m2.group(1)), str(run_log)

    raise RuntimeError(f"Join failed for {case_name}. Attempts: {' | '.join(attempted_errors)}")


def _build_latex_table(rows: List[Dict[str, object]]) -> str:
    def _latex_escape(text: object) -> str:
        s = str(text)
        replacements = {
            "\\": r"\textbackslash{}",
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
            "{": r"\{",
            "}": r"\}",
            "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}",
        }
        return "".join(replacements.get(ch, ch) for ch in s)

    source_info = {
        "Nuclei_1": r"TDBase simulator nuclei dataset (\url{https://github.com/tengdj/tdbase})",
        "Nuclei_2": r"TDBase simulator nuclei dataset (\url{https://github.com/tengdj/tdbase})",
        "Vessel_1": r"TDBase simulator vessel dataset (\url{https://github.com/tengdj/tdbase})",
        "Neurons_1": r"MICrONS neuron mesh aggregate (\url{https://www.microns-explorer.org/})",
        "Neurons_2": r"MICrONS neuron mesh aggregate (\url{https://www.microns-explorer.org/})",
        "Cubes_1": r"Synthetic cube dataset generated by benchmark scripts",
        "Cubes_2": r"Synthetic cube dataset generated by benchmark scripts",
    }

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{lrrp{4.6cm}rrr}",
        r"\toprule",
        r"Dataset Name & \#Objects & Size on Disk & Source Description & \#Triangles & Preprocessing Time (Pierce/TDBase) & Loading Time (Pierce/TDBase) \\",
        r"\midrule",
    ]

    for row in rows:
        preprocess_pierce_s = (
            f"{(float(row['preprocess_pierce_ms']) / 1000.0):.2f}"
            if row.get("preprocess_pierce_ms") is not None
            else "-"
        )
        preprocess_tdbase_s = (
            f"{(float(row['preprocess_tdbase_ms']) / 1000.0):.2f}"
            if row.get("preprocess_tdbase_ms") is not None
            else "-"
        )
        loading_pierce_ms = (
            f"{float(row['loading_pierce_ms']):.2f}"
            if row.get("loading_pierce_ms") is not None
            else "-"
        )
        loading_tdbase_ms = (
            f"{float(row['loading_tdbase_ms']):.2f}"
            if row.get("loading_tdbase_ms") is not None
            else "-"
        )
        dataset_id = str(row["dataset_id"])
        source_desc = source_info.get(dataset_id, "Dataset used in benchmark")
        lines.append(
            f"{_latex_escape(dataset_id)} & {row['objects']} & {_latex_escape(row['dataset_size_human'])} & "
            f"{source_desc} & {row['triangles']} & {preprocess_pierce_s} / {preprocess_tdbase_s} & "
            f"{loading_pierce_ms} / {loading_tdbase_ms} \\\\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Dataset statistics and preprocessing/loading costs used in the benchmark.}",
            r"\label{tab:dataset_benchmark}",
            r"\end{table*}",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset benchmark table generator for overall overlap benchmark datasets")
    parser.add_argument("--nu", type=int, default=400, help="NU count used for large_nu_v/large_nu_nn overall benchmark point")
    parser.add_argument("--microns-size-gb", type=int, default=4, help="MICrONS size used in overall benchmark point")
    parser.add_argument("--cube-count-b", type=int, default=1000000, help="Cubes count for dataset B used in overall benchmark point")
    args = parser.parse_args()
    _log(
        f"dataset benchmark start nu={args.nu} microns_size_gb={args.microns_size_gb} "
        f"cube_count_b={args.cube_count_b}"
    )

    run_layout = create_benchmark_run_layout(SCRIPT_DIR, "dataset_table_benchmark")
    logs_dir = Path(run_layout["logs_dir"])
    _log(f"run_dir={run_layout['run_dir']}")
    _log(f"logs_dir={logs_dir}")

    dataset_dirs = get_shared_data_dirs("dataset_table_benchmark")
    _log(f"shared dataset dirs: {dataset_dirs}")

    nu_dirs = get_shared_data_dirs("large_nu_nn_scalability")
    mic_dirs = get_shared_data_dirs("microns_overlap")
    cube_dirs = get_shared_data_dirs("cube_scalability")

    vessel_path = canonical_nu_pair_paths(nu_dirs["raw"], nu=args.nu, nv=750, prefix="tdbase_large")[1]
    nuclei1_path = canonical_nu_pair_paths(nu_dirs["raw"], nu=args.nu, nv=750, prefix="tdbase_large")[0]
    nuclei_nn_1, nuclei_nn_2 = canonical_nn_pair_paths(nu_dirs["raw"], nu=args.nu, nv=750, prefix="tdbase_large")
    neurons_1, neurons_2 = canonical_microns_aggregated_paths(mic_dirs["raw"], args.microns_size_gb)
    cubes_1, cubes_2 = canonical_cube_pair_paths(
        cube_dirs["raw"],
        num_cubes_a=200000,
        num_cubes_b=args.cube_count_b,
        min_size=1.0,
        max_size=2.0,
        selectivity=0.001,
        seed=42,
        grid_cell_size=5.0,
    )

    ensure_cube_pair_dataset(
        cubes_1,
        cubes_2,
        num_cubes_a=200000,
        num_cubes_b=args.cube_count_b,
        min_size=1.0,
        max_size=2.0,
        selectivity=0.001,
        seed=42,
    )
    _log(f"cube files ensured: {cubes_1} | {cubes_2}")

    datasets: List[DatasetRow] = [
        DatasetRow("Nuclei_1", nuclei1_path, 200.0),
        DatasetRow("Vessel_1", vessel_path, 200.0),
        DatasetRow("Nuclei_2", nuclei_nn_2, 200.0),
        DatasetRow("Neurons_1", neurons_1, 700.0),
        DatasetRow("Neurons_2", neurons_2, 700.0),
        DatasetRow("Cubes_1", cubes_1, 5.0),
        DatasetRow("Cubes_2", cubes_2, 5.0),
    ]

    for ds in datasets:
        if not ds.source_path.exists():
            raise FileNotFoundError(f"Dataset file not found for {ds.dataset_id}: {ds.source_path}")
        _log(f"dataset present: {ds.dataset_id} -> {ds.source_path}")

    preprocess_stats: Dict[str, Dict[str, float | int | str]] = {}
    _log("=== Stage: preprocessing all datasets ===")
    for ds in datasets:
        preprocess_stats[ds.dataset_id] = _run_preprocess(
            dataset=ds,
            preprocessed_dir=dataset_dirs["preprocessed"],
            timings_dir=dataset_dirs["timings"],
            logs_dir=logs_dir,
        )
    _log("=== Stage complete: preprocessing ===")

    loading_ms: Dict[str, float] = {}
    join_logs: Dict[str, str] = {}

    _log("=== Stage: join + loading extraction (nuclei_vessel) ===")
    l1, l2, log = _run_join_and_parse_loading(
        case_name="nuclei_vessel",
        mesh1=vessel_path,
        mesh2=nuclei1_path,
        grid_cell_size=200.0,
        preprocessed_dir=dataset_dirs["preprocessed"],
        timings_dir=dataset_dirs["timings"],
        logs_dir=logs_dir,
    )
    loading_ms["Vessel_1"] = l1
    loading_ms["Nuclei_1"] = l2
    join_logs["nuclei_vessel"] = log

    _log("=== Stage: join + loading extraction (nuclei_nuclei) ===")
    l1, l2, log = _run_join_and_parse_loading(
        case_name="nuclei_nuclei",
        mesh1=nuclei_nn_1,
        mesh2=nuclei_nn_2,
        grid_cell_size=200.0,
        preprocessed_dir=dataset_dirs["preprocessed"],
        timings_dir=dataset_dirs["timings"],
        logs_dir=logs_dir,
    )
    loading_ms.setdefault("Nuclei_1", l1)
    loading_ms["Nuclei_2"] = l2
    join_logs["nuclei_nuclei"] = log

    _log("=== Stage: join + loading extraction (neurons_neurons) ===")
    l1, l2, log = _run_join_and_parse_loading(
        case_name="neurons_neurons",
        mesh1=neurons_1,
        mesh2=neurons_2,
        grid_cell_size=700.0,
        preprocessed_dir=dataset_dirs["preprocessed"],
        timings_dir=dataset_dirs["timings"],
        logs_dir=logs_dir,
    )
    loading_ms["Neurons_1"] = l1
    loading_ms["Neurons_2"] = l2
    join_logs["neurons_neurons"] = log

    _log("=== Stage: join + loading extraction (cubes_cubes) ===")
    l1, l2, log = _run_join_and_parse_loading(
        case_name="cubes_cubes",
        mesh1=cubes_1,
        mesh2=cubes_2,
        grid_cell_size=5.0,
        preprocessed_dir=dataset_dirs["preprocessed"],
        timings_dir=dataset_dirs["timings"],
        logs_dir=logs_dir,
    )
    loading_ms["Cubes_1"] = l1
    loading_ms["Cubes_2"] = l2
    join_logs["cubes_cubes"] = log
    _log("=== Stage complete: join + loading extraction ===")

    rows: List[Dict[str, object]] = []
    for ds in datasets:
        stats = preprocess_stats[ds.dataset_id]
        rows.append(
            {
                "dataset_id": ds.dataset_id,
                "source_path": str(ds.source_path),
                "dataset_size_bytes": ds.source_path.stat().st_size,
                "dataset_size_human": _human_bytes(ds.source_path.stat().st_size),
                "objects": int(stats["objects"]),
                "triangles": int(stats["triangles"]),
                "preprocess_pierce_ms": float(stats["preprocess_ms"]),
                "preprocess_tdbase_ms": None,
                "loading_pierce_ms": float(loading_ms.get(ds.dataset_id, 0.0)),
                "loading_tdbase_ms": None,
                "preprocess_log": stats["log_path"],
                "preprocess_timing_json": stats["timing_path"],
            }
        )

    latex_table = _build_latex_table(rows)
    latex_path = Path(run_layout["run_dir"]) / "dataset_table.tex"
    latex_path.write_text(latex_table + "\n", encoding="utf-8")
    _log(f"latex table written: {latex_path}")

    payload = {
        "metadata": {
            "timestamp": run_layout["timestamp"],
            "run_name": run_layout["run_name"],
            "run_dir": str(run_layout["run_dir"]),
            "nu": args.nu,
            "microns_size_gb": args.microns_size_gb,
            "cube_count_b": args.cube_count_b,
            "grid_sizes": {"nu": 200.0, "microns": 700.0, "cubes": 5.0},
            "join_logs": join_logs,
            "latex_table_path": str(latex_path),
        },
        "rows": rows,
        "latex_table": latex_table,
    }

    write_json(Path(run_layout["results_json"]), payload)
    _log(f"results json written: {run_layout['results_json']}")

    print(f"Saved results: {run_layout['results_json']}")
    print(f"Saved LaTeX table: {latex_path}")


if __name__ == "__main__":
    main()
