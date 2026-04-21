import time
import json
import numpy as np
from typing import Dict, Any, Optional, List
from pathlib import Path

# Fix potential import issues for standalone script use
import sys
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))

from benchmarks.common.adapters.base import OverlapBenchmarkAdapter, run_command_streaming


def _extract_named_phases(data: Dict[str, Any]) -> Dict[str, float]:
    phases = data.get("phases", {})
    if not isinstance(phases, dict):
        return {}

    named: Dict[str, float] = {}

    for key, value in phases.items():
        if not isinstance(value, dict):
            continue
        duration_ms = float(value.get("duration_ms", 0.0))
        key_base = key.rsplit("_", 1)[0]
        named[key_base] = named.get(key_base, 0.0) + duration_ms

    return named


def _extract_counters(data: Dict[str, Any]) -> Dict[str, float]:
    counters = data.get("counters", {})
    if not isinstance(counters, dict):
        return {}

    out: Dict[str, float] = {}
    for key, value in counters.items():
        try:
            out[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def _build_breakdown(named: Dict[str, float], query_type: str) -> Dict[str, float]:
    base = {
        "selectivity_estimation_ms": named.get("selectivity estimation", 0.0),
        "deduplication_ms": named.get("compact_hash_table_pairs", 0.0),
        "download_results_ms": named.get("download results", 0.0),
    }

    if query_type == "overlap":
        base["raytrace_hash_mesh1_to_mesh2_ms"] = named.get("raytrace_hash_mesh1tomesh2", 0.0)
        base["raytrace_hash_mesh2_to_mesh1_ms"] = named.get("raytrace_hash_mesh2tomesh1", 0.0)
        base["query_ms"] = named.get("execute hash query", 0.0)
    else:
        # New intersection timing split (correct logic: overlap + containment in both directions).
        base["raytrace_overlap_hash_mesh1_to_mesh2_ms"] = named.get("raytrace_overlap_hash_mesh1tomesh2", 0.0)
        base["raytrace_overlap_hash_mesh2_to_mesh1_ms"] = named.get("raytrace_overlap_hash_mesh2tomesh1", 0.0)
        base["raytrace_containment_hash_mesh1_to_mesh2_ms"] = named.get("raytrace_containment_hash_mesh1tomesh2", 0.0)
        base["raytrace_containment_hash_mesh2_to_mesh1_ms"] = named.get("raytrace_containment_hash_mesh2tomesh1", 0.0)

        # Backward-compatible fallback for legacy estimated-intersection binaries.
        if base["raytrace_overlap_hash_mesh1_to_mesh2_ms"] == 0.0 and base["raytrace_overlap_hash_mesh2_to_mesh1_ms"] == 0.0:
            base["raytrace_overlap_hash_mesh1_to_mesh2_ms"] = named.get("raytrace_hash_mesh1tomesh2", 0.0)
            base["raytrace_overlap_hash_mesh2_to_mesh1_ms"] = named.get("raytrace_hash_mesh2tomesh1", 0.0)

        base["query_ms"] = named.get("query", 0.0)

    return base


def _mean_breakdown(run_breakdowns: List[Dict[str, float]]) -> Dict[str, float]:
    if not run_breakdowns:
        return {}

    keys = set()
    for item in run_breakdowns:
        keys.update(item.keys())

    out: Dict[str, float] = {}
    for key in sorted(keys):
        values = [float(item.get(key, 0.0)) for item in run_breakdowns]
        out[key] = float(np.mean(values))
    return out


def _mean_counters(run_counters: List[Dict[str, float]]) -> Dict[str, float]:
    if not run_counters:
        return {}

    keys = set()
    for item in run_counters:
        keys.update(item.keys())

    out: Dict[str, float] = {}
    for key in sorted(keys):
        values = [float(item.get(key, 0.0)) for item in run_counters]
        out[key] = float(np.mean(values))
    return out

class ComparisonAdapter(OverlapBenchmarkAdapter):
    def __init__(
        self,
        rayspace_dir: str,
        query_type: str = "intersection", # "intersection" or "overlap"
        mode: str = "direct_estimation",
        preprocessed_dir: str = "preprocessed",
        timings_dir: str = "timings",
        grid_cell_size: int = 10,
        warmup_runs: int = 2,
        intersection_extra_args: Optional[List[str]] = None,
    ):
        super().__init__(f"RaySpace3D_{query_type}")
        self.rayspace_dir = Path(rayspace_dir)
        self.query_type = query_type
        self.mode = mode
        self.preprocessed_dir = Path(preprocessed_dir)
        self.timings_dir = Path(timings_dir)
        self.grid_cell_size = grid_cell_size
        self.warmup_runs = warmup_runs
        self.intersection_extra_args = intersection_extra_args or []
        
        # Ensure directories exist
        self.timings_dir.mkdir(parents=True, exist_ok=True)
        self.preprocessed_dir.mkdir(parents=True, exist_ok=True)
        
        query_bin_dir = self.rayspace_dir / "query" / "build" / "bin"
        if self.query_type == "intersection":
            self.executable = query_bin_dir / "raytracer_intersection_estimated"
        else: # overlap
            self.executable = query_bin_dir / "raytracer_overlap_direct_estimation"

        self.preprocess_exec = self.rayspace_dir / "preprocess" / "build" / "bin" / "preprocess_dataset"
        
    def check_preprocessed(self, file_path: str) -> bool:
        input_path = Path(file_path)
        pre_file = self.preprocessed_dir / input_path.with_suffix('.pre').name
        return pre_file.exists()

    def preprocess_from_source(self, source_file: str, dt_file: str, log_dir: Optional[str] = None):
        source_path = Path(source_file)
        dt_path = Path(dt_file)
        output_geometry = self.preprocessed_dir / dt_path.with_suffix('.pre').name
        output_timing = self.timings_dir / (dt_path.stem + '_timing.json')
        
        mode = "dt" if source_path.suffix == ".dt" else "mesh"
        
        cmd = [
            str(self.preprocess_exec),
            "--mode", mode,
            "--dataset", str(source_path),
            "--output-geometry", str(output_geometry),
            "--output-timing", str(output_timing),
            "--generate-grid",
            "--grid-cell-size", str(self.grid_cell_size)
        ]
        
        run_command_streaming(cmd, timeout=None, log_path=None, prefix=f"[{self.name}] Preprocess")

    def run_overlap(self, *args, **kwargs):
        return self.run_query(*args, **kwargs)

    def run_query(
        self,
        file1: str,
        file2: str,
        num_runs: int,
        run_id: str,
        log_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self.executable.exists():
            return {"error": f"Executable not found: {self.executable}"}

        input_path1 = Path(file1)
        input_path2 = Path(file2)
        p1 = self.preprocessed_dir / input_path1.with_suffix('.pre').name
        p2 = self.preprocessed_dir / input_path2.with_suffix('.pre').name
        
        f1 = str(p1) if p1.exists() else file1
        f2 = str(p2) if p2.exists() else file2

        runtimes: List[float] = []
        per_run: List[Dict[str, Any]] = []

        run_timings_dir = self.timings_dir / run_id / self.query_type
        run_timings_dir.mkdir(parents=True, exist_ok=True)

        for run_idx in range(num_runs):
            json_output = run_timings_dir / f"dataset_run_{run_idx:03d}.json"
            if json_output.exists():
                json_output.unlink()
            
            cmd = [
                str(self.executable),
                "--mesh1", f1,
                "--mesh2", f2,
                "--runs", "1",
                "--warmup-runs", str(self.warmup_runs),
                "--output", str(json_output)
            ]
            
            # Estimated intersection binary does not support --runs/--warmup-runs arguments.
            if self.query_type == "intersection":
                cmd = [
                    str(self.executable),
                    "--mesh1", f1,
                    "--mesh2", f2,
                    "--output", str(json_output)
                ]
                if self.intersection_extra_args:
                    cmd.extend(self.intersection_extra_args)

            run_command_streaming(cmd, timeout=None, log_path=None, prefix=f"[{self.name}]")
            
            if json_output.exists():
                with open(json_output, 'r') as f:
                    data = json.load(f)

                    named = _extract_named_phases(data)
                    counters = _extract_counters(data)
                    phase_breakdown = _build_breakdown(named, self.query_type)
                    query_time = phase_breakdown.get("query_ms", 0.0)
                    runtimes.append(query_time)

                    per_run.append({
                        "run_index": run_idx,
                        "timing_file": str(json_output),
                        "query_ms": query_time,
                        "breakdown": phase_breakdown,
                        "counters": counters,
                        "raw_named_phases_ms": named,
                    })
            else:
                print(f"[{self.name}] Warning: No timing JSON found at {json_output}")

        avg_breakdown = _mean_breakdown([run["breakdown"] for run in per_run])
        avg_counters = _mean_counters([run.get("counters", {}) for run in per_run])
        avg_time_ms = float(np.mean(runtimes)) if runtimes else 0.0
        std_time_ms = float(np.std(runtimes)) if runtimes else 0.0

        notes = []
        if self.query_type == "intersection":
            notes.append(
                "Estimated intersection runs both overlap and containment raytracing (both directions) in hash mode, then deduplicates results."
            )
        
        return {
            "query_type": self.query_type,
            "binary": str(self.executable),
            "timings_dir": str(run_timings_dir),
            "num_runs": len(per_run),
            "avg_time_ms": avg_time_ms,
            "std_time_ms": std_time_ms,
            "breakdown_avg_ms": avg_breakdown,
            "counters_avg": avg_counters,
            "runs": per_run,
            "notes": notes,
        }
