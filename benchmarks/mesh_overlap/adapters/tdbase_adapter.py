import subprocess
import time
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path
from .base import OverlapBenchmarkAdapter, run_command_streaming
from benchmarks.common.adapters.tdbase_common import (
    TDBASE_TIMING_MODE_INDEX_COMPUTE_EVALUATE,
    parse_tdbase_run_metrics,
    query_time_for_mode,
    validate_tdbase_timing_mode,
)

class TDBaseAdapter(OverlapBenchmarkAdapter):
    def __init__(
        self,
        tdbase_dir: str,
        preprocessed_dir: Optional[str] = None,
        threads: Optional[int] = None,
        compute_threads: int = 1,
        query_timing_mode: str = TDBASE_TIMING_MODE_INDEX_COMPUTE_EVALUATE,
    ):
        super().__init__("TDBase")
        self.tdbase_dir = Path(tdbase_dir)
        legacy_build_dir = self.tdbase_dir / "src" / "build"
        patch_build_dir = self.tdbase_dir.parent / "tdbase_patch" / "build"
        direct_build_dir = self.tdbase_dir / "build"

        tdbase_candidates = [
            patch_build_dir / "tdbase",
            legacy_build_dir / "tdbase",
            direct_build_dir / "tdbase",
        ]
        obj_to_dt_candidates = [
            patch_build_dir / "obj_to_dt",
            legacy_build_dir / "obj_to_dt",
            direct_build_dir / "obj_to_dt",
        ]

        self.executable = next((p for p in tdbase_candidates if p.exists()), tdbase_candidates[0])
        self.obj_to_dt_exec = next((p for p in obj_to_dt_candidates if p.exists()), obj_to_dt_candidates[0])
        self.preprocessed_dir = Path(preprocessed_dir) if preprocessed_dir else None
        self.threads = threads
        self.compute_threads = compute_threads
        self.query_timing_mode = validate_tdbase_timing_mode(query_timing_mode)
        
        if self.preprocessed_dir:
            self.preprocessed_dir.mkdir(parents=True, exist_ok=True)

    def check_preprocessed(self, file_path: str) -> bool:
        """Check if .dt file exists for the given file path in preprocessed dir."""
        input_path = Path(file_path)
        if self.preprocessed_dir:
             dt_file = self.preprocessed_dir / input_path.with_suffix('.dt').name
        else:
             dt_file = input_path.with_suffix('.dt')
        return dt_file.exists()

    def preprocess_from_source(self, source_file: str, dt_file: str, log_dir: Optional[str] = None):
        """Convert .obj to .dt using obj_to_dt tool."""
        source_path = Path(source_file)
        dt_path = Path(dt_file)

        if self.preprocessed_dir:
            output_dt = self.preprocessed_dir / dt_path.with_suffix(".dt").name
        else:
            output_dt = dt_path.with_suffix(".dt")
            
        # Ensure output directory exists (if not using preprocessed_dir, or if it was just created)
        output_dt.parent.mkdir(parents=True, exist_ok=True)
        
        # If it's already a .dt file, just copy it to the preprocessed dir if needed
        if source_path.suffix == '.dt':
            if source_path.resolve() != output_dt.resolve():
                import shutil
                print(f"[{self.name}] Copying {source_path.name} to {output_dt.name}...")
                shutil.copyfile(source_path, output_dt)
            return

        if not self.obj_to_dt_exec.exists():
            print(f"[{self.name}] Error: obj_to_dt tool not found at {self.obj_to_dt_exec}")
            return

        cmd = [str(self.obj_to_dt_exec), str(source_path), str(output_dt)]

        print(f"[{self.name}] Converting {source_path.name} to {output_dt.name}...")
        
        if log_dir:
            adapter_log_dir = Path(log_dir) / self.name
            adapter_log_dir.mkdir(parents=True, exist_ok=True)
            log_path = adapter_log_dir / f"preprocess_{dt_path.stem}_{int(time.time())}.log"
            run_command_streaming(cmd, timeout=None, log_path=str(log_path), prefix=f"[{self.name}]")
        else:
            run_command_streaming(cmd, timeout=None, log_path=None, prefix=f"[{self.name}]")

    def run_overlap(
        self,
        file1: str,
        file2: str,
        num_runs: int,
        timeout: Optional[float] = None,
        log_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run tdbase overlap join."""
        if not self.executable.exists():
            return {"error": f"Executable not found: {self.executable}"}
            
        # Determine actual input files (use preprocessed if available)
        input_path1 = Path(file1)
        input_path2 = Path(file2)
        f1 = file1
        f2 = file2
        
        if self.preprocessed_dir:
            p1 = self.preprocessed_dir / input_path1.with_suffix('.dt').name
            p2 = self.preprocessed_dir / input_path2.with_suffix('.dt').name
            if p1.exists(): f1 = str(p1)
            if p2.exists(): f2 = str(p2)

        # Run TDBase once per run with multiple -l flags (progressive LODs) and GPU enabled
        lods = [20, 40, 60, 80, 100]
        runtimes = []
        preprocessing_times = []
        loading_times = []
        run_metrics = []

        # Build command with repeated -l flags as recommended by TDBase README
        cmd_base = [
            str(self.executable),
            "join",
            "-q", "intersect",
            "--tile1", f1,
            "--tile2", f2,
        ]
        if self.threads:
            cmd_base.extend(["-t", str(self.threads)])
        cmd_base.extend(["--cn", str(self.compute_threads)])
        for lod in lods:
            cmd_base.extend(["-l", str(lod)])
        cmd_base.append("-g")

        print(
            f"[{self.name}] Running TDBase with LODs {lods} (GPU) "
            f"using {self.threads or 'all available'} join threads and "
            f"{self.compute_threads} compute threads..."
        )

        adapter_log_dir = None
        if log_dir:
            adapter_log_dir = Path(log_dir) / self.name
            adapter_log_dir.mkdir(parents=True, exist_ok=True)

        for run_idx in range(num_runs):
            try:
                log_path = None
                if adapter_log_dir is not None:
                    log_path = str(adapter_log_dir / f"run_{run_idx:03d}.log")
                stdout_text, stderr_text = run_command_streaming(
                    cmd_base,
                    timeout=timeout,
                    log_path=log_path,
                    prefix=f"[{self.name}]",
                )
                output = stdout_text + stderr_text
                metrics = parse_tdbase_run_metrics(output)
                query_time_ms = query_time_for_mode(metrics, self.query_timing_mode)
                runtimes.append(query_time_ms)
                preprocessing_times.append(metrics["preprocessing_ms"])
                loading_times.append(metrics["loading_ms"])
                run_metrics.append(
                    {
                        **metrics,
                        "query_time_ms": query_time_ms,
                        "query_timing_mode": self.query_timing_mode,
                    }
                )
            except subprocess.TimeoutExpired:
                print(f"[{self.name}] Timeout reached ({timeout}s)")
                return {"error": f"Timeout reached ({timeout}s)"}
            except subprocess.CalledProcessError as e:
                return {"error": f"TDBase failed with exit code {e.returncode}: {e.stderr}"}
            except RuntimeError as e:
                print(f"[{self.name}] Error: {e}. Result:\n{output}")
                return {"error": str(e)}

        if not runtimes:
            return {"error": "No timing results collected for TDBase"}

        # Return aggregate stats over the runs (each run processed all LODs)
        mean_prep = float(np.mean(preprocessing_times)) if preprocessing_times else 0.0
        mean_loading = float(np.mean(loading_times)) if loading_times else 0.0
        return {
            "mean": float(np.mean(runtimes)),
            "min": float(np.min(runtimes)),
            "max": float(np.max(runtimes)),
            "std": float(np.std(runtimes)),
            "raw_times": [float(x) for x in runtimes],
            "mean_preprocessing": mean_prep,
            "mean_loading": mean_loading,
            "run_metrics": run_metrics,
            "query_timing_mode": self.query_timing_mode,
            "lods": lods,
            "gpu": True
        }
