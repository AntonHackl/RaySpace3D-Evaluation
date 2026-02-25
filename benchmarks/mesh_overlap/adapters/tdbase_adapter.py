import subprocess
import time
import re
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path
from .base import OverlapBenchmarkAdapter, run_command_streaming

class TDBaseAdapter(OverlapBenchmarkAdapter):
    def __init__(self, tdbase_dir: str, preprocessed_dir: Optional[str] = None):
        super().__init__("TDBase")
        self.tdbase_dir = Path(tdbase_dir)
        legacy_build_dir = self.tdbase_dir / "src" / "build"
        patch_build_dir = self.tdbase_dir.parent / "tdbase_patch" / "build"
        direct_build_dir = self.tdbase_dir / "build"

        if (patch_build_dir / "tdbase").exists() and (patch_build_dir / "obj_to_dt").exists():
            build_dir = patch_build_dir
        elif (legacy_build_dir / "tdbase").exists() and (legacy_build_dir / "obj_to_dt").exists():
            build_dir = legacy_build_dir
        else:
            build_dir = direct_build_dir

        self.executable = build_dir / "tdbase"
        self.obj_to_dt_exec = build_dir / "obj_to_dt"
        self.preprocessed_dir = Path(preprocessed_dir) if preprocessed_dir else None
        
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
            output_dt = self.preprocessed_dir / dt_path.name
        else:
            output_dt = dt_path
            
        # Ensure output directory exists (if not using preprocessed_dir, or if it was just created)
        output_dt.parent.mkdir(parents=True, exist_ok=True)

        if not self.obj_to_dt_exec.exists():
            print(f"[{self.name}] Error: obj_to_dt tool not found at {self.obj_to_dt_exec}")
            return

        cmd = [
            str(self.obj_to_dt_exec),
            str(source_path),
            str(output_dt)
        ]

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

        # Build command with repeated -l flags as recommended by TDBase README
        cmd_base = [
            str(self.executable),
            "join",
            "-q", "intersect",
            "--tile1", f1,
            "--tile2", f2,
        ]
        for lod in lods:
            cmd_base.extend(["-l", str(lod)])
        cmd_base.append("-g")

        print(f"[{self.name}] Running TDBase with LODs {lods} (GPU) ...")

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
                # Parse: "compute:        1.48621 s(8.46801%)" or "compute:        50.123 ms(..."
                match = re.search(r"compute:\s+([\d.]+)\s+(s|ms)", output)
                if match:
                    val = float(match.group(1))
                    unit = match.group(2)
                    if unit == 's':
                        val *= 1000.0
                    runtimes.append(val)
                else:
                    # Fallback to the individual computation lines if summary is missing?
                    # "computation for checking intersection takes 813.053000 ms"
                    comp_matches = re.finditer(r"computation for checking intersection takes ([\d.]+) ms", output)
                    comp_times = [float(m.group(1)) for m in comp_matches]
                    if comp_times:
                        runtimes.append(sum(comp_times))
                    else:
                        print(f"[{self.name}] Error: Could not find computation timing in output. Result:\n{output}")
                        return {"error": "Computation timing not found"}
            except subprocess.TimeoutExpired:
                print(f"[{self.name}] Timeout reached ({timeout}s)")
                return {"error": f"Timeout reached ({timeout}s)"}
            except subprocess.CalledProcessError as e:
                return {"error": f"TDBase failed with exit code {e.returncode}: {e.stderr}"}

        if not runtimes:
            return {"error": "No computation timing results collected for TDBase"}

        # Return aggregate stats over the runs (each run processed all LODs)
        return {
            "mean": float(np.mean(runtimes)),
            "min": float(np.min(runtimes)),
            "max": float(np.max(runtimes)),
            "std": float(np.std(runtimes)),
            "raw_times": [float(x) for x in runtimes],
            "lods": lods,
            "gpu": True
        }
