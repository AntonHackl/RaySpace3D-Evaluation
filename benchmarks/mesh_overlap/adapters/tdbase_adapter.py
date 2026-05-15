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
                # Parse either the legacy summary line with units or the newer plain numeric stats.
                match = re.search(r"compute:\s+([\d.]+)\s+(s|ms)", output)
                comp_time = 0.0
                if match:
                    val = float(match.group(1))
                    unit = match.group(2)
                    if unit == 's':
                        val *= 1000.0
                    comp_time = val
                else:
                    computation_match = re.search(r"computation:\s*([\d.]+)", output)
                    if computation_match:
                        comp_time = float(computation_match.group(1))
                    else:
                        comp_matches = re.finditer(r"computation for checking intersection takes ([\d.]+) ms", output)
                        comp_times = [float(m.group(1)) for m in comp_matches]
                        if comp_times:
                            comp_time = sum(comp_times)
                            
                total_match = re.search(r"total:\s*([\d.]+)\s+(s|ms)?", output)
                total_time = 0.0
                if total_match:
                    val = float(total_match.group(1))
                    if total_match.group(2) == 's':
                        val *= 1000.0
                    # TDBase sometimes logs total without unit assuming seconds
                    elif total_match.group(2) is None and val < 1000:
                         val *= 1000.0
                    total_time = val
                    
                if comp_time > 0:
                    runtimes.append(comp_time)
                    # Hack: store the PREPROCESSING time as a property on self, or append it to a list
                    if not hasattr(self, 'preprocessing_times'):
                        self.preprocessing_times = []
                    
                    prep_time = 0.0
                    decode_m = re.search(r"decode:\s+([\d.]+)\s+(s|ms)", output)
                    if decode_m:
                        v = float(decode_m.group(1))
                        if decode_m.group(2) == 's': v *= 1000.0
                        prep_time += v
                    
                    index_m = re.search(r"index:\s+([\d.]+)\s+(s|ms)", output)
                    if index_m:
                        v = float(index_m.group(1))
                        if index_m.group(2) == 's': v *= 1000.0
                        prep_time += v
                        
                    prepare_m = re.search(r"prepare:\s+([\d.]+)\s+(s|ms)", output)
                    if prepare_m:
                        v = float(prepare_m.group(1))
                        if prepare_m.group(2) == 's': v *= 1000.0
                        prep_time += v
                        
                    # Also load tiles counts as query setup in TDBase since it's inside the join execution
                    load_m = re.search(r"load tiles takes\s+([\d.]+)\s+(s|ms)", output)
                    if load_m:
                        v = float(load_m.group(1))
                        if load_m.group(2) == 's': v *= 1000.0
                        prep_time += v

                    self.preprocessing_times.append(prep_time)
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
        mean_prep = np.mean(self.preprocessing_times) if hasattr(self, 'preprocessing_times') and self.preprocessing_times else 0.0
        return {
            "mean": float(np.mean(runtimes)),
            "min": float(np.min(runtimes)),
            "max": float(np.max(runtimes)),
            "std": float(np.std(runtimes)),
            "raw_times": [float(x) for x in runtimes],
            "mean_preprocessing": float(mean_prep),
            "lods": lods,
            "gpu": True
        }
