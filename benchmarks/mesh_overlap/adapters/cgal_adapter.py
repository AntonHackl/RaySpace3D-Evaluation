import subprocess
import time
import re
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path
from .base import OverlapBenchmarkAdapter, run_command_streaming

class CGALAdapter(OverlapBenchmarkAdapter):
    def __init__(self, cgal_dir: str, preprocessed_dir: str = "preprocessed", threads: int = None):
        super().__init__("CGAL")
        self.cgal_dir = Path(cgal_dir)
        self.preprocessed_dir = Path(preprocessed_dir)
        self.executable = self.cgal_dir / "build" / "cgal_overlap"
        self.threads = threads

    def run_overlap(
        self,
        file1: str,
        file2: str,
        num_runs: int,
        timeout: float = None,
        log_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run CGAL overlap join."""
        if not self.executable.exists():
            return {"error": f"Executable not found: {self.executable}"}

        # Use preprocessed files from the preprocessed directory
        p1 = self.preprocessed_dir / Path(file1).with_suffix('.pre').name
        p2 = self.preprocessed_dir / Path(file2).with_suffix('.pre').name

        if not p1.exists() or not p2.exists():
            return {"error": f"CGAL requires .pre files. One of these does not exist: {p1}, {p2}"}

        runtimes = []
        # CGAL overlap usage: <datasetA.bin> <datasetB.bin> [threads]
        cmd = [str(self.executable), str(p1), str(p2)]
        if self.threads:
            cmd.append(str(self.threads))
        
        print(f"[{self.name}] Running benchmark on {p1.name} and {p2.name} using {self.threads or 'all available'} threads...")

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
                    cmd,
                    timeout=timeout,
                    log_path=log_path,
                    prefix=f"[{self.name}]",
                )
                output = stdout_text + stderr_text
                match = re.search(r"Query Time:.*?\(([\d.]+) ms\)", output)
                if match:
                    runtimes.append(float(match.group(1)))
                else:
                    print(f"[{self.name}] Error: Could not find 'Query Time' in output. Result:\n{output}")
                    return {"error": "Timing string not found in output"}
            except subprocess.TimeoutExpired:
                print(f"[{self.name}] Timeout reached ({timeout}s)")
                return {"error": f"Timeout reached ({timeout}s)"}
            except subprocess.CalledProcessError as e:
                return {"error": f"CGAL failed with exit code {e.returncode}: {e.stderr}"}

        if not runtimes:
            return {"error": "No timing results collected"}

        return {
            "mean": np.mean(runtimes),
            "min": np.min(runtimes),
            "max": np.max(runtimes),
            "std": np.std(runtimes),
            "raw_times": runtimes
        }
