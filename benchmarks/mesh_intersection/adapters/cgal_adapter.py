import re
import statistics
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from benchmarks.common.adapters.base import IntersectionBenchmarkAdapter, run_command_streaming


class CGALIntersectionAdapter(IntersectionBenchmarkAdapter):
    def __init__(
        self,
        cgal_dir: str,
        preprocessed_dir: str = "preprocessed",
        threads: Optional[int] = None,
        grid_cell_size: float = 10.0,
    ):
        super().__init__("Face")
        self.cgal_dir = Path(cgal_dir)
        self.preprocessed_dir = Path(preprocessed_dir)
        self.executable = self.cgal_dir / "build" / "cgal_intersection"
        self.threads = threads
        self.grid_cell_size = grid_cell_size

    def _get_preprocessed_path(self, file_path: str) -> Path:
        input_path = Path(file_path)
        grid_token = str(self.grid_cell_size).replace(".", "_")
        modern = self.preprocessed_dir / f"{input_path.stem}_g{grid_token}.pre"
        legacy = self.preprocessed_dir / f"{input_path.stem}.pre"
        return modern if modern.exists() else legacy

    def run_intersection(
        self,
        file1: str,
        file2: str,
        num_runs: int,
        timeout: Optional[float] = None,
        log_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self.executable.exists():
            return {"error": f"Executable not found: {self.executable}"}

        p1 = self._get_preprocessed_path(file1)
        p2 = self._get_preprocessed_path(file2)

        if not p1.exists() or not p2.exists():
            return {"error": f"Face requires .pre files. Missing one of: {p1}, {p2}"}

        cmd = [str(self.executable), str(p1), str(p2)]
        if self.threads:
            cmd.append(str(self.threads))

        runtimes = []
        num_intersections = 0

        adapter_log_dir = None
        if log_dir:
            adapter_log_dir = Path(log_dir) / self.name
            adapter_log_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"[{self.name}] Running benchmark on {p1.name} and {p2.name} "
            f"using {self.threads or 'all available'} threads..."
        )

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
                if not match:
                    return {"error": "Timing string not found in output"}
                runtimes.append(float(match.group(1)))

                if run_idx == 0:
                    count_match = re.search(r"Unique intersecting object pairs:\s*(\d+)", output)
                    if not count_match:
                        count_match = re.search(r"Total Overlaps:\s*(\d+)", output)
                    if count_match:
                        num_intersections = int(count_match.group(1))

            except subprocess.TimeoutExpired:
                return {"error": f"Timeout reached ({timeout}s)"}
            except subprocess.CalledProcessError as e:
                return {"error": f"Face failed with exit code {e.returncode}: {e.stderr}"}

        if not runtimes:
            return {"error": "No timing results collected"}

        std = statistics.pstdev(runtimes) if len(runtimes) > 1 else 0.0
        return {
            "mean": float(statistics.mean(runtimes)),
            "min": float(min(runtimes)),
            "max": float(max(runtimes)),
            "std": float(std),
            "raw_times": runtimes,
            "num_intersections": int(num_intersections),
        }
