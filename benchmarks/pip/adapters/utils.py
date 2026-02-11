"""Utility functions for adapters."""

import sys
import subprocess
from typing import Optional, Tuple

import numpy as np


def compute_obj_bbox(obj_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """Compute bounding box of an OBJ mesh.
    
    Args:
        obj_file: Path to OBJ file
        
    Returns:
        Tuple of (bbox_min, bbox_max) as numpy arrays
    """
    vertices = []
    with open(obj_file, 'r') as f:
        for line in f:
            line_stripped = line.strip()
            if line_stripped.startswith('v '):
                parts = line_stripped.split()
                if len(parts) >= 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    
    if not vertices:
        raise ValueError(f"No vertices found in {obj_file}")
    
    vertices_array = np.array(vertices)
    return vertices_array.min(axis=0), vertices_array.max(axis=0)


def translate_obj(input_obj: str, output_obj: str, translation: np.ndarray):
    """Translate OBJ mesh by given vector."""
    with open(input_obj, 'r') as f_in, open(output_obj, 'w') as f_out:
        for line in f_in:
            line_stripped = line.strip()
            if line_stripped.startswith('v '):
                parts = line_stripped.split()
                if len(parts) >= 4:
                    x = float(parts[1]) + translation[0]
                    y = float(parts[2]) + translation[1]
                    z = float(parts[3]) + translation[2]
                    f_out.write(f"v {x} {y} {z}\n")
                else:
                    f_out.write(line)
            else:
                f_out.write(line)


def wkt_points_to_csv(wkt_file: str, csv_file: str):
    """Convert WKT points to CSV format (x,y,z) for SQL loading."""
    # Fast streaming parser: do a quick first pass to count POINT lines so we can
    # log progress every 5% during conversion. Then do the conversion with
    # batched writes to minimize syscalls.
    batch_size = 10000
    out_lines = []

    # First pass: count total POINT lines (fast checks)
    total_points = 0
    with open(wkt_file, 'r', buffering=1 << 20) as f_in:
        for line in f_in:
            if not line:
                continue
            if line[0] != 'P':
                continue
            if line.startswith('POINT'):
                total_points += 1

    # Prepare reporting thresholds
    if total_points > 0:
        report_every = max(1, total_points // 20)  # 5% increments -> 20 reports
    else:
        report_every = None

    processed = 0
    next_report = report_every if report_every is not None else None

    with open(wkt_file, 'r', buffering=1 << 20) as f_in, open(csv_file, 'w', buffering=1 << 20) as f_out:
        f_out.write("x,y,z\n")
        for line in f_in:
            if not line:
                continue
            if line[0] != 'P':
                continue
            if not line.startswith('POINT'):
                continue

            lp = line.find('(')
            if lp == -1:
                continue
            rp = line.find(')', lp + 1)
            if rp == -1:
                continue
            inner = line[lp + 1:rp]
            parts = inner.split()
            if len(parts) < 3:
                continue

            out_lines.append(parts[0] + ',' + parts[1] + ',' + parts[2] + '\n')
            processed += 1

            # Batch flush
            if len(out_lines) >= batch_size:
                f_out.write(''.join(out_lines))
                out_lines = []

            # Progress logging every 5%
            if next_report is not None and processed >= next_report:
                pct = int((processed / total_points) * 100)
                print(f"[wkt->csv] {pct}% ({processed}/{total_points})")
                # advance next_report to next multiple
                while next_report is not None and processed >= next_report:
                    next_report += report_every

        # flush remaining
        if out_lines:
            f_out.write(''.join(out_lines))

    # Final report
    if total_points > 0:
        print(f"[wkt->csv] 100% ({processed}/{total_points}) - conversion complete")


def run_subprocess_streaming(
    cmd,
    prefix: str = "",
    timeout: Optional[float] = None,
    check: bool = True,
    **kwargs
) -> Tuple[subprocess.CompletedProcess, str, str]:
    """Run subprocess with real-time output streaming.
    
    Args:
        cmd: Command to run (list or string)
        prefix: Prefix to add to each output line (e.g., "[SQL]")
        timeout: Timeout in seconds
        check: If True, raise CalledProcessError on non-zero exit
        **kwargs: Additional arguments passed to subprocess.Popen
    
    Returns:
        Tuple of (CompletedProcess, stdout, stderr) where stdout/stderr are full captured output
    """
    import sys
    
    if isinstance(cmd, str):
        shell = True
    else:
        shell = False
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        shell=shell,
        **kwargs
    )
    
    stdout_lines = []
    
    try:
        for line in process.stdout:
            line = line.rstrip()
            if prefix:
                print(f"{prefix} {line}", flush=True)
            else:
                print(line, flush=True)
            stdout_lines.append(line + '\n')
        
        process.wait(timeout=timeout)
    
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        raise
    
    stdout = ''.join(stdout_lines)
    stderr = ''  # stderr is merged into stdout via stderr=subprocess.STDOUT
    
    result = subprocess.CompletedProcess(
        process.args,
        process.returncode,
        stdout,
        stderr
    )
    
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode,
            result.args,
            result.stdout,
            result.stderr
        )
    
    return result, stdout, stderr
