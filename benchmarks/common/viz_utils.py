import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

APPROACH_STYLES = {
    "exact": {"label": "Pierce (Two Pass)", "color": "#1f77b4", "marker": "o"},
    "direct_estimation": {"label": "Pierce", "color": "#1f77b4", "marker": "s"},
    "estimated": {"label": "Pierce", "color": "#1f77b4", "marker": "s"},
    "estimated_mem10": {"label": "Pierce (Selectivity Estimation, 10 GiB Hash Table)", "color": "#1f77b4", "marker": "v"},
    "cgal": {"label": "CGAL", "color": "#ff7f0e", "marker": "D"},
    "touch": {"label": "TOUCH", "color": "#2ca02c", "marker": "^"},
    "tdbase": {"label": "TDBase", "color": "#d62728", "marker": "X"},
}


def apply_paper_style():
    plt.rcParams.update({
        "font.size": 13,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "axes.linewidth": 1.6,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 13,
        "lines.linewidth": 2.8,
        "lines.markersize": 9,
    })


def style_for(approach: str):
    return APPROACH_STYLES.get(approach, {"label": approach, "color": "#444444", "marker": "o"})


def plot_mean_series(ax, xs, ys, approach: str):
    st = style_for(approach)
    ax.plot(xs, ys, linestyle="-", marker=st["marker"], color=st["color"], label=st["label"])


def generate_scalability_figure(results, approaches, figures_dir: Path, timestamp: str,
                                scenario_name: str, x_axis_key: str, x_axis_label: str,
                                y_axis_label: str, title: str):
    """Generate a line plot for scalability from successful runs."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    apply_paper_style()

    plt.figure(figsize=(10, 6.8))
    has_any_series = False

    for approach in approaches:
        x_vals = []
        y_vals = []
        for row in results:
            res = row.get(approach)
            if not isinstance(res, dict) or "error" in res:
                continue
            mean = res.get("mean")
            if mean is None:
                continue
            x_vals.append(row.get(x_axis_key))
            y_vals.append(mean)

        if not x_vals:
            continue

        has_any_series = True
        sorted_points = sorted(zip(x_vals, y_vals), key=lambda t: (t[0] if t[0] is not None else 0))
        xs = [p[0] for p in sorted_points]
        ys = [p[1] for p in sorted_points]
        plot_mean_series(plt.gca(), xs, ys, approach)

    if not has_any_series:
        print(f"No successful approach results available; skipping {scenario_name} scalability figure.")
        plt.close()
        return

    plt.yscale("log")
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.grid(True, which="both", linestyle="-", alpha=0.2)
    plt.legend()
    plt.tight_layout()

    output_base = figures_dir / f"{scenario_name}_scalability_{timestamp}"
    plt.savefig(f"{output_base}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{output_base}.pdf", bbox_inches="tight")
    plt.close()

    print(f"Saved figure: {output_base}.png")


def generate_breakdown_figure(results, approaches, figures_dir: Path, timestamp: str,
                              scenario_name: str, x_axis_key: str, x_axis_label: str,
                              y_axis_label: str, title: str):
    """Generate a stacked bar chart showing the runtime breakdown."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    apply_paper_style()

    for approach in approaches:
        x_labels = []
        breakdown_keys = set()
        data_points = []

        for row in results:
            res = row.get(approach)
            if not isinstance(res, dict) or "error" in res or "breakdown" not in res:
                continue
            x_labels.append(str(row.get(x_axis_key)))
            data_points.append(res["breakdown"])
            breakdown_keys.update(res["breakdown"].keys())

        if not data_points:
            continue

        sorted_keys = sorted(list(breakdown_keys))

        # Preferred intersection phase order (bottom -> top in stacked bars).
        phase_order = [
            "selectivity estimation",
            "raytrace_overlap_hash_mesh1tomesh2",
            "raytrace_overlap_hash_mesh2tomesh1",
            "raytrace_containment_hash_mesh1tomesh2",
            "raytrace_containment_hash_mesh2tomesh1",
            "compact_hash_table_pairs",
        ]
        phase_labels = {
            "selectivity estimation": "Selectivity Estimation",
            "raytrace_overlap_hash_mesh1tomesh2": "Edge (M1->M2)",
            "raytrace_overlap_hash_mesh2tomesh1": "Edge (M2 -> M1)",
            "raytrace_containment_hash_mesh1tomesh2": "Containment (M1-M2)",
            "raytrace_containment_hash_mesh2tomesh1": "Containment (M2 -> M1)",
            "compact_hash_table_pairs": "Download results",
            "download results": "Download results",
        }

        present_preferred = [k for k in phase_order if k in breakdown_keys]
        remaining = [k for k in sorted_keys if k not in present_preferred]
        display_order = present_preferred + remaining

        plt.figure(figsize=(12, 7.2))
        bottom = np.zeros(len(x_labels))
        colors = plt.cm.tab10.colors

        # Draw in bottom->top order so the first phase is at the bottom.
        for i, key in enumerate(display_order):
            vals = np.array([dp.get(key, 0.0) for dp in data_points])
            plt.bar(
                x_labels,
                vals,
                bottom=bottom,
                label=phase_labels.get(key, key),
                color=colors[i % len(colors)],
            )
            bottom += vals

        plt.xlabel(x_axis_label)
        plt.ylabel(y_axis_label)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(axis='y', linestyle='-', alpha=0.3)
        plt.tight_layout()

        output_base = figures_dir / f"{scenario_name}_breakdown_{approach}_{timestamp}"
        plt.savefig(f"{output_base}.png", dpi=300, bbox_inches="tight")
        plt.savefig(f"{output_base}.pdf", bbox_inches="tight")
        plt.close()
        print(f"Saved figure: {output_base}.png")
