#!/usr/bin/env python3
"""Convert MICrONS mesh subset (.npz files) to a single GLB file for Blender viewing.

Each mesh becomes a separate object in the GLB with a distinct color for segmentation.

Usage:
  # Convert local .npz files to GLB
  python convert_microns_to_blender.py --input-dir ./tmp/microns_mesh_subset_lod0 --output blender_microns_subset.glb

  # Or download from GCS and convert
  python convert_microns_to_blender.py --download-from-gcs --output blender_microns_subset.glb
"""

from __future__ import annotations

import argparse
import colorsys
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import trimesh
except ImportError:
    raise ImportError("trimesh is required. Install with: pip install trimesh")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    script_dir = Path(__file__).parent
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=script_dir / "microns_data" / "microns_mesh_subset_lod0",
        help="Directory containing .npz mesh files. If not provided, uses --download-from-gcs.",
    )
    parser.add_argument(
        "--download-from-gcs",
        action="store_true",
        help="Download .npz files from GCS before converting.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "microns_data" / "microns_mesh_subset.glb",
        help="Output GLB filename (when using --combine). Or output directory (when using --separate).",
    )
    parser.add_argument(
        "--combine",
        action="store_true",
        default=True,
        help="Combine all meshes into a single GLB file (default).",
    )
    parser.add_argument(
        "--separate",
        action="store_true",
        help="Export each mesh as a separate GLB file in --output directory.",
    )
    parser.add_argument(
        "--mesh-source",
        default="precomputed://gs://iarpa_microns/minnie/minnie65/seg_m1300",
        help="CloudVolume segmentation source (used with --download-from-gcs).",
    )
    parser.add_argument(
        "--lod",
        type=int,
        default=0,
        help="Mesh level of detail (used with --download-from-gcs).",
    )
    parser.add_argument(
        "--root-ids",
        nargs="+",
        type=int,
        default=[
            864691135569592300,
            864691135685661367,
            864691135361291591,
            864691136813553523,
            864691136025333561,
            864691135463999294,
            864691136335276211,
            864691135430460720,
            864691135777381805,
            864691135777521837,
            864691136052291827,
            864691135688375264,
            864691136012739747,
            864691135503367517,
            864691135114295961,
            864691136663371742,
            864691135359010904,
            864691135341516741,
        ],
        help="Root IDs to load (default: the 18 downloaded in the subset).",
    )
    return parser.parse_args()


def load_npz_mesh(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load vertices and faces from an .npz file."""
    data = np.load(npz_path)
    vertices = data["vertices"]
    faces = data["faces"]
    return vertices, faces


def generate_colors(count: int) -> list[tuple[float, float, float]]:
    """Generate N distinct colors using HSL space."""
    colors = []
    for i in range(count):
        hue = i / count
        saturation = 0.7 + 0.2 * (i % 2)
        lightness = 0.5
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        colors.append(rgb)
    return colors


def load_meshes_from_npz_dir(input_dir: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Load all .npz files from input directory."""
    meshes = {}
    npz_files = sorted(input_dir.glob("*.npz"))
    print(f"Found {len(npz_files)} .npz files in {input_dir}")
    for npz_file in npz_files:
        try:
            root_id = int(npz_file.stem)
            vertices, faces = load_npz_mesh(npz_file)
            meshes[root_id] = (vertices, faces)
            print(f"  loaded {root_id}: {vertices.shape[0]} verts, {faces.shape[0]} faces")
        except (ValueError, KeyError) as e:
            print(f"  skipped {npz_file.name}: {e}")
    return meshes


def load_meshes_from_gcs(
    root_ids: list[int],
    mesh_source: str,
    lod: int,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Download and load meshes from GCS CloudVolume."""
    try:
        from cloudvolume import CloudVolume
    except ImportError:
        raise ImportError("cloudvolume is required for --download-from-gcs. Install with: pip install cloud-volume")

    print(f"Connecting to {mesh_source} (LOD={lod})...")
    cv = CloudVolume(mesh_source, progress=False, use_https=True)

    meshes = {}
    for root_id in root_ids:
        try:
            mesh_dict = cv.mesh.get(root_id, lod=lod)
            mesh = mesh_dict.get(root_id)
            if mesh is None:
                print(f"  {root_id}: mesh not found")
                continue
            meshes[root_id] = (np.asarray(mesh.vertices), np.asarray(mesh.faces))
            print(f"  {root_id}: {mesh.vertices.shape[0]} verts, {mesh.faces.shape[0]} faces")
        except Exception as e:
            print(f"  {root_id}: {e}")
    return meshes


def export_meshes_individually(
    meshes: dict[int, tuple[np.ndarray, np.ndarray]],
    output_dir: Path,
    rescale_nm_to_um: bool = True,
) -> None:
    """Export each mesh as a separate GLB file in output_dir.
    
    Args:
        meshes: Dict of root_id -> (vertices, faces)
        output_dir: Output directory for GLB files
        rescale_nm_to_um: If True, rescale from nanometers to micrometers.
    """
    if not meshes:
        raise ValueError("No meshes provided")

    output_dir.mkdir(parents=True, exist_ok=True)
    root_ids = sorted(meshes.keys())
    colors = generate_colors(len(root_ids))

    print(f"Exporting {len(root_ids)} meshes individually to {output_dir}...")
    print(f"  (rescaled from nanometers to micrometers for Blender viewing)\n")

    for (root_id, (vertices, faces)), color in zip(meshes.items(), colors):
        verts = np.asarray(vertices, dtype=np.float32)
        
        # Rescale from nanometers to micrometers (1000 nm = 1 um)
        if rescale_nm_to_um:
            verts = verts / 1000.0
        
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        # Assign a unique color for segmentation visualization
        mesh.visual.vertex_colors = [
            tuple(int(c * 255) for c in color) + (255,) for _ in range(len(verts))
        ]
        
        glb_file = output_dir / f"neuron_{root_id}.glb"
        mesh.export(glb_file, file_type="glb")
        file_size_mb = glb_file.stat().st_size / (1024 ** 2)
        print(f"  ✓ neuron_{root_id}.glb ({file_size_mb:.1f} MB)")

    print(f"\n✓ All meshes exported to {output_dir}")
    print(f"  Each neuron is a separate GLB file ready to import into Blender.")
    print(f"  You can import them one at a time or all at once.")


def combine_meshes_to_glb(
    meshes: dict[int, tuple[np.ndarray, np.ndarray]],
    output_path: Path,
    rescale_nm_to_um: bool = True,
) -> None:
    """Combine multiple mesh (vertices, faces) pairs into a single GLB file.
    
    Args:
        meshes: Dict of root_id -> (vertices, faces)
        output_path: Output GLB path
        rescale_nm_to_um: If True, rescale from nanometers to micrometers for better Blender viewing.
    """
    if not meshes:
        raise ValueError("No meshes provided")

    root_ids = sorted(meshes.keys())
    colors = generate_colors(len(root_ids))

    scene = trimesh.Scene()
    for (root_id, (vertices, faces)), color in zip(meshes.items(), colors):
        verts = np.asarray(vertices, dtype=np.float32)
        
        # Rescale from nanometers to micrometers (1000 nm = 1 um)
        # This makes the mesh visible in Blender at a reasonable scale
        if rescale_nm_to_um:
            verts = verts / 1000.0
        
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        # Assign a unique color for segmentation visualization
        mesh.visual.vertex_colors = [
            tuple(int(c * 255) for c in color) + (255,) for _ in range(len(verts))
        ]
        scene.add_geometry(mesh, node_name=f"neuron_{root_id}")

    print(f"Exporting {len(root_ids)} meshes to {output_path}...")
    print(f"  (rescaled from nanometers to micrometers for Blender viewing)")
    scene.export(output_path, file_type="glb")
    print(f"Done! GLB file saved to {output_path}")
    print(f"You can now open {output_path} in Blender.")
    print(f"  - Use Home key or View > Frame All to focus the view")
    print(f"  - Check the Outliner panel to see all neuron objects")


def main() -> None:
    args = parse_args()

    if args.input_dir:
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        meshes = load_meshes_from_npz_dir(input_dir)
    elif args.download_from_gcs:
        meshes = load_meshes_from_gcs(args.root_ids, args.mesh_source, args.lod)
    else:
        raise ValueError("Either --input-dir or --download-from-gcs must be specified")

    if not meshes:
        raise RuntimeError("No meshes loaded")

    output_path = Path(args.output)
    
    if args.separate:
        output_path.mkdir(parents=True, exist_ok=True)
        export_meshes_individually(meshes, output_path)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combine_meshes_to_glb(meshes, output_path)


if __name__ == "__main__":
    main()
