"""Dataset validation utility for checking 3D mesh integrity."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import open3d as o3d

LOG = logging.getLogger(__name__)

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

class MeshValidator:
    """Validator for 3D geometric fragments."""

    def __init__(self, mesh_path: Path):
        self.path = mesh_path
        self.mesh: Optional[o3d.geometry.TriangleMesh] = None
        self.stats: Dict[str, any] = {}
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def load(self) -> bool:
        """Attempt to load the mesh."""
        try:
            self.mesh = o3d.io.read_triangle_mesh(str(self.path))
            if self.mesh.is_empty():
                self.errors.append("Mesh is empty")
                return False
            return True
        except Exception as e:
            self.errors.append(f"Failed to load mesh: {e}")
            return False

    def validate(self):
        """Perform comprehensive integrity checks."""
        if not self.mesh:
            return

        # Basic stats
        n_vertices = len(self.mesh.vertices)
        n_triangles = len(self.mesh.triangles)
        self.stats["n_vertices"] = n_vertices
        self.stats["n_triangles"] = n_triangles

        # Consistency checks
        if n_vertices < 3:
            self.errors.append(f"Insufficient vertices: {n_vertices}")
        if n_triangles == 0:
            self.errors.append("No faces detected (point cloud only?)")

        # Topological checks
        is_edge_manifold = self.mesh.is_edge_manifold(allow_boundary_edges=True)
        is_edge_manifold_watertight = self.mesh.is_edge_manifold(allow_boundary_edges=False)
        is_vertex_manifold = self.mesh.is_vertex_manifold()
        is_self_intersecting = self.mesh.is_self_intersecting()
        
        self.stats["is_edge_manifold"] = is_edge_manifold
        self.stats["is_watertight"] = is_edge_manifold_watertight
        self.stats["is_vertex_manifold"] = is_vertex_manifold
        self.stats["is_self_intersecting"] = is_self_intersecting

        if not is_edge_manifold:
            self.errors.append("Mesh is not edge-manifold")
        if not is_vertex_manifold:
            self.errors.append("Mesh is not vertex-manifold")
        if is_self_intersecting:
            self.warnings.append("Mesh has self-intersecting faces")

        # Texture/Color checks
        has_vertex_colors = self.mesh.has_vertex_colors()
        has_textures = self.mesh.has_textures()
        self.stats["has_vertex_colors"] = has_vertex_colors
        self.stats["has_textures"] = has_textures

        if not (has_vertex_colors or has_textures):
            self.warnings.append("No color or texture data found")

        # Coordinate consistency (check scale/bounding box)
        bbox = self.mesh.get_axis_aligned_bounding_box()
        extent = bbox.get_max_bound() - bbox.get_min_bound()
        self.stats["bbox_extent"] = extent.tolist()

        if np.any(extent < 1e-6):
            self.errors.append("Degenerate bounding box (zero volume)")

    def get_report(self) -> Dict[str, any]:
        return {
            "path": str(self.path.name),
            "stats": self.stats,
            "errors": self.errors,
            "warnings": self.warnings,
            "valid": len(self.errors) == 0
        }

def validate_directory(data_dir: Path) -> List[Dict[str, any]]:
    """Scan and validate all meshes in a directory."""
    results = []
    extensions = {".ply", ".obj"}
    
    mesh_files = [p for p in data_dir.glob("**/*") if p.suffix.lower() in extensions]
    LOG.info("Found %d mesh files in %s", len(mesh_files), data_dir)

    for f in mesh_files:
        LOG.info("Validating %s...", f.name)
        validator = MeshValidator(f)
        if validator.load():
            validator.validate()
        results.append(validator.get_report())
    
    return results

def write_markdown_report(results: List[Dict[str, any]], output_path: Path):
    """Generate the DATASET_INTEGRITY_REPORT.md."""
    with output_path.open("w", encoding="utf-8") as f:
        f.write("# Dataset Integrity Report\n\n")
        
        total = len(results)
        valid_count = sum(1 for r in results if r["valid"])
        f.write(f"**Total Files Scanned:** {total}\n")
        f.write(f"**Valid Meshes:** {valid_count}\n")
        f.write(f"**Integrity Ratio:** {(valid_count/total if total > 0 else 0):.2%}\n\n")

        f.write("## 1. File Summary\n\n")
        f.write("| File | Status | Vertices | Triangles | Notes |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        
        for r in results:
            status = "✅ PASS" if r["valid"] else "❌ FAIL"
            v = r["stats"].get("n_vertices", 0)
            t = r["stats"].get("n_triangles", 0)
            notes = "; ".join(r["errors"] + r["warnings"])
            f.write(f"| {r['path']} | {status} | {v:,} | {t:,} | {notes} |\n")

        f.write("\n## 2. Detailed Findings\n\n")
        for r in results:
            if r["errors"] or r["warnings"]:
                f.write(f"### {r['path']}\n")
                if r["errors"]:
                    f.write("**Errors:**\n")
                    for e in r["errors"]:
                        f.write(f"- {e}\n")
                if r["warnings"]:
                    f.write("**Warnings:**\n")
                    for w in r["warnings"]:
                        f.write(f"- {w}\n")
                f.write("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate mesh dataset integrity.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing fragments")
    parser.add_argument("--output", type=Path, default=Path("DATASET_INTEGRITY_REPORT.md"), help="Report output path")
    args = parser.parse_args()

    setup_logging()
    results = validate_directory(args.data_dir)
    write_markdown_report(results, args.output)
    LOG.info("Report written to %s", args.output)
