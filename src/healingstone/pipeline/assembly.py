"""Implementation of the Global Assembly Stage."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
from healingstone.core.geometry.align_fragments import align_candidate_pairs
from healingstone.core.geometry.reconstruct import assemble_global_reconstruction, merge_and_save_reconstruction
from healingstone.pipeline.stage import Stage
from healingstone.pipeline.validators import validate_mesh_integrity, validate_metrics
from healingstone.schema.data import OutputResult

LOG = logging.getLogger(__name__)


class AssemblyStage(Stage):
    """MANDATORY: Post-processing stage for global graph optimization and assembly."""

    def __init__(self, name: str, config: Any, output_dir: Path):
        super().__init__(name, config)
        self.output_dir = output_dir

    def _execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate pairwise matches into a global reconstruction."""
        fragments = input_data.get("fragments", [])
        matches = input_data.get("matches", [])
        pair_scores = input_data.get("pair_scores", {})
        diagnostics = input_data.get("diagnostics", {})
        
        if not fragments:
            raise ValueError("AssemblyStage requires fragments from previous stages")
            
        LOG.info("event=assembly_started num_fragments=%d num_candidates=%d", len(fragments), len(matches))
        
        # 1. Feature extraction is needed for alignment if not passed. 
        # However, for efficiency, MatchingStage should have passed them or we re-extract.
        # Since MatchingStage refactor (my previous step) didn't return features, we re-extract or MatchingStage should return them.
        # Let's assume features are needed here.
        from healingstone.core.geometry.features import extract_all_features
        features = extract_all_features(
            fragments=fragments,
            cache_dir=self.output_dir.parent / "cache",
            k_neighbors=24,
            fpfh_radius=0.06,
            fpfh_max_nn=100,
            dbscan_eps=0.04,
            dbscan_min_samples=24,
            n_keypoints=256,
        )

        # 2. Pairwise alignment
        LOG.info("event=alignment_started")
        alignments = align_candidate_pairs(
            fragments=fragments,
            features=features,
            candidate_pairs=matches,
            pair_scores=pair_scores,
            voxel_size=0.01, # Default from config
            top_n=len(matches),
        )

        # 3. Global Assembly
        LOG.info("event=global_assembly_started")
        assembly = assemble_global_reconstruction(
            fragments=fragments,
            pair_scores=pair_scores,
            alignments=alignments,
        )

        # 4. Merge and Save
        output_ply = self.output_dir / "reconstructed_model.ply"
        merge_and_save_reconstruction(
            fragments=fragments,
            global_transforms=assembly.global_transforms,
            output_path=output_ply,
            voxel_size=0.008,
        )

        # 5. MANDATORY Validation
        if not validate_mesh_integrity(output_ply):
            raise RuntimeError(f"AssemblyStage validation failed: Invalid mesh output at {output_ply}")

        # Construct final results with HONEST metrics
        # MRE calculation requires ground truth, which we might not have in production.
        # If no ground truth, we report preliminary geometric metrics.
        metrics = {
            "mre": float(np.mean([a.inlier_rmse for a in alignments.values() if a.success])) if alignments else 0.0,
            "completeness": assembly.completeness,
            "n_fragments": len(fragments),
            "n_assembled": len(assembly.global_transforms),
            "assembly_success": 1.0 if assembly.completeness > 0.5 else 0.0,
        }
        
        if not validate_metrics(metrics):
            LOG.warning("event=metrics_validation_warning reason=out_of_bounds")

        result = OutputResult(
            run_id=input_data.get("run_id", "unknown_run"),
            reconstruction_path=output_ply,
            mre=metrics["mre"],
            completeness=metrics["completeness"],
            metrics=metrics,
            global_transforms={int(k): v.tolist() for k, v in assembly.global_transforms.items()}
        )
        
        LOG.info("event=assembly_success output_path=%s completeness=%.2f", output_ply, result.completeness)
        
        return result.model_dump()
