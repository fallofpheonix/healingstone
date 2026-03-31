"""MANDATORY: Research-grade metrics for 3D fragment reassembly."""

import numpy as np
from scipy.spatial import cKDTree


def compute_mre(pred_points, gt_points):
    """
    MRE = sqrt( (1/N) * Σ || T(p_i) - p_i^* ||^2 )
    Includes KD-tree correspondence and 95th percentile outlier rejection.
    """
    tree = cKDTree(gt_points)
    distances, _ = tree.query(pred_points, k=1)
    
    # outlier rejection (95 percentile)
    threshold = np.percentile(distances, 95)
    distances = distances[distances <= threshold]
    
    return np.sqrt(np.mean(distances ** 2))


def compute_weighted_mre(mre_list, surface_areas):
    """
    WMRE = Σ (w_i * MRE_i), where w_i = A_i / Σ A_j
    """
    if not mre_list or np.sum(surface_areas) == 0:
        return 0.0
    weights = surface_areas / np.sum(surface_areas)
    return float(np.sum(weights * np.array(mre_list)))


def compute_precision_recall(pred_matches, gt_matches):
    """
    Formally calculated Precision, Recall, and F1.
    """
    pred_set = set(pred_matches)
    gt_set = set(gt_matches)

    tp = len(pred_set & gt_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def compute_completeness(correct_fragments, total_fragments):
    """
    AC = (correctly assembled) / (total fragments)
    """
    return correct_fragments / (total_fragments + 1e-8)
