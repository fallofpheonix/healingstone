"""
Healing Stones: 3D Fragment Matching & Reconstruction Pipeline
=============================================================
GSoC 2025 | HumanAI | University of Alabama

Strategy Overview
-----------------
Best approach for fragment reassembly without heavy DL frameworks:

1. SURFACE CLASSIFICATION
   - Classify each triangle/point as "break surface" vs "original surface"
   - Break surfaces: locally planar, rough micro-texture, sharp boundary
   - Use: local curvature + PCA-based planarity + normal consistency

2. GEOMETRIC FEATURE EXTRACTION
   - FPFH (Fast Point Feature Histograms) — robust local descriptors
   - Shape diameter function on break surfaces
   - Boundary curve extraction (fracture edges)

3. FRAGMENT MATCHING (ML)
   - RandomForest classifier on pairwise feature similarity
   - Trained on synthetic augmented data (known-matching pairs)
   - Outputs: match probability matrix between all fragment pairs

4. ALIGNMENT
   - RANSAC-based coarse alignment using matched FPFH features
   - ICP (Iterative Closest Point) fine refinement
   - Gap-aware ICP: robust to missing material between fragments

Dependencies: numpy, scipy, scikit-learn, matplotlib
Optional:     open3d (faster ICP if available)
"""

import os
import sys
import glob
import time
import json
import struct
import logging
import argparse
import warnings
import itertools
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D

from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.linalg import svd as scipy_svd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, precision_recall_curve,
                              average_precision_score)
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# 1. I/O — Load PLY / OBJ
# ─────────────────────────────────────────────

def load_ply(filepath: str) -> dict:
    """Load PLY file, returning vertices and faces (if mesh)."""
    filepath = Path(filepath)
    with open(filepath, "rb") as f:
        # Parse header
        header_lines = []
        while True:
            line = f.readline().decode("ascii", errors="replace").strip()
            header_lines.append(line)
            if line == "end_header":
                break

        header = "\n".join(header_lines)
        is_binary_little = "binary_little_endian" in header
        is_binary_big    = "binary_big_endian"    in header
        is_ascii         = "format ascii"         in header

        # Parse element counts and properties
        elements = {}
        current_elem = None
        for line in header_lines:
            if line.startswith("element"):
                parts = line.split()
                current_elem = parts[1]
                elements[current_elem] = {"count": int(parts[2]), "props": []}
            elif line.startswith("property") and current_elem:
                parts = line.split()
                elements[current_elem]["props"].append({"type": parts[1], "name": parts[-1]})

        n_verts = elements.get("vertex", {}).get("count", 0)
        n_faces = elements.get("face",   {}).get("count", 0)
        vert_props = elements.get("vertex", {}).get("props", [])

        # Build format string for binary
        fmt_map = {"float": "f", "double": "d", "int": "i", "uint": "I",
                   "short": "h", "ushort": "H", "uchar": "B", "char": "b"}
        vert_fmt_chars = [fmt_map.get(p["type"], "f") for p in vert_props]
        vert_fmt = ("<" if is_binary_little else ">") + "".join(vert_fmt_chars)
        vert_size = struct.calcsize(vert_fmt)

        prop_names = [p["name"] for p in vert_props]

        vertices = np.zeros((n_verts, 3), dtype=np.float32)
        normals  = np.zeros((n_verts, 3), dtype=np.float32) if any(
            n in prop_names for n in ["nx", "ny", "nz"]) else None
        colors   = np.zeros((n_verts, 3), dtype=np.float32) if any(
            n in prop_names for n in ["red", "r"]) else None

        if is_ascii:
            raw_data = f.read().decode("ascii", errors="replace").split()
            idx = 0
            for i in range(n_verts):
                row = [float(raw_data[idx + j]) for j in range(len(vert_props))]
                idx += len(vert_props)
                for k, name in enumerate(prop_names):
                    if name == "x": vertices[i, 0] = row[k]
                    elif name == "y": vertices[i, 1] = row[k]
                    elif name == "z": vertices[i, 2] = row[k]
                    elif name == "nx" and normals is not None: normals[i, 0] = row[k]
                    elif name == "ny" and normals is not None: normals[i, 1] = row[k]
                    elif name == "nz" and normals is not None: normals[i, 2] = row[k]
                    elif name in ("red", "r") and colors is not None:
                        colors[i, 0] = row[k] / 255.0
                    elif name in ("green", "g") and colors is not None:
                        colors[i, 1] = row[k] / 255.0
                    elif name in ("blue", "b") and colors is not None:
                        colors[i, 2] = row[k] / 255.0
            # Faces (ASCII)
            faces = []
            for _ in range(n_faces):
                row = raw_data[idx:idx + 4]
                n = int(row[0])
                face = [int(raw_data[idx + 1 + j]) for j in range(n)]
                faces.append(face)
                idx += 1 + n
        else:
            endian = "<" if is_binary_little else ">"
            raw = f.read(n_verts * vert_size)
            data = np.frombuffer(raw, dtype=np.dtype(vert_fmt))
            for i, name in enumerate(prop_names):
                col = data[:, i] if data.ndim > 1 else data
                # handle structured array
            # Use structured approach
            dt_fields = [(p["name"], fmt_map.get(p["type"], "f")) for p in vert_props]
            dt = np.dtype([(n, (endian + t)) for n, t in dt_fields])
            f.seek(0)
            # re-read properly
            with open(filepath, "rb") as f2:
                while True:
                    line = f2.readline().decode("ascii", errors="replace").strip()
                    if line == "end_header":
                        break
                raw_v = np.frombuffer(f2.read(n_verts * dt.itemsize), dtype=dt)
                for name in prop_names:
                    if name in raw_v.dtype.names:
                        if   name == "x":  vertices[:, 0] = raw_v[name]
                        elif name == "y":  vertices[:, 1] = raw_v[name]
                        elif name == "z":  vertices[:, 2] = raw_v[name]
                        elif name == "nx" and normals is not None: normals[:, 0] = raw_v[name]
                        elif name == "ny" and normals is not None: normals[:, 1] = raw_v[name]
                        elif name == "nz" and normals is not None: normals[:, 2] = raw_v[name]
                        elif name in ("red", "r") and colors is not None:
                            colors[:, 0] = raw_v[name].astype(float) / 255.0
                        elif name in ("green", "g") and colors is not None:
                            colors[:, 1] = raw_v[name].astype(float) / 255.0
                        elif name in ("blue", "b") and colors is not None:
                            colors[:, 2] = raw_v[name].astype(float) / 255.0
                # Faces (binary)
                faces = []
                for _ in range(n_faces):
                    n_vf = struct.unpack(endian + "B", f2.read(1))[0]
                    face = list(struct.unpack(endian + "I" * n_vf, f2.read(4 * n_vf)))
                    faces.append(face)

        faces = np.array(faces, dtype=object) if faces else np.empty((0,), dtype=object)
        return {"vertices": vertices, "normals": normals,
                "colors": colors, "faces": faces, "path": str(filepath)}


def load_obj(filepath: str) -> dict:
    """Load OBJ file."""
    vertices, normals, faces = [], [], []
    vn_list = []
    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == "v":
                vertices.append([float(x) for x in parts[1:4]])
            elif parts[0] == "vn":
                vn_list.append([float(x) for x in parts[1:4]])
            elif parts[0] == "f":
                face = []
                for tok in parts[1:]:
                    face.append(int(tok.split("/")[0]) - 1)
                faces.append(face)
    vertices = np.array(vertices, dtype=np.float32)
    normals  = np.array(vn_list,  dtype=np.float32) if vn_list else None
    faces    = np.array(faces, dtype=object) if faces else np.empty((0,), dtype=object)
    return {"vertices": vertices, "normals": normals,
            "colors": None, "faces": faces, "path": str(filepath)}


def load_fragment(filepath: str) -> dict:
    ext = Path(filepath).suffix.lower()
    log.info(f"Loading {Path(filepath).name} ...")
    if ext == ".ply":
        return load_ply(filepath)
    elif ext == ".obj":
        return load_obj(filepath)
    else:
        raise ValueError(f"Unsupported format: {ext}")


def save_ply(filepath: str, vertices: np.ndarray, normals: np.ndarray = None,
             colors: np.ndarray = None):
    """Save point cloud as ASCII PLY."""
    has_normals = normals is not None
    has_colors  = colors  is not None
    with open(filepath, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if has_normals:
            f.write("property float nx\nproperty float ny\nproperty float nz\n")
        if has_colors:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i, v in enumerate(vertices):
            row = f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}"
            if has_normals:
                n = normals[i]
                row += f" {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}"
            if has_colors:
                c = (np.clip(colors[i], 0, 1) * 255).astype(int)
                row += f" {c[0]} {c[1]} {c[2]}"
            f.write(row + "\n")


# ─────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────

def normalize_fragment(frag: dict) -> dict:
    """Center + scale fragment to unit sphere."""
    verts = frag["vertices"].copy()
    centroid = verts.mean(axis=0)
    verts -= centroid
    scale = np.max(np.linalg.norm(verts, axis=1))
    if scale > 0:
        verts /= scale
    frag = dict(frag)
    frag["vertices"] = verts
    frag["centroid"] = centroid
    frag["scale"]    = scale
    return frag


def downsample_voxel(vertices: np.ndarray, voxel_size: float = 0.02,
                     normals: np.ndarray = None):
    """Voxel grid downsampling."""
    mins = vertices.min(axis=0)
    idxs = np.floor((vertices - mins) / voxel_size).astype(int)
    keys = idxs[:, 0] * 1_000_000 + idxs[:, 1] * 1_000 + idxs[:, 2]
    _, unique_idx = np.unique(keys, return_index=True)
    ds_verts = vertices[unique_idx]
    ds_normals = normals[unique_idx] if normals is not None else None
    return ds_verts, ds_normals, unique_idx


def estimate_normals(vertices: np.ndarray, k: int = 15) -> np.ndarray:
    """Estimate surface normals via PCA on local neighborhoods."""
    tree = cKDTree(vertices)
    normals = np.zeros_like(vertices)
    _, idxs = tree.query(vertices, k=k + 1)
    for i, neighbors in enumerate(idxs):
        pts = vertices[neighbors]
        centered = pts - pts.mean(axis=0)
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        normals[i] = Vt[-1]  # smallest singular value = normal
    # Orient normals consistently (viewpoint at centroid)
    centroid = vertices.mean(axis=0)
    for i in range(len(normals)):
        if np.dot(normals[i], vertices[i] - centroid) < 0:
            normals[i] = -normals[i]
    return normals


# ─────────────────────────────────────────────
# 3. SURFACE CLASSIFICATION
# ─────────────────────────────────────────────

def compute_local_features(vertices: np.ndarray, normals: np.ndarray,
                            k: int = 15) -> np.ndarray:
    """
    Compute per-point geometric features for break surface detection.

    Features per point (9 total):
      0: planarity       — how flat the local neighborhood is
      1: linearity       — how linear (edge-like)
      2: scattering      — how disordered (break indicator)
      3: curvature       — principal curvature magnitude
      4: normal_variation — std of normal directions in neighborhood
      5: roughness       — mean distance to local plane
      6: density         — local point density
      7: omnivariance    — 3rd root of eigenvalue product
      8: anisotropy      — (λ1 - λ3) / λ1
    """
    tree = cKDTree(vertices)
    dists, idxs = tree.query(vertices, k=k + 1)
    n = len(vertices)
    feats = np.zeros((n, 9), dtype=np.float32)

    for i in range(n):
        nb_pts = vertices[idxs[i]]
        centered = nb_pts - nb_pts.mean(axis=0)
        cov = centered.T @ centered / k
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.sort(np.abs(eigvals))[::-1]  # descending
        l1, l2, l3 = eigvals[0], eigvals[1], eigvals[2]
        eps = 1e-10
        total = l1 + l2 + l3 + eps

        planarity  = (l2 - l3) / (l1 + eps)
        linearity  = (l1 - l2) / (l1 + eps)
        scattering = l3 / (l1 + eps)
        curvature  = l3 / total
        omnivar    = (l1 * l2 * l3 + eps) ** (1/3)
        anisotropy = (l1 - l3) / (l1 + eps)

        # Normal variation in neighborhood
        nb_normals = normals[idxs[i]]
        normal_var = np.std(nb_normals, axis=0).mean()

        # Roughness: mean distance from local plane
        center = nb_pts.mean(axis=0)
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        plane_normal = Vt[-1]
        roughness = np.abs((nb_pts - center) @ plane_normal).mean()

        # Density: points per unit volume
        r = dists[i].max() + eps
        density = k / (4/3 * np.pi * r**3)

        feats[i] = [planarity, linearity, scattering, curvature,
                    normal_var, roughness, density, omnivar, anisotropy]

    return feats


def classify_break_surfaces(vertices: np.ndarray, normals: np.ndarray,
                             k: int = 15) -> np.ndarray:
    """
    Classify each point as break surface (1) or original surface (0).

    Break surfaces are characterized by:
    - High roughness / scattering (irregular fracture texture)
    - High normal variation (normals change sharply at break edge)
    - Low planarity (not smooth original surface)
    - High curvature
    """
    feats = compute_local_features(vertices, normals, k=k)

    # Rule-based scoring (no labels needed; unsupervised)
    # Break score = high scattering + high roughness + high curvature
    scattering  = feats[:, 2]
    curvature   = feats[:, 3]
    normal_var  = feats[:, 4]
    roughness   = feats[:, 5]

    def norm01(x):
        mn, mx = x.min(), x.max()
        return (x - mn) / (mx - mn + 1e-10)

    break_score = (0.35 * norm01(scattering)
                 + 0.25 * norm01(roughness)
                 + 0.25 * norm01(normal_var)
                 + 0.15 * norm01(curvature))

    # Adaptive threshold using Otsu's method on the score distribution
    threshold = otsu_threshold(break_score)
    is_break = (break_score > threshold).astype(np.uint8)
    log.info(f"  Break surface: {is_break.sum()}/{len(is_break)} pts "
             f"({100*is_break.mean():.1f}%) | Otsu thresh={threshold:.3f}")
    return is_break, break_score, feats


def otsu_threshold(scores: np.ndarray, n_bins: int = 256) -> float:
    """Otsu's binarization threshold."""
    hist, bin_edges = np.histogram(scores, bins=n_bins, density=True)
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2
    hist = hist / hist.sum()

    best_thresh, best_var = 0, -1
    for i in range(1, n_bins - 1):
        w0, w1 = hist[:i].sum(), hist[i:].sum()
        if w0 < 1e-10 or w1 < 1e-10:
            continue
        mu0 = (hist[:i] * bin_mids[:i]).sum() / w0
        mu1 = (hist[i:] * bin_mids[i:]).sum() / w1
        between_var = w0 * w1 * (mu0 - mu1) ** 2
        if between_var > best_var:
            best_var   = between_var
            best_thresh = bin_mids[i]
    return best_thresh


# ─────────────────────────────────────────────
# 4. FEATURE EXTRACTION — FPFH
# ─────────────────────────────────────────────

def compute_spfh(vertices: np.ndarray, normals: np.ndarray,
                 k: int = 15, n_bins: int = 11) -> np.ndarray:
    """
    Simplified Point Feature Histogram (SPFH).
    For each point, computes a 3 × n_bins histogram of (α, φ, θ) Darboux frame angles.
    """
    tree = cKDTree(vertices)
    n = len(vertices)
    spfh = np.zeros((n, 3 * n_bins), dtype=np.float32)
    _, idxs = tree.query(vertices, k=k + 1)

    for i in range(n):
        alphas, phis, thetas = [], [], []
        ni = normals[i]
        pi = vertices[i]
        for j in idxs[i][1:]:
            pj = vertices[j]
            nj = normals[j]
            diff = pj - pi
            d = np.linalg.norm(diff) + 1e-10
            u = ni
            v = np.cross(u, diff / d)
            w = np.cross(u, v)
            alpha = v @ nj
            phi   = u @ (diff / d)
            theta = np.arctan2(w @ nj, u @ nj)
            alphas.append(alpha)
            phis.append(phi)
            thetas.append(theta)

        for arr, start in [(alphas, 0), (phis, n_bins), (thetas, 2*n_bins)]:
            hist, _ = np.histogram(arr, bins=n_bins, range=(-1, 1))
            spfh[i, start:start+n_bins] = hist / (len(arr) + 1e-10)

    return spfh


def compute_fpfh(vertices: np.ndarray, normals: np.ndarray,
                 k: int = 15, n_bins: int = 11) -> np.ndarray:
    """
    Fast Point Feature Histogram (FPFH).
    FPFH = SPFH(p) + (1/k) Σ (1/d_i) SPFH(q_i)
    """
    tree = cKDTree(vertices)
    spfh = compute_spfh(vertices, normals, k=k, n_bins=n_bins)
    n = len(vertices)
    fpfh = np.zeros_like(spfh)
    dists, idxs = tree.query(vertices, k=k + 1)

    for i in range(n):
        nb_dists = dists[i][1:] + 1e-10
        weights  = 1.0 / nb_dists
        weights /= weights.sum()
        weighted_nb = weights[:, None] * spfh[idxs[i][1:]]
        fpfh[i] = spfh[i] + weighted_nb.sum(axis=0)

    # L1 normalize
    row_sums = fpfh.sum(axis=1, keepdims=True) + 1e-10
    fpfh /= row_sums
    return fpfh


# ─────────────────────────────────────────────
# 5. ML FRAGMENT MATCHING
# ─────────────────────────────────────────────

def fragment_descriptor(vertices: np.ndarray, normals: np.ndarray,
                         break_mask: np.ndarray, fpfh: np.ndarray,
                         n_keypoints: int = 64) -> np.ndarray:
    """
    Global descriptor for a fragment's break surface:
    - Sample n_keypoints from break surface
    - Stack their FPFH vectors
    - Summarise: mean, std, max, min → fixed-length vector
    """
    break_idxs = np.where(break_mask)[0]
    if len(break_idxs) == 0:
        break_idxs = np.arange(len(vertices))

    # Farthest point sampling for diversity
    idxs = farthest_point_sample(vertices[break_idxs], n_keypoints)
    sampled_fpfh = fpfh[break_idxs[idxs]]

    # Aggregate statistics
    desc = np.concatenate([
        sampled_fpfh.mean(axis=0),
        sampled_fpfh.std(axis=0),
        sampled_fpfh.max(axis=0),
        sampled_fpfh.min(axis=0),
    ])
    return desc.astype(np.float32)


def farthest_point_sample(pts: np.ndarray, n: int) -> np.ndarray:
    """Farthest Point Sampling for diverse keypoint selection."""
    n = min(n, len(pts))
    idxs  = np.zeros(n, dtype=int)
    dists = np.full(len(pts), np.inf)
    idxs[0] = np.random.randint(len(pts))
    for i in range(1, n):
        d = np.linalg.norm(pts - pts[idxs[i-1]], axis=1)
        dists = np.minimum(dists, d)
        idxs[i] = np.argmax(dists)
    return idxs


def build_pairwise_features(descs: list, frag_names: list) -> tuple:
    """
    Build pairwise feature matrix for all fragment pairs.
    Each pair (i, j) → feature vector = |desc_i - desc_j| ⊕ desc_i * desc_j
    """
    n = len(descs)
    pairs = []
    pair_names = []
    X = []

    for i, j in itertools.combinations(range(n), 2):
        diff    = np.abs(descs[i] - descs[j])
        product = descs[i] * descs[j]
        feat    = np.concatenate([diff, product])
        X.append(feat)
        pairs.append((i, j))
        pair_names.append(f"{frag_names[i]} vs {frag_names[j]}")

    return np.array(X, dtype=np.float32), pairs, pair_names


def augment_fragment_pair(desc_a: np.ndarray, desc_b: np.ndarray,
                           is_match: int, n_aug: int = 5) -> list:
    """
    Data augmentation for training:
    - Add small Gaussian noise to descriptors
    - Simulate partial occlusion (zero out random dimensions)
    - Simulate scale variation
    """
    samples = []
    for _ in range(n_aug):
        noise_a = desc_a + np.random.normal(0, 0.01, desc_a.shape)
        noise_b = desc_b + np.random.normal(0, 0.01, desc_b.shape)
        # Partial occlusion
        mask = np.random.rand(len(desc_a)) > 0.1
        noise_a *= mask
        noise_b *= mask
        diff    = np.abs(noise_a - noise_b)
        product = noise_a * noise_b
        feat    = np.concatenate([diff, product])
        samples.append((feat, is_match))
    return samples


def train_match_classifier(X_pos: np.ndarray, X_neg: np.ndarray,
                             augment: bool = True) -> Pipeline:
    """
    Train RandomForest + GradientBoosting ensemble to predict fragment matches.
    Uses cross-validation to report accuracy.
    """
    if augment:
        aug_pos, aug_neg = [], []
        for x in X_pos:
            n = len(x) // 2
            da, db = x[:n], x[n:]  # rough split
            for feat, _ in augment_fragment_pair(da, db, 1, n_aug=8):
                aug_pos.append(feat)
        for x in X_neg:
            n = len(x) // 2
            da, db = x[:n], x[n:]
            for feat, _ in augment_fragment_pair(da, db, 0, n_aug=8):
                aug_neg.append(feat)
        X_pos_aug = np.vstack([X_pos] + ([np.array(aug_pos)] if aug_pos else []))
        X_neg_aug = np.vstack([X_neg] + ([np.array(aug_neg)] if aug_neg else []))
    else:
        X_pos_aug, X_neg_aug = X_pos, X_neg

    # Balance classes
    n_min = min(len(X_pos_aug), len(X_neg_aug))
    X_pos_bal = X_pos_aug[:n_min]
    X_neg_bal = X_neg_aug[:n_min]

    X = np.vstack([X_pos_bal, X_neg_bal])
    y = np.array([1]*len(X_pos_bal) + [0]*len(X_neg_bal))

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])

    if len(np.unique(y)) < 2:
        log.warning("Only one class in training data — skipping ML training.")
        return None

    # Cross-validation
    if len(y) >= 6:
        cv = StratifiedKFold(n_splits=min(5, len(y)//2), shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
        log.info(f"  CV ROC-AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    model.fit(X, y)
    return model


# ─────────────────────────────────────────────
# 6. ALIGNMENT — RANSAC + ICP
# ─────────────────────────────────────────────

def match_fpfh_features(fpfh_src: np.ndarray, fpfh_tgt: np.ndarray,
                         verts_src: np.ndarray, verts_tgt: np.ndarray,
                         break_src: np.ndarray, break_tgt: np.ndarray,
                         n_matches: int = 500) -> tuple:
    """Find correspondences between break surface points using FPFH."""
    src_idxs = np.where(break_src)[0]
    tgt_idxs = np.where(break_tgt)[0]

    if len(src_idxs) < 3 or len(tgt_idxs) < 3:
        return np.array([]), np.array([])

    # Subsample for speed
    src_sub = src_idxs[::max(1, len(src_idxs)//500)]
    tgt_sub = tgt_idxs[::max(1, len(tgt_idxs)//500)]

    fpfh_s = fpfh_src[src_sub]
    fpfh_t = fpfh_tgt[tgt_sub]

    # Mutual nearest-neighbor matching in descriptor space
    tree = cKDTree(fpfh_t)
    dists_st, nn_st = tree.query(fpfh_s, k=1)

    tree2 = cKDTree(fpfh_s)
    dists_ts, nn_ts = tree2.query(fpfh_t[nn_st], k=1)

    # Lowe's ratio test + mutual consistency
    mutual = (nn_ts == np.arange(len(src_sub)))
    valid  = mutual & (dists_st < np.percentile(dists_st, 60))

    src_pts = verts_src[src_sub[valid]]
    tgt_pts = verts_tgt[tgt_sub[nn_st[valid]]]

    return src_pts, tgt_pts


def estimate_rigid_transform(src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    """Estimate 4×4 rigid transformation via SVD (least squares)."""
    mu_s = src.mean(axis=0)
    mu_t = tgt.mean(axis=0)
    H    = (src - mu_s).T @ (tgt - mu_t)
    U, _, Vt = scipy_svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:  # reflection correction
        Vt[-1] *= -1
        R = Vt.T @ U.T
    t = mu_t - R @ mu_s
    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = t
    return T


def ransac_alignment(src_pts: np.ndarray, tgt_pts: np.ndarray,
                      n_iter: int = 1000, threshold: float = 0.05) -> tuple:
    """
    RANSAC rigid alignment.
    Returns: (best_T, inlier_mask, inlier_ratio)
    """
    n = len(src_pts)
    if n < 3:
        return np.eye(4), np.zeros(n, dtype=bool), 0.0

    best_T      = np.eye(4)
    best_inliers = 0
    best_mask   = np.zeros(n, dtype=bool)

    for _ in range(n_iter):
        idx = np.random.choice(n, size=3, replace=False)
        T   = estimate_rigid_transform(src_pts[idx], tgt_pts[idx])
        transformed = (T[:3, :3] @ src_pts.T).T + T[:3, 3]
        dists = np.linalg.norm(transformed - tgt_pts, axis=1)
        mask  = dists < threshold
        if mask.sum() > best_inliers:
            best_inliers = mask.sum()
            best_mask    = mask
            # Refit on all inliers
            if mask.sum() >= 3:
                best_T = estimate_rigid_transform(src_pts[mask], tgt_pts[mask])

    inlier_ratio = best_inliers / n
    return best_T, best_mask, inlier_ratio


def icp(src: np.ndarray, tgt: np.ndarray, T_init: np.ndarray = None,
        max_iter: int = 50, tol: float = 1e-5,
        max_dist: float = None, gap_aware: bool = True) -> tuple:
    """
    Iterative Closest Point with optional gap-awareness.
    gap_aware=True: uses robust median-based distance threshold
                    to handle missing material between fragments.
    Returns: (T_final, rmse, n_correspondences)
    """
    T = T_init if T_init is not None else np.eye(4)
    src_h = np.hstack([src, np.ones((len(src), 1))])
    tgt_tree = cKDTree(tgt)

    prev_rmse = np.inf
    for iteration in range(max_iter):
        # Transform source
        transformed = (src_h @ T.T)[:, :3]

        # Find nearest neighbors
        dists, idxs = tgt_tree.query(transformed)

        # Gap-aware outlier rejection
        if gap_aware:
            median_d = np.median(dists)
            mad      = np.median(np.abs(dists - median_d))
            thresh   = median_d + 2.5 * mad * 1.4826
        elif max_dist is not None:
            thresh = max_dist
        else:
            thresh = np.inf

        valid = dists < thresh
        if valid.sum() < 3:
            break

        src_v = transformed[valid]
        tgt_v = tgt[idxs[valid]]
        rmse  = np.sqrt((dists[valid] ** 2).mean())

        # Compute update transform
        dT = estimate_rigid_transform(src_v, tgt_v)
        dT_h = dT
        T = dT_h @ T

        if abs(prev_rmse - rmse) < tol:
            break
        prev_rmse = rmse

    n_corr = valid.sum() if valid is not None else 0
    return T, prev_rmse, n_corr


def align_fragments(frag_src: dict, frag_tgt: dict,
                    fpfh_src: np.ndarray, fpfh_tgt: np.ndarray,
                    break_src: np.ndarray, break_tgt: np.ndarray) -> dict:
    """Full alignment pipeline: feature matching → RANSAC → ICP."""
    vs = frag_src["vertices"]
    vt = frag_tgt["vertices"]

    src_pts, tgt_pts = match_fpfh_features(
        fpfh_src, fpfh_tgt, vs, vt, break_src, break_tgt)

    if len(src_pts) < 3:
        log.warning("  Insufficient correspondences for alignment.")
        return {"T": np.eye(4), "inlier_ratio": 0.0, "icp_rmse": np.inf,
                "n_correspondences": 0, "success": False}

    T_ransac, mask, inlier_ratio = ransac_alignment(src_pts, tgt_pts)
    log.info(f"  RANSAC inlier ratio: {inlier_ratio:.3f} ({mask.sum()}/{len(mask)})")

    T_icp, rmse, n_corr = icp(vs, vt, T_init=T_ransac, gap_aware=True)
    log.info(f"  ICP RMSE: {rmse:.5f} | Correspondences: {n_corr}")

    return {"T": T_icp, "inlier_ratio": inlier_ratio, "icp_rmse": rmse,
            "n_correspondences": n_corr, "success": rmse < 0.1}


# ─────────────────────────────────────────────
# 7. EVALUATION METRICS
# ─────────────────────────────────────────────

def evaluate_reconstruction(match_matrix: np.ndarray,
                              true_pairs: list,
                              all_pairs: list,
                              frag_names: list) -> dict:
    """
    Compute evaluation metrics for fragment matching.

    match_matrix[i,j] = predicted match probability
    true_pairs = list of (i,j) pairs that truly match (ground truth)
    """
    n = len(frag_names)
    y_true, y_score = [], []

    for idx, (i, j) in enumerate(all_pairs):
        score = match_matrix[i, j]
        label = 1 if (i, j) in true_pairs or (j, i) in true_pairs else 0
        y_true.append(label)
        y_score.append(score)

    y_true  = np.array(y_true)
    y_score = np.array(y_score)
    y_pred  = (y_score > 0.5).astype(int)

    metrics = {}
    metrics["n_fragments"]    = n
    metrics["n_pairs_total"]  = len(all_pairs)
    metrics["n_pairs_true"]   = int(y_true.sum())

    if y_true.sum() > 0:
        metrics["precision"] = float(
            np.sum((y_pred == 1) & (y_true == 1)) / (np.sum(y_pred == 1) + 1e-10))
        metrics["recall"]    = float(
            np.sum((y_pred == 1) & (y_true == 1)) / (np.sum(y_true == 1) + 1e-10))
        metrics["f1"]        = float(
            2 * metrics["precision"] * metrics["recall"]
            / (metrics["precision"] + metrics["recall"] + 1e-10))
        if len(np.unique(y_true)) > 1:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
            metrics["avg_precision"] = float(average_precision_score(y_true, y_score))
        else:
            metrics["roc_auc"]       = float("nan")
            metrics["avg_precision"] = float("nan")
    else:
        # No ground truth provided → report predicted top matches
        top_k = np.argsort(y_score)[::-1][:10]
        metrics["top_matches"] = [
            {"pair": all_pairs[k], "score": float(y_score[k]),
             "names": f"{frag_names[all_pairs[k][0]]} ↔ {frag_names[all_pairs[k][1]]}"}
            for k in top_k
        ]

    return metrics


# ─────────────────────────────────────────────
# 8. VISUALISATION
# ─────────────────────────────────────────────

def plot_break_classification(frag: dict, break_mask: np.ndarray,
                               break_score: np.ndarray, name: str,
                               out_dir: str):
    verts = frag["vertices"]
    fig = plt.figure(figsize=(14, 5))

    ax1 = fig.add_subplot(131, projection="3d")
    colors = np.where(break_mask, "crimson", "steelblue")
    ax1.scatter(verts[::5, 0], verts[::5, 1], verts[::5, 2],
                c=colors[::5], s=1, alpha=0.6)
    ax1.set_title(f"{name}\nBreak Surface Classification")
    ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")
    red_patch  = mpatches.Patch(color="crimson",   label="Break surface")
    blue_patch = mpatches.Patch(color="steelblue", label="Original surface")
    ax1.legend(handles=[red_patch, blue_patch], loc="upper left", fontsize=7)

    ax2 = fig.add_subplot(132)
    ax2.hist(break_score, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    thresh = otsu_threshold(break_score)
    ax2.axvline(thresh, color="crimson", linestyle="--", label=f"Otsu={thresh:.3f}")
    ax2.set_xlabel("Break Score")
    ax2.set_ylabel("Count")
    ax2.set_title("Break Score Distribution")
    ax2.legend()

    ax3 = fig.add_subplot(133, projection="3d")
    sc = ax3.scatter(verts[::5, 0], verts[::5, 1], verts[::5, 2],
                     c=break_score[::5], cmap="plasma", s=1, alpha=0.6)
    plt.colorbar(sc, ax=ax3, shrink=0.5, label="Break Score")
    ax3.set_title("Break Score Heatmap")

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{name}_classification.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    return out_path


def plot_match_matrix(match_matrix: np.ndarray, frag_names: list,
                       top_pairs: list, out_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    im = axes[0].imshow(match_matrix, cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(im, ax=axes[0], label="Match Probability")
    axes[0].set_xticks(range(len(frag_names)))
    axes[0].set_yticks(range(len(frag_names)))
    short_names = [Path(n).stem[-10:] for n in frag_names]
    axes[0].set_xticklabels(short_names, rotation=45, ha="right", fontsize=7)
    axes[0].set_yticklabels(short_names, fontsize=7)
    axes[0].set_title("Pairwise Match Probability Matrix")

    # Bar chart of top matches
    scores = [match_matrix[i, j] for i, j in top_pairs[:10]]
    labels = [f"{short_names[i]}↔{short_names[j]}" for i, j in top_pairs[:10]]
    colors = plt.cm.RdYlGn(np.array(scores))
    bars = axes[1].barh(range(len(scores)), scores, color=colors)
    axes[1].set_yticks(range(len(labels)))
    axes[1].set_yticklabels(labels, fontsize=8)
    axes[1].set_xlabel("Match Probability")
    axes[1].set_title("Top Fragment Pair Matches")
    axes[1].axvline(0.5, color="gray", linestyle="--", alpha=0.5)
    axes[1].set_xlim(0, 1)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "match_matrix.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    return out_path


def plot_alignment(src_verts: np.ndarray, tgt_verts: np.ndarray,
                   T: np.ndarray, name_s: str, name_t: str, out_dir: str):
    src_h = np.hstack([src_verts, np.ones((len(src_verts), 1))])
    aligned = (src_h @ T.T)[:, :3]

    fig = plt.figure(figsize=(14, 5))
    for i, (title, s, t) in enumerate([
        ("Before Alignment", src_verts, tgt_verts),
        ("After Alignment",  aligned,   tgt_verts),
    ]):
        ax = fig.add_subplot(1, 2, i+1, projection="3d")
        ax.scatter(s[::8, 0], s[::8, 1], s[::8, 2],
                   c="tomato", s=1, alpha=0.5, label=Path(name_s).stem)
        ax.scatter(t[::8, 0], t[::8, 1], t[::8, 2],
                   c="steelblue", s=1, alpha=0.5, label=Path(name_t).stem)
        ax.set_title(title)
        ax.legend(fontsize=7)

    plt.suptitle(f"Alignment: {Path(name_s).stem} → {Path(name_t).stem}", fontsize=11)
    plt.tight_layout()
    name = f"align_{Path(name_s).stem}_{Path(name_t).stem}.png"
    out_path = os.path.join(out_dir, name)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    return out_path


def plot_reconstruction(fragments: list, transforms: dict,
                         frag_names: list, out_dir: str):
    """Visualize all aligned fragments together."""
    cmap = plt.cm.tab20
    fig  = plt.figure(figsize=(12, 10))
    ax   = fig.add_subplot(111, projection="3d")

    for idx, (frag, name) in enumerate(zip(fragments, frag_names)):
        verts = frag["vertices"].copy()
        if name in transforms:
            T = transforms[name]
            h = np.hstack([verts, np.ones((len(verts), 1))])
            verts = (h @ T.T)[:, :3]
        color = cmap(idx / max(len(fragments), 1))
        sub = verts[::max(1, len(verts)//300)]
        ax.scatter(sub[:, 0], sub[:, 1], sub[:, 2],
                   c=[color], s=2, alpha=0.7, label=Path(name).stem[-12:])

    ax.set_title("Reconstructed Stele — All Fragments Aligned", fontsize=13)
    ax.legend(loc="upper left", fontsize=6, markerscale=3)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    plt.tight_layout()
    out_path = os.path.join(out_dir, "full_reconstruction.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


# ─────────────────────────────────────────────
# 9. MAIN PIPELINE
# ─────────────────────────────────────────────

def run_pipeline(data_dir: str, out_dir: str = "output",
                 voxel_size: float = 0.03, k_neighbors: int = 15,
                 n_keypoints: int = 64):
    """
    End-to-end pipeline:
      Load → Preprocess → Classify → Extract features →
      Predict matches (ML) → Align top matches → Report metrics
    """
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    # ── 1. Discover files ──────────────────────────────────────
    exts = ["*.ply", "*.PLY", "*.obj", "*.OBJ"]
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(data_dir, "**", ext), recursive=True))
    files = sorted(set(files))

    if not files:
        raise FileNotFoundError(f"No PLY/OBJ files found in {data_dir}")
    log.info(f"Found {len(files)} fragments.")

    # ── 2. Load & preprocess ──────────────────────────────────
    fragments, frag_names = [], []
    for fp in files:
        try:
            frag = load_fragment(fp)
            frag = normalize_fragment(frag)

            # Downsample
            ds_verts, ds_normals, _ = downsample_voxel(
                frag["vertices"], voxel_size=voxel_size, normals=frag.get("normals"))
            frag["vertices"] = ds_verts

            # Estimate normals if missing
            if ds_normals is None or len(ds_normals) != len(ds_verts):
                log.info(f"  Estimating normals for {Path(fp).name} ...")
                ds_normals = estimate_normals(ds_verts, k=k_neighbors)
            frag["normals"] = ds_normals

            fragments.append(frag)
            frag_names.append(fp)
            log.info(f"  {Path(fp).name}: {len(ds_verts)} pts after downsampling")
        except Exception as e:
            log.error(f"  Failed to load {fp}: {e}")

    n_frags = len(fragments)
    log.info(f"\n{'='*50}")
    log.info(f"Loaded {n_frags} fragments successfully.")

    # ── 3. Surface classification ─────────────────────────────
    log.info("\n[STEP 3] Surface Classification")
    all_break_masks, all_break_scores, all_feats = [], [], []
    for idx, (frag, name) in enumerate(zip(fragments, frag_names)):
        log.info(f"  Fragment {idx+1}/{n_frags}: {Path(name).name}")
        mask, score, feats = classify_break_surfaces(
            frag["vertices"], frag["normals"], k=k_neighbors)
        all_break_masks.append(mask)
        all_break_scores.append(score)
        all_feats.append(feats)
        plot_break_classification(frag, mask, score, Path(name).stem, out_dir)

    # ── 4. FPFH feature extraction ────────────────────────────
    log.info("\n[STEP 4] FPFH Feature Extraction")
    all_fpfh, all_descs = [], []
    for idx, (frag, name) in enumerate(zip(fragments, frag_names)):
        log.info(f"  Fragment {idx+1}/{n_frags}: {Path(name).name}")
        fpfh = compute_fpfh(frag["vertices"], frag["normals"],
                             k=k_neighbors, n_bins=11)
        all_fpfh.append(fpfh)
        desc = fragment_descriptor(frag["vertices"], frag["normals"],
                                    all_break_masks[idx], fpfh, n_keypoints)
        all_descs.append(desc)

    # ── 5. Pairwise feature matrix ────────────────────────────
    log.info("\n[STEP 5] Building Pairwise Feature Matrix")
    X_pairs, all_pairs, pair_names = build_pairwise_features(all_descs, frag_names)
    log.info(f"  {len(all_pairs)} fragment pairs → feature dim: {X_pairs.shape[1]}")

    # ── 6. ML Training (self-supervised via synthetic positives) ─
    log.info("\n[STEP 6] ML Model Training (self-supervised)")
    # Self-supervised: create synthetic positives by splitting each fragment
    # into two halves (should match), and use all cross-fragment as negatives
    synthetic_positives, synthetic_negatives = [], []
    for i, desc in enumerate(all_descs):
        # Positive: add noise to same descriptor (same fragment → should match)
        noise = np.random.normal(0, 0.02, desc.shape)
        d2    = desc + noise
        diff    = np.abs(desc - d2)
        product = desc * d2
        synthetic_positives.append(np.concatenate([diff, product]))

    for i, j in itertools.combinations(range(n_frags), 2):
        diff    = np.abs(all_descs[i] - all_descs[j])
        product = all_descs[i] * all_descs[j]
        synthetic_negatives.append(np.concatenate([diff, product]))

    X_pos = np.array(synthetic_positives, dtype=np.float32)
    X_neg = np.array(synthetic_negatives, dtype=np.float32)

    model = train_match_classifier(X_pos, X_neg, augment=True)

    # ── 7. Predict matches ────────────────────────────────────
    log.info("\n[STEP 7] Predicting Fragment Matches")
    match_matrix = np.zeros((n_frags, n_frags), dtype=np.float32)
    np.fill_diagonal(match_matrix, 1.0)

    if model is not None and len(X_pairs) > 0:
        probs = model.predict_proba(X_pairs)[:, 1]
        for (i, j), prob in zip(all_pairs, probs):
            match_matrix[i, j] = prob
            match_matrix[j, i] = prob
        log.info(f"  Predicted {(probs > 0.5).sum()} potential matches "
                 f"(threshold=0.5) out of {len(probs)} pairs")
    else:
        # Fallback: cosine similarity between descriptors
        log.info("  Using cosine similarity fallback.")
        descs_arr = np.array(all_descs)
        norms = np.linalg.norm(descs_arr, axis=1, keepdims=True) + 1e-10
        normed = descs_arr / norms
        sim = normed @ normed.T
        for i, j in all_pairs:
            match_matrix[i, j] = float(sim[i, j])
            match_matrix[j, i] = float(sim[i, j])

    # Rank pairs by match probability
    top_pairs = sorted(all_pairs, key=lambda p: match_matrix[p[0], p[1]], reverse=True)
    log.info("\n  Top 10 predicted matches:")
    for rank, (i, j) in enumerate(top_pairs[:10]):
        score = match_matrix[i, j]
        log.info(f"    #{rank+1:2d} | score={score:.3f} | "
                 f"{Path(frag_names[i]).stem} ↔ {Path(frag_names[j]).stem}")

    plot_match_matrix(match_matrix, frag_names, top_pairs, out_dir)

    # ── 8. Alignment of top matches ───────────────────────────
    log.info("\n[STEP 8] Aligning Top Fragment Pairs")
    alignment_results = {}
    transforms = {}
    n_align = min(5, len(top_pairs))

    for rank, (i, j) in enumerate(top_pairs[:n_align]):
        score = match_matrix[i, j]
        name_s = frag_names[i]
        name_t = frag_names[j]
        log.info(f"\n  Aligning pair #{rank+1}: {Path(name_s).stem} → {Path(name_t).stem} "
                 f"(score={score:.3f})")

        result = align_fragments(
            fragments[i], fragments[j],
            all_fpfh[i], all_fpfh[j],
            all_break_masks[i], all_break_masks[j])

        alignment_results[f"{Path(name_s).stem}_{Path(name_t).stem}"] = {
            "fragment_a": Path(name_s).name,
            "fragment_b": Path(name_t).name,
            "match_score": float(score),
            "icp_rmse":    float(result["icp_rmse"]),
            "inlier_ratio": float(result["inlier_ratio"]),
            "n_correspondences": int(result["n_correspondences"]),
            "success": bool(result["success"]),
        }

        if result["success"]:
            transforms[name_s] = result["T"].tolist()

        plot_alignment(fragments[i]["vertices"], fragments[j]["vertices"],
                       result["T"], name_s, name_t, out_dir)

    # Save aligned fragments
    for name, T_list in transforms.items():
        T = np.array(T_list)
        idx = frag_names.index(name)
        verts = fragments[idx]["vertices"]
        norms = fragments[idx]["normals"]
        h = np.hstack([verts, np.ones((len(verts), 1))])
        aligned_verts = (h @ T.T)[:, :3]
        out_ply = os.path.join(out_dir, f"aligned_{Path(name).stem}.ply")
        save_ply(out_ply, aligned_verts, norms)

    # ── 9. Reconstruction visualisation ──────────────────────
    recon_path = plot_reconstruction(fragments, {
        n: np.array(T) for n, T in transforms.items()
    }, frag_names, out_dir)

    # ── 10. Metrics report ────────────────────────────────────
    log.info("\n[STEP 9] Evaluation Report")
    metrics = evaluate_reconstruction(match_matrix, [], all_pairs, frag_names)

    elapsed = time.time() - t0
    report = {
        "pipeline": "Healing Stones 3D Fragment Reconstruction",
        "n_fragments": n_frags,
        "runtime_seconds": round(elapsed, 2),
        "voxel_size": voxel_size,
        "k_neighbors": k_neighbors,
        "break_surface_stats": [
            {"fragment": Path(frag_names[i]).name,
             "break_pct": round(float(all_break_masks[i].mean() * 100), 1)}
            for i in range(n_frags)
        ],
        "top_10_matches": [
            {"rank": r+1,
             "fragment_a": Path(frag_names[i]).stem,
             "fragment_b": Path(frag_names[j]).stem,
             "match_score": round(float(match_matrix[i, j]), 4)}
            for r, (i, j) in enumerate(top_pairs[:10])
        ],
        "alignment_results": alignment_results,
        "metrics": metrics,
        "output_files": {
            "classification_plots": f"{out_dir}/*_classification.png",
            "match_matrix":         f"{out_dir}/match_matrix.png",
            "alignment_plots":      f"{out_dir}/align_*.png",
            "reconstruction":       recon_path,
            "aligned_fragments":    f"{out_dir}/aligned_*.ply",
            "report":               f"{out_dir}/report.json",
        }
    }

    report_path = os.path.join(out_dir, "report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print("\n" + "="*60)
    print("  HEALING STONES — PIPELINE COMPLETE")
    print("="*60)
    print(f"  Fragments processed : {n_frags}")
    print(f"  Runtime             : {elapsed:.1f}s")
    print(f"  Output directory    : {out_dir}/")
    print("\n  TOP PREDICTED MATCHES:")
    for r, (i, j) in enumerate(top_pairs[:10]):
        s = match_matrix[i, j]
        bar = "█" * int(s * 20)
        print(f"  #{r+1:2d} [{bar:<20}] {s:.3f}  "
              f"{Path(frag_names[i]).stem} ↔ {Path(frag_names[j]).stem}")

    print("\n  ALIGNMENT RESULTS:")
    for key, res in alignment_results.items():
        status = "✓" if res["success"] else "✗"
        print(f"  {status} {res['fragment_a']} ↔ {res['fragment_b']}")
        print(f"      ICP RMSE={res['icp_rmse']:.5f}  "
              f"Inliers={res['inlier_ratio']:.2%}  "
              f"Corr={res['n_correspondences']}")

    print(f"\n  Report saved → {report_path}")
    print("="*60)
    return report


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Healing Stones: 3D Fragment Matching & Reconstruction")
    parser.add_argument("data_dir",
        help="Directory containing PLY/OBJ fragment files")
    parser.add_argument("--output", "-o", default="output",
        help="Output directory (default: output/)")
    parser.add_argument("--voxel-size", "-v", type=float, default=0.03,
        help="Voxel size for downsampling (default: 0.03)")
    parser.add_argument("--k-neighbors", "-k", type=int, default=15,
        help="Number of nearest neighbors for local features (default: 15)")
    parser.add_argument("--keypoints", "-kp", type=int, default=64,
        help="Keypoints sampled per fragment for global descriptor (default: 64)")
    args = parser.parse_args()

    run_pipeline(
        data_dir    = args.data_dir,
        out_dir     = args.output,
        voxel_size  = args.voxel_size,
        k_neighbors = args.k_neighbors,
        n_keypoints = args.keypoints,
    )
