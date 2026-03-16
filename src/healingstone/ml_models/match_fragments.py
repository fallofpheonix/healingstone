"""Fragment matching using learned Siamese embeddings."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..core.features import FeatureBundle, build_augmented_descriptor
from ..core.preprocess import Fragment
from .train_model import SiameseModelBundle, cosine_similarity_matrix, encode_descriptors, train_siamese_model

LOG = logging.getLogger(__name__)


def load_pair_labels(labels_csv: Optional[Path], fragments: List[Fragment]) -> Dict[Tuple[int, int], int]:
    """Load optional supervised labels from CSV columns: frag_a, frag_b, label."""
    if labels_csv is None:
        return {}
    if not labels_csv.exists():
        LOG.warning("Labels file not found: %s", labels_csv)
        return {}

    lookup = {}
    for frag in fragments:
        lookup[str(frag.path)] = frag.idx
        lookup[frag.path.name] = frag.idx
        lookup[frag.path.stem] = frag.idx

    labels: Dict[Tuple[int, int], int] = {}
    with labels_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"frag_a", "frag_b", "label"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"labels csv must contain columns {sorted(required)}")

        for row in reader:
            a_name = (row.get("frag_a") or "").strip()
            b_name = (row.get("frag_b") or "").strip()
            if a_name not in lookup or b_name not in lookup:
                continue
            a, b = lookup[a_name], lookup[b_name]
            if a == b:
                continue
            raw_label = (row.get("label") or "").strip()
            try:
                y = 1 if int(float(raw_label)) > 0 else 0
            except Exception:
                # Skip unlabeled or malformed rows.
                continue
            pair = (min(a, b), max(a, b))
            labels[pair] = y

    LOG.info("Loaded %d labeled pairs", len(labels))
    return labels


def _all_pairs(n: int) -> List[Tuple[int, int]]:
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j))
    return pairs


def _descriptor_matrix(features: Dict[int, FeatureBundle], n: int) -> np.ndarray:
    desc = []
    for i in range(n):
        if i not in features:
            raise KeyError(f"Missing features for fragment idx {i}")
        desc.append(features[i].descriptor)
    return np.vstack(desc).astype(np.float32)


def _build_self_supervised_pairs(
    fragments: List[Fragment],
    features: Dict[int, FeatureBundle],
    labels: Dict[Tuple[int, int], int],
    augment_rotations: bool,
    augment_count: int,
    rng: np.random.Generator,
    k_neighbors: int,
    fpfh_radius: float,
    fpfh_max_nn: int,
    dbscan_eps: float,
    dbscan_min_samples: int,
    n_keypoints: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    """Build training pairs for contrastive learning."""
    n = len(fragments)
    desc = _descriptor_matrix(features, n=n)

    x1, x2, y = [], [], []

    # Supervised pairs if available.
    for (i, j), lbl in labels.items():
        x1.append(desc[i])
        x2.append(desc[j])
        y.append(float(lbl))

    # Positive self-supervised pairs via augmentation.
    for frag in fragments:
        base = desc[frag.idx]
        x1.append(base)
        x2.append(base + rng.normal(0.0, 0.01, size=base.shape).astype(np.float32))
        y.append(1.0)

        if augment_rotations:
            for _ in range(max(1, augment_count)):
                aug_desc = build_augmented_descriptor(
                    fragment=frag,
                    k_neighbors=k_neighbors,
                    fpfh_radius=fpfh_radius,
                    fpfh_max_nn=fpfh_max_nn,
                    dbscan_eps=dbscan_eps,
                    dbscan_min_samples=dbscan_min_samples,
                    n_keypoints=n_keypoints,
                    rng=rng,
                )
                x1.append(base)
                x2.append(aug_desc)
                y.append(1.0)

    # Hard negatives from high descriptor cosine among different fragments.
    all_pairs = _all_pairs(n)
    neg_candidates = []
    for i, j in all_pairs:
        if (i, j) in labels and labels[(i, j)] == 1:
            continue
        di = desc[i]
        dj = desc[j]
        sim = float(np.dot(di, dj) / ((np.linalg.norm(di) * np.linalg.norm(dj)) + 1e-12))
        neg_candidates.append((sim, i, j))

    neg_candidates.sort(key=lambda t: t[0], reverse=True)
    n_pos = int(np.sum(np.array(y, dtype=np.float32) > 0.5))
    n_target_neg = max(n_pos, int(1.5 * n_pos))

    for _, i, j in neg_candidates[: min(len(neg_candidates), n_target_neg)]:
        x1.append(desc[i])
        x2.append(desc[j])
        y.append(0.0)

    x1_arr = np.vstack(x1).astype(np.float32)
    x2_arr = np.vstack(x2).astype(np.float32)
    y_arr = np.array(y, dtype=np.float32)

    stats = {
        "n_train_pairs": int(y_arr.shape[0]),
        "n_pos_pairs": int(np.sum(y_arr > 0.5)),
        "n_neg_pairs": int(np.sum(y_arr <= 0.5)),
    }
    return x1_arr, x2_arr, y_arr, stats


def reciprocal_topk_pairs(similarity: np.ndarray, top_k: int) -> List[Tuple[int, int]]:
    """Keep reciprocal top-k fragment candidates."""
    n = similarity.shape[0]
    top_sets = []
    for i in range(n):
        idx = np.argsort(similarity[i])[::-1]
        idx = [j for j in idx if j != i][: max(1, min(top_k, n - 1))]
        top_sets.append(set(idx))

    pairs = []
    for i in range(n):
        for j in top_sets[i]:
            if i in top_sets[j] and i < j:
                pairs.append((i, j))
    return pairs


def evaluate_pair_accuracy(
    similarity: np.ndarray,
    labels: Dict[Tuple[int, int], int],
    threshold: float = 0.5,
) -> Optional[float]:
    """Evaluate pairwise match accuracy on labeled pairs."""
    if not labels:
        return None
    correct = 0
    total = 0
    for (i, j), gt in labels.items():
        pred = 1 if similarity[i, j] >= threshold else 0
        correct += int(pred == gt)
        total += 1
    return float(correct / max(1, total))


def evaluate_pair_metrics(
    similarity: np.ndarray,
    labels: Dict[Tuple[int, int], int],
    threshold: float,
) -> Dict[str, float]:
    """Compute pairwise classification metrics at selected threshold."""
    if not labels:
        return {
            "threshold": float(threshold),
            "accuracy": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "n_labeled_pairs": 0,
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0,
        }

    tp = fp = tn = fn = 0
    for (i, j), gt in labels.items():
        pred = 1 if similarity[i, j] >= threshold else 0
        if pred == 1 and gt == 1:
            tp += 1
        elif pred == 1 and gt == 0:
            fp += 1
        elif pred == 0 and gt == 0:
            tn += 1
        else:
            fn += 1

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / max(1, total)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "n_labeled_pairs": int(total),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def calibrate_threshold(
    similarity: np.ndarray,
    labels: Dict[Tuple[int, int], int],
    default_threshold: float = 0.5,
    objective: str = "accuracy",
) -> float:
    """Select threshold that maximizes selected metric on labeled pairs."""
    if not labels:
        return float(default_threshold)

    scores = np.array([similarity[i, j] for (i, j) in labels], dtype=np.float32)
    if scores.size == 0:
        return float(default_threshold)
    lo, hi = float(np.min(scores)), float(np.max(scores))
    if hi - lo < 1e-9:
        return float(default_threshold)

    best_thr = float(default_threshold)
    best_acc = -1.0
    best_f1 = -1.0
    for thr in np.linspace(lo, hi, 201):
        m = evaluate_pair_metrics(similarity, labels, threshold=float(thr))
        f1 = m["f1"] if np.isfinite(m["f1"]) else -1.0
        acc = m["accuracy"] if np.isfinite(m["accuracy"]) else -1.0
        if objective == "f1":
            if f1 > best_f1 or (abs(f1 - best_f1) <= 1e-9 and acc > best_acc):
                best_f1 = f1
                best_acc = acc
                best_thr = float(thr)
        else:
            if acc > best_acc or (abs(acc - best_acc) <= 1e-9 and f1 > best_f1):
                best_acc = acc
                best_f1 = f1
                best_thr = float(thr)
    return best_thr


def write_labeling_candidates(
    output_csv: Path,
    fragments: List[Fragment],
    pair_scores: Dict[Tuple[int, int], float],
    top_n: int = 50,
) -> None:
    """Write highest-confidence and most-ambiguous candidate pairs for annotation."""
    if not pair_scores:
        return
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    ranked = sorted(pair_scores.items(), key=lambda kv: kv[1], reverse=True)
    by_uncertainty = sorted(pair_scores.items(), key=lambda kv: abs(kv[1] - 0.5))

    selected = []
    seen = set()
    for (pair, score) in ranked[:top_n]:
        if pair not in seen:
            selected.append((pair, score, "high_confidence"))
            seen.add(pair)
    for (pair, score) in by_uncertainty[:top_n]:
        if pair not in seen:
            selected.append((pair, score, "ambiguous"))
            seen.add(pair)

    name_by_idx = {f.idx: f.name for f in fragments}
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["frag_a", "frag_b", "label", "score", "source"])
        writer.writeheader()
        for (i, j), score, source in selected:
            writer.writerow(
                {
                    "frag_a": name_by_idx[i],
                    "frag_b": name_by_idx[j],
                    "label": "",
                    "score": f"{float(score):.6f}",
                    "source": source,
                }
            )
    LOG.info("Wrote labeling candidates to %s", output_csv)


def train_and_match_fragments(
    fragments: List[Fragment],
    features: Dict[int, FeatureBundle],
    models_dir: Path,
    output_dir: Path,
    labels_csv: Optional[Path] = None,
    augment_rotations: bool = True,
    augment_count: int = 2,
    candidate_top_k: int = 4,
    label_suggestions_top_n: int = 50,
    threshold_objective: str = "accuracy",
    k_neighbors: int = 24,
    fpfh_radius: float = 0.06,
    fpfh_max_nn: int = 100,
    dbscan_eps: float = 0.04,
    dbscan_min_samples: int = 24,
    n_keypoints: int = 256,
    emb_dim: int = 64,
    epochs: int = 120,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    margin: float = 1.0,
    seed: int = 42,
    device: str = "cpu",
) -> Tuple[np.ndarray, List[Tuple[int, int]], Dict[Tuple[int, int], float], Dict[str, float], SiameseModelBundle]:
    """Train Siamese matcher and produce similarity matrix + candidate pairs."""
    rng = np.random.default_rng(seed)
    labels = load_pair_labels(labels_csv, fragments)

    x1, x2, y, train_stats = _build_self_supervised_pairs(
        fragments=fragments,
        features=features,
        labels=labels,
        augment_rotations=augment_rotations,
        augment_count=augment_count,
        rng=rng,
        k_neighbors=k_neighbors,
        fpfh_radius=fpfh_radius,
        fpfh_max_nn=fpfh_max_nn,
        dbscan_eps=dbscan_eps,
        dbscan_min_samples=dbscan_min_samples,
        n_keypoints=n_keypoints,
    )

    bundle = train_siamese_model(
        x1=x1,
        x2=x2,
        y=y,
        models_dir=models_dir,
        emb_dim=emb_dim,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        margin=margin,
        device=device,
    )

    descriptors = _descriptor_matrix(features, n=len(fragments))
    embeddings = encode_descriptors(descriptors, bundle=bundle, device=device)
    similarity = cosine_similarity_matrix(embeddings)

    pair_scores: Dict[Tuple[int, int], float] = {}
    for i in range(similarity.shape[0]):
        for j in range(i + 1, similarity.shape[0]):
            pair_scores[(i, j)] = float(similarity[i, j])

    candidate_pairs = reciprocal_topk_pairs(similarity, top_k=candidate_top_k)
    if not candidate_pairs:
        candidate_pairs = sorted(pair_scores, key=pair_scores.get, reverse=True)[: max(1, len(fragments) - 1)]

    write_labeling_candidates(
        output_csv=output_dir / "labeling_candidates.csv",
        fragments=fragments,
        pair_scores=pair_scores,
        top_n=max(10, label_suggestions_top_n),
    )

    threshold = calibrate_threshold(
        similarity,
        labels=labels,
        default_threshold=0.5,
        objective=threshold_objective,
    )
    calibrated = evaluate_pair_metrics(similarity, labels=labels, threshold=threshold)
    base = evaluate_pair_metrics(similarity, labels=labels, threshold=0.5)

    pair_acc = calibrated["accuracy"] if np.isfinite(calibrated["accuracy"]) else None
    diagnostics = {
        **train_stats,
        "pairwise_match_accuracy": float(pair_acc) if pair_acc is not None else float("nan"),
        "n_labeled_pairs": int(len(labels)),
        "n_candidate_pairs": int(len(candidate_pairs)),
        "selected_threshold": float(threshold),
        "metrics_at_selected_threshold": calibrated,
        "metrics_at_threshold_0_5": base,
        "threshold_objective": threshold_objective,
    }

    LOG.info(
        "Training pairs=%d (pos=%d, neg=%d), candidate pairs=%d",
        diagnostics["n_train_pairs"],
        diagnostics["n_pos_pairs"],
        diagnostics["n_neg_pairs"],
        diagnostics["n_candidate_pairs"],
    )

    return similarity, candidate_pairs, pair_scores, diagnostics, bundle
