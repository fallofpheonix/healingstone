"""Generate all technical diagrams for the GSoC proposal using Matplotlib."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), "images")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Shared style ──────────────────────────────────────────────────────────────
NAVY = "#1B2A4A"
LIGHT_BLUE = "#D6E4F0"
MID_BLUE = "#4A7FB5"
ACCENT = "#E8913A"
WHITE = "#FFFFFF"
LIGHT_GRAY = "#F0F2F5"
DARK_GRAY = "#3A3A3A"
FONT = {"family": "sans-serif", "size": 9}
matplotlib.rc("font", **FONT)


def _rounded_box(ax, x, y, w, h, text, fill=LIGHT_BLUE, edge=NAVY, fontsize=8,
                 text_color=DARK_GRAY, bold=False, subtext=None):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                         facecolor=fill, edgecolor=edge, linewidth=1.2)
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    if subtext:
        ax.text(x + w / 2, y + h * 0.62, text, ha="center", va="center",
                fontsize=fontsize, color=text_color, fontweight=weight)
        ax.text(x + w / 2, y + h * 0.3, subtext, ha="center", va="center",
                fontsize=max(6, fontsize - 2), color="#666666", style="italic")
    else:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, color=text_color, fontweight=weight,
                wrap=True)
    return box


def _arrow(ax, x1, y1, x2, y2, color=NAVY):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.3))


# ── 1. Pipeline Diagram ──────────────────────────────────────────────────────
def generate_pipeline():
    fig, ax = plt.subplots(1, 1, figsize=(14, 4.5), dpi=200)
    ax.set_xlim(-0.5, 14.5)
    ax.set_ylim(-0.5, 4.5)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(WHITE)

    stages = [
        ("Raw\nFragments", ".ply / .obj / .png"),
        ("Preprocessing", "Denoise · Normals\nDownsample"),
        ("Feature\nExtraction", "FPFH + PointNet\nEmbeddings"),
        ("Pairwise\nMatching", "Similarity Matrix\nTop-K Candidates"),
        ("Alignment", "RANSAC → ICP\nRefinement"),
        ("Validation\nGate", "RMSE · Overlap\nNormal Check"),
        ("Graph\nAssembly", "Pose Graph\nOptimization"),
        ("Reconstructed\nModel", "Merged .ply"),
    ]

    bw, bh = 1.5, 1.4
    gap = 0.35
    y_center = 2.0

    for i, (title, sub) in enumerate(stages):
        x = i * (bw + gap)
        fill = ACCENT if "Gate" in title else (NAVY if i == 0 or i == len(stages) - 1 else LIGHT_BLUE)
        tc = WHITE if fill in (NAVY, ACCENT) else DARK_GRAY
        _rounded_box(ax, x, y_center - bh / 2, bw, bh, title, fill=fill,
                     fontsize=8, text_color=tc, bold=True, subtext=sub)
        if i > 0:
            _arrow(ax, x - gap + 0.02, y_center, x - 0.02, y_center)

    # Reject branch from validation gate
    gate_idx = 5
    gate_x = gate_idx * (bw + gap) + bw / 2
    gate_y_bottom = y_center - bh / 2
    ax.annotate("Reject", xy=(gate_x, gate_y_bottom - 0.6),
                xytext=(gate_x, gate_y_bottom - 0.05),
                fontsize=7, color="#CC3333", ha="center",
                arrowprops=dict(arrowstyle="-|>", color="#CC3333", lw=1.0))
    ax.text(gate_x, gate_y_bottom - 0.75, "⊘ Discard Pair", ha="center",
            fontsize=7, color="#CC3333", style="italic")

    ax.set_title("Pipeline Flow: Fragment Reconstruction System", fontsize=11,
                 fontweight="bold", color=NAVY, pad=15)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "pipeline.png"), bbox_inches="tight",
                facecolor=WHITE, dpi=200)
    plt.close(fig)
    print("  ✓ pipeline.png")


# ── 2. Architecture Diagram ──────────────────────────────────────────────────
def generate_architecture():
    fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=200)
    ax.set_xlim(-0.5, 12.5)
    ax.set_ylim(-0.5, 8.5)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(WHITE)

    # Layer definitions: (y, height, items)
    layers = [
        (7.0, 0.8, [("CLI / API Layer", 12.0, NAVY, WHITE)]),
        (5.6, 0.9, [("Pipeline Orchestrator (run_pipeline.py)", 12.0, MID_BLUE, WHITE)]),
        (3.8, 1.2, [
            ("Preprocessing\nModule", 3.6, LIGHT_BLUE, DARK_GRAY),
            ("ML Models\n(PointNet)", 3.6, LIGHT_BLUE, DARK_GRAY),
            ("Alignment Engine\n(RANSAC + ICP)", 3.6, LIGHT_BLUE, DARK_GRAY),
        ]),
        (2.2, 1.0, [
            ("Runtime Config", 2.4, LIGHT_GRAY, DARK_GRAY),
            ("Metrics Schema", 2.4, LIGHT_GRAY, DARK_GRAY),
            ("Path Resolution", 2.4, LIGHT_GRAY, DARK_GRAY),
            ("Seed Mgmt", 2.4, LIGHT_GRAY, DARK_GRAY),
            # Note: we'll handle 4 items fitting in 12 width
        ]),
        (0.8, 1.0, [
            ("Fragment\nLoader", 3.6, "#E8E8E8", DARK_GRAY),
            ("Config Parser\n(YAML)", 3.6, "#E8E8E8", DARK_GRAY),
            ("Artifact\nWriter", 3.6, "#E8E8E8", DARK_GRAY),
        ]),
    ]

    for y, h, items in layers:
        n = len(items)
        total_w = sum(it[1] for it in items)
        if n == 1:
            gap_total = 0
        else:
            gap_total = (12.0 - total_w) / (n - 1) if total_w < 12 else 0
        x = 0.0
        for label, w_item, fill, tc in items:
            actual_w = w_item if n > 1 else 12.0
            _rounded_box(ax, x, y, actual_w, h, label, fill=fill, edge=NAVY,
                         fontsize=8 if n <= 3 else 7, text_color=tc, bold=(n == 1))
            x += actual_w + gap_total

    # Fix layer 3 (4 items at y=2.2): recalculate
    # Actually let me handle the 4-item layer manually
    y4, h4 = 2.2, 1.0
    labels4 = [
        ("Runtime\nConfig", LIGHT_GRAY),
        ("Metrics\nSchema", LIGHT_GRAY),
        ("Path\nResolution", LIGHT_GRAY),
        ("Seed\nManagement", LIGHT_GRAY),
    ]
    item_w4 = 2.7
    gap4 = (12.0 - 4 * item_w4) / 3
    for i, (lbl, fill) in enumerate(labels4):
        x4 = i * (item_w4 + gap4)
        _rounded_box(ax, x4, y4, item_w4, h4, lbl, fill=fill, edge=NAVY,
                     fontsize=7, text_color=DARK_GRAY)

    # Layer labels
    layer_labels = ["Entry Point", "Orchestration", "Processing", "Core Services", "I/O"]
    layer_ys = [7.4, 6.05, 4.4, 2.7, 1.3]
    for lbl, ly in zip(layer_labels, layer_ys):
        ax.text(-0.4, ly, lbl, fontsize=6, color="#999999", ha="right",
                va="center", rotation=0, style="italic")

    # Inter-layer arrows
    for y_from, y_to in [(7.0, 6.5), (5.6, 5.0), (3.8, 3.2), (2.2, 1.8)]:
        _arrow(ax, 6.0, y_from, 6.0, y_to, color="#AAAAAA")

    ax.set_title("System Architecture: Healing Stone", fontsize=12,
                 fontweight="bold", color=NAVY, pad=15)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "architecture.png"), bbox_inches="tight",
                facecolor=WHITE, dpi=200)
    plt.close(fig)
    print("  ✓ architecture.png")


# ── 3. Data Flow Diagram ─────────────────────────────────────────────────────
def generate_dataflow():
    fig, ax = plt.subplots(1, 1, figsize=(14, 5), dpi=200)
    ax.set_xlim(-0.5, 14.5)
    ax.set_ylim(-1.0, 5.0)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(WHITE)

    stages = [
        ("Mesh\nTopology", "V vertices\nF faces"),
        ("Point\nCloud", "N × 3"),
        ("Normals +\nFeatures", "N × (3+33)"),
        ("Learned\nEmbeddings", "d-dim per\nfragment"),
        ("Similarity\nMatrix", "K × K"),
        ("Candidate\nPairs", "List of\n(i, j) tuples"),
        ("SE(3)\nTransforms", "4 × 4 per\npair"),
        ("Pose\nGraph", "K nodes\nE edges"),
        ("Final\nAssembly", "M × 3\nmerged"),
    ]

    bw, bh = 1.3, 1.6
    gap = 0.28
    y_center = 2.0

    for i, (title, dim) in enumerate(stages):
        x = i * (bw + gap)
        fill = NAVY if (i == 0 or i == len(stages) - 1) else LIGHT_BLUE
        tc = WHITE if fill == NAVY else DARK_GRAY

        # Two-part box: title on top, dimension below
        _rounded_box(ax, x, y_center - bh / 2, bw, bh * 0.55, title,
                     fill=fill, fontsize=7, text_color=tc, bold=True)
        _rounded_box(ax, x, y_center - bh / 2 + bh * 0.58, bw, bh * 0.38,
                     dim, fill=WHITE, edge=MID_BLUE, fontsize=6.5,
                     text_color=MID_BLUE)

        if i > 0:
            _arrow(ax, x - gap + 0.02, y_center, x - 0.02, y_center, color=MID_BLUE)

    # Transformation labels on arrows
    transforms = ["Sample", "Estimate", "Embed", "Cosine\nSim", "Top-K", "RANSAC\n+ICP", "Optimize", "Merge"]
    for i, t in enumerate(transforms):
        x = i * (bw + gap) + bw + gap / 2
        ax.text(x, y_center + 1.0, t, fontsize=5.5, ha="center", va="center",
                color=ACCENT, fontweight="bold")

    ax.set_title("Data Representation Flow: Fragment → Reconstruction",
                 fontsize=11, fontweight="bold", color=NAVY, pad=15)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "dataflow.png"), bbox_inches="tight",
                facecolor=WHITE, dpi=200)
    plt.close(fig)
    print("  ✓ dataflow.png")


if __name__ == "__main__":
    print("Generating GSoC proposal diagrams...")
    generate_pipeline()
    generate_architecture()
    generate_dataflow()
    print("Done. Output in:", OUT_DIR)
