# Healing Stones: 3D Fragment Reconstruction

## GSoC 2025 | HumanAI

AI-powered reconstruction of fragmented cultural heritage artifacts (Mayan stele) using 3D and 2D digital scan data.

## Quick Start

```bash
git clone https://github.com/fallofpheonix/healingstone.git
cd healingstone
pip install -r requirements.txt

# Validate with synthetic data (no dataset needed)
python test_pipeline.py --n-fragments 12

# Run on real data
python healing_stones.py DataSet/3D/ --data2d DataSet/2D/ --output output/
```

## Project Structure

```
healingstone/
├── healing_stones.py     # Main pipeline (3D+2D reconstruction)
├── test_pipeline.py      # Synthetic validator
├── requirements.txt      # Dependencies
├── DataSet/              # Not tracked in git (too large)
│   ├── 3D/               # PLY fragment files (from CERNBox)
│   └── 2D/               # PNG image files (from CERNBox)
└── output/               # Results (plots, aligned PLYs, report.json)
```

## Data Sources

| Format | Link |
|--------|------|
| 3D PLY fragments | [CERNBox 3D](https://cernbox.cern.ch/s/hQO24HxuKi6VeQo) |
| 2D image fragments | [CERNBox 2D](https://cernbox.cern.ch/s/kOdhPJxQrMzGdTN) |

## Pipeline Overview

| Stage | Method |
|-------|--------|
| Surface Classification | Otsu-thresholded break-score (curvature + roughness + normal variation) |
| Feature Extraction | FPFH descriptors (rotation-invariant, 33-d per point) |
| Fragment Matching | RandomForest on pairwise descriptor similarity |
| Coarse Alignment | RANSAC on FPFH correspondences |
| Fine Alignment | Gap-aware ICP (robust to missing material) |
| 2D Support | HOG + LBP + edge features fused with 3D scores |

## Outputs

- `output/match_matrix.png` — Pairwise fragment similarity heatmap
- `output/full_reconstruction.png` — 3D reconstruction visualization
- `output/aligned_*.ply` — Aligned fragment point clouds
- `output/report.json` — Full metrics (RMSE, inlier ratios, top matches)
- `healing_stones.log` — Detailed run log

## Contact

Questions: [human-ai@cern.ch](mailto:human-ai@cern.ch) — Subject: `GSoC Healing Stones`
