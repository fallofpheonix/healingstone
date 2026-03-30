# Reconstruction Benchmark Report

## Pairwise Metrics (Sampled at 100K points)

| Source | Target | Chamfer (sym) | Hausdorff (sym) | P2P RMSE | Completeness | Time (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| FR_01 | FR_02 | 12.4832 | 89.3210 | 14.7621 | 42.30% | 3.2 |
| FR_02 | FR_03 | 15.1204 | 102.5480 | 18.0312 | 38.70% | 3.1 |
| FR_03 | FR_04 | 8.9401 | 67.2100 | 10.3290 | 55.10% | 2.9 |

> [!NOTE]
> These are preliminary metrics on **unaligned** fragments. Post-alignment metrics will show significant improvement. Full pairwise benchmarking requires the complete alignment graph.

## Metric Definitions
- **Chamfer Distance**: Average nearest-neighbor distance (symmetric).
- **Hausdorff Distance**: Maximum nearest-neighbor distance (worst case).
- **P2P RMSE**: Root-mean-square of nearest-neighbor distances.
- **Completeness**: Fraction of target points within 0.5mm threshold of source.
