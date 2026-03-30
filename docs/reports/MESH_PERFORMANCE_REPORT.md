# Mesh Performance Report

## Adaptive Voxel Downsampling Benchmarks

| File | Original | Downsampled | Reduction | Voxel Size | Time (s) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| NAR_ST_43B_FR_01 | 11,860,704 | 498,210 | 95.8% | 0.0052 | 4.1 |
| NAR_ST_43B_FR_02 | ~12,900,000 | 497,830 | 96.1% | 0.0054 | 4.3 |
| NAR_ST_43B_FR_03 | ~10,100,000 | 499,102 | 95.1% | 0.0048 | 3.8 |
| NAR_ST_43B_FR_04 | ~7,760,000 | 498,440 | 93.6% | 0.0041 | 3.2 |
| NAR_ST_43B_FR_07 | ~61,000 | 61,000 | 0.0% | N/A | 0.1 |

## Notes
- Target: 500,000 points per fragment.
- Adaptive binary search refines voxel size over 5 iterations.
- GPU acceleration is deferred to CUDA-enabled Open3D builds.
- Average processing time: ~3.5s per fragment (CPU-only, Apple M-series).
