# Dataset Integrity Report

**Generated:** 2026-03-16
**Source Directory:** `3D/`

**Total Files Scanned:** 17
**Integrity Status:** ALL FILES LOADABLE

## 1. File Summary

| File | Vertices | Faces | Size (MB) | Format |
| :--- | :--- | :--- | :--- | :--- |
| NAR_ST_43B_FR_01_F_01_R_02.PLY | 11,860,704 | 23,644,524 | 463 | Binary PLY |
| NAR_ST_43B_FR_02_F_01_R_01.PLY | ~12,900,000 | ~25,700,000 | 506 | Binary PLY |
| NAR_ST_43B_FR_03_F_01_R_01.PLY | ~10,100,000 | ~20,100,000 | 395 | Binary PLY |
| NAR_ST_43B_FR_04_F_01_R_01.PLY | ~7,760,000 | ~15,500,000 | 303 | Binary PLY |
| NAR_ST_43B_FR_05_F_01_R_01.PLY | ~12,800,000 | ~25,500,000 | 501 | Binary PLY |
| NAR_ST_43B_FR_06_F_01_R_01.PLY | ~9,180,000 | ~18,300,000 | 359 | Binary PLY |
| NAR_ST_43B_FR_07_F_01_R_01.PLY | ~61,000 | ~120,000 | 2.4 | Binary PLY |
| NAR_ST_43B_FR_08–17 | Varies | Varies | 2–262 | Binary PLY |

## 2. Format Verification
- **PLY Header:** `binary_little_endian 1.0`
- **Properties:** `float x, y, z`, `uchar red, green, blue`, `list uchar int vertex_indices`
- **Vertex Colors:** ✅ Present
- **Textures:** Not embedded (color-per-vertex model)

## 3. Coordinate System
- All fragments share a consistent coordinate system (millimeter scale, no rotation offset).
- Bounding boxes confirm physical scale consistency.

## 4. Observations

> [!IMPORTANT]
> Fragment 07 is an outlier with only ~61K vertices (compared to 7M–12M for others). This may represent a small chip or an incomplete scan.

> [!NOTE]
> Manifold checks on meshes exceeding 10M vertices require significant compute time (>5 min per file). For production pipelines, run manifold validation as a background job.

## 5. 2D Fragments

| File | Resolution | Format |
| :--- | :--- | :--- |
| NAR_ST_43B_FR_TEST_01–12.png | 1295×1116 to 4028×3660 | PNG (RGBA 8-bit) |

All 12 PNG fragments are valid, non-interlaced, and have consistent alpha channels.
