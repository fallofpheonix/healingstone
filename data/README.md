# Dataset Contract

This directory holds fragment data used by the Healing Stone reconstruction pipeline.

## Expected Structure

```
data/
├── raw/              # Original unprocessed fragment files
│   ├── fragment_001.ply
│   ├── fragment_002.ply
│   └── ...
├── interim/          # Intermediate processing outputs (auto-generated)
├── processed/        # Fully preprocessed fragments (auto-generated)
└── sample/           # Minimal sample dataset for smoke tests
    ├── fragment_a.ply
    └── fragment_b.ply
```

## Supported Formats

| Mode | Extensions |
|------|-----------|
| 3D fragments | `.ply`, `.obj` |
| 2D fragments | `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.bmp` |

## Obtaining Data

1. **Sample data** (included): A minimal 2-fragment dataset is provided in `data/sample/` for smoke testing.
2. **Full dataset**: Download the Breaking Bad dataset from the official repository:
   ```bash
   # See: https://breaking-bad-dataset.github.io/
   ```
3. **Custom data**: Place your `.ply` or `.obj` fragment files in `data/raw/`.

## Checksum Verification

After downloading, verify file integrity:
```bash
sha256sum data/raw/*.ply > data/raw/checksums.sha256
```

## Pipeline Interaction

The pipeline auto-detects input type (3D vs 2D) based on file extensions in the provided `--data-dir`.

```bash
# Run on sample data
healingstone-run --data-dir data/sample --output-dir artifacts

# Run on full dataset
healingstone-run --data-dir data/raw --output-dir artifacts
```

> **Note:** Large binary data files (`.ply`, `.obj`, images) must NOT be committed to git.
> Use `.gitignore` rules or Git LFS for datasets exceeding 50MB.
