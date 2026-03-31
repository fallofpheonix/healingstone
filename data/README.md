# Data Contract

## Layout

```text
data/
├── sample/
│   └── 3d/
│       ├── fragment_a.ply
│       └── fragment_b.ply
├── raw/
│   └── 3d/               # local large dataset root
├── interim/              # generated
└── processed/            # generated
```

## Supported Inputs

- 3D: `.ply`, `.obj`
- 2D: `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.bmp`

## Smoke Dataset

`data/sample/3d/` is committed and intended for:

- CLI smoke execution
- CI runtime validation
- deterministic metrics regression checks

Run it with:

```bash
healingstone-run --data-dir data/sample/3d --output-dir artifacts --min-required-accuracy 0 --allow-overwrite-run
```

## Large Data Policy

- Keep only the tiny sample dataset in git.
- Treat `data/raw/3d/` as local-only runtime input.
- Do not commit large meshes, archives, or generated derivatives.
