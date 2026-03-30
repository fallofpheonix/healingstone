# Setup

## Supported Environments

| Goal | Python | Install |
| --- | --- | --- |
| full 3D runtime | `3.10` to `3.12` | `pip install -e '.[dev,runtime]'` |
| light development checks | `3.10` to `3.13` | `pip install -e '.[dev]'` |
| 2D-only experiments | `3.10` to `3.13` | `pip install -e '.[dev,pipeline2d]'` |

## Recommended Full Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,runtime]'
```

## Minimal Developer Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

## Verification Commands

```bash
pytest -q
ruff check .
mypy
```

## Runtime Commands

### 3D

```bash
healingstone-run --data-dir data/raw/3d --output-dir artifacts --min-required-accuracy 0
```

### 2D

```bash
healingstone-run --data-dir /path/to/2d_fragments --output-dir artifacts
```

## Common Failure Modes

| Failure | Cause | Fix |
| --- | --- | --- |
| `ModuleNotFoundError: open3d` | Python `3.13` or missing runtime extras | use Python `3.12` and install `.[runtime]` |
| `ModuleNotFoundError: torch` | runtime extras not installed | reinstall with `.[runtime]` |
| empty dataset error | wrong `--data-dir` or missing fragments | verify suffixes and path |
| accuracy gate failure | no labels or poor labeled performance | provide `labels.csv` or lower gate for smoke runs |

## Reproducibility Rules

- run from repo root
- keep `CLI > ENV > YAML` precedence intact
- do not write outputs outside `artifacts/`
- treat `artifacts/submission/` as generated output, not source
