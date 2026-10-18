# Installation

## Reference environment

The tested GPU environment is Linux x86_64, CPython 3.11, PyTorch 2.8.0 and CUDA 12.8 wheels. BF16 inference used one RTX 6000 Ada 48 GB with CPU offload. This describes a working configuration, not a minimum memory requirement. CPU-only inference and other GPU sizes have not been tested.

Allocate approximately 57.72 GB for the base components and 472 MB for the restoration adapter, in addition to the Python environment and download cache.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --require-hashes -r requirements-demo-linux-cu128-py311.txt
python -m pip install --no-deps -e .
```

Use `requirements-linux-cu128-py311.txt` instead if you do not need the Gradio interface. The locks bind exact wheels to their official PyPI SHA-256 values and target this Linux/Python/CUDA combination. The package declares Python 3.10–3.13 support, but only Python 3.11 received the fresh GPU environment check.

## Base components

```bash
python scripts/download_base_models.py --output weights/base --workers 4
```

The helper reads the revision-pinned inventory in `configs/base-models.json`, downloads the required public files and verifies recorded sizes and hashes. An interrupted download can be retried using the same cache.

```bash
export DIFFSYNTH_MODEL_BASE_PATH=weights/base
export DIFFSYNTH_SKIP_DOWNLOAD=true
```

Offline discovery resolves the local cache and fails early if a required pattern is absent. Prepare the separate restoration adapter using [Models](models.md).

## Check the installation

```bash
demoire-infer --help
demoire-train --help
demoire-demo --help
python -m unittest discover -s tests -v
```

CPU checks do not establish that a GPU can load the complete model. See [Reproducibility](reproducibility.md) for the actual GPU check and [Known issues](issues.md) for the offline installation workaround.
