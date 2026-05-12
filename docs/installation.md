# Installation

SNAPpy installs like a standard Python package. The command users run after installation is `mrsnappy`.

## Recommended Install

Install SNAPpy from PyPI with:

```bash
python -m pip install mrsnappy
```

Confirm the install:

```bash
mrsnappy --help
```

To install the current GitHub version instead of the PyPI release:

```bash
python -m pip install "git+https://github.com/marcorojas-cessa/SNAPpy.git"
```

## Clean Environment Install

Use a virtual environment if you want SNAPpy isolated from your system Python:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install mrsnappy
```

SNAPpy does not require a GPU. For routine detection, use a modern CPU-only computer with at least 8-16 GB RAM; 16 GB or more is recommended for larger z-stacks or batch processing. See [hardware guidance](hardware.md) for details.

On Windows, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Local Developer Install

Use this when editing the source code:

```bash
git clone https://github.com/marcorojas-cessa/SNAPpy.git
cd SNAPpy
python -m pip install -e ".[dev]"
pytest
```

## Name Reference

| Name | Where it is used |
|---|---|
| `SNAPpy` | Repository and project name |
| `mrsnappy` | PyPI distribution name used with `pip install` |
| `mrsnappy` | Terminal command installed by the package |
| `mrsnappy` | Python import name |

Example:

```bash
mrsnappy detect --model /path/to/model/model.joblib --input image.tif --output detections.csv
```

```python
from mrsnappy import detect

detect(model="/path/to/model/model.joblib", input_path="image.tif", output="detections.csv")
```

## Why Not `python install mrsnappy`?

Python packages are installed with `pip`, usually through:

```bash
python -m pip install ...
```

The name `mrsnappy` is the command-line tool installed by this package. It is not the install command.

Also, PyPI already has an unrelated package at the normalized name `snappy`, so SNAPpy uses `mrsnappy` as both its install name and command-line tool.
