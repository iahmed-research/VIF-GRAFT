# Reproduction

Run all commands from the repository root.

Install and test:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e ".[test]"
python -m pytest -q
graft doctor --device cuda
```

Prepare data:

```bash
graft download --raw data/vtac-1.0
graft prepare --raw data/vtac-1.0 --out outputs/reproduction --device cuda
```

Run the complete experiment:

```bash
graft all --raw data/vtac-1.0 --out outputs/reproduction --device cuda
```

The pipeline writes receipts after each completed stage and resumes completed tracks after verifying their bindings. Use a new output directory when changing source code, configuration or runtime environment.
