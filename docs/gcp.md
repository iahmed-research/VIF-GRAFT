# GCP Run

Use a GPU VM with a working NVIDIA driver. Confirm `nvidia-smi` succeeds before starting the full run.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e ".[test]"
graft doctor --device cuda
```

Then run:

```bash
bash scripts/run_gcp.sh benchmark
bash scripts/run_gcp.sh one-epoch
bash scripts/run_gcp.sh all
```

Set `GRAFT_RAW` and `GRAFT_OUT` to override the default data and output locations.
