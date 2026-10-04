# GRAFT reproducibility package

GRAFT is a gated residual anchored fusion method for ventricular tachycardia alarm classification. It extends a calibrated multimodal anchor with an optional supplementary ECG channel. The anchor provides the baseline probability from ECG, photoplethysmography and arterial blood pressure, while GRAFT learns a bounded residual correction from compact waveform descriptors and cross-ECG agreement.

The repository reproduces the experiments for:

- QMF anchor training and calibration.
- GRAFT residual-head fitting.
- Internal validation scoring.
- Matched ablations and a second multimodal anchor.
- Supplementary-ECG robustness analysis.
- Exact score decomposition and grouped Shapley explanations.

## Results

The completed run reported the following headline results on VTaC 1.0:

| Evaluation | Comparison | Result |
|---|---|---:|
| Development | GRAFT vs QMF anchor | 1.96% lower record-balanced log loss |
| Internal validation | GRAFT vs QMF anchor | 2.82% lower record-balanced log loss |
| High-sensitivity operating point | False positives | 72 to 65 |
| Second multimodal anchor | GRAFT residual interface | 1.97% lower record-balanced log loss |

The expected headline values and provenance are stored in `reference/graft_final_results.json`. A full run writes `outputs/reproduction/ALL_RESULTS.md` and `outputs/reproduction/ALL_COMPLETE.json`.

## Repository Layout

```text
configs/       Fixed cohort, preprocessing, training and quality settings
docs/          Method, data, reproduction, results and XAI notes
scripts/       GCP helper and source packaging utility
src/graft/     GRAFT implementation
tests/         Unit and orchestration tests
reference/     Expected headline results
```

Generated data, model checkpoints, logs and reproduction outputs are intentionally ignored by Git.

## Installation

Python 3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[test]"
python -m pytest -q
graft doctor --device cpu
```

For an NVIDIA GPU with CUDA 12.8, install the CUDA PyTorch wheel instead:

```bash
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
```

## Data

Use VTaC version 1.0. The raw-data root should contain:

```text
waveforms/
event_labels.csv
benchmark_data_split.csv
SHA256SUMS.txt
```

You can point the commands at an existing copy or download the selected train/validation waveforms:

```bash
graft download --raw data/vtac-1.0
```

The downloader verifies publisher checksums. The official test waveforms are not downloaded by this command.

## Quick Checks

```bash
graft prepare --raw data/vtac-1.0 --out outputs/reproduction --device cpu
graft benchmark --raw data/vtac-1.0 --out outputs/reproduction --device cpu
```

For a GPU smoke test:

```bash
graft train --family QMF --fold 0 --seed 20260911 --epoch-limit 1 --raw data/vtac-1.0 --out outputs/reproduction --device cuda
```

The one-epoch command is a runtime check and writes a resumable checkpoint. It is not a scientific completion run.

## Complete Reproduction

```bash
graft all --raw data/vtac-1.0 --out outputs/reproduction --device cuda
```

The complete experiment trains 50 neural anchor models, then fits GRAFT residual heads, scores validation predictions, runs robustness analysis and writes explanation outputs. On completion, inspect:

```text
outputs/reproduction/ALL_RESULTS.md
outputs/reproduction/ALL_COMPLETE.json
outputs/reproduction/validation/RESULTS.json
outputs/reproduction/robustness/RESULTS.json
outputs/reproduction/explanations/
```

## Citation and Data License

Code is released under the MIT License. Please cite VTaC 1.0 and QMF when using this repository. VTaC data are licensed separately by the dataset provider; see `LICENSE-DATA.txt` and `docs/data.md`.
