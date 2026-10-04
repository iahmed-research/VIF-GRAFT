# Explanations

GRAFT uses a linear pre-bounded residual score, so each event has an exact additive score decomposition. The implementation groups terms into:

- QMF confidence
- Original ECG
- Supplementary ECG
- PPG
- ABP
- ECG agreement

The explanation stage also writes computational interventions and grouped Shapley summaries for the probability-space correction. Outputs are written under `outputs/reproduction/explanations/`.
