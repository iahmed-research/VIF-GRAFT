# Results

The completed GRAFT experiment reported:

| Evaluation | QMF anchor | GRAFT | Relative improvement |
|---|---:|---:|---:|
| Development | 0.23865641 | 0.23396759 | 1.96% |
| Internal validation | 0.25370810 | 0.24655219 | 2.82% |

At the development-derived high-sensitivity threshold, false positives decreased from 72 to 65 without increasing missed true VT alarms. The same residual interface improved a second multimodal anchor by 1.97%.

A fresh run writes detailed metrics to `outputs/reproduction/ALL_RESULTS.md`, `outputs/reproduction/validation/RESULTS.json` and `outputs/reproduction/corrections/*/DEVELOPMENT.json`. The expected headline values and run provenance are summarized in `reference/graft_final_results.json`.
