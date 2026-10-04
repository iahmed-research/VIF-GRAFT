# Data

The experiments use VTaC version 1.0. The repository expects the raw data root to contain `waveforms/`, `event_labels.csv`, `benchmark_data_split.csv` and `SHA256SUMS.txt`.

The fixed cohort contains 4,060 development alarms and 495 internal-validation alarms. Record-group folds are listed in `configs/cohort.csv`. The preprocessing window is the 10-second interval ending immediately before the alarm, corresponding to samples 72,500 through 74,999 at 250 Hz.

The repository does not redistribute clinical waveforms or labels. Use the dataset provider's access and citation requirements when preparing a local copy.
