# Method

GRAFT, Gated Residual Anchored Fusion, extends a calibrated multimodal VT alarm classifier with a supplementary ECG channel. The anchor remains responsible for the complete baseline prediction. GRAFT estimates how eligible supplementary ECG evidence should refine that prediction.

For event \(i\), let \(\hat p_{Q,i}\) be the calibrated QMF anchor probability and \(\eta_{Q,i}=\operatorname{logit}(\hat p_{Q,i})\). GRAFT extracts descriptor vector \(u_i\) from the supplementary ECG, the primary physiological channels and cross-ECG agreement. A linear residual score is fitted on the calibration fold:

\[
r_i = \beta_0 + \beta^\top \tilde u_i.
\]

The final logit is:

\[
\eta_{G,i} = \eta_{Q,i} + e_i \tanh(r_i),
\]

where \(e_i\) is the supplementary-ECG eligibility indicator. If the supplementary ECG is not eligible, \(e_i=0\), and the GRAFT probability is exactly the QMF anchor probability.

The bounded residual gives the supplementary ECG a controlled role: it can refine the anchor but cannot freely replace it. The same interface is also evaluated with a second multimodal anchor and matched ablations.
