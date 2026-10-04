"""Matched correction arms, fitting and record-balanced evaluation."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from . import features as cp

ARMS = (
    'qmf_anchor',
    'qmf_recalibration',
    'graft_ecg2',
    'graft',
    'graft_without_agreement',
    'graft_without_supplementary_ecg',
    'graft_unbounded',
)
REPETITIONS=10000
SEED=20260927

def feature_specs():
    if len(cp.CHANNEL_NAMES) != 12 or len(cp.PAIR_NAMES) != 3:
        raise ValueError('Feature schema changed')
    result = {}
    for arm in ARMS[1:]:
        source = 'A' if arm == 'graft_ecg2' else 'B'
        slots = ['ECG1', 'ECG2' if source == 'A' else 'chest', 'PPG', 'ABP']
        names = [slot + '.' + f for slot in slots for f in cp.CHANNEL_NAMES]
        names += ['ECG_pair.' + f for f in cp.PAIR_NAMES]
        indices = list(range(51))
        if arm == 'qmf_recalibration': indices = []
        if arm == 'graft_without_agreement': indices = list(range(48))
        if arm == 'graft_without_supplementary_ecg': indices = list(range(12)) + list(range(24, 48))
        result[arm] = {'source': source, 'waveform_indices': indices,
                       'predictor_names': ['qmf_logit'] + [names[i] for i in indices],
                       'bounded': arm != 'graft_unbounded'}
    return result


def logit(q):
    q = np.clip(q, 1e-6, 1-1e-6)
    return np.log(q) - np.log1p(-q)


def matrix(data, arm, ix, q):
    spec = feature_specs()[arm]
    features = data[spec['source']]
    if features.shape != (len(data['event']), 51):
        raise ValueError('Expected 51 waveform descriptors')
    x = np.c_[logit(q), features[np.ix_(ix, spec['waveform_indices'])]]
    if not np.isfinite(x).all(): raise ValueError('Nonfinite predictors')
    return x


def fit_head(x, y, records, q, eligible, bounded):
    if bounded:
        model = cp.fit(x, y, records, q, eligible, .1, True, 500)
    else:
        # cp.fit(..., bounded=False) fits a standalone classifier, not an
        # unbounded QMF residual. Preserve the QMF offset and eligibility here.
        mean, scale = cp.design_fit(x, records)
        z = cp.design(x, mean, scale)
        result = minimize(cp.objective, np.zeros(z.shape[1]),
                          args=(z, y, cp.weights(records), logit(q), eligible, .1, False),
                          method='L-BFGS-B', jac=True,
                          options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-6})
        if not result.success or not np.isfinite(result.fun) or not np.isfinite(result.x).all():
            raise RuntimeError('Unbounded fit failed: ' + str(result.message))
        model = {'mean': mean, 'scale': scale, 'theta': result.x, 'iterations': int(result.nit)}
    if not all(np.isfinite(model[k]).all() for k in ('mean', 'scale', 'theta')):
        raise RuntimeError('Nonfinite fitted head')
    return {**model, 'converged': True, 'bounded': bounded}


def predict(model, x, q, eligible):
    raw = cp.design(x, model['mean'], model['scale']) @ model['theta']
    correction = np.tanh(raw) if model['bounded'] else raw
    offset = logit(q)
    p = np.clip(q + (expit(offset + correction) - expit(offset)), 0, 1)
    return np.where(eligible, p, q)


def weighted_auc(y, p, w):
    order = np.argsort(p, kind='stable'); y, p, w = y[order], p[order], w[order]
    positive = w * (y == 1); negative = w * (y == 0)
    if positive.sum() == 0 or negative.sum() == 0: return None
    starts = np.r_[0, np.flatnonzero(np.diff(p)) + 1]
    pos = np.add.reduceat(positive, starts); neg = np.add.reduceat(negative, starts)
    return float(pos @ (np.cumsum(neg) - .5 * neg) / (pos.sum() * neg.sum()))


def losses(y, p):
    p = np.clip(p, 1e-15, 1-1e-15)
    return -(y*np.log(p) + (1-y)*np.log1p(-p))


def metrics(y, p, records, q):
    w = cp.weights(records); loss = losses(y, p)
    classes = {}
    for label in (0, 1):
        mask = y == label
        classes[str(label)] = {'events': int(mask.sum()), 'records': len(set(records[mask])),
                              'logloss': float(w[mask] @ loss[mask] / w[mask].sum()) if mask.any() else None}
    return {'record_balanced_logloss': float(w @ loss),
            'relative_gain_vs_qmf': 1-float(w @ loss)/cp.logloss(y, q, records),
            'record_weighted_Brier': float(w @ (p-y)**2),
            'record_weighted_AUROC': weighted_auc(y, p, w), 'class_metrics': classes,
            'class_weighting': 'Global record-balanced event weights, normalized within class.',
            'correction': {'weighted_mean_absolute_probability_change': float(w @ np.abs(p-q)),
                           'maximum_absolute_probability_change': float(np.max(np.abs(p-q)))}}


def comparisons(y, predictions, records, repetitions=REPETITIONS, seed=SEED):
    keys = list(predictions); _, inv, counts = np.unique(records, return_inverse=True, return_counts=True)
    per_record = np.column_stack([np.bincount(inv, weights=losses(y, predictions[k]))/counts for k in keys])
    rng = np.random.Generator(np.random.PCG64(seed)); sampled = []
    for start in range(0, repetitions, 100):
        ix = rng.integers(0, len(counts), size=(min(100, repetitions-start), len(counts)))
        sampled.append(per_record[ix].mean(axis=1))
    sampled = np.concatenate(sampled); means = per_record.mean(axis=0); b = keys.index('graft')
    result = {}
    for j, key in enumerate(keys):
        if key == 'graft': continue
        diff = per_record[:, b]-per_record[:, j]
        result['graft_minus_'+key] = {
            'candidate_minus_reference': float(diff.mean()),
            'difference_ci95': np.quantile(sampled[:, b]-sampled[:, j], [.025,.975]).tolist(),
            'relative_gain': float(1-means[b]/means[j]),
            'relative_gain_ci95': np.quantile(1-sampled[:, b]/sampled[:, j], [.025,.975]).tolist(),
            'record_groups_improved': int((diff < -1e-12).sum()),
            'record_groups_worsened': int((diff > 1e-12).sum()),
            'record_groups_unchanged': int((np.abs(diff) <= 1e-12).sum())}
    return result
