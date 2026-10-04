"""Exact finite-reference grouped Shapley attribution; fixed QMF context."""
import hashlib, math
import numpy as np
from scipy.special import expit
GROUPS=('Joint ECG/chest/agreement','PPG','ABP')
INDICES=(list(range(24))+list(range(48,51)),list(range(24,36)),list(range(36,48)))
REFERENCES=('all_matched_records','record_half_0','record_half_1')

def order(value):
    return hashlib.sha256(('GRAFT|' + str(value)).encode()).hexdigest()


def weights(records):
    _, inv, counts = np.unique(records, return_inverse=True, return_counts=True)
    w = 1.0 / counts[inv]
    return w / w.sum()


def logit(q):
    q = np.clip(q, 1e-6, 1-1e-6)
    return np.log(q) - np.log1p(-q)


def delta(q, score, eligible=True):
    offset = logit(q)
    p = np.clip(q + (expit(offset + np.tanh(score)) - expit(offset)), 0, 1)
    return np.where(eligible, p-q, 0.0)


def components(model, features, q):
    mean, scale, theta = (np.asarray(model[k]) for k in ('mean', 'scale', 'theta'))
    z = np.clip((features-mean[1:])/scale[1:], -8, 8)
    terms = z*theta[2:]
    groups = np.column_stack([terms[:, ix].sum(axis=1) for ix in INDICES])
    context = theta[0] + theta[1]*np.clip((logit(q)-mean[0])/scale[0], -8, 8)
    return groups, context


def shapley(values, n):
    """Enumerate every coalition, with exact factorial weights."""
    values = np.asarray(values, float)
    if values.shape != (2**n,) or not np.isfinite(values).all():
        raise ValueError('Invalid coalition values')
    phi = np.zeros(n)
    for j in range(n):
        for mask in range(2**n):
            if mask & (1 << j):
                continue
            size = mask.bit_count()
            weight = math.factorial(size)*math.factorial(n-size-1)/math.factorial(n)
            phi[j] += weight*(values[mask | (1 << j)]-values[mask])
    return phi


def explain(target, donors, w, q, context, eligible=True):
    donors = np.asarray(donors, float)
    w = np.asarray(w, float)
    if donors.ndim != 2 or donors.shape[1] != len(target) or len(donors) == 0:
        raise ValueError('Invalid reference pool')
    if w.shape != (len(donors),) or np.any(w < 0) or not np.isclose(w.sum(), 1):
        raise ValueError('Invalid reference weights')
    if not np.isfinite(donors).all() or not np.isfinite(target).all():
        raise ValueError('Nonfinite contributions')
    values = []
    for mask in range(2**len(target)):
        keep = np.array([bool(mask & (1 << j)) for j in range(len(target))])
        # All absent groups use the SAME donor row, preserving their joint reference.
        scores = context + np.where(keep, target, donors).sum(axis=1)
        values.append(float(w @ delta(q, scores, eligible)))
    values = np.asarray(values)
    phi = shapley(values, len(target))
    if abs(values[0] + phi.sum() - values[-1]) > 1e-10:
        raise ValueError('Shapley accounting failed')
    return phi, values


def selected(data, per_fold):
    chosen = []
    for fold in range(5):
        ix = np.flatnonzero((data['fold'] == fold) & data['eligible'])
        representatives = {}
        for i in sorted(ix, key=lambda i: order(data['event'][i])):
            representatives.setdefault(str(data['record'][i]), int(i))
        chosen.extend(sorted(representatives.values(), key=lambda i: order(data['record'][i]))[:per_fold])
        missing = np.flatnonzero((data['fold'] == fold) & ~data['eligible'])
        if len(missing):
            chosen.append(int(min(missing, key=lambda i: order(data['event'][i]))))
    return chosen


def reference_sets(data, i, calibration_fold):
    present = data['B'][:, [0, 12, 24, 36]]
    same = np.all(present == present[i], axis=1)
    pool = np.flatnonzero((data['fold'] == calibration_fold) & data['eligible'] & same)
    records = sorted(set(data['record'][pool].tolist()), key=order)
    if len(records) < 4:
        return None
    if str(data['record'][i]) in set(data['record'][pool].tolist()):
        raise ValueError('Evaluation record leaked into reference')
    sets = [pool]
    for half in (0, 1):
        sets.append(pool[np.isin(data['record'][pool], records[half::2])])
    return sets
