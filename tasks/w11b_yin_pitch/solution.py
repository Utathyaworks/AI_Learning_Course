import numpy as np

def yin_pitch(frame, sr, fmin=60.0, fmax=800.0, threshold=0.1):
    x = np.asarray(frame, float)
    if np.sum(x * x) < 1e-10:
        return 0.0
    tmin, tmax = int(sr / fmax), int(np.ceil(sr / fmin))
    W = len(x) - tmax
    if W <= 0:
        raise ValueError("frame too short for fmin")
    d = np.array([np.sum((x[:W] - x[t:t + W]) ** 2) for t in range(tmax + 1)])
    cm = np.ones_like(d)
    cs = np.cumsum(d[1:])
    cm[1:] = d[1:] * np.arange(1, tmax + 1) / np.where(cs == 0, 1e-12, cs)
    tau = None
    for t in range(max(tmin, 1), tmax):
        if cm[t] < threshold:
            while t + 1 < tmax and cm[t + 1] < cm[t]:
                t += 1
            tau = t
            break
    if tau is None:
        t = max(tmin, 1) + int(np.argmin(cm[max(tmin, 1):tmax]))
        if cm[t] >= 0.5:
            return 0.0
        tau = t
    if 1 <= tau < tmax:
        a, b, c = cm[tau - 1], cm[tau], cm[tau + 1]
        den = a - 2 * b + c
        shift = 0.5 * (a - c) / den if den != 0 else 0.0
        tau = tau + float(np.clip(shift, -1, 1))
    return float(sr / tau)
