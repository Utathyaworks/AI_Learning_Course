"""Coding tasks, part 5 (Weeks 1B coordinate geometry, 4B probability, 5B statistics)."""

T = []

T.append(dict(
id="w01b_lines_polygons", title="Distances, hyperplanes and polygons",
prompt="""Implement loop-free NumPy geometry helpers.

- `pairwise_dist(X, Y)`: (N, M) Euclidean distances between rows of X (N, d) and Y (M, d), using
  ‖x−y‖² = ‖x‖² + ‖y‖² − 2x·y (clip tiny negatives to 0 before the square root).
- `hyperplane_signed_distance(X, w, b)`: signed distance of each row of X to the hyperplane w·x + b = 0.
- `point_line_distance(P, a, b)`: distance from each 2-D point in P (N, 2) to the **infinite** line through points a and b.
- `polygon_area(V)`: signed area of a polygon with vertices V (n, 2) in order (shoelace formula;
  counter-clockwise = positive).
- `convex_hull(P)`: the convex hull of 2-D points as an (h, 2) array in counter-clockwise order, starting from the
  lowest-x (then lowest-y) point, without collinear points on the edges (Andrew's monotone chain).""",
starter='''import numpy as np

def pairwise_dist(X, Y):
    raise NotImplementedError

def hyperplane_signed_distance(X, w, b):
    raise NotImplementedError

def point_line_distance(P, a, b):
    raise NotImplementedError

def polygon_area(V) -> float:
    raise NotImplementedError

def convex_hull(P):
    raise NotImplementedError
''',
solution='''import numpy as np

def pairwise_dist(X, Y):
    X = np.asarray(X, float); Y = np.asarray(Y, float)
    d2 = (X ** 2).sum(1)[:, None] + (Y ** 2).sum(1)[None, :] - 2 * X @ Y.T
    return np.sqrt(np.clip(d2, 0, None))

def hyperplane_signed_distance(X, w, b):
    w = np.asarray(w, float)
    return (np.asarray(X, float) @ w + b) / np.linalg.norm(w)

def point_line_distance(P, a, b):
    P = np.asarray(P, float); a = np.asarray(a, float); b = np.asarray(b, float)
    d = b - a
    cross = d[0] * (P[:, 1] - a[1]) - d[1] * (P[:, 0] - a[0])
    return np.abs(cross) / np.linalg.norm(d)

def polygon_area(V):
    V = np.asarray(V, float)
    x, y = V[:, 0], V[:, 1]
    return float(0.5 * (x @ np.roll(y, -1) - y @ np.roll(x, -1)))

def convex_hull(P):
    pts = sorted(set(map(tuple, np.asarray(P, float))))
    if len(pts) <= 2:
        return np.array(pts)
    cross = lambda o, a, b: (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])
''',
tests='''
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist

def test_pairwise():
    rng = np.random.default_rng(0)
    X, Y = rng.normal(size=(30, 5)), rng.normal(size=(20, 5))
    assert np.allclose(pairwise_dist(X, Y), cdist(X, Y), atol=1e-8)
    assert np.allclose(np.diag(pairwise_dist(X, X)), 0, atol=1e-6)

def test_hyperplane():
    X = np.array([[3.0, 4.0], [0.0, 0.0], [-3.0, -4.0]])
    w, b = np.array([3.0, 4.0]), -5.0
    assert np.allclose(hyperplane_signed_distance(X, w, b), [4.0, -1.0, -6.0])
    assert np.allclose(hyperplane_signed_distance(X, 10 * w, 10 * b), [4.0, -1.0, -6.0])

def test_point_line():
    P = np.array([[0.0, 1.0], [5.0, -2.0], [2.0, 2.0]])
    assert np.allclose(point_line_distance(P, [0, 0], [1, 0]), [1, 2, 2])
    assert np.allclose(point_line_distance(np.array([[0.0, 0.0]]), [1, 0], [0, 1]), [np.sqrt(0.5)])

def test_area_sign():
    sq = np.array([[0, 0], [2, 0], [2, 2], [0, 2]])
    assert polygon_area(sq) == 4.0 and polygon_area(sq[::-1]) == -4.0
    tri = np.array([[0, 0], [4, 0], [0, 3]])
    assert polygon_area(tri) == 6.0

def test_hull():
    rng = np.random.default_rng(1)
    P = rng.random((200, 2))
    h = convex_hull(P)
    ref = P[ConvexHull(P).vertices]
    assert len(h) == len(ref)
    assert {tuple(np.round(p, 12)) for p in h} == {tuple(np.round(p, 12)) for p in ref}
    assert polygon_area(h) > 0
    sq = np.array([[0, 0], [1, 0], [2, 0], [2, 2], [0, 2], [1, 1]], float)
    assert np.array_equal(convex_hull(sq), [[0, 0], [2, 0], [2, 2], [0, 2]])
'''))

T.append(dict(
id="w01b_rotations_camera", title="Rotations, quaternions and the pinhole camera",
prompt="""Implement:

- `rodrigues(axis, angle)`: 3×3 rotation matrix for a rotation by `angle` (radians, right-hand rule) about `axis`
  (normalize it first).
- `quat_to_matrix(q)`: rotation matrix of a quaternion `q = (w, x, y, z)` (normalize it first).
- `quat_multiply(p, q)`: Hamilton product, so that `quat_to_matrix(quat_multiply(p, q)) == R(p) @ R(q)`.
- `homogeneous(R, t)`: the 4×4 matrix `[[R, t], [0, 1]]`.
- `project(X, K, R, t)`: pinhole projection of world points X (N, 3) to pixels (N, 2): `x ~ K (R X + t)`, divide by depth.""",
starter='''import numpy as np

def rodrigues(axis, angle):
    raise NotImplementedError

def quat_to_matrix(q):
    raise NotImplementedError

def quat_multiply(p, q):
    raise NotImplementedError

def homogeneous(R, t):
    raise NotImplementedError

def project(X, K, R, t):
    raise NotImplementedError
''',
solution='''import numpy as np

def rodrigues(axis, angle):
    k = np.asarray(axis, float); k = k / np.linalg.norm(k)
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * Kx + (1 - np.cos(angle)) * Kx @ Kx

def quat_to_matrix(q):
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])

def quat_multiply(p, q):
    a1, b1, c1, d1 = p; a2, b2, c2, d2 = q
    return np.array([a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2,
                     a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2,
                     a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2,
                     a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2])

def homogeneous(R, t):
    H = np.eye(4); H[:3, :3] = R; H[:3, 3] = np.asarray(t, float)
    return H

def project(X, K, R, t):
    Xc = np.asarray(X, float) @ np.asarray(R, float).T + np.asarray(t, float)
    x = Xc @ np.asarray(K, float).T
    return x[:, :2] / x[:, 2:3]
''',
tests='''
from scipy.spatial.transform import Rotation

def test_rodrigues_matches_scipy():
    rng = np.random.default_rng(0)
    for _ in range(20):
        ax = rng.normal(size=3); ang = rng.uniform(-3, 3)
        ref = Rotation.from_rotvec(ax / np.linalg.norm(ax) * ang).as_matrix()
        R = rodrigues(ax, ang)
        assert np.allclose(R, ref) and np.allclose(R.T @ R, np.eye(3)) and np.isclose(np.linalg.det(R), 1)

def test_z_rotation():
    assert np.allclose(rodrigues([0, 0, 2], np.pi / 2) @ [1, 0, 0], [0, 1, 0])

def test_quaternion():
    rng = np.random.default_rng(1)
    for _ in range(20):
        q = rng.normal(size=4)
        ref = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        assert np.allclose(quat_to_matrix(q), ref)
        assert np.allclose(quat_to_matrix(-q), ref)          # double cover
        p = rng.normal(size=4); p /= np.linalg.norm(p); qn = q / np.linalg.norm(q)
        assert np.allclose(quat_to_matrix(quat_multiply(p, qn)), quat_to_matrix(p) @ quat_to_matrix(qn))

def test_homogeneous_composes():
    R1, R2 = rodrigues([1, 2, 3], 0.4), rodrigues([0, 1, 0], -1.1)
    t1, t2 = np.array([1.0, 2, 3]), np.array([-2.0, 0, 5])
    x = np.array([0.3, -0.7, 2.0, 1.0])
    lhs = homogeneous(R1, t1) @ homogeneous(R2, t2) @ x
    assert np.allclose(lhs[:3], R1 @ (R2 @ x[:3] + t2) + t1)

def test_projection():
    K = np.array([[500.0, 0, 320], [0, 500, 240], [0, 0, 1]])
    X = np.array([[0.0, 0, 5], [1, 0, 5], [0, 1, 10]])
    assert np.allclose(project(X, K, np.eye(3), np.zeros(3)), [[320, 240], [420, 240], [320, 290]])
    assert np.allclose(project(2 * X, K, np.eye(3), np.zeros(3)), project(X, K, np.eye(3), np.zeros(3)))
'''))

T.append(dict(
id="w04b_gaussian_hoeffding", title="Gaussian conditioning and concentration bounds",
prompt="""Implement:

- `gaussian_condition(mu, S, idx_obs, x_obs)`: for x ~ N(mu, S), return `(mean, cov)` of the unobserved coordinates
  given that the coordinates `idx_obs` equal `x_obs` (Schur complement; solve, don't invert).
- `kl_gaussians(mu0, S0, mu1, S1)`: KL( N(mu0, S0) ‖ N(mu1, S1) ).
- `hoeffding_n(eps, delta)`: the smallest integer n such that Hoeffding's two-sided bound guarantees
  P(|mean − p| ≥ eps) ≤ delta for i.i.d. variables in [0, 1], i.e. `2·exp(−2 n eps²) ≤ delta`.
- `hoeffding_halfwidth(n, delta)`: the matching half-width eps for a given n.""",
starter='''import numpy as np

def gaussian_condition(mu, S, idx_obs, x_obs):
    raise NotImplementedError

def kl_gaussians(mu0, S0, mu1, S1) -> float:
    raise NotImplementedError

def hoeffding_n(eps, delta) -> int:
    raise NotImplementedError

def hoeffding_halfwidth(n, delta) -> float:
    raise NotImplementedError
''',
solution='''import numpy as np

def gaussian_condition(mu, S, idx_obs, x_obs):
    mu = np.asarray(mu, float); S = np.asarray(S, float)
    o = np.asarray(idx_obs); u = np.setdiff1d(np.arange(len(mu)), o)
    Soo, Suo, Suu = S[np.ix_(o, o)], S[np.ix_(u, o)], S[np.ix_(u, u)]
    A = np.linalg.solve(Soo, Suo.T).T
    mean = mu[u] + A @ (np.asarray(x_obs, float) - mu[o])
    cov = Suu - A @ Suo.T
    return mean, cov

def kl_gaussians(mu0, S0, mu1, S1):
    mu0, mu1 = np.asarray(mu0, float), np.asarray(mu1, float)
    S0, S1 = np.asarray(S0, float), np.asarray(S1, float)
    k = len(mu0); d = mu1 - mu0
    tr = np.trace(np.linalg.solve(S1, S0))
    quad = d @ np.linalg.solve(S1, d)
    _, ld1 = np.linalg.slogdet(S1); _, ld0 = np.linalg.slogdet(S0)
    return float(0.5 * (tr + quad - k + ld1 - ld0))

def hoeffding_n(eps, delta):
    return int(np.ceil(np.log(2 / delta) / (2 * eps ** 2)))

def hoeffding_halfwidth(n, delta):
    return float(np.sqrt(np.log(2 / delta) / (2 * n)))
''',
tests='''
def _spd(rng, d):
    A = rng.normal(size=(d, d)); return A @ A.T + d * np.eye(d)

def test_bivariate_closed_form():
    mu = np.array([1.0, 2.0]); S = np.array([[4.0, 1.2], [1.2, 1.0]])
    m, c = gaussian_condition(mu, S, [1], [3.0])
    assert np.allclose(m, [1 + 1.2 * (3 - 2)]) and np.allclose(c, [[4 - 1.44]])

def test_conditioning_matches_precision_form():
    rng = np.random.default_rng(0)
    S = _spd(rng, 5); mu = rng.normal(size=5)
    m, c = gaussian_condition(mu, S, [0, 3], [0.5, -1.0])
    P = np.linalg.inv(S); u = [1, 2, 4]
    assert np.allclose(c, np.linalg.inv(P[np.ix_(u, u)]))
    assert np.allclose(m, mu[u] - c @ P[np.ix_(u, [0, 3])] @ (np.array([0.5, -1.0]) - mu[[0, 3]]))

def test_kl():
    rng = np.random.default_rng(1)
    S = _spd(rng, 3); mu = rng.normal(size=3)
    assert np.isclose(kl_gaussians(mu, S, mu, S), 0)
    assert np.isclose(kl_gaussians([0.0], [[1.0]], [1.0], [[4.0]]), 0.5 * (0.25 + 0.25 - 1 + np.log(4)))
    assert kl_gaussians(mu, S, mu + 1, 2 * S) > 0

def test_hoeffding():
    n = hoeffding_n(0.01, 0.05)
    assert n == 18445
    assert 2 * np.exp(-2 * n * 0.01 ** 2) <= 0.05 < 2 * np.exp(-2 * (n - 1) * 0.01 ** 2)
    assert np.isclose(hoeffding_halfwidth(n, 0.05), 0.01, atol=1e-6)
'''))

T.append(dict(
id="w04b_mcmc_importance", title="Importance sampling and Metropolis–Hastings",
prompt="""Implement:

- `importance_estimate(f, logp, q_sample, q_logpdf, n, rng)`: self-normalized importance-sampling estimate of
  E_p[f(X)] where `logp` is an **unnormalized** log-density. Draw `x = q_sample(n, rng)` (shape (n,) or (n, d)),
  weights `w ∝ exp(logp(x) − q_logpdf(x))` (stabilize with the max). Return `(estimate, ess)` where
  `ess = (Σw)² / Σw²`.
- `metropolis_hastings(logp, x0, n, step, rng)`: random-walk MH with Gaussian proposals of std `step`. Return
  `(samples, acceptance_rate)` with `samples` of shape (n, d) (the chain after each step, including repeats).""",
starter='''import numpy as np

def importance_estimate(f, logp, q_sample, q_logpdf, n, rng):
    raise NotImplementedError

def metropolis_hastings(logp, x0, n, step, rng):
    raise NotImplementedError
''',
solution='''import numpy as np

def importance_estimate(f, logp, q_sample, q_logpdf, n, rng):
    x = q_sample(n, rng)
    lw = logp(x) - q_logpdf(x)
    w = np.exp(lw - lw.max())
    est = float(np.sum(w * f(x)) / w.sum())
    ess = float(w.sum() ** 2 / np.sum(w ** 2))
    return est, ess

def metropolis_hastings(logp, x0, n, step, rng):
    x = np.atleast_1d(np.asarray(x0, float)).copy()
    lp = logp(x)
    out = np.empty((n, x.size)); acc = 0
    for i in range(n):
        y = x + step * rng.normal(size=x.size)
        ly = logp(y)
        if np.log(rng.random()) < ly - lp:
            x, lp = y, ly; acc += 1
        out[i] = x
    return out, acc / n
''',
tests='''
from scipy import stats

def test_importance_gaussian_mean():
    rng = np.random.default_rng(0)
    logp = lambda x: -0.5 * (x - 2.0) ** 2            # N(2, 1), unnormalized
    qs = lambda n, r: r.normal(0, 3, size=n)
    ql = lambda x: stats.norm(0, 3).logpdf(x)
    est, ess = importance_estimate(lambda x: x, logp, qs, ql, 50000, rng)
    assert abs(est - 2.0) < 0.05 and 0 < ess <= 50000

def test_light_tailed_proposal_has_low_ess():
    rng = np.random.default_rng(1)
    logp = lambda x: -0.5 * (x / 3) ** 2
    _, good = importance_estimate(lambda x: x, logp, lambda n, r: r.normal(0, 4, n), stats.norm(0, 4).logpdf, 20000, rng)
    _, bad = importance_estimate(lambda x: x, logp, lambda n, r: r.normal(0, 0.7, n), stats.norm(0, 0.7).logpdf, 20000, rng)
    assert bad < 0.2 * good

def test_mh_targets_correlated_gaussian():
    rng = np.random.default_rng(2)
    C = np.array([[1.0, 0.8], [0.8, 1.0]]); P = np.linalg.inv(C)
    logp = lambda x: -0.5 * x @ P @ x
    s, acc = metropolis_hastings(logp, [3.0, -3.0], 40000, 0.8, rng)
    s = s[5000:]
    assert s.shape == (35000, 2) and 0.2 < acc < 0.8
    assert np.allclose(s.mean(0), 0, atol=0.12)
    assert np.allclose(np.cov(s.T), C, atol=0.15)
'''))

T.append(dict(
id="w05b_intervals_tests", title="Wilson interval, McNemar and the paired t-test",
prompt="""Implement statistics for comparing models (you may use `scipy.stats` distributions, not its tests).

- `wilson_interval(k, n, conf=0.95)`: Wilson score interval for a proportion k/n.
- `mcnemar_exact(a_correct, b_correct)`: two boolean arrays (same test examples). With b = #(A right, B wrong) and
  c = #(A wrong, B right), return the exact two-sided p-value `min(1, 2·P(Bin(b+c, 0.5) ≤ min(b, c)))`
  (1.0 when b + c = 0).
- `paired_t(x, y)`: return `(t, p)` for the two-sided paired t-test of mean(x − y) = 0.
- `cramer_rao_bernoulli(p, n)`: the Cramér–Rao lower bound on the variance of an unbiased estimator of p from n
  Bernoulli trials.""",
starter='''import numpy as np
from scipy import stats

def wilson_interval(k, n, conf=0.95):
    raise NotImplementedError

def mcnemar_exact(a_correct, b_correct) -> float:
    raise NotImplementedError

def paired_t(x, y):
    raise NotImplementedError

def cramer_rao_bernoulli(p, n) -> float:
    raise NotImplementedError
''',
solution='''import numpy as np
from scipy import stats

def wilson_interval(k, n, conf=0.95):
    z = stats.norm.ppf(0.5 + conf / 2)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (float(c - h), float(c + h))

def mcnemar_exact(a_correct, b_correct):
    a = np.asarray(a_correct, bool); b_ = np.asarray(b_correct, bool)
    b = int(np.sum(a & ~b_)); c = int(np.sum(~a & b_))
    if b + c == 0:
        return 1.0
    return float(min(1.0, 2 * stats.binom.cdf(min(b, c), b + c, 0.5)))

def paired_t(x, y):
    d = np.asarray(x, float) - np.asarray(y, float)
    n = len(d)
    t = d.mean() / (d.std(ddof=1) / np.sqrt(n))
    return float(t), float(2 * stats.t.sf(abs(t), n - 1))

def cramer_rao_bernoulli(p, n):
    return float(p * (1 - p) / n)
''',
tests='''
from scipy import stats
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.contingency_tables import mcnemar

def test_wilson_matches_statsmodels():
    for k, n in [(0, 20), (3, 20), (91, 100), (1800, 2000)]:
        assert np.allclose(wilson_interval(k, n), proportion_confint(k, n, method="wilson"))

def test_wilson_stays_inside_unit_interval():
    lo, hi = wilson_interval(20, 20)
    assert 0 <= lo < 1 and hi <= 1 + 1e-12

def test_mcnemar():
    rng = np.random.default_rng(0)
    a = rng.random(500) < 0.85; b = a.copy(); flip = rng.random(500) < 0.08; b[flip] = ~b[flip]
    t = [[np.sum(a & b), np.sum(a & ~b)], [np.sum(~a & b), np.sum(~a & ~b)]]
    assert np.isclose(mcnemar_exact(a, b), mcnemar(t, exact=True).pvalue)
    assert mcnemar_exact(a, a) == 1.0

def test_paired_t():
    rng = np.random.default_rng(1)
    x = rng.normal(size=12); y = x + rng.normal(0.3, 0.2, size=12)
    r = stats.ttest_rel(x, y)
    assert np.allclose(paired_t(x, y), (r.statistic, r.pvalue))

def test_crlb():
    assert np.isclose(cramer_rao_bernoulli(0.9, 1000), 9e-5)
'''))

T.append(dict(
id="w05b_fdr_bayes_ab", title="Multiple testing and a Bayesian A/B test",
prompt="""Implement:

- `holm(pvals, alpha=0.05)`: boolean array, True where Holm's step-down procedure rejects.
- `benjamini_hochberg(pvals, q=0.05)`: boolean array, True where BH rejects (largest k with p_(k) ≤ k·q/m; reject
  the k smallest).
- `beta_posterior(successes, trials, a=1.0, b=1.0)`: return `(alpha_post, beta_post, mean, (lo, hi))` for a
  Beta(a, b) prior on a Bernoulli rate, with an equal-tailed 95% credible interval.
- `prob_b_beats_a(sa, na, sb, nb)`: P(θ_B > θ_A) under independent Beta(1, 1) priors, computed by numerical
  integration over θ (not sampling), accurate to 1e-4.""",
starter='''import numpy as np
from scipy import stats

def holm(pvals, alpha=0.05):
    raise NotImplementedError

def benjamini_hochberg(pvals, q=0.05):
    raise NotImplementedError

def beta_posterior(successes, trials, a=1.0, b=1.0):
    raise NotImplementedError

def prob_b_beats_a(sa, na, sb, nb) -> float:
    raise NotImplementedError
''',
solution='''import numpy as np
from scipy import stats, integrate

def holm(pvals, alpha=0.05):
    p = np.asarray(pvals, float); m = len(p)
    order = np.argsort(p); rej = np.zeros(m, bool)
    for i, j in enumerate(order):
        if p[j] <= alpha / (m - i):
            rej[j] = True
        else:
            break
    return rej

def benjamini_hochberg(pvals, q=0.05):
    p = np.asarray(pvals, float); m = len(p)
    order = np.argsort(p)
    ok = np.nonzero(p[order] <= q * np.arange(1, m + 1) / m)[0]
    rej = np.zeros(m, bool)
    if len(ok):
        rej[order[:ok[-1] + 1]] = True
    return rej

def beta_posterior(successes, trials, a=1.0, b=1.0):
    ap, bp = a + successes, b + trials - successes
    d = stats.beta(ap, bp)
    return ap, bp, float(d.mean()), (float(d.ppf(0.025)), float(d.ppf(0.975)))

def prob_b_beats_a(sa, na, sb, nb):
    A = stats.beta(1 + sa, 1 + na - sa); B = stats.beta(1 + sb, 1 + nb - sb)
    lo, hi = min(A.ppf(1e-12), B.ppf(1e-12)), max(A.ppf(1 - 1e-12), B.ppf(1 - 1e-12))
    val, _ = integrate.quad(lambda t: B.pdf(t) * A.cdf(t), lo, hi, limit=500, points=[A.mean(), B.mean()])
    return float(val)
''',
tests='''
from statsmodels.stats.multitest import multipletests

def test_holm_and_bh_match_statsmodels():
    rng = np.random.default_rng(0)
    for _ in range(20):
        p = np.concatenate([rng.uniform(0, 0.01, 5), rng.uniform(0, 1, 25)])
        rng.shuffle(p)
        assert np.array_equal(holm(p), multipletests(p, 0.05, "holm")[0])
        assert np.array_equal(benjamini_hochberg(p), multipletests(p, 0.05, "fdr_bh")[0])

def test_bh_step_up():
    p = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    assert benjamini_hochberg(p).all() and holm(p).sum() == 1

def test_beta_posterior():
    ap, bp, mean, (lo, hi) = beta_posterior(30, 100)
    assert (ap, bp) == (31, 71) and np.isclose(mean, 31 / 102) and lo < mean < hi

def test_prob_b_beats_a():
    assert np.isclose(prob_b_beats_a(50, 100, 50, 100), 0.5, atol=1e-4)
    rng = np.random.default_rng(1)
    a = rng.beta(1 + 120, 1 + 880, 400000); b = rng.beta(1 + 150, 1 + 850, 400000)
    assert np.isclose(prob_b_beats_a(120, 1000, 150, 1000), np.mean(b > a), atol=3e-3)
    assert prob_b_beats_a(10, 1000, 100, 1000) > 0.9999
'''))
