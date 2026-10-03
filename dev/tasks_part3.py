"""Coding tasks, part 3 (Weeks 16–23)."""

T = []

T.append(dict(
id="w16_hedge", title="Hedge (multiplicative weights) and its regret",
prompt="""Implement `hedge(losses, eta)` for `losses` of shape (T, n) with entries in [0, 1].

Start with uniform weights. At round t, play the distribution `p_t ∝ w_t`, suffer expected loss `p_t · ℓ_t`, then
update `w ← w · exp(−eta · ℓ_t)`. Return `(P, regret)`: `P` (T, n) the distributions played and
`regret = Σ_t p_t·ℓ_t − min_i Σ_t ℓ_t[i]`. Keep it numerically stable for long horizons (normalize each round).""",
starter='''import numpy as np

def hedge(losses, eta: float):
    raise NotImplementedError
''',
solution='''import numpy as np

def hedge(losses, eta):
    L = np.asarray(losses, float)
    T, n = L.shape
    logw = np.zeros(n)
    P = np.empty((T, n))
    for t in range(T):
        p = np.exp(logw - logw.max()); p /= p.sum()
        P[t] = p
        logw -= eta * L[t]
    regret = float((P * L).sum() - L.sum(0).min())
    return P, regret
''',
tests='''
def test_first_round_uniform():
    P, _ = hedge(np.array([[1.0, 0.0, 0.5]]), 0.5)
    assert np.allclose(P[0], 1 / 3)

def test_matches_formula():
    rng = np.random.default_rng(0)
    L = rng.random((20, 4)); eta = 0.3
    P, _ = hedge(L, eta)
    cum = np.vstack([np.zeros(4), np.cumsum(L, 0)[:-1]])
    ref = np.exp(-eta * cum); ref /= ref.sum(1, keepdims=True)
    assert np.allclose(P, ref)

def test_regret_bound():
    rng = np.random.default_rng(1)
    T, n = 5000, 10
    L = rng.random((T, n)); L[:, 3] *= 0.8
    eta = np.sqrt(8 * np.log(n) / T)
    _, R = hedge(L, eta)
    assert R <= np.sqrt(T * np.log(n) / 2) + 1e-9

def test_long_horizon_stable():
    L = np.ones((20000, 3)); L[:, 0] = 0
    P, R = hedge(L, 1.0)
    assert np.all(np.isfinite(P)) and P[-1, 0] > 0.999
'''))

T.append(dict(
id="w16_shapley", title="Exact Shapley values",
prompt="""Implement `shapley_values(value, n)` where `value` is a function taking a `frozenset` of player indices
(0..n−1) and returning a number. Compute exact Shapley values from the subset formula
`φ_i = Σ_{S ⊆ N∖{i}} |S|!(n−|S|−1)!/n! · (v(S∪{i}) − v(S))`. Return a NumPy array of length n.
(n will be at most 10.)""",
starter='''import numpy as np

def shapley_values(value, n: int):
    raise NotImplementedError
''',
solution='''import numpy as np
from itertools import combinations
from math import factorial

def shapley_values(value, n):
    phi = np.zeros(n)
    cache = {}
    def v(S):
        if S not in cache:
            cache[S] = value(S)
        return cache[S]
    for i in range(n):
        others = [j for j in range(n) if j != i]
        for k in range(n):
            w = factorial(k) * factorial(n - k - 1) / factorial(n)
            for S in combinations(others, k):
                S = frozenset(S)
                phi[i] += w * (v(S | {i}) - v(S))
    return phi
''',
tests='''
def test_additive_game():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    phi = shapley_values(lambda S: sum(a[i] for i in S), 4)
    assert np.allclose(phi, a)

def test_glove_game():
    # players 0,1 own left gloves, player 2 a right glove; a pair is worth 1
    v = lambda S: float(min(len(S & {0, 1}), len(S & {2})))
    phi = shapley_values(v, 3)
    assert np.allclose(phi, [1 / 6, 1 / 6, 2 / 3])

def test_efficiency_and_null():
    rng = np.random.default_rng(0)
    w = rng.random(5); w[4] = 0
    v = lambda S: (sum(w[i] for i in S)) ** 2
    phi = shapley_values(v, 5)
    assert np.isclose(phi.sum(), v(frozenset(range(5))) - v(frozenset()))
    assert abs(phi[4]) < 1e-12
'''))

T.append(dict(
id="w17_value_iteration", title="Value iteration on a tabular MDP",
prompt="""Implement `value_iteration(P, R, gamma, tol=1e-10)`.

- `P` (S, A, S): transition probabilities; `R` (S, A): expected immediate reward.
- Iterate the Bellman optimality operator until the max change is below `tol`.
Return `(V, policy)` where `policy[s]` is a greedy action (lowest index on ties).""",
starter='''import numpy as np

def value_iteration(P, R, gamma: float, tol: float = 1e-10):
    raise NotImplementedError
''',
solution='''import numpy as np

def value_iteration(P, R, gamma, tol=1e-10):
    V = np.zeros(P.shape[0])
    while True:
        Q = R + gamma * P @ V
        V2 = Q.max(1)
        if np.max(np.abs(V2 - V)) < tol:
            V = V2
            break
        V = V2
    Q = R + gamma * P @ V
    return V, np.argmax(Q, axis=1)
''',
tests='''
def test_two_state_closed_form():
    # state 0: action 0 stays (+1), action 1 goes to 1 (+0); state 1 absorbing with +2
    P = np.zeros((2, 2, 2)); P[0, 0, 0] = 1; P[0, 1, 1] = 1; P[1, :, 1] = 1
    R = np.array([[1.0, 0.0], [2.0, 2.0]])
    V, pi = value_iteration(P, R, 0.9)
    assert np.allclose(V[1], 2 / 0.1) and np.allclose(V[0], 0.9 * 20)
    assert pi[0] == 1

def test_bellman_fixed_point_random():
    rng = np.random.default_rng(0)
    S, A = 6, 3
    P = rng.random((S, A, S)); P /= P.sum(-1, keepdims=True)
    R = rng.normal(size=(S, A))
    V, pi = value_iteration(P, R, 0.95)
    assert np.allclose(V, (R + 0.95 * P @ V).max(1), atol=1e-8)
    Ppi = P[np.arange(S), pi]; Rpi = R[np.arange(S), pi]
    assert np.allclose(V, np.linalg.solve(np.eye(S) - 0.95 * Ppi, Rpi), atol=1e-6)
'''))

T.append(dict(
id="w17_q_learning", title="Q-learning and double Q-learning updates",
prompt="""Implement two in-place-free update rules (return new arrays; don't modify inputs):

- `q_update(Q, s, a, r, s2, done, alpha, gamma)`: standard Q-learning target `r + γ·max_a' Q[s2, a']`
  (no bootstrap when `done`).
- `double_q_update(QA, QB, s, a, r, s2, done, alpha, gamma)`: update **QA** using
  `a* = argmax QA[s2]` evaluated by `QB[s2, a*]`. Return the new QA.""",
starter='''import numpy as np

def q_update(Q, s, a, r, s2, done, alpha, gamma):
    raise NotImplementedError

def double_q_update(QA, QB, s, a, r, s2, done, alpha, gamma):
    raise NotImplementedError
''',
solution='''import numpy as np

def q_update(Q, s, a, r, s2, done, alpha, gamma):
    Q = np.array(Q, float)
    target = r + (0.0 if done else gamma * Q[s2].max())
    Q[s, a] += alpha * (target - Q[s, a])
    return Q

def double_q_update(QA, QB, s, a, r, s2, done, alpha, gamma):
    QA = np.array(QA, float)
    if done:
        target = r
    else:
        a_star = int(np.argmax(QA[s2]))
        target = r + gamma * QB[s2, a_star]
    QA[s, a] += alpha * (target - QA[s, a])
    return QA
''',
tests='''
def test_q_update_value_and_no_mutation():
    Q = np.zeros((3, 2)); Q[2] = [1.0, 5.0]; Q0 = Q.copy()
    Q2 = q_update(Q, 0, 1, 1.0, 2, False, 0.5, 0.9)
    assert np.isclose(Q2[0, 1], 0.5 * (1 + 0.9 * 5))
    assert np.array_equal(Q, Q0)

def test_terminal():
    Q = np.ones((2, 2)) * 10
    assert np.isclose(q_update(Q, 0, 0, 1.0, 1, True, 1.0, 0.9)[0, 0], 1.0)

def test_double_q_uses_other_table():
    QA = np.zeros((2, 2)); QA[1] = [3.0, 1.0]
    QB = np.zeros((2, 2)); QB[1] = [-2.0, 7.0]
    new = double_q_update(QA, QB, 0, 0, 0.0, 1, False, 1.0, 1.0)
    assert np.isclose(new[0, 0], -2.0)

def test_converges_on_bandit():
    rng = np.random.default_rng(0)
    Q = np.zeros((1, 2))
    for _ in range(5000):
        a = rng.integers(2); r = rng.normal(1.0 if a else 0.0, 0.1)
        Q = q_update(Q, 0, a, r, 0, True, 0.01, 0.0)
    assert abs(Q[0, 1] - 1.0) < 0.05 and abs(Q[0, 0]) < 0.05
'''))

T.append(dict(
id="w18_gae", title="Generalized advantage estimation",
prompt="""Implement `gae(rewards, values, dones, last_value, gamma, lam)` for one rollout of length T.

- `values[t]` is V(s_t); `last_value` is V(s_T) used to bootstrap after the final step.
- `dones[t] = True` means the episode ended **after** step t (no bootstrapping across it).
- `δ_t = r_t + γ·V(s_{t+1})·(1 − done_t) − V(s_t)`; `A_t = δ_t + γλ(1 − done_t)·A_{t+1}`.
Return `(advantages, returns)` with `returns = advantages + values`.""",
starter='''import numpy as np

def gae(rewards, values, dones, last_value, gamma, lam):
    raise NotImplementedError
''',
solution='''import numpy as np

def gae(rewards, values, dones, last_value, gamma, lam):
    r = np.asarray(rewards, float); v = np.asarray(values, float); d = np.asarray(dones, float)
    T = len(r)
    adv = np.zeros(T)
    nxt_v, nxt_a = last_value, 0.0
    for t in reversed(range(T)):
        nonterm = 1.0 - d[t]
        delta = r[t] + gamma * nxt_v * nonterm - v[t]
        nxt_a = delta + gamma * lam * nonterm * nxt_a
        adv[t] = nxt_a
        nxt_v = v[t]
    return adv, adv + v
''',
tests='''
def test_lambda_zero_is_td():
    r = np.array([1.0, 0.0, 2.0]); v = np.array([0.5, 0.2, 0.1]); d = np.array([False, False, False])
    adv, _ = gae(r, v, d, 0.3, 0.9, 0.0)
    assert np.allclose(adv, r + 0.9 * np.array([0.2, 0.1, 0.3]) - v)

def test_lambda_one_is_mc():
    r = np.array([1.0, 0.0, 2.0]); v = np.zeros(3); d = np.array([False, False, True])
    adv, ret = gae(r, v, d, 99.0, 0.5, 1.0)
    assert np.allclose(ret, [1 + 0.5 * 0 + 0.25 * 2, 0 + 0.5 * 2, 2])

def test_done_cuts_bootstrap():
    r = np.array([1.0, 1.0]); v = np.array([0.0, 0.0]); d = np.array([True, False])
    adv, _ = gae(r, v, d, 10.0, 1.0, 1.0)
    assert np.isclose(adv[0], 1.0) and np.isclose(adv[1], 11.0)
'''))

T.append(dict(
id="w18_ppo_clip", title="PPO clipped surrogate loss and gradient",
prompt="""Implement `ppo_clip_loss(logp_new, logp_old, adv, eps=0.2)`.

With ratio `ρ = exp(logp_new − logp_old)`, the loss to **minimize** is
`−mean( min(ρ·A, clip(ρ, 1−eps, 1+eps)·A) )`. Return `(loss, dlogp_new, clip_fraction)` where
`dlogp_new` is the gradient of the loss w.r.t. `logp_new` and `clip_fraction` is the fraction of samples where
the clipped term is the one selected **and** differs from the unclipped one.""",
starter='''import numpy as np

def ppo_clip_loss(logp_new, logp_old, adv, eps=0.2):
    raise NotImplementedError
''',
solution='''import numpy as np

def ppo_clip_loss(logp_new, logp_old, adv, eps=0.2):
    ln = np.asarray(logp_new, float); lo = np.asarray(logp_old, float); A = np.asarray(adv, float)
    rho = np.exp(ln - lo)
    unclipped = rho * A
    clipped = np.clip(rho, 1 - eps, 1 + eps) * A
    use_unclipped = unclipped <= clipped
    obj = np.where(use_unclipped, unclipped, clipped)
    n = len(A)
    grad = np.where(use_unclipped, -rho * A / n, 0.0)
    clip_frac = float(np.mean(~use_unclipped))
    return float(-obj.mean()), grad, clip_frac
''',
tests='''
def test_no_change_equals_minus_mean_adv():
    A = np.array([1.0, -2.0, 0.5])
    loss, g, cf = ppo_clip_loss(np.zeros(3), np.zeros(3), A)
    assert np.isclose(loss, -A.mean()) and np.allclose(g, -A / 3) and cf == 0.0

def test_clipping_positive_advantage():
    loss, g, cf = ppo_clip_loss(np.log([1.5]), np.zeros(1), np.array([2.0]), 0.2)
    assert np.isclose(loss, -1.2 * 2.0) and g[0] == 0.0 and cf == 1.0

def test_negative_advantage_not_clipped_when_ratio_high():
    loss, g, cf = ppo_clip_loss(np.log([1.5]), np.zeros(1), np.array([-2.0]), 0.2)
    assert np.isclose(loss, 1.5 * 2.0) and np.isclose(g[0], 1.5 * 2.0) and cf == 0.0

def test_gradient_numeric():
    rng = np.random.default_rng(0)
    ln, lo, A = rng.normal(0, 0.1, 6), rng.normal(0, 0.1, 6), rng.normal(size=6)
    _, g, _ = ppo_clip_loss(ln, lo, A)
    for i in range(6):
        e = np.zeros(6); e[i] = 1e-7
        num = (ppo_clip_loss(ln + e, lo, A)[0] - ppo_clip_loss(ln - e, lo, A)[0]) / 2e-7
        assert abs(num - g[i]) < 1e-5
'''))

T.append(dict(
id="w19_top_p", title="Temperature and nucleus (top-p) sampling filter",
prompt="""Implement `sampling_probs(logits, temperature=1.0, top_p=1.0)` for a 1-D logits vector.

1. Divide logits by `temperature` and apply a stable softmax.
2. Keep the smallest set of most-probable tokens whose cumulative probability is ≥ `top_p` (always keep at least
   the top token; break probability ties by lower index), zero out the rest, and renormalize.
Return the final probability vector.""",
starter='''import numpy as np

def sampling_probs(logits, temperature=1.0, top_p=1.0):
    raise NotImplementedError
''',
solution='''import numpy as np

def sampling_probs(logits, temperature=1.0, top_p=1.0):
    z = np.asarray(logits, float) / temperature
    z = z - z.max()
    p = np.exp(z); p /= p.sum()
    order = np.lexsort((np.arange(len(p)), -p))
    cum = np.cumsum(p[order])
    k = int(np.searchsorted(cum, top_p - 1e-12) + 1)
    keep = order[:max(1, k)]
    out = np.zeros_like(p); out[keep] = p[keep]
    return out / out.sum()
''',
tests='''
def test_no_filter():
    l = np.array([1.0, 2.0, 3.0])
    p = sampling_probs(l)
    assert np.allclose(p, np.exp(l) / np.exp(l).sum())

def test_top_p_keeps_nucleus():
    l = np.log(np.array([0.5, 0.3, 0.15, 0.05]))
    p = sampling_probs(l, top_p=0.8)
    assert np.allclose(p, [0.5 / 0.8, 0.3 / 0.8, 0, 0])

def test_temperature_sharpens():
    l = np.array([1.0, 2.0])
    assert sampling_probs(l, temperature=0.1)[1] > sampling_probs(l, temperature=1.0)[1]

def test_tiny_top_p_is_greedy():
    p = sampling_probs(np.array([0.1, 3.0, 2.9]), top_p=1e-6)
    assert np.allclose(p, [0, 1, 0])
'''))

T.append(dict(
id="w19_dpo_loss", title="DPO loss and implicit reward accuracy",
prompt="""Implement `dpo_loss(pi_chosen, pi_rejected, ref_chosen, ref_rejected, beta)` for batches of sequence
log-probabilities (1-D arrays).

With implicit rewards `r_c = β(π_c − ref_c)` and `r_r = β(π_r − ref_r)`, the loss is
`mean(−log σ(r_c − r_r))`, computed stably. Return `(loss, reward_accuracy, grad_chosen, grad_rejected)` where
`reward_accuracy` is the fraction with `r_c > r_r`, and the gradients are w.r.t. `pi_chosen` and `pi_rejected`.""",
starter='''import numpy as np

def dpo_loss(pi_chosen, pi_rejected, ref_chosen, ref_rejected, beta):
    raise NotImplementedError
''',
solution='''import numpy as np

def dpo_loss(pi_chosen, pi_rejected, ref_chosen, ref_rejected, beta):
    pc, pr = np.asarray(pi_chosen, float), np.asarray(pi_rejected, float)
    rc = beta * (pc - np.asarray(ref_chosen, float))
    rr = beta * (pr - np.asarray(ref_rejected, float))
    m = rc - rr
    loss = np.mean(np.logaddexp(0.0, -m))
    s = np.exp(-np.logaddexp(0.0, m))  # sigma(-m)
    n = len(m)
    return float(loss), float(np.mean(rc > rr)), -beta * s / n, beta * s / n
''',
tests='''
def test_equal_policies_log2():
    z = np.zeros(4)
    loss, acc, gc, gr = dpo_loss(z, z, z, z, 0.1)
    assert np.isclose(loss, np.log(2)) and acc == 0.0
    assert np.allclose(gc, -0.1 * 0.5 / 4) and np.allclose(gr, 0.1 * 0.5 / 4)

def test_numeric_gradient_and_acc():
    rng = np.random.default_rng(0)
    pc, pr, rc, rr = (rng.normal(size=5) for _ in range(4))
    loss, acc, gc, gr = dpo_loss(pc, pr, rc, rr, 0.5)
    for i in range(5):
        e = np.zeros(5); e[i] = 1e-6
        assert abs((dpo_loss(pc + e, pr, rc, rr, 0.5)[0] - dpo_loss(pc - e, pr, rc, rr, 0.5)[0]) / 2e-6 - gc[i]) < 1e-6
        assert abs((dpo_loss(pc, pr + e, rc, rr, 0.5)[0] - dpo_loss(pc, pr - e, rc, rr, 0.5)[0]) / 2e-6 - gr[i]) < 1e-6
    assert acc == np.mean(0.5 * (pc - rc) > 0.5 * (pr - rr))

def test_stable_large_margin():
    loss, *_ = dpo_loss(np.array([-1e4]), np.array([1e4]), np.zeros(1), np.zeros(1), 1.0)
    assert np.isfinite(loss) and np.isclose(loss, 2e4)
'''))

T.append(dict(
id="w20_lora", title="LoRA forward, merge and parameter count",
prompt="""Implement LoRA for a linear layer `y = x W` with frozen `W` (d_in, d_out).

- `lora_forward(x, W, A, B, alpha)`: `A` is (d_in, r), `B` is (r, d_out); return `x W + (alpha / r) · x A B`.
- `lora_merge(W, A, B, alpha)`: return the merged weight so that `x @ merged` equals `lora_forward`.
- `lora_init(d_in, d_out, r, seed=0)`: return `(A, B)` with `A ~ N(0, 1/r)` (any seeded Gaussian) and
  **B all zeros**, so the adapted layer starts identical to the base layer.
- `lora_param_count(d_in, d_out, r)`: number of trainable parameters.""",
starter='''import numpy as np

def lora_forward(x, W, A, B, alpha):
    raise NotImplementedError

def lora_merge(W, A, B, alpha):
    raise NotImplementedError

def lora_init(d_in, d_out, r, seed=0):
    raise NotImplementedError

def lora_param_count(d_in, d_out, r):
    raise NotImplementedError
''',
solution='''import numpy as np

def lora_forward(x, W, A, B, alpha):
    r = A.shape[1]
    return x @ W + (alpha / r) * (x @ A) @ B

def lora_merge(W, A, B, alpha):
    return W + (alpha / A.shape[1]) * A @ B

def lora_init(d_in, d_out, r, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1 / np.sqrt(r), (d_in, r)), np.zeros((r, d_out))

def lora_param_count(d_in, d_out, r):
    return r * (d_in + d_out)
''',
tests='''
def test_init_is_identity():
    rng = np.random.default_rng(0)
    W = rng.normal(size=(8, 6)); x = rng.normal(size=(3, 8))
    A, B = lora_init(8, 6, 2)
    assert A.shape == (8, 2) and B.shape == (2, 6) and np.all(B == 0)
    assert np.allclose(lora_forward(x, W, A, B, 16), x @ W)

def test_merge_matches_forward():
    rng = np.random.default_rng(1)
    W, A, B, x = rng.normal(size=(8, 6)), rng.normal(size=(8, 4)), rng.normal(size=(4, 6)), rng.normal(size=(5, 8))
    assert np.allclose(x @ lora_merge(W, A, B, 8), lora_forward(x, W, A, B, 8))

def test_update_rank_and_count():
    rng = np.random.default_rng(2)
    A, B = rng.normal(size=(64, 3)), rng.normal(size=(3, 32))
    delta = lora_merge(np.zeros((64, 32)), A, B, 3)
    assert np.linalg.matrix_rank(delta) == 3
    assert lora_param_count(4096, 4096, 16) == 131072
'''))

T.append(dict(
id="w20_prefix_attention", title="Prefix tuning as extra keys and values",
prompt="""Prefix tuning prepends learned key/value vectors to every attention layer.

Implement `prefix_attention(Q, K, V, P_k, P_v)` for single-head attention: `Q` (Lq, d), `K` and `V` (L, d),
prefixes `P_k` and `P_v` (p, d). Attention runs over the concatenated keys `[P_k; K]` and values `[P_v; V]`
(no mask). Return `(out, prefix_weight)` where `prefix_weight` (Lq,) is the total attention mass each query puts
on the prefix positions.""",
starter='''import numpy as np

def prefix_attention(Q, K, V, P_k, P_v):
    raise NotImplementedError
''',
solution='''import numpy as np

def prefix_attention(Q, K, V, P_k, P_v):
    Kc = np.vstack([P_k, K]); Vc = np.vstack([P_v, V])
    s = Q @ Kc.T / np.sqrt(Q.shape[1])
    s = s - s.max(1, keepdims=True)
    w = np.exp(s); w /= w.sum(1, keepdims=True)
    return w @ Vc, w[:, :len(P_k)].sum(1)
''',
tests='''
def _attn(Q, K, V):
    s = Q @ K.T / np.sqrt(Q.shape[1]); s -= s.max(1, keepdims=True)
    w = np.exp(s); w /= w.sum(1, keepdims=True)
    return w @ V

def test_matches_concatenated_attention():
    rng = np.random.default_rng(0)
    Q, K, V = rng.normal(size=(4, 8)), rng.normal(size=(6, 8)), rng.normal(size=(6, 8))
    Pk, Pv = rng.normal(size=(3, 8)), rng.normal(size=(3, 8))
    out, pw = prefix_attention(Q, K, V, Pk, Pv)
    assert np.allclose(out, _attn(Q, np.vstack([Pk, K]), np.vstack([Pv, V])))
    assert pw.shape == (4,) and np.all((pw > 0) & (pw < 1))

def test_dominant_prefix():
    Q = np.ones((1, 4)); K = np.zeros((2, 4)); V = np.zeros((2, 4))
    Pk = np.full((1, 4), 20.0); Pv = np.full((1, 4), 7.0)
    out, pw = prefix_attention(Q, K, V, Pk, Pv)
    assert pw[0] > 0.999 and np.allclose(out, 7.0, atol=1e-3)
'''))

T.append(dict(
id="w21_ctc_loss", title="CTC loss by the forward algorithm",
prompt="""Implement `ctc_loss(log_probs, target, blank=0)`.

- `log_probs` (T, C): per-frame log-probabilities (already log-softmaxed). `target`: list of label ids (no blanks).
- Return the negative log-likelihood `−log p(target | x)`, summing over all alignments that collapse to the target
  (merge repeats, then remove blanks), computed with the forward (α) recursion **in log space**.
- Return `inf` if no alignment is possible (target too long for T).""",
starter='''import numpy as np

def ctc_loss(log_probs, target, blank=0) -> float:
    raise NotImplementedError
''',
solution='''import numpy as np

def ctc_loss(log_probs, target, blank=0):
    lp = np.asarray(log_probs, float)
    T = lp.shape[0]
    ext = [blank]
    for c in target:
        ext += [c, blank]
    S = len(ext)
    alpha = np.full(S, -np.inf)
    alpha[0] = lp[0, ext[0]]
    if S > 1:
        alpha[1] = lp[0, ext[1]]
    for t in range(1, T):
        new = np.full(S, -np.inf)
        for s in range(S):
            terms = [alpha[s]]
            if s >= 1:
                terms.append(alpha[s - 1])
            if s >= 2 and ext[s] != blank and ext[s] != ext[s - 2]:
                terms.append(alpha[s - 2])
            new[s] = np.logaddexp.reduce(terms) + lp[t, ext[s]]
        alpha = new
    ends = [alpha[S - 1]] + ([alpha[S - 2]] if S > 1 else [])
    ll = np.logaddexp.reduce(ends)
    return float(-ll) if np.isfinite(ll) else float("inf")
''',
tests='''
import itertools

def _collapse(path, blank=0):
    out, prev = [], None
    for p in path:
        if p != prev and p != blank:
            out.append(p)
        prev = p
    return out

def _brute(lp, target):
    T, C = lp.shape
    tot = -np.inf
    for path in itertools.product(range(C), repeat=T):
        if _collapse(path) == list(target):
            tot = np.logaddexp(tot, sum(lp[t, p] for t, p in enumerate(path)))
    return -tot

def _rand_lp(rng, T, C):
    z = rng.normal(size=(T, C))
    return z - np.log(np.exp(z).sum(1, keepdims=True))

def test_against_brute_force():
    rng = np.random.default_rng(0)
    for target in [[1], [1, 2], [1, 1], [2, 1, 2]]:
        lp = _rand_lp(rng, 5, 3)
        assert np.isclose(ctc_loss(lp, target), _brute(lp, target))

def test_impossible():
    rng = np.random.default_rng(1)
    assert ctc_loss(_rand_lp(rng, 2, 3), [1, 1, 1]) == float("inf")

def test_long_sequence_finite():
    rng = np.random.default_rng(2)
    assert np.isfinite(ctc_loss(_rand_lp(rng, 400, 20), list(rng.integers(1, 20, 50))))
'''))

T.append(dict(
id="w21_wer", title="Word error rate by edit distance",
prompt="""Implement `wer(reference, hypothesis)` for whitespace-tokenized strings.

WER = (substitutions + deletions + insertions) / (number of reference words), using the minimum-edit-distance
dynamic program over words. Return a float. If the reference is empty, return 0.0 when the hypothesis is also
empty and `inf` otherwise.""",
starter='''def wer(reference: str, hypothesis: str) -> float:
    raise NotImplementedError
''',
solution='''def wer(reference, hypothesis):
    r, h = reference.split(), hypothesis.split()
    if not r:
        return 0.0 if not h else float("inf")
    prev = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        cur = [i] + [0] * len(h)
        for j in range(1, len(h) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r[i - 1] != h[j - 1]))
        prev = cur
    return prev[-1] / len(r)
''',
tests='''
def test_basic():
    assert wer("the cat sat", "the cat sat") == 0.0
    assert abs(wer("the cat sat on the mat", "the cat sit on mat") - 2 / 6) < 1e-12

def test_insertions_can_exceed_one():
    assert wer("hi", "oh hi there friend") == 3.0

def test_empty_reference():
    assert wer("", "") == 0.0 and wer("", "x") == float("inf")

def test_whitespace():
    assert wer("a  b\\tc", "a b c") == 0.0
'''))

T.append(dict(
id="w22_q_sample", title="Diffusion forward process in closed form",
prompt="""Implement the DDPM forward process.

- `alpha_bars(betas)`: cumulative products `ᾱ_t = Π_{s≤t} (1 − β_s)` (index t from 0).
- `q_sample(x0, t, betas, noise)`: `x_t = √ᾱ_t · x0 + √(1 − ᾱ_t) · noise`.
- `linear_betas(T, beta_1=1e-4, beta_T=0.02)`: the linear schedule as an array of length T.
- `posterior_mean(x0, xt, t, betas)` (t ≥ 1): mean of q(x_{t−1} | x_t, x0)
  `= √ᾱ_{t−1} β_t/(1−ᾱ_t) · x0 + √(1−β_t)(1−ᾱ_{t−1})/(1−ᾱ_t) · x_t`.""",
starter='''import numpy as np

def linear_betas(T, beta_1=1e-4, beta_T=0.02):
    raise NotImplementedError

def alpha_bars(betas):
    raise NotImplementedError

def q_sample(x0, t, betas, noise):
    raise NotImplementedError

def posterior_mean(x0, xt, t, betas):
    raise NotImplementedError
''',
solution='''import numpy as np

def linear_betas(T, beta_1=1e-4, beta_T=0.02):
    return np.linspace(beta_1, beta_T, T)

def alpha_bars(betas):
    return np.cumprod(1 - np.asarray(betas, float))

def q_sample(x0, t, betas, noise):
    ab = alpha_bars(betas)[t]
    return np.sqrt(ab) * x0 + np.sqrt(1 - ab) * noise

def posterior_mean(x0, xt, t, betas):
    ab = alpha_bars(betas); b = np.asarray(betas, float)
    c0 = np.sqrt(ab[t - 1]) * b[t] / (1 - ab[t])
    ct = np.sqrt(1 - b[t]) * (1 - ab[t - 1]) / (1 - ab[t])
    return c0 * x0 + ct * xt
''',
tests='''
def test_schedule():
    b = linear_betas(1000)
    assert len(b) == 1000 and np.isclose(b[0], 1e-4) and np.isclose(b[-1], 0.02)
    ab = alpha_bars(b)
    assert np.all(np.diff(ab) < 0) and ab[-1] < 1e-3

def test_closed_form_matches_iterative():
    rng = np.random.default_rng(0)
    b = linear_betas(50); x0 = rng.normal(size=10000)
    x = x0.copy()
    for t in range(30):
        x = np.sqrt(1 - b[t]) * x + np.sqrt(b[t]) * rng.normal(size=x.shape)
    ab = alpha_bars(b)[29]
    xs = q_sample(x0, 29, b, rng.normal(size=x0.shape))
    assert abs(np.corrcoef(x, x0)[0, 1] - np.corrcoef(xs, x0)[0, 1]) < 0.03
    assert abs(x.var() - xs.var()) < 0.05

def test_posterior_mean_consistency():
    b = linear_betas(100); t = 40
    x0 = np.array([1.0, -1.0]); eps = np.array([0.3, 0.7])
    xt = q_sample(x0, t, b, eps)
    ab = alpha_bars(b)
    mu = posterior_mean(x0, xt, t, b)
    alt = (xt - b[t] / np.sqrt(1 - ab[t]) * eps) / np.sqrt(1 - b[t])
    assert np.allclose(mu, alt)
'''))

T.append(dict(
id="w22_frechet", title="Fréchet distance between Gaussians (FID core)",
prompt="""Implement `frechet_distance(mu1, S1, mu2, S2)` =
`||mu1 − mu2||² + Tr(S1 + S2 − 2·(S1^{1/2} S2 S1^{1/2})^{1/2})` for symmetric PSD covariances.

Compute matrix square roots through **symmetric eigendecompositions** (clip tiny negative eigenvalues to 0);
don't use `scipy.linalg.sqrtm`. Also implement `stats(features)` returning `(mean, covariance)` with the
`n − 1` denominator.""",
starter='''import numpy as np

def stats(features):
    raise NotImplementedError

def frechet_distance(mu1, S1, mu2, S2) -> float:
    raise NotImplementedError
''',
solution='''import numpy as np

def _sqrtm_psd(S):
    w, U = np.linalg.eigh((S + S.T) / 2)
    return (U * np.sqrt(np.clip(w, 0, None))) @ U.T

def stats(features):
    f = np.asarray(features, float)
    return f.mean(0), np.cov(f, rowvar=False)

def frechet_distance(mu1, S1, mu2, S2):
    r1 = _sqrtm_psd(S1)
    mid = _sqrtm_psd(r1 @ S2 @ r1)
    d = np.asarray(mu1) - np.asarray(mu2)
    return float(d @ d + np.trace(S1 + S2 - 2 * mid))
''',
tests='''
def test_identical_is_zero():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(5, 5)); S = A @ A.T
    assert abs(frechet_distance(np.ones(5), S, np.ones(5), S)) < 1e-8

def test_diagonal_closed_form():
    s1, s2 = np.array([1.0, 4.0]), np.array([9.0, 1.0])
    d = frechet_distance(np.zeros(2), np.diag(s1), np.array([1.0, 2.0]), np.diag(s2))
    assert np.isclose(d, 5 + ((np.sqrt(s1) - np.sqrt(s2)) ** 2).sum())

def test_stats():
    rng = np.random.default_rng(1)
    f = rng.normal(size=(200, 3))
    m, S = stats(f)
    assert np.allclose(m, f.mean(0)) and np.allclose(S, np.cov(f.T))

def test_symmetric():
    rng = np.random.default_rng(2)
    A, B = rng.normal(size=(4, 4)), rng.normal(size=(4, 4))
    S1, S2 = A @ A.T, B @ B.T
    m1, m2 = rng.normal(size=4), rng.normal(size=4)
    assert np.isclose(frechet_distance(m1, S1, m2, S2), frechet_distance(m2, S2, m1, S1))
'''))

T.append(dict(
id="w23_clip_loss", title="CLIP contrastive loss with a learnable temperature",
prompt="""Implement `clip_loss(img, txt, logit_scale)` for paired embeddings `img`, `txt` (N, D).

L2-normalize both, form `logits = exp(logit_scale) · img @ txt.T`, and return
`(loss, d_logit_scale)`: loss = mean of image→text and text→image cross-entropies with the diagonal as targets;
`d_logit_scale` is the derivative of the loss with respect to `logit_scale`.""",
starter='''import numpy as np

def clip_loss(img, txt, logit_scale: float):
    raise NotImplementedError
''',
solution='''import numpy as np

def _ce_and_dlogits(L):
    Z = L - L.max(1, keepdims=True)
    logp = Z - np.log(np.exp(Z).sum(1, keepdims=True))
    n = len(L)
    loss = -np.mean(np.diag(logp))
    g = np.exp(logp); g[np.arange(n), np.arange(n)] -= 1
    return loss, g / n

def clip_loss(img, txt, logit_scale):
    a = img / np.linalg.norm(img, axis=1, keepdims=True)
    b = txt / np.linalg.norm(txt, axis=1, keepdims=True)
    S = a @ b.T
    s = np.exp(logit_scale)
    L = s * S
    l1, g1 = _ce_and_dlogits(L)
    l2, g2 = _ce_and_dlogits(L.T)
    loss = 0.5 * (l1 + l2)
    dL = 0.5 * (g1 + g2.T)
    return float(loss), float((dL * S).sum() * s)
''',
tests='''
def test_uniform_when_scale_tiny():
    rng = np.random.default_rng(0)
    i, t = rng.normal(size=(8, 16)), rng.normal(size=(8, 16))
    loss, _ = clip_loss(i, t, np.log(1e-6))
    assert abs(loss - np.log(8)) < 1e-4

def test_aligned_low_loss():
    rng = np.random.default_rng(1)
    i = rng.normal(size=(8, 64))
    loss, _ = clip_loss(i, i + 0.01 * rng.normal(size=i.shape), np.log(100))
    assert loss < 0.01

def test_scale_gradient_numeric():
    rng = np.random.default_rng(2)
    i, t = rng.normal(size=(6, 5)), rng.normal(size=(6, 5))
    _, g = clip_loss(i, t, 1.3)
    num = (clip_loss(i, t, 1.3 + 1e-6)[0] - clip_loss(i, t, 1.3 - 1e-6)[0]) / 2e-6
    assert abs(g - num) < 1e-6
'''))

T.append(dict(
id="w23_inverse_variance_fusion", title="Uncertainty-weighted sensor fusion",
prompt="""Several sensors estimate the same quantity with Gaussian noise. Implement
`fuse(means, variances, available=None)`:

- `means`, `variances`: arrays of shape (M,) or (M, N) (M sensors, N independent estimates).
- `available`: optional boolean mask of the same shape; missing sensors are ignored.
- Return `(fused_mean, fused_variance)` using inverse-variance weighting: weights `1/σ²`, fused variance
  `1 / Σ (1/σ²)`. If no sensor is available for an estimate, return `nan` mean and `inf` variance there.""",
starter='''import numpy as np

def fuse(means, variances, available=None):
    raise NotImplementedError
''',
solution='''import numpy as np

def fuse(means, variances, available=None):
    m = np.asarray(means, float); v = np.asarray(variances, float)
    a = np.ones_like(m, bool) if available is None else np.asarray(available, bool)
    prec = np.where(a, 1.0 / v, 0.0)
    tot = prec.sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = np.where(tot > 0, (prec * np.where(a, m, 0.0)).sum(0) / np.where(tot > 0, tot, 1.0), np.nan)
        var = np.where(tot > 0, 1.0 / np.where(tot > 0, tot, 1.0), np.inf)
    return mean, var
''',
tests='''
def test_two_sensors():
    m, v = fuse(np.array([0.0, 10.0]), np.array([1.0, 4.0]))
    assert np.isclose(m, 2.0) and np.isclose(v, 0.8)

def test_matches_kalman_update():
    # fusing a prior N(x, p) with a measurement N(z, r) equals one Kalman update
    x, p, z, r = 1.0, 2.0, 3.0, 0.5
    m, v = fuse(np.array([x, z]), np.array([p, r]))
    k = p / (p + r)
    assert np.isclose(m, x + k * (z - x)) and np.isclose(v, (1 - k) * p)

def test_missing_sensors():
    means = np.array([[1.0, 5.0], [3.0, 7.0]]); var = np.ones((2, 2))
    avail = np.array([[True, False], [True, False]])
    m, v = fuse(means, var, avail)
    assert np.isclose(m[0], 2.0) and np.isclose(v[0], 0.5)
    assert np.isnan(m[1]) and np.isinf(v[1])
'''))
