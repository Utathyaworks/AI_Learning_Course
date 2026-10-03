# Q-learning and double Q-learning updates

Implement two in-place-free update rules (return new arrays; don't modify inputs):

- `q_update(Q, s, a, r, s2, done, alpha, gamma)`: standard Q-learning target `r + γ·max_a' Q[s2, a']`
  (no bootstrap when `done`).
- `double_q_update(QA, QB, s, a, r, s2, done, alpha, gamma)`: update **QA** using
  `a* = argmax QA[s2]` evaluated by `QB[s2, a*]`. Return the new QA.
