# Value Iteration on a Small Markov Decision Process

Intermediate | reinforcement-learning

### The problem, from first principles

Every technique in this section — RLHF, PPO, DPO, reward modeling — is built on top of one foundational idea from classical reinforcement learning: given an environment where taking an action in a state leads (possibly randomly) to a new state and a reward, what's the best possible long-term strategy? Value iteration answers this exactly, for small enough environments, by repeatedly applying one simple update until it provably converges to the true optimal answer.

### From theory to code

Implement `bellman_backup` (one update step), `value_iteration` (repeat it to convergence), and `policy_evaluation_exact` (an independent, closed-form check on the result), for a Markov Decision Process described by transition probabilities `P` and rewards `R`.

### Constraints

- `P` has shape `(num_states, num_actions, num_states)`: `P[s, a, s']` is the probability of landing in `s'` after taking action `a` in state `s`. `R` has the same shape, giving the reward for that specific transition.
- `bellman_backup(V, P, R, gamma)` returns `(new_V, Q)`, where `Q[s, a] = sum_{s'} P[s, a, s'] * (R[s, a, s'] + gamma * V[s'])` and `new_V[s] = max_a Q[s, a]`.
- `value_iteration` repeats `bellman_backup` starting from `V = 0` until the largest per-state change drops below `theta`, then returns `(V, policy)` with `policy[s] = argmax_a Q[s, a]` from the final backup.
- `policy_evaluation_exact(policy, P, R, gamma)` solves `V = (I - gamma * P_pi)^-1 @ R_pi` directly via `np.linalg.solve`, where `P_pi`/`R_pi` restrict `P`/`R` to the actions `policy` actually takes.

### Hints

Open one at a time. Each gives away a little more than the last.

<details>
<summary>Hint 1</summary>

`Q[s, a] = np.sum(P[s, a] * (R[s, a] + gamma * V))` works directly as vectorized NumPy — `P[s, a]` and `R[s, a]` are both length-`num_states` vectors, and `V` is too, so the whole sum-over-`s'` collapses into one line per `(s, a)` pair.

</details>

<details>
<summary>Hint 2</summary>

`value_iteration`'s stopping condition is `np.max(np.abs(new_V - V)) < theta` — once no state's value estimate is changing by more than `theta`, you've essentially found the fixed point of the Bellman optimality equation.

</details>
