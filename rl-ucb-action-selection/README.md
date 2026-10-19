# Upper Confidence Bound (UCB) Action Selection

Intermediate | rl | bandits | exploration | confidence-bounds

Upper Confidence Bound is a principled exploration strategy that selects actions based on an optimistic estimate of their value, balancing exploitation and uncertainty-driven exploration.

### The problem, from first principles

In UCB, each action has an estimated value Q(a) and an uncertainty measure based on how many times it's been tried. The intuition: "if I'm unsure about an action, it might be the best—let me try it." Formally:

A_t = argmax_a [ Q(a) + c·sqrt(ln(t) / N(a)) ]

where:

- Q(a): average reward of action a
- N(a): times action a has been selected
- t: total time steps elapsed
- c: exploration constant

The second term is the confidence bonus. Actions with high uncertainty (low N(a)) or early in learning (high t initially) get larger bonuses.

### From theory to code

Implement `select_ucb_action(q_values, counts, t, c)` which:

- Takes action values Q(a) for each action
- Takes visit counts N(a) for each action
- Takes the current time step t (≥ 1)
- Takes exploration constant c > 0
- Returns the selected action index

Select the action with the highest upper confidence bound.

### Constraints

- q_values and counts have the same length (≥ 1)
- counts[a] ≥ 0, with at least one action having count > 0
- t ≥ 1
- c > 0 (typically 1 or sqrt(2))
- Handle the case where an action has never been tried (count=0): the UCB is +∞, so always pick it

### Hints

<details>
<summary>Hint 1: Confidence bonus formula</summary>
bonus[a] = c * sqrt(ln(t) / N(a))

But if N(a) = 0, return np.inf.
</details>

<details>
<summary>Hint 2: Handle zero counts</summary>
Use np.where or a loop to avoid division by zero.
</details>

<details>
<summary>Hint 3: Compute UCB for all actions</summary>
ucb = q_values + bonus, then argmax.
</details>
