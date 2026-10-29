# Tabular Q-Learning

Intermediate | reinforcement-learning

### The problem, from first principles

`01-value-iteration-mdp` found the exact optimal values — but it needed the _entire_ MDP handed to it up front (`P` and `R`, in full). Q-learning solves the more realistic problem: an agent that can only ever take one action at a time and observe what happens, with no idea what the transition probabilities or reward function actually are. It has to learn the same optimal values purely from lived experience.

### From theory to code

Implement `epsilon_greedy_action` (balancing exploring new actions against exploiting known-good ones), `q_learning_update` (the core temporal-difference learning rule), and `train_q_learning`, which runs the full learning loop against an MDP the agent only ever _samples_ from, never inspects directly.

### Constraints

- `epsilon_greedy_action(Q, state, epsilon, rng)` returns a uniformly random action with probability `epsilon`, otherwise `argmax(Q[state])`.
- `q_learning_update(Q, state, action, reward, next_state, alpha, gamma, done)` updates `Q[state, action] += alpha * (target - Q[state, action])`, where `target = reward` if `done`, else `reward + gamma * max(Q[next_state])`.
- `train_q_learning` never reads `P`/`R` to plan — it only ever samples one transition (`next_state ~ P[state, action]`, `reward = R[state, action, next_state]`) at a time, exactly as a real environment interaction would work.
- After enough episodes, `train_q_learning`'s learned `Q` should approximate the same optimal values `value_iteration` finds exactly on the identical MDP.

### Hints

Open one at a time. Each gives away a little more than the last.

<details>
<summary>Hint 1</summary>

The Q-learning update's `target` is exactly the right-hand side of one Bellman backup (`01-value-iteration-mdp`'s `bellman_backup`), but evaluated using a single _sampled_ transition instead of the full expectation over all possible next states — that's the whole difference between model-based and model-free learning.

</details>

<details>
<summary>Hint 2</summary>

`train_q_learning`'s episode loop is: pick a random start state, then repeatedly (pick action via epsilon-greedy, sample the next state with `rng.choice(num_states, p=P[state, action])`, look up the reward, run `q_learning_update`, move to the next state) until landing in a terminal state.

</details>
