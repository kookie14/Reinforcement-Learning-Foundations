# Book Map — *Reinforcement Learning: Foundations* (Mannor, Mansour, Tamar, Oct 2025)

Phase 1 output (survey). Source: table of contents, Ch. 1, Ch. 2, Ch. 9, Fig. 1.1.

## 1. Shape of the book

The book has two halves: **planning** (you know the rules of the world) and **learning** (you don't).

```text
PLANNING (model is known)                  LEARNING (model is unknown / too big)
─────────────────────────                  ─────────────────────────────────────
Ch 2  Why MDPs? (philosophy)               Ch 9   Why RL? (philosophy)
Ch 3  Deterministic decisions, DP          Ch 10  Model-based RL (learn the rules)
Ch 4  Markov chains                        Ch 11  Model-free RL (MC, TD, Q-learning, SARSA)
Ch 5  MDPs, finite horizon DP, Q           Ch 12  Value function approximation, DQN
Ch 6  Discounted MDPs, VI, PI              Ch 13  Policy gradient, REINFORCE, PPO
Ch 7  Episodic MDPs                        Ch 14  Bandits, regret, UCB
Ch 8  Linear programming
                      App A: Dynamic programming · App B: ODEs
```

Dependency graph from the book (Fig. 1.1), simplified:

```text
3 Deterministic ─┐
4 Markov chains ─┴─► 5 MDP ─► 6 Discounted ─► 7 Episodic
                                   │             │
                                   ├─► 8 LP      │
                                   ▼             ▼
                              10 Model-based   11 Model-free ◄── B ODEs
                                                 │
                                   12 Value approx ─► 13 Policy gradient ─► 14 Regret
```

## 2. The ideas that actually matter (first pass)

| # | Concept | Book | Priority | Equation | Visual idea | Post |
|---|---|---|---|---|---|---|
| 1 | Agent–environment loop | 1, 9.1 | P0 | — | loop diagram | 01 |
| 2 | State / action / reward / transition | 2, 3.1, 5.1 | P0 | $s_{t+1}\sim p(\cdot\mid s_t,a_t)$ | coffee shop / grid world | 01, 02 |
| 3 | Return (sum of rewards) | 3.2.1, 5.2 | P0 | $G=\sum_t r_t$ | timeline bar chart | 01, 02 |
| 4 | Policy | 3.2.3, 5.1 | P0 | $a=\pi(s)$ | lookup table / arrows on grid | 02 |
| 5 | Dynamic programming, principle of optimality | 3.3, App A, 5.4 | P0 | backward recursion | fill table right-to-left | 03 |
| 6 | Shortest path (Bellman-Ford, Dijkstra, A*) | 3.4 | P1 | $d(u)=\min_v c(u,v)+d(v)$ | animated graph | 03 |
| 7 | Markov chains, stationary distribution | 4 | P1 | $\mu = \mu P$ | random walk histogram | 04 |
| 8 | MDP, value function $V$, Q function | 5 | P0 | $V^\pi, Q^\pi$ | heatmap on grid | 05 |
| 9 | Discount factor $\gamma$ | 6.1, 10.1 | P0 | $\sum \gamma^t r_t$ | decaying weights plot | 06 |
| 10 | Bellman equation | 6.2, 6.5 | P0 | $V=r+\gamma PV$ | "one step + rest" diagram | 06 |
| 11 | Value iteration / policy iteration | 6.6–6.8 | P0 | $V_{k+1}=TV_k$ | convergence animation | 06 |
| 12 | Contraction, Banach fixed point | 6.4 | P2 | $\|TV-TU\|\le\gamma\|V-U\|$ | shrinking distance | 06 (box) |
| 13 | Episodic MDPs | 7 | P1 | — | — | 06 (box) |
| 14 | Linear programming for MDPs | 8 | P2 | LP | — | skip / appendix |
| 15 | Exploration vs exploitation, regret | 9.1, 14 | P0 | Regret$(N)$ | slot machines | 07 |
| 16 | UCB | 14.5.3 | P1 | $\hat\mu + \sqrt{\cdot}$ | confidence bars | 07 |
| 17 | Model-based learning, R-MAX | 10 | P1 | count-based estimates | — | 08 |
| 18 | Monte Carlo evaluation | 11.3 | P0 | average of returns | — | 09 |
| 19 | TD(0), bootstrapping | 11.5.1 | P0 | $V\leftarrow V+\alpha(r+\gamma V'-V)$ | MC vs TD figure (11.5) | 09 |
| 20 | Stochastic approximation (convergence) | 11.4 | P2 | ODE method | — | 09 (box) |
| 21 | Q-learning, SARSA, on/off policy | 11.5.2–11.6 | P0 | Q update | cliff walk | 10 |
| 22 | Function approximation, projected Bellman, LSTD | 12.1–12.3 | P1 | $V_\theta\approx V$ | projection picture | 11 |
| 23 | Fitted Q, DQN | 12.5 | P0 | loss on TD target | pipeline diagram | 11 |
| 24 | Policy gradient theorem, REINFORCE | 13.5–13.6 | P0 | $\nabla J=E[\nabla\log\pi\cdot Q]$ | "push up good actions" | 12 |
| 25 | PPO | 13.8 | P1 | clipped ratio | clip plot | 12 |
| 26 | LQR, iLQR | 3.6 | P2 | Riccati | — | skip / optional |
| 27 | Mixing time, reversibility | 4.3 | P2 | — | — | skip |

## 3. Proposed blog series (normal-reader path, not book order)

```text
Part 1 — The Problem
  01  What is reinforcement learning?                     (Ch 1, 2, 9.1)
  02  States, actions, rewards, policies                  (Ch 3.1–3.2, 5.1)

Part 2 — Solving it when you know the rules (planning)
  03  Dynamic programming: think backwards                (Ch 3.3–3.4, App A)
  04  Adding dice: Markov chains in one post              (Ch 4, light)
  05  MDPs and the value of a situation (V and Q)         (Ch 5)
  06  Discounting, the Bellman equation, VI and PI        (Ch 6, 7; Ch 8 skipped)

Part 3 — Learning when you don't know the rules
  07  Explore or exploit? Bandits and UCB                 (Ch 14, moved earlier: it's the gentlest learning problem)
  08  Learn the model first: model-based RL               (Ch 10, light)
  09  Learn from experience: Monte Carlo vs TD            (Ch 11.3–11.5.1)
  10  Q-learning and SARSA                                (Ch 11.5.2–11.6)

Part 4 — Big worlds
  11  Too many states: function approximation and DQN     (Ch 12)
  12  Learn the policy directly: policy gradient and PPO  (Ch 13)
```

Running example across posts: **the coffee shop** (post 01–02, inventory is Example 2.3 in the book)
and **a small grid world** (from post 03 on, Fig. 3.6). Reuse them so readers don't have to learn a
new toy problem in every post.

## 4. Notation (use the same symbols in every post)

| Symbol | Meaning |
|---|---|
| $t$ | time step (0, 1, 2, …) |
| $s_t$ | state at time $t$ |
| $a_t$ | action at time $t$ |
| $r_t$ | reward received after taking $a_t$ in $s_t$ |
| $p(s'\mid s,a)$ | probability of the next state being $s'$ |
| $\pi$ | policy (the rule for choosing actions) |
| $G$ | return: total (possibly discounted) reward |
| $\gamma$ | discount factor, $0\le\gamma<1$ |
| $V^\pi(s)$, $Q^\pi(s,a)$ | value of a state / of a state-action pair under $\pi$ |

The book uses **costs** in Ch. 3 and **rewards** elsewhere. The blog always uses rewards (cost = −reward).

## 5. Open questions / to check while deep reading

- Ch. 7 (episodic): does it need its own post, or does a box inside post 06 cover it?
- Ch. 11.4 (stochastic approximation): is there any intuition a normal reader needs, beyond "a small step size averages out the noise"?
- Ch. 13.3 performance difference lemma: P1 or P2?
