"""Post 04 — Markov chains: what a random process does in the long run.

Checks every number used in the post:
  1. the weather chain: p_t = p_0 P^t, the stationary distribution, return times
  2. why irreducible + aperiodic are needed (two broken examples)
  3. the coffee shop under two fixed policies: stationary distribution and average profit
  4. how fast it settles (mixing), and a simulation check
Run: python 04_markov_chains.py
"""
import random


def step_distribution(p, P):
    """One step: p_next[j] = sum_i p[i] * P[i][j]."""
    return [sum(p[i] * P[i][j] for i in range(len(p))) for j in range(len(p))]


def run(p0, P, steps):
    p, history = p0, [p0]
    for _ in range(steps):
        p = step_distribution(p, P)
        history.append(p)
    return history


def stationary(P, tol=1e-12):
    """Power iteration: run the chain from a uniform start until it stops changing."""
    n = len(P)
    p = [1 / n] * n
    for _ in range(100_000):
        q = step_distribution(p, P)
        if max(abs(a - b) for a, b in zip(p, q)) < tol:
            return q
        p = q
    return p


# ---------------------------------------------------------------------------
# 1. Weather: Rainy / Sunny
# ---------------------------------------------------------------------------
WEATHER = [[0.4, 0.6],    # from Rainy
           [0.2, 0.8]]    # from Sunny

print("1. Weather chain (Rainy, Sunny)")
for start, name in (([1.0, 0.0], "start Rainy"), ([0.0, 1.0], "start Sunny")):
    hist = run(start, WEATHER, 6)
    print(f"   {name}: " + " -> ".join(f"({p[0]:.3f}, {p[1]:.3f})" for p in hist))
mu = stationary(WEATHER)
print(f"   stationary mu = ({mu[0]:.4f}, {mu[1]:.4f})")
print(f"   check mu P = ({step_distribution(mu, WEATHER)[0]:.4f}, {step_distribution(mu, WEATHER)[1]:.4f})")
print(f"   average gap between rainy days E[T] = 1/mu = {1 / mu[0]:.2f} days")

# total variation distance to the stationary distribution
print("   distance to mu from a rainy start:")
for t, p in enumerate(run([1.0, 0.0], WEATHER, 8)):
    tv = 0.5 * sum(abs(a - b) for a, b in zip(p, mu))
    print(f"     t={t}: ({p[0]:.4f}, {p[1]:.4f})  distance {tv:.4f}")

# ---------------------------------------------------------------------------
# 2. The two conditions, broken on purpose
# ---------------------------------------------------------------------------
print("\n2. When the chain does not settle")
FLIP = [[0.0, 1.0], [1.0, 0.0]]                 # periodic: swaps every step
print("   periodic (swap every step), start in state 0:")
print("     " + " -> ".join(f"({p[0]:.1f},{p[1]:.1f})" for p in run([1.0, 0.0], FLIP, 5)))
SPLIT = [[1.0, 0.0, 0.0], [0.5, 0.0, 0.5], [0.0, 0.0, 1.0]]   # two absorbing states
print("   reducible (two traps), the end depends on where you start:")
for start in ([1, 0, 0], [0, 1, 0], [0, 0, 1]):
    end = run(start, SPLIT, 50)[-1]
    print(f"     from {start}: ({end[0]:.2f}, {end[1]:.2f}, {end[2]:.2f})")

# ---------------------------------------------------------------------------
# 3. Coffee shop: a policy turns the shop into a Markov chain
# ---------------------------------------------------------------------------
PRICE, COST, CAPACITY = 3, 1, 2
DEMAND = {1: 0.5, 2: 0.5}

POLICIES = {
    "fill the fridge  (order up to 2 every day)": {0: 2, 1: 1},
    "bulk order       (order 2 only when empty)": {0: 2, 1: 0},
}

print("\n3. Coffee shop under a fixed policy")
for name, pi in POLICIES.items():
    P = [[0.0, 0.0], [0.0, 0.0]]
    rewards = [0.0, 0.0]
    for s in (0, 1):
        a = pi[s]
        for d, prob in DEMAND.items():
            sold = min(s + a, d)
            rewards[s] += prob * (PRICE * sold - COST * a)
            P[s][s + a - sold] += prob
    mu = stationary(P)
    avg = sum(mu[s] * rewards[s] for s in (0, 1))
    print(f"   {name}")
    print(f"     P = {P},  daily profit per state = {rewards}")
    print(f"     mu = ({mu[0]:.4f}, {mu[1]:.4f})  ->  average profit {avg:.4f} per day")

    random.seed(1)
    s, total, days = 0, 0.0, 200_000
    for _ in range(days):
        a = pi[s]
        d = random.choice([1, 2])
        sold = min(s + a, d)
        total += PRICE * sold - COST * a
        s = s + a - sold
    print(f"     simulated over {days:,} days: {total / days:.4f}")
