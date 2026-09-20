"""Post 04 — checks for the theory half: classification, recurrence, invariant distribution.

  1. communicating classes, and which are recurrent / transient
  2. the period of a state, from the lengths of its return loops
  3. the random walk on the integers: recurrent, but with infinite expected return time
  4. the random walk with jumps: positive recurrent
  5. mu_i = 1 / E[T_i], checked by simulation
Run: python 04b_chain_theory.py
"""
import math
import random
from itertools import combinations


def reachable(P, i):
    """States reachable from i in one or more steps (a walk on the transition graph)."""
    seen, stack = set(), [i]
    while stack:
        u = stack.pop()
        for j, p in enumerate(P[u]):
            if p > 0 and j not in seen:
                seen.add(j)
                stack.append(j)
    return seen


def classes(P):
    """Communicating classes: i ~ j when each is reachable from the other."""
    n = len(P)
    reach = {i: reachable(P, i) for i in range(n)}
    remaining, out = set(range(n)), []
    while remaining:
        i = min(remaining)
        group = {j for j in remaining if (j == i) or (j in reach[i] and i in reach[j])}
        out.append(sorted(group))
        remaining -= group
    return out


def is_closed(P, group):
    """A class is closed (recurrent) if no transition leaves it."""
    return all(P[i][j] == 0 for i in group for j in range(len(P)) if j not in group)


def period(P, i, max_len=40):
    """gcd of the lengths of all loops from i back to i."""
    n, lengths = len(P), []
    probs = [1.0 if k == i else 0.0 for k in range(n)]
    for m in range(1, max_len + 1):
        probs = [sum(probs[u] * P[u][v] for u in range(n)) for v in range(n)]
        if probs[i] > 1e-12:
            lengths.append(m)
    return math.gcd(*lengths) if lengths else 0


# --- 1. a chain with a transient class and a recurrent class ---------------
#     0 <-> 1  can leak into  2 <-> 3 (which is closed)
CHAIN = [[0.5, 0.4, 0.1, 0.0],
         [0.5, 0.5, 0.0, 0.0],
         [0.0, 0.0, 0.3, 0.7],
         [0.0, 0.0, 0.6, 0.4]]

print("1. Classification of a 4-state chain")
for group in classes(CHAIN):
    kind = "recurrent (closed)" if is_closed(CHAIN, group) else "transient (can leak out)"
    print(f"   class {group}: {kind}")

random.seed(0)
visits, s = [0] * 4, 0
for t in range(200_000):
    s = random.choices(range(4), weights=CHAIN[s])[0]
    visits[s] += 1
print(f"   time spent in each state over 200,000 steps: "
      f"{[round(v / 200_000, 4) for v in visits]}  (states 0,1 are visited finitely often)")

# --- 2. periods -------------------------------------------------------------
FLIP = [[0.0, 1.0], [1.0, 0.0]]
WEATHER = [[0.4, 0.6], [0.2, 0.8]]
TRIANGLE = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]
print("\n2. Period of state 0")
for name, P in (("swap chain", FLIP), ("3-cycle", TRIANGLE), ("weather", WEATHER)):
    print(f"   {name}: d = {period(P, 0)}")

# --- 3. random walk on the integers: recurrent, E[T] = infinity -------------
print("\n3. Random walk on the integers (+1 / -1, fair coin)")
print("   probability of being back at 0 after 2k steps, and the running sum:")
total = 0.0
for k in range(1, 9):
    p = math.comb(2 * k, k) * 2.0 ** (-2 * k)
    total += p
    print(f"     k={k}: p_00^({2 * k}) = {p:.4f}  (1/sqrt(pi k) = {1 / math.sqrt(math.pi * k):.4f})   sum so far {total:.3f}")
print("   the sum diverges like sum 1/sqrt(k), so state 0 is recurrent")

random.seed(1)
print("   simulated return times (capped, so the average is an underestimate):")
for cap in (100, 1_000, 10_000, 100_000):
    times, reached = [], 0
    for _ in range(4000):
        pos, t = 0, 0
        while t < cap:
            pos += random.choice((-1, 1))
            t += 1
            if pos == 0:
                break
        times.append(t)
        reached += pos == 0
    print(f"     cap {cap:>6}: returned within the cap {reached / 4000:.1%} of the time, "
          f"average capped return time {sum(times) / len(times):.1f}")

# --- 4. random walk with jumps: positive recurrent --------------------------
print("\n4. Random walk with jumps (move to i+1 or back to 0, 50/50)")
random.seed(2)
for target in (0, 1, 3):
    times, pos = [], target
    for _ in range(200_000):
        t = 0
        while True:
            pos = pos + 1 if random.random() < 0.5 else 0
            t += 1
            if pos == target:
                break
        times.append(t)
    bound = 2 + 2 * 2 ** target
    print(f"   E[T_{target}] simulated {sum(times) / len(times):.2f}   (book's bound: {bound})")

# --- 5. mu_i = 1 / E[T_i] ---------------------------------------------------
print("\n5. mu_i = 1 / E[T_i] on the weather chain")
random.seed(3)
gaps, s, last = [], 0, None
for t in range(400_000):
    s = random.choices((0, 1), weights=WEATHER[s])[0]
    if s == 0:
        if last is not None:
            gaps.append(t - last)
        last = t
print(f"   simulated E[T_rainy] = {sum(gaps) / len(gaps):.3f}   theory 1/0.25 = 4")
print(f"   fraction of rainy days = {len(gaps) / 400_000:.4f}   theory mu_rainy = 0.25")

# --- 6. detailed balance: a discrete-time queue ----------------------------
# Arrivals with probability p, service with probability q, queue length 0,1,2,...
p, q = 0.3, 0.5
lam, eta = p * (1 - q), (1 - p) * q
rho = lam / eta
N = 60                                            # truncate the (infinite) queue to check numerically
Q = [[0.0] * N for _ in range(N)]
for i in range(N):
    up = lam if i + 1 < N else 0.0
    down = eta if i > 0 else 0.0
    Q[i][i + 1 if i + 1 < N else i] += up
    if i > 0:
        Q[i][i - 1] += down
    Q[i][i] += 1 - up - down

mu = [1 / N] * N
for _ in range(200_000):
    mu = [sum(mu[i] * Q[i][j] for i in range(N)) for j in range(N)]
print(f"\n6. Queue with p={p}, q={q}: rho = lam/eta = {lam:.2f}/{eta:.2f} = {rho:.4f}")
print("   i:      numeric mu_i     (1-rho) rho^i")
for i in range(5):
    print(f"   {i}:      {mu[i]:.6f}        {(1 - rho) * rho ** i:.6f}")

# --- 7. stationary distribution of the random walk with jumps --------------
print("\n7. Random walk with jumps: mu_i should be 2^-(i+1), so E[T_i] = 2^(i+1)")
random.seed(4)
counts, pos = [0] * 6, 0
for _ in range(400_000):
    pos = pos + 1 if random.random() < 0.5 else 0
    if pos < 6:
        counts[pos] += 1
for i in range(4):
    print(f"   i={i}: simulated mu_i = {counts[i] / 400_000:.4f}, theory {2.0 ** -(i + 1):.4f}, "
          f"E[T_i] = {2 ** (i + 1)}")
