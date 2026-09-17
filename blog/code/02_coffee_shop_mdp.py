"""Post 02 — the coffee shop as a proper decision model.

Checks every number used in the post:
  * the probability tree for the "fill the fridge" policy
  * fixed plans vs. a policy that reacts to the fridge
  * a coin-flip (stochastic) policy
  * brute force over every plan and every policy for a 2-day horizon
Run: python 02_coffee_shop_mdp.py
"""
import itertools
import random

PRICE, COST, CAPACITY = 3, 1, 2
DEMAND = {1: 0.5, 2: 0.5}          # tomorrow's customers want 1 or 2 cartons
STATES = [0, 1]                     # cartons left in the fridge in the morning
HORIZON = 2                         # we plan two days


def actions(s):
    """You can't order more than fits in the fridge."""
    return list(range(CAPACITY - s + 1))


def step(s, a, d):
    """One day: returns (reward, next state) for demand d."""
    stock = s + a
    sold = min(stock, d)
    return PRICE * sold - COST * a, stock - sold


def trajectories(policy, s0=0, horizon=HORIZON):
    """Every possible history with its probability and return.

    policy(t, s) returns a dict {action: probability}.
    """
    paths = [(1.0, 0, s0, [])]
    for t in range(horizon):
        new = []
        for prob, ret, s, hist in paths:
            for a, pa in policy(t, s).items():
                for d, pd in DEMAND.items():
                    r, s_next = step(s, a, d)
                    new.append((prob * pa * pd, ret + r, s_next, hist + [(s, a, d, r)]))
        paths = new
    return paths


def value(policy, s0=0):
    return sum(p * g for p, g, _, _ in trajectories(policy, s0))


def deterministic(rule):
    return lambda t, s: {rule(t, s): 1.0}


# --- the "fill the fridge" policy and its probability tree ------------------
fill = deterministic(lambda t, s: CAPACITY - s)
print("Policy 'fill the fridge': pi(0) = 2, pi(1) = 1")
for p, g, _, hist in trajectories(fill):
    days = "  then  ".join(f"s={s} order {a} demand {d} r={r:+d}" for s, a, d, r in hist)
    print(f"  prob {p:.2f}  return {g:+d}   {days}")
print(f"  V(0) = {value(fill):.2f}")

# --- fixed plans: the order for each day is decided in advance --------------
print("\nFixed plans (same orders no matter what happens)")
best_plan = None
for plan in itertools.product(range(CAPACITY + 1), repeat=HORIZON):
    ok, total = True, 0.0
    for p, g, _, hist in trajectories(deterministic(lambda t, s, plan=plan: plan[t])):
        if any(a not in actions(s) for s, a, _, _ in hist):
            ok = False
            break
        total += p * g
    label = f"{total:.2f}" if ok else "impossible (fridge overflows on some days)"
    print(f"  plan {plan}: {label}")
    if ok and (best_plan is None or total > best_plan[1]):
        best_plan = (plan, total)
print(f"  best plan: {best_plan[0]} with {best_plan[1]:.2f}")

# --- a stochastic policy -----------------------------------------------------
coin = lambda t, s: {1: 0.5, 2: 0.5} if s == 0 else {1: 1.0}
one_day = lambda pol: sum(p * g for p, g, _, _ in trajectories(pol, horizon=1))
print("\nOne day from an empty fridge")
print(f"  always order 1: {one_day(deterministic(lambda t, s: 1)):.2f}")
print(f"  always order 2: {one_day(deterministic(lambda t, s: 2)):.2f}")
print(f"  flip a coin   : {one_day(coin):.2f}")

# --- brute force over all Markov deterministic policies ---------------------
rules = [dict(zip(STATES, choice)) for choice in itertools.product(*(actions(s) for s in STATES))]
best = max(
    ((r0, r1) for r0 in rules for r1 in rules),
    key=lambda pair: value(deterministic(lambda t, s, pair=pair: pair[t][s])),
)
print(f"\nPolicies that may change by day: {len(rules) ** HORIZON}")
print(f"  best: day 0 {best[0]}, day 1 {best[1]}, "
      f"V(0) = {value(deterministic(lambda t, s: best[t][s])):.2f}")

# --- the book's counting example --------------------------------------------
n = m = T = 10
print(f"\nn = m = T = 10: {m ** T:.0e} paths per start state, 10^{n * T} policies")

# --- simulate to double-check V(0) = 5.5 ------------------------------------
random.seed(0)
runs, total = 100_000, 0
for _ in range(runs):
    s = 0
    for t in range(HORIZON):
        r, s = step(s, CAPACITY - s, random.choice([1, 2]))
        total += r
print(f"\nSimulated average return of 'fill the fridge': {total / runs:.3f}")
