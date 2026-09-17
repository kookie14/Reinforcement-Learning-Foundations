"""Post 01 — the coffee-shop milk problem.

Checks every number used in the post and runs a tiny agent-environment loop.
Run: python 01_coffee_shop.py
"""
import random

PRICE = 3  # we earn $3 for every carton we sell
COST = 1   # we pay $1 for every carton we order


def profit(order, demand):
    """One day's reward: sales minus what we paid for milk. Leftovers spoil."""
    sold = min(order, demand)
    return PRICE * sold - COST * order


# --- Part 1: greedy vs. planning (demand tomorrow is known: 2 cartons) -----
print("Two-day plan, demand tomorrow = 2")
for order in range(4):
    today = -COST * order                  # pay today, nothing to sell yet
    tomorrow = PRICE * min(order, 2)       # milk arrives, we sell
    print(f"  order {order}: today {today:+d}, tomorrow {tomorrow:+d}, "
          f"total {today + tomorrow:+d}")

# --- Part 2: uncertain demand (1 or 3 cartons, 50/50) ----------------------
print("\nUncertain demand: 1 or 3 cartons, each with probability 0.5")
for order in range(5):
    low, high = profit(order, 1), profit(order, 3)
    expected = 0.5 * low + 0.5 * high
    print(f"  order {order}: if demand=1 -> {low:+d}, if demand=3 -> {high:+d}, "
          f"expected {expected:+.1f}")

# --- Part 3: the agent-environment loop ------------------------------------
def environment_step(order):
    demand = random.choice([1, 3])         # the world rolls its dice
    return profit(order, demand)            # ... and hands back a reward


def run(policy, days=10_000, seed=0):
    random.seed(seed)
    total = 0
    for _ in range(days):
        action = policy()                   # agent decides
        total += environment_step(action)   # environment responds
    return total / days


print("\nAverage daily profit over 10,000 simulated days")
for order in range(5):
    print(f"  always order {order}: {run(lambda: order):+.2f}")
