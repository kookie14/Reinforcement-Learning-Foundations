"""Post 03 — dynamic programming: think backwards.

Checks every number used in the post:
  1. the drive to the airport (backward DP on a staged graph, vs. greedy and brute force)
  2. the coffee shop solved backwards (2 days, then 30 days)
  3. Bellman-Ford round by round on the airport graph
  4. Dijkstra vs. A* on a grid with a wall
  5. DP outside control: Fibonacci and the maximum contiguous sum
Run: python 03_dynamic_programming.py
"""
import heapq
import itertools
import math

# ---------------------------------------------------------------------------
# 1. Drive to the airport: minutes on each road, travelled stage by stage
# ---------------------------------------------------------------------------
ROADS = {
    "Home": {"A": 10, "B": 15},
    "A": {"C": 20, "D": 25},
    "B": {"C": 10, "D": 15},
    "C": {"Airport": 30},
    "D": {"Airport": 10},
    "Airport": {},
}
STAGES = [["Home"], ["A", "B"], ["C", "D"], ["Airport"]]

print("1. Drive to the airport")
V, best_next = {"Airport": 0}, {}
for stage in reversed(STAGES[:-1]):
    for town in stage:
        options = {nxt: minutes + V[nxt] for nxt, minutes in ROADS[town].items()}
        best_next[town] = min(options, key=options.get)
        V[town] = options[best_next[town]]
        print(f"   V({town}) = min({', '.join(f'{m}+{V[n]}' for n, m in ROADS[town].items())}) = {V[town]}  -> go to {best_next[town]}")

route, town = ["Home"], "Home"
while town != "Airport":
    town = best_next[town]
    route.append(town)
print(f"   best route {' -> '.join(route)}: {V['Home']} min")

town, greedy, total = "Home", ["Home"], 0
while ROADS[town]:
    nxt = min(ROADS[town], key=ROADS[town].get)
    total += ROADS[town][nxt]
    town = nxt
    greedy.append(town)
print(f"   greedy (shortest next road) {' -> '.join(greedy)}: {total} min")

paths = [["Home", a, b, "Airport"] for a in ROADS["Home"] for b in ROADS[a]]
lengths = {tuple(p): sum(ROADS[x][y] for x, y in zip(p, p[1:])) for p in paths}
print(f"   brute force over {len(paths)} routes: best {min(lengths.values())}")
print(f"   sub-route from B: V(B) = {V['B']}, and B -> D -> Airport = {ROADS['B']['D'] + ROADS['D']['Airport']}")

# ---------------------------------------------------------------------------
# 2. Coffee shop (same model as post 02) solved backwards
# ---------------------------------------------------------------------------
PRICE, COST, CAPACITY = 3, 1, 2
DEMAND = {1: 0.5, 2: 0.5}
STATES = [0, 1]


def outcomes(s, a):
    """(probability, reward, next state) for each possible demand."""
    for d, p in DEMAND.items():
        sold = min(s + a, d)
        yield p, PRICE * sold - COST * a, s + a - sold


def solve(horizon, verbose=False):
    V = {s: 0.0 for s in STATES}                   # V_T(s) = 0: leftovers are worth nothing at the end
    policy = []
    for t in reversed(range(horizon)):
        Q = {s: {a: sum(p * (r + V[s2]) for p, r, s2 in outcomes(s, a))
                 for a in range(CAPACITY - s + 1)} for s in STATES}
        V = {s: max(Q[s].values()) for s in STATES}
        pi = {s: max(Q[s], key=Q[s].get) for s in STATES}
        policy.insert(0, pi)
        if verbose:
            for s in STATES:
                opts = ", ".join(f"order {a}: {q:.2f}" for a, q in Q[s].items())
                print(f"   day {t + 1}, fridge {s}: {opts}  -> V = {V[s]:.2f}, order {pi[s]}")
    return V, policy


print("\n2. Coffee shop, 2 days")
V2, _ = solve(2, verbose=True)
calcs = sum(CAPACITY - s + 1 for s in STATES) * 2
print(f"   V_0(0) = {V2[0]:.2f} using {calcs} one-step comparisons (post 02 needed 36 policies)")

V30, pol30 = solve(30)
print("\n   Coffee shop, 30 days")
print(f"   V_0(0) = {V30[0]:.2f}, V_0(1) = {V30[1]:.2f}")
distinct = {tuple(sorted(p.items())) for p in pol30}
print(f"   distinct daily rules over 30 days: {distinct}")
print(f"   comparisons: {sum(CAPACITY - s + 1 for s in STATES) * 30}; policies: 6^30 = {6 ** 30:.2e}")

# ---------------------------------------------------------------------------
# 3. Bellman-Ford: distance to the airport after each round
# ---------------------------------------------------------------------------
print("\n3. Bellman-Ford on the airport graph")
d = {v: (0 if v == "Airport" else math.inf) for v in ROADS}
for rnd in range(1, len(ROADS)):
    new = {v: (0 if v == "Airport" else min((c + d[u] for u, c in ROADS[v].items()), default=math.inf)) for v in ROADS}
    changed = new != d
    d = new
    print(f"   round {rnd}: " + ", ".join(f"{v}={'∞' if d[v] == math.inf else d[v]}" for v in ["Home", "A", "B", "C", "D"]))
    if not changed:
        print("   no change -> stop early")
        break

# ---------------------------------------------------------------------------
# 4. Dijkstra vs. A* on a grid
# ---------------------------------------------------------------------------
W, H = 20, 12
WALL = {(10, y) for y in range(0, 9)}              # a wall with a gap at the bottom
START, GOAL = (2, 3), (17, 3)


def neighbours(cell):
    x, y = cell
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        n = (x + dx, y + dy)
        if 0 <= n[0] < W and 0 <= n[1] < H and n not in WALL:
            yield n


def search(heuristic):
    dist, expanded, queue, done = {START: 0}, 0, [(heuristic(START), START)], set()
    while queue:
        _, u = heapq.heappop(queue)
        if u in done:
            continue
        done.add(u)
        expanded += 1
        if u == GOAL:
            return dist[u], expanded
        for v in neighbours(u):
            if dist[u] + 1 < dist.get(v, math.inf):
                dist[v] = dist[u] + 1
                heapq.heappush(queue, (dist[v] + heuristic(v), v))


manhattan = lambda c: abs(c[0] - GOAL[0]) + abs(c[1] - GOAL[1])
print(f"\n4. Grid {W}x{H} with a wall ({W * H - len(WALL)} open cells)")
print("   Dijkstra: length %d, cells expanded %d" % search(lambda c: 0))
print("   A*      : length %d, cells expanded %d" % search(manhattan))

# ---------------------------------------------------------------------------
# 5. DP outside control
# ---------------------------------------------------------------------------
calls = 0


def fib_naive(n):
    global calls
    calls += 1
    return n if n < 2 else fib_naive(n - 1) + fib_naive(n - 2)


print("\n5. Fibonacci(30)")
print(f"   naive recursion: {fib_naive(30)} after {calls:,} calls")
a, b = 0, 1
for _ in range(30):
    a, b = b, a + b
print(f"   bottom-up DP   : {a} after 30 additions")

xs = [2, -5, 3, 1, -2, 4, -6, 1]
V, best = [xs[0]], xs[0]
for x in xs[1:]:
    V.append(max(V[-1] + x, x))
print(f"\n   Max contiguous sum of {xs}")
print(f"   V_t = {V} -> best {max(V)}")
brute = max(sum(xs[i:j + 1]) for i, j in itertools.combinations_with_replacement(range(len(xs)), 2))
print(f"   brute force check: {brute}")
