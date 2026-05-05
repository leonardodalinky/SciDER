---
name: computer-science-theory
description: CS theory for research — algorithm complexity analysis, data structure selection, rigorous benchmarking discipline, distributed systems fundamentals, and formal verification concepts. Use when reasoning about algorithmic correctness, efficiency, or system design.
allowed_agents: [experiment, ideation]
---

# Computer Science Theory

## Overview

This skill provides the theoretical CS foundations needed for rigorous research: complexity analysis, data structure selection, benchmarking methodology, and system design principles. Use it to make principled algorithmic choices and ensure benchmarks are scientifically valid.

## When to Use This Skill

- Analyzing or comparing algorithm complexity
- Choosing the right data structure for a performance-critical path
- Designing a benchmarking study with statistical rigor
- Reasoning about distributed ML training or data pipelines
- Validating algorithm correctness with invariants and property-based tests

---

## 1. Algorithm Complexity

### Big-O Quick Reference

| Complexity | Example | Max n for 1s (rough) |
|---|---|---|
| O(1) | Hash table lookup | Any |
| O(log n) | Binary search | Any |
| O(n) | Linear scan | ~10⁸ |
| O(n log n) | Merge sort, FFT | ~10⁷ |
| O(n²) | Nested loops, naive DP | ~10⁴ |
| O(n³) | Matrix multiplication (naive) | ~10³ |
| O(2ⁿ) | Exponential, backtracking | ~25 |

```python
import time, math, numpy as np

def measure_complexity(func, sizes, repeats=5):
    """Empirically measure complexity by timing at different input sizes."""
    times = {}
    for n in sizes:
        data = list(range(n))
        elapsed = []
        for _ in range(repeats):
            start = time.perf_counter()
            func(data)
            elapsed.append(time.perf_counter() - start)
        times[n] = np.median(elapsed)

    # Log-log plot slope estimates complexity class
    log_n = np.log(list(times.keys()))
    log_t = np.log(list(times.values()))
    slope = np.polyfit(log_n, log_t, 1)[0]
    print(f"Empirical complexity slope: {slope:.2f}  (1.0=linear, 2.0=quadratic)")
    return times

# Example: verify that your sort is O(n log n)
def my_sort(data): return sorted(data)
times = measure_complexity(my_sort, [100, 1000, 10000, 100000])
```

### Amortized Analysis
Some operations appear O(n) in worst case but O(1) amortized:
- **Dynamic array append**: occasional resize is O(n), but amortized O(1)
- **Union-Find with path compression**: nearly O(1) per operation
- Don't judge a data structure by its worst-case single operation — think about sequences

### NP-Hardness and Approximation
```
If a problem is NP-hard:
1. Is n small? → Exact algorithm (brute force, dynamic programming)
2. Is structure exploitable? → Special-case polynomial algorithm
3. General large n → Approximation algorithm with provable ratio
               OR → Heuristic with empirical quality bound (report on benchmarks)
               OR → ILP solver (Branch & Bound) for moderate n

Common NP-hard research problems:
- Graph coloring, TSP, set cover, Steiner tree
- Many scheduling and assignment problems
- Subset sum, knapsack
```

---

## 2. Data Structure Selection

```python
import heapq
from collections import deque, defaultdict
from sortedcontainers import SortedList  # pip install sortedcontainers

# Decision guide:
# ────────────────────────────────────────────────────────
# Need O(1) insert + O(1) lookup by key?  → dict (hash map)
# Need O(1) insert + O(log n) min/max?   → heapq (min-heap)
# Need O(1) insert/remove at both ends?  → deque
# Need O(log n) insert + sorted iteration? → SortedList
# Need sparse matrix operations?          → scipy.sparse
# Need disjoint set union?                → union-find (custom)

# Priority queue with heapq (min-heap)
pq = []
heapq.heappush(pq, (priority, item))  # O(log n)
top_item = heapq.heappop(pq)           # O(log n)

# Deque for sliding window problems
window = deque(maxlen=k)   # auto-pops oldest when full
window.append(new_val)

# Union-Find with path compression + union by rank
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # path compression
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py: return False
        if self.rank[px] < self.rank[py]: px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]: self.rank[px] += 1
        return True
```

### Cache-Friendliness

```python
import numpy as np

# Array of Structures (AoS) — bad for vectorized access
class PointAoS:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z

points_aos = [PointAoS(i, i, i) for i in range(1000)]
# Accessing all x values: [p.x for p in points_aos]  ← many cache misses

# Structure of Arrays (SoA) — cache friendly for vectorized ops
points_x = np.arange(1000, dtype=np.float32)
points_y = np.arange(1000, dtype=np.float32)
points_z = np.arange(1000, dtype=np.float32)
# Access all x: points_x  ← contiguous memory, vectorizable
```

---

## 3. Benchmarking Discipline

Benchmarks must be reproducible and statistically valid. Poor benchmarks are a form of scientific error.

```python
import timeit, time, numpy as np, statistics

def rigorous_benchmark(func, args, n_warmup=3, n_runs=30, unit="ms"):
    """Statistically rigorous timing benchmark."""
    # Warmup: eliminate JIT compilation, cold cache effects
    for _ in range(n_warmup):
        func(*args)

    # Timed runs
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        func(*args)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    scale = {"s": 1, "ms": 1e3, "us": 1e6, "ns": 1e9}[unit]
    times_scaled = [t * scale for t in times]

    return {
        "mean": statistics.mean(times_scaled),
        "median": statistics.median(times_scaled),
        "stdev": statistics.stdev(times_scaled),
        "min": min(times_scaled),
        "max": max(times_scaled),
        "unit": unit,
        "n_runs": n_runs,
    }

# Comparing two implementations: use statistical test
from scipy.stats import wilcoxon

def compare_implementations(func1, func2, args, n_runs=30, unit="ms"):
    r1 = rigorous_benchmark(func1, args, n_runs=n_runs, unit=unit)
    r2 = rigorous_benchmark(func2, args, n_runs=n_runs, unit=unit)

    # Wilcoxon signed-rank test (paired, non-parametric)
    times1 = [rigorous_benchmark(func1, args, n_runs=1, unit=unit)["mean"] for _ in range(n_runs)]
    times2 = [rigorous_benchmark(func2, args, n_runs=1, unit=unit)["mean"] for _ in range(n_runs)]
    stat, p_value = wilcoxon(times1, times2)

    print(f"Func1: {r1['mean']:.3f} ± {r1['stdev']:.3f} {unit}")
    print(f"Func2: {r2['mean']:.3f} ± {r2['stdev']:.3f} {unit}")
    print(f"Wilcoxon p-value: {p_value:.4f} {'(significant)' if p_value < 0.05 else '(not significant)'}")
    print(f"Speedup: {r1['mean'] / r2['mean']:.2f}x")
```

**Benchmarking rules**:
1. **Warmup first**: discard first N runs to eliminate JIT, cache cold starts
2. **Report distribution**: mean ± std (or median + IQR), not just min
3. **Statistical test**: use Wilcoxon for comparing two implementations (n≥30 runs each)
4. **Multiple input sizes**: run at 3-5 different n values; plot log-log to confirm complexity
5. **Isolate**: close other processes, pin to one CPU core if measuring single-thread performance
6. **Report hardware**: CPU model, RAM, OS version — timing is hardware-dependent

---

## 4. Formal Verification Concepts

### Loop Invariants
```python
def binary_search(arr, target):
    """Binary search with explicit invariant documentation."""
    lo, hi = 0, len(arr) - 1

    # Invariant: if target is in arr, it is in arr[lo..hi]
    while lo <= hi:
        mid = lo + (hi - lo) // 2  # avoid overflow (vs (lo+hi)//2)
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1  # target is in arr[mid+1..hi]
        else:
            hi = mid - 1  # target is in arr[lo..mid-1]
    return -1  # invariant: lo > hi, so target not in arr
```

### Property-Based Testing with Hypothesis
```python
from hypothesis import given, strategies as st, settings
from hypothesis import assume

@given(st.lists(st.integers(), min_size=0, max_size=100))
def test_sort_is_sorted(lst):
    result = sorted(lst)
    assert len(result) == len(lst)               # length preserved
    assert all(result[i] <= result[i+1] for i in range(len(result)-1))  # sorted

@given(st.lists(st.integers(), min_size=1), st.integers())
def test_binary_search(lst, target):
    sorted_lst = sorted(lst)
    idx = binary_search(sorted_lst, target)
    if target in sorted_lst:
        assert idx >= 0 and sorted_lst[idx] == target
    else:
        assert idx == -1
```

### Pre/Post-Conditions
```python
def sqrt_newton(x: float, tol: float = 1e-10) -> float:
    """Newton-Raphson square root.

    Pre:  x >= 0
    Post: |result² - x| < tol
    """
    assert x >= 0, f"Precondition failed: x={x} must be non-negative"
    if x == 0:
        return 0.0
    guess = x / 2.0
    for _ in range(100):
        guess = (guess + x / guess) / 2.0
        if abs(guess * guess - x) < tol:
            break
    assert abs(guess * guess - x) < tol * 10, "Postcondition failed: did not converge"
    return guess
```

---

## 5. Distributed Systems Fundamentals

Relevant when working with multi-GPU training or large-scale data pipelines:

### CAP Theorem
You can have at most 2 of: **C**onsistency + **A**vailability + **P**artition tolerance.
In distributed training: parameter servers sacrifice strict consistency for performance (eventual consistency).

### Distributed ML Training Patterns
```
AllReduce (e.g., NCCL, Horovod):
  - Each worker computes gradients on its shard
  - AllReduce aggregates gradients across all workers (sum/avg)
  - All workers update identically → strong consistency
  - PyTorch: torch.distributed.all_reduce()

Parameter Server:
  - Workers push gradients to PS, PS updates params, workers pull
  - Allows asynchronous updates → higher throughput, stale gradients
  - Use when AllReduce communication is the bottleneck
```

### Idempotency for Fault Tolerance
```python
# Non-idempotent: calling twice gives wrong result
count = 0
def increment():
    global count; count += 1  # NOT idempotent

# Idempotent: safe to retry on failure
def upsert_record(db, record_id, data):
    """INSERT OR REPLACE — calling twice is safe."""
    db.execute("INSERT OR REPLACE INTO records VALUES (?, ?)", (record_id, data))
```

**Checkpointing for long experiments**:
```python
import torch, os

def save_checkpoint(model, optimizer, epoch, loss, path):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }, path)

def load_checkpoint(path, model, optimizer):
    if os.path.exists(path):
        ckpt = torch.load(path)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        return ckpt["epoch"], ckpt["loss"]
    return 0, None  # start from scratch

# Save every N epochs to tolerate failures
for epoch in range(start_epoch, total_epochs):
    train_one_epoch(model, optimizer)
    if epoch % 10 == 0:
        save_checkpoint(model, optimizer, epoch, loss, f"ckpt_{epoch}.pt")
```
