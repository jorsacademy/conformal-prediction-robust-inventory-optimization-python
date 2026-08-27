# Conformal Prediction + Robust Inventory Optimization

A decision-making-under-uncertainty project that couples **simultaneous split conformal prediction** with an exact small-horizon **box-robust inventory LP**.

The Industrial Engineering / Operations Research pipeline is:

```text
contextual demand data
        ↓
ExtraTrees multi-period demand predictor
        ↓
proper training / calibration split
        ↓
simultaneous conformal demand trajectory box
        ↓
exact robust multi-period procurement LP
        ↓
held-out inventory cost / fill rate / stockout evaluation
```

The repository deliberately separates three questions:

1. How accurate is the demand predictor?
2. Does the uncertainty band attain the intended statistical coverage?
3. What operational decisions result from optimizing against that band?

A conformal coverage statement is **not** relabeled as a service-level guarantee.

---

## Planning problem

A planner commits to orders `q_t` for a short multi-period horizon before the demand trajectory is observed.

For period `t`:

```text
inventory_t =
    initial_inventory
    + cumulative_orders_t
    - cumulative_demand_t
```

Positive ending inventory incurs holding cost and negative inventory represents backlog/shortage.

The plan trades:

```text
procurement cost
holding cost
shortage cost
```

subject to period-specific order-capacity limits.

The current benchmark is a six-period pre-commitment problem. It does not contain adaptive recourse after demand is revealed.

---

## Synthetic contextual demand model

Each sample is one complete demand trajectory together with planning-time context.

Context contains stylized drivers such as:

- market size;
- local activity;
- promotion intensity;
- trend;
- seasonal phase;
- macro factor;
- channel mix.

The hidden demand generator contains:

- nonlinear contextual mean effects;
- seasonal structure;
- heteroskedastic noise;
- serially correlated disturbances;
- common trajectory shocks;
- occasional positive demand surges.

The predictor never receives the hidden generator.

The synthetic generator is used so that out-of-sample experiments and an information-advantaged distribution reference can be constructed without claiming access to real industrial demand data.

---

## Demand predictor

The point predictor is a multi-output `ExtraTreesRegressor`.

It predicts the full horizon:

```text
x  →  [d_hat_1, ..., d_hat_H]
```

The purpose of the repository is not to claim that ExtraTrees is the best forecasting architecture. It is intentionally a reasonably capable nonlinear predictor so the uncertainty/optimization layer remains the main research object.

---

# Simultaneous split conformal prediction

The data are split into:

```text
proper training
calibration
held-out test
```

The forecasting model is fitted only on the proper-training split.

A period scale `s_t > 0` is also estimated **only from the proper-training residuals**.

For calibration trajectory `i`, the nonconformity score is

```text
score_i =
max_t |d_i,t - d_hat_i,t| / s_t
```

For calibration size `n`, the conformal radius is the order statistic at

```text
ceil((n + 1) * (1 - alpha))
```

using the standard finite-sample split-conformal correction.

The resulting trajectory set is

```text
d_t ∈ [
    max(d_hat_t - q * s_t, 0),
    d_hat_t + q * s_t
]
```

for every horizon period.

Because one maximum score is calibrated for the whole trajectory, the target is **simultaneous trajectory coverage**, not merely periodwise marginal coverage.

---

## What the coverage guarantee means

Under the standard exchangeability assumptions for proper-training/calibration/test trajectory samples, split conformal provides a finite-sample marginal coverage statement for a new trajectory.

It does **not** imply:

- conditional coverage for every context value;
- robustness to arbitrary distribution shift;
- that empirical coverage on every finite held-out test set must be at least the nominal level;
- an inventory fill-rate guarantee;
- a stockout-probability guarantee.

For example, the fixed seed-42 development test obtained `89.1%` empirical trajectory coverage for a `90%` target. That is compatible with the conformal guarantee; empirical coverage on one finite test sample is random.

---

# Comparison uncertainty sets

The experiment compares four planning approaches.

### 1. Deterministic forecast

Optimize against the point prediction only.

```text
d = d_hat
```

No uncertainty protection.

### 2. Classical sigma box

A fixed box based on proper-training residual standard deviations:

```text
d_hat_t ± 2 * sigma_t
```

This is a classical uncertainty heuristic. It has no finite-sample simultaneous conformal guarantee.

### 3. Marginal residual-quantile box

Each period independently receives a split-conformal-style absolute-residual quantile at the requested marginal level.

This can achieve strong pointwise coverage while having much lower **joint trajectory coverage** because all periods must be covered simultaneously.

### 4. Simultaneous conformal box

One trajectory-level max-normalized score is calibrated.

This is the uncertainty set with the declared simultaneous split-conformal coverage target.

---

# Exact box-robust inventory LP

For an uncertainty box:

```text
lower_t <= demand_t <= upper_t
```

the implementation enumerates all `2^H` box vertices.

For every vertex scenario `s` and period `t`:

```text
h[s,t] - b[s,t]
=
initial_inventory
+ cumulative_orders_t
- cumulative_demand[s,t]
```

with:

```text
h[s,t] >= 0
b[s,t] >= 0
```

The epigraph variable `z` satisfies:

```text
z >=
sum_t holding_cost_t * h[s,t]
+
sum_t shortage_cost_t * b[s,t]
```

for every box vertex.

The robust objective is:

```text
min
    procurement_cost(q)
    + z
```

subject to order capacities.

For fixed orders, the piecewise-linear holding/backlog penalty is convex in demand. The maximum of this convex function over the rectangular demand polytope is attained at a box vertex. Therefore the vertex-enumeration LP is exact for the **declared continuous box-robust model**.

This exactness claim is intentionally small-horizon. Vertex enumeration scales as `2^H` and is not presented as a long-horizon robust-optimization architecture.

The LP is solved with SciPy/HiGHS.

---

# Independent robust-LP oracle

A one-period fixture is checked without trusting the LP formulation alone.

For a one-period uncertainty interval, the test:

1. solves the box-robust LP;
2. evaluates a dense independent grid of 20,001 feasible order quantities;
3. computes the true maximum cost over both interval endpoints;
4. requires the LP optimum to match the independent grid optimum to tolerance.

A second test evaluates every box vertex at the returned robust plan and verifies the epigraph's reported worst-case inventory cost exactly.

---

# Inventory operational metrics

For a realized demand trajectory, the evaluator reports:

- total procurement + inventory/backlog cost;
- immediate fill rate;
- stockout-period rate;
- total ordered quantity;
- cost gap versus an information-advantaged clairvoyant realized-demand plan.

Backlogged demand from an earlier period consumes later incoming inventory before new-period demand is counted as immediately filled.

Therefore fill rate is not inferred from ending inventory alone.

---

# Clairvoyant realized-demand lower reference

For every held-out realized trajectory, a deterministic LP is also solved with the *jactual future demand** supplied to the optimizer.

That plan has information the practical policies do not have.

Its realized cost is used only as an information-advantaged reference for reporting a relative cost gap.

It is not a deployable policy.

---

# True-distribution SAA reference

Because the benchmark demand generator is synthetic, an additional research control is possible.

For a small number of held-out contexts:

```text
context
   ↓
true hidden conditional demand generator
   ↓
large scenario sample
   ↓
expected-cost SAA inventory LP
```

The SAA plan is then evaluated on a disjoint Monte Carlo sample from the same hidden conditional generator.

Practical methods do **not** receive this generator.

This reference helps expose the cost of robustness:

```text
distribution-aware expected-cost optimization
vs.
deterministic forecast planning
vs.
robust uncertainty-set planning
```

The SAA reference is information-advantaged and finite-sample. It is not called an exact stochastic optimum or a lower bound.

A regression test confirms that the SAA solution is optimal relative to a fixed alternative on its own supplied scenario sample.

---

# Development benchmark

Fixed seed-42 development configuration:

```text
horizon                       6
proper-training trajectories 700
calibration trajectories     250
test trajectories            700
inventory planning tests      80
ExtraTrees                   200 trees
alpha                         0.10
target trajectory coverage    0.90
```

Prediction:

```text
train RMSE         6.796
calibration RMSE  11.301
test RMSE         10.601
```

Uncertainty bands:

```text
                               trajectory    pointwise     mean width

simultaneous conformal            89.1%         97.5%        49.596
marginal residual-quantile        73.3%         92.4%        37.976
classical 2-sigma                 44.6%          --          27.122
```

The conformal normalized trajectory radius was:

```text
3.660
```

Again, `89.1%` empirical test coverage is not interpreted as a violation of a `90%` split-conformal avg coverage theorem.

---

## Held-out inventory decisions

80 held-out contexts:

```text
method                       mean cost   fill rate   stockout    order qty

Deterministic                  2535.30      0.8846     0.4042      374.26
Classical sigma robust         2430.50      0.9866     0.0750      414.68
Marginal quantile robust       2556.37      0.9953     0.0271      431.81
Conformal robust                2727.92      0.9974     0.0187      451.33
```

Mean cost gap versus the per-realization clairvoyant information-advantaged plan:

```text
Deterministic                  36.96%
Classical sigma robust         36.21%
Marginal quantile robust       43.79%
Conformal robust               53.75%
```

This run illustrates the expected trade-off rather than a universal ranking.

The simultaneous conformal plan produced the highest fill rate and lowest stockout-period rate, but it also ordered more inventory and had the highest mean cost among the four practical policies in this fixture.

There is no claim that conformal robust optimization minimizes expected cost.

---

# Information-advantaged distribution reference

Five held-out contexts were evaluated with:

```text
SAA scenarios per context        192
independent MC evaluation        384
```

Result:

```text
method                       MC mean cost   fill rate   stockout   order qty

Distribution-aware SAA           2556.39      0.9512     0.1536     417.61
Deterministic                    2950.73      0.8598     0.4944     417.76
Classical sigma robust           2688.50      0.9792     0.1008     459.34
Marginal quantile robust         2810.14      0.9891     0.0585     480.35
Conformal robust                2958.53      0.9919     0.0468      498.83
```

The information-advantaged SAA reference achieved the lowest mean cost in this small Monte Carlo fixture.

The conformal robust plan bought substantially more protection and achieved the highest fill / lowest stockout, at a higher expected cost.

That is a decision trade-off, not an implementation failure.

---

# Regression tests

The suite currently checks:

1. deterministic synthetic-data generation;
2. nonnegative demand;
3. hand-checked finite-sample conformal quantile rank;
4. positive proper-training residual scales;
5. conformal band construction;
6. complete box-vertex enumeration;
7. deterministic LP order-capacity feasibility;
8. robust LP vs independent one-period dense-grid oracle;
9. robust epigraph vs explicit all-vertex cost evaluation;
10. SAA optimality against a fixed alternative on the same scenarios;
11. immediate fill-rate / stockout accounting;
12. short end-to-end prediction → calibration → robust planning smoke.

`unittest` groups some checks inside the same test method; the executable suite currently reports **11 tests**.

---

# Run

Install:

```bash
pip install -r requirements.txt
```

Self-test:

```bash
python run_conformal_inventory.py --self-test
```

Tests:

```bash
python -m unittest discover -s tests -v
```

Development experiment:

```bash
python run_conformal_inventory.py \
  --seed 42 \
  --horizon 6 \
  --train-samples 700 \
  --calibration-samples 250 \
  --test-samples 700 \
  --planning-samples 80 \
  --trees 200 \
  --alpha 0.10 \
  --distribution-reference-contexts 5 \
  --saa-scenarios 192 \
  --reference-eval-scenarios 384
```

DdD �