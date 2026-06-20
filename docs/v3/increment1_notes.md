# v3 Increment 1 — Solver + VFI oracle (M0-M5)

Status notes for the first build increment of the DF26 TF/TFP port. Scope was fixed
with the user: economic primitives, the VFI benchmark oracle, the FiLM Block-1
networks, and Block-2 grid refinement validated against VFI. Synthetic-only (no
Compustat), serial execution, two-tier precision. Full spec: `DF26_implementation_spec.md`.

## What was built (`src/v3/`)

| Area | Modules | Spec |
|---|---|---|
| Config | `config.py` (ModelParams, ExternalParams, ParamBounds=Table A1, Grid/Network/Train/Run configs) | Sec 8, 9 |
| Common | `common/{precision,seeding,quadrature,normalization}.py` | Sec 3.2, 3.6, 10, 11 |
| Economics | `economics/{production,adjustment,dividends,debt,bounds,tauchen}.py` | Sec 1 |
| Networks | `networks/{film,value,policy,bundle}.py` (FiLM, identity init, target net) | Sec 3.1, 3.3 |
| Block 1 | `solver/{bellman,default_prob,train_step,trainer,sampling}.py` | Sec 3.4, 3.5, 3.7 |
| Grids/interp | `solver/{grid,interp}.py` | Sec 3.2, 4.1, 4.2 |
| Block 2 refine | `solver/{refine_improve,refine_evaluate,refine}.py` | Sec 4.1 |
| VFI oracle | `validation/vfi.py` | Sec 12.1 |
| Validation | `validation/{evaluate,policy_slices,figures,properties}.py` | Sec 12.3, 12.4 |
| Output | `output/{artifacts,checkpoints}.py` | Sec 12.5 |
| Tests | `src/v3/tests/test_*.py` | Sec 12 |

Demo notebook: `docs/v3/01_block1_vs_vfi_validation.ipynb` (thin: configures a run,
calls the library, renders Fig V3 + deviation report + property checks; logic and
gating live in `src/v3`).

## Test status

65 tests pass (`pytest src/v3/tests`, ~2 min; 49 fast + 16 slow oracle/refinement gates). Includes
Increment 2 gates: dense solve reproduces the VFI fixed point, Howard PFI matches plain VI, and
`bilinear_corners` matches `interp_grid`.

## How to run

- Tests: `pytest src/v3/tests -m "not slow"` (fast) or drop the flag for the full
  oracle/refinement gates.
- Demo: open `docs/v3/01_block1_vs_vfi_validation.ipynb` (kernel `dl_corp_finance`).
  `PROFILE="MEDIUM"` (default, ~3-4 min) demonstrates the Sec 12.3 value gate (Howard VFI + dense
  refine); `"SMOKE"` (~1-2 min) is the quick path; `"BASE"` is the full grid (tens of min).

## Key findings / decisions

- **`tf.cast(python_float, float64)` silently routes through float32** (~2e-8 error).
  Fixed with `precision.as_float`, used wherever scalar inputs are converted. (ENV-2.)
- **Apple Metal has no reliable float64**; TF places some batched float64 ops on
  Metal and loses precision even under a CPU device context. The float64 numeric
  tier (Tauchen, VFI, refinement, interp) runs **CPU-only** via
  `precision.use_cpu_only()` (called in `conftest.py` and the notebook). (HW-1.)
- **Legacy Keras is forced** (`TF_USE_LEGACY_KERAS=1`, set in `src/v3/__init__.py`
  and `common/precision.py`) so TFP works with Keras 3 installed. (ENV-1.)
- **Higher-order autodiff**: the default-prob Taylor uses nested tapes; reductions
  must stay inside the recording tape (a `tf.reduce_sum` placed outside silently
  yields `None` gradients). (TF-3.)
- **VFI benchmark** (Sec 12.1) is mechanically **Howard PFI**: global policy improvement (full
  control-grid argmax) + exact policy evaluation via the dense linear solve, stopping when the policy
  is stable. `mode="howard"` (spec-faithful, ~8 sweeps) is the benchmark; `mode="value_iteration"`
  (~765 sweeps) is kept as a small-grid cross-check. **Refinement** (Sec 4.1) is the same but with
  **local** 9^3 improvement. Both reach the same fixed point: tests prove `policy_evaluate(VFI policy)`
  reproduces VFI's V to 1e-9 (both VI and dense) and the VFI policy is a fixed point of the local
  improvement. See "Policy-evaluation linear solve" below.
- **Accuracy**: at the MEDIUM grid (7x10x15 / 25x25x15) the refined **value** is within ~0.2% of VFI
  (inside the Sec 12.3 1% gate); the raw network is ~27% off, so the refinement does the work. Refined
  **policy** max relative deviation is bounded below by the control-grid spacing (reported, not gated);
  mean deviations are tiny. The strict 1% *policy* gate needs the full BASE control grid.

## Policy-evaluation linear solve (Increment 2, DF26 Sec 4.1 step 2b / Sec 11)

Implemented in `solver/refine_evaluate.py` (`policy_evaluate(method="dense")`), used by both the
refinement (Sec 4.1) and the Howard-PFI VFI benchmark (Sec 12.1). The spec mandates a semismooth-Newton
linear solve; dense `tf.linalg.solve` is the Sec-11-sanctioned realization (GMRES behind a flag, not
needed). Concrete construction, under a fixed policy per state s (controls k'(s), b'(s), c'(s); next
net debt bpp(s)=b'(s)-c'(s); beta=1/(1+r_f); Tauchen P):

- Off-grid next value (bilinear over log k, b): Vtilde(s,z') = sum_{c in 4 corners} w_c(s) V[z',
  kb_c(s)]; corners/weights from `solver/interp.bilinear_corners` (verified to match `interp_grid`
  to 1e-15), computed once from the fixed policy.
- Active set a(s,z') = 1{Vtilde(s,z') > 0}. With a fixed, max{Vtilde,0} = a*Vtilde is linear in V,
  giving **(I - beta A) V = D**, A[s, flat(z', kb_c(s))] += P[z(s),z'] a(s,z') w_c(s) (<= 4 corners x
  n_z' nonzeros per row), assembled with `tf.scatter_nd`. Solve dense (float64, CPU). Recompute a;
  repeat until the partition stabilizes (safeguard ||dV||_inf/(1+||V||_inf) < 1e-10, cap 50 Newton
  iters). At the VFI fixed point this converges in **1 Newton iteration**.
- The bond price q (hence the exact dividend D) is held fixed during the solve and recomputed between
  rounds (Sec 4.1 step 3) so the system stays linear in V.
- **Unclamped-V convention:** the dense path stores V per Eq 10 (no outer max), so V<0 in the default
  region lets the active set identify default; the *reported* value clamps to max{V,0} (equity to
  shareholders). VI and dense agree in the good region (V>0); cross-checks compare there.
- Result: Howard PFI converges in ~8 sweeps vs ~765 for plain VI (same policy; value agrees to ~5e-7),
  making finer grids feasible.

## Status (M6-M10)

- **Done:** M6 (simulation + 11 moments), M7 (surrogates + native batched Levenberg-Marquardt +
  Erickson-Whited weighting on the simulated panel), M9 (end-to-end recovery pipeline runs; R2 gated
  on collection scale, see Increment 3). Increment 3 (GPU-default batched collection, below).
- **Remaining:** M8 (controller + adaptive shrinkage, Sec 6/7), M10 (regression golden test + async
  engine stub, Sec 12.5/7). Strict 1% *policy* gate (vs *value*) needs the full BASE 81x91x71 grid via
  Howard, now feasible but off by default.

## Increment 3: GPU-default batched collection (Sec 11)

**Why.** The Monte-Carlo recovery (Sec 12.2) is gated on collection scale: the moment surrogate over
the 8-D parameter box needs thousands of (beta, moments) rows, and the serial collector refines +
simulates one draw at a time. Sec 11 sanctions vectorizing "over the batch of parameter vectors
processed together"; this increment builds that batched path as the default and keeps the serial path
as a CPU-run option.

**Device policy** (`common/precision.py`). `configure_devices(mode)` sets GPU *visibility* only; all
numeric code is device-agnostic float64. `"auto"` -> GPU on CUDA, CPU on Apple Metal or no-GPU (Metal
has no reliable float64 kernels, review HW-1). `use_cpu_only()` is now the `"cpu"` alias (back-compat;
conftest still calls it). On a CUDA box the float64 batched solve/einsums run on the GPU; on this M1
they run CPU-only, so the GPU speedup is realized only on CUDA deployment and cannot be benchmarked here.

**Batched modules.** `solver/batched.py` carries a leading `[B, ...]` parameter-batch dim through
grids, network-on-grid, bilinear interp, local policy improvement, and the dense `(I - beta A) V = D`
solve (now `[B, S, S]` via `tf.linalg.solve`). `simulation/batched.py` does the same for the panel
simulation and the 11 moments (the good-obs filter becomes a masked reduction since the kept count
varies per parameter). `estimation/collector.py::collect_dataset_batch` draws the *same* per-row
parameters and sim seeds as the serial collector (so `batch_size=1` reproduces it) but
refines/simulates `batch_size` draws at once. `recovery.run_recovery` takes `collect_batch_size`
(default 16) and uses the batched collector for the bottleneck collection. Only z (Tauchen), k, k'
grids depend on the parameters; the b/b'/c' grids are parameter-free and stay shared.

**Validation** (`tests/test_batched.py`, 7 tests). Batched == single, cross-checked: grids / interp /
moment-formulas bit-identical (<1e-12); batched refine reproduces single refine (value <1e-9 in the
good region, identical policy nodes); simulation reproducible and grouping-invariant. **CPU speedup
~3.3x** at batch_size 6 on the SMOKE grid (10.0s -> 3.0s for 12 draws); larger on a real GPU.

**Latent precision fix.** `solver/grid.py::build_grids` passed Python-float parameters to
`economics.bounds`, where `tf.sqrt(1 - rho*rho)` took the float32 detour (~2e-7) on the k bounds (same
class as the `as_float` ENV-2 fix, but in the bounds, not Tauchen). Casting the four parameters to
float64 before the bounds call fixes it; batched and single grids now agree to ~1e-16 (was ~2e-7), and
the single VFI/refine grids are correspondingly more accurate.

**Boundary-sensitivity finding (documented, benign).** End-to-end the batched and serial collectors
agree EXCEPT for rare draws (~1/12 on SMOKE) where a firm sits within ~1e-13 of the limited-liability
default boundary (V < 0). There the batched `[B,S,S]` LAPACK solve and the single `[S,S]` solve differ
at float precision, the discontinuity flips one firm, and a noisy OLS-slope moment (10/11) moves by
~0.15. This does NOT shrink with panel size (it is the model's V<0 discontinuity, not small-sample OLS
noise) and is reproducible for a fixed `batch_size`. The components are proven bit-identical; only the
end-to-end composition crosses the discontinuity. For the deterministic regression slice (M10), pin a
fixed `batch_size` (or the serial path).

## Increment 4: M8 controller + M10 regression/async + Sec 12.4 + scale/Colab

Closes the spec to "fully and faithfully implemented." A full spec-vs-code audit confirmed M8 and
M10 were the remaining milestones plus three in-scope faithfulness gaps; all are now closed.

**M8 -- adaptive controller (Sec 6), `src/v3/controller/`.**
- `min_loss.py` -- minimum loss functions (Eq 44): 31-point profile per parameter, inner LM over
  beta^{-j} (one parameter pinned per row, the same sigmoid-reparam/surrogate Jacobian path as
  estimation), median + recentered across-fold SD. The same routine serves Eq 45.
- `shrinkage.py` -- the level rule + identification guard (90th-pct loss >= min + 3 SD) + containment
  guard (50-closest-of-500 recent by GMM norm, and all 30x10 LM estimates) + volume-fraction search
  (smallest admissible v in [0.05, 1.0], reject non-shrinks). Renormalize-on-shrink is a scaler swap.
- `identification.py` -- Eq 45 diagnostic: re-profile against simulated-at-beta-hat moments; sharpness
  (3-SD) per parameter + self-recovery of the generating beta.
- `dataset.py` -- the 10k-row ring buffer (Sec 5.3/7). `loop.py` -- the serial Engine/WeightStore
  schedule (train -> collect -> surrogate -> estimate -> shrink), the runnable equivalent of the async
  design. Tests: `test_controller.py` (7, incl. an end-to-end loop) -- identification guard skips flat
  curves, level rule + volume search, containment vetoes an outside estimate, min-loss recovers the
  truth, self-recovery.

**M10 -- regression + async (Sec 12.5, 7).**
- `tests/test_regression.py` -- the 5 deterministic pieces (network forward, q+D, one refinement round,
  fixed-seed sim+moments+default count, one LM solve) compared to committed goldens
  (`tests/regression_fixtures/goldens.npz`, 3 KB) at rtol 1e-5 / atol 1e-7, default count exact, under
  `configure_determinism`. Uses a seed-frozen network (reproducible init == a committed checkpoint, but
  light). First run writes the goldens and skips; commit them to guard drift.
- `controller/async_engine.py` -- the Sec-7 1-trainer/3-collector design as a documented stub plus a
  `WeightStore` (publish/snapshot) primitive; the serial `loop.py` is the runnable form on one device.

**Sec 12.4 economic property tests -- `validation/properties.py` extended, `tests/test_property.py`.**
Group 1 (bounds + Tauchen rows) and Group 2 (now incl. i decreasing in k); Group 3 (`check_mechanics`:
Bellman residual small on the refined solution, q in bounds, accounting identity k'=(1+i-d)k, finite;
`check_weighting_spd`); Group 4 (`check_corner_cases`: finite production constants, nu>0, k_min>0 over
all 256 Table A1 corners); Group 6 (`check_comparative_statics`: value-network V down c_f/delta/gamma1/
gamma0, V up chi, and q non-decreasing in chi). Group 6's subtler network signs (delta, gamma1) are a
converged-network gate exercised at FULL scale; at SMOKE the robust signs (c_f, chi) + q-in-chi are
asserted, the rest reported (the coarse net has not converged on them).

**In-scope faithfulness gaps closed.** (1) Sec 5.3 / Table A2 **10,000-most-recent surrogate cap** now
in `surrogate.train(max_obs=...)` + `ControllerConfig.surrogate_max_obs` + the dataset ring buffer.
(2) **Table 1B targets** and reference SEs stored (`config.REFERENCE_TARGETS`, `REFERENCE_ESTIMATE_SE`).
(3) Stale `refine_evaluate.py` docstring fixed (dense solve is the implemented default, not "future").

**Scale + device, `src/v3/profiles.py` + `src/v3/run.py` + `docs/v3/03_recovery_colab.ipynb`.**
`Profile` (SMOKE / MEDIUM / FULL) is the single knob over every size (grids, training length, panel,
collection, surrogate/LM, controller resolution); FULL is the Table A2 paper scale. `run.py`
(`train_and_recover`, `run_adaptive_controller`) is one entry point for M1 and Colab, parameterized by
profile + `device` (`configure_devices('auto')` -> GPU on CUDA, CPU on Metal). The Colab notebook is a
thin demo producing Figs V1/V2; pinned stack tf 2.16.2 / tfp 0.24.0 / tf-keras 2.16.0. The Fig V1
(moment R^2 >= 0.99) / V2 (param R^2 >= 0.95 for >= 7/8) gates are reached at FULL scale on a CUDA GPU.

## Isolation

`src/v3` never imports `src.v2` or `src._legacy`; enforced by
`src/v3/tests/test_isolation.py` (AST scan).
