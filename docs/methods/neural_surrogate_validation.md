# Neural Surrogate Policy — Validation Track

> Scope: validation only. Wiring into Bayesian Section 3 / "Phase B" is deferred until the surrogate passes the comparative-statics tests in this plan.
> Target model: **frictional basic investment** (Bayesian.md §1.3 model #2). The frictionless case is a validation special case where we have a closed-form reference.

---

## Context

[`docs/paper/Bayesian.md`](../paper/Bayesian.md) §3 (lines 259–316) proposes a **neural surrogate policy** `φ_NN(k, z, β; θ)` — a single network trained once across the prior support of β so that any candidate β inside MCMC becomes a forward pass, not a full model re-solve. This replaces the current per-iteration solver bottleneck.

**The eventual production target is the frictional basic investment model** (Bayesian.md §1.3 model #2), which has no closed-form policy. The frictionless model is used here only as a validation special case — at φ_quad = φ_prop = 0 the analytical policy is known in closed form (Strebulaev & Whited 2012 §3.1), so we can test whether the surrogate recovers it.

**Parameter vector β = (α, ρ, σ_ε, φ_quad, φ_prop) ∈ ℝ⁵.** The surrogate is trained across the full 5-D prior support. Frictions are sampled from their priors (with positive mass at small values) so the φ = 0 slice is well-covered for the validation comparison.

**Adjustment cost**: `ψ(k, I) = 0.5 · φ_quad · I²/k + φ_prop · |I|` (quadratic + proportional, Strebulaev-Whited convention). No fixed cost — the discrete indicator at I = 0 breaks autodiff.

Two facts from codebase exploration shape this plan:

1. **The policy network change is the easy half.** All v2 trainers ([`src/v2/trainers/`](../../src/v2/trainers/)) consume a [`PolicyNetwork`](../../src/v2/networks/policy.py) whose input dim they don't care about. A subclass that accepts `[s_endo, s_exo, β]` is drop-in.
2. **The hard half is the env API.** ER and SHAC compute losses via `env.euler_residual(...)`, `env.reward(...)`, `env.endogenous_transition(...)`. These currently read β off immutable `EconomicParams` / `ShockParams` on the env instance ([`src/v2/environments/basic_investment.py:254-288`](../../src/v2/environments/basic_investment.py)). For a β-conditional surrogate, each minibatch sample must have its own β flowing through the env, not a fixed instance attribute.

A third, sharper observation: **in the Phase A frictionless model the policy does not enter the Kalman likelihood at all** — capital is treated as observed without error and enters the observation equation as a known offset (Bayesian.md §2.3 line 110; confirmed in [`src/v2/estimation/bayesian_basic_investment.py`](../../src/v2/estimation/bayesian_basic_investment.py) lines 138–168). Swapping in a surrogate changes the *simulator* side only. Therefore: validating a frictionless surrogate against the closed-form policy is a sanity check, not a Bayesian-pipeline stress test. The surrogate becomes load-bearing for the **full basic model with frictions** (no closed-form) and any model where capital is latent. The validation notebook proposed here is calibrated to that reality — it validates the surrogate as a *policy approximator over (k, z, β)*, not as a Bayesian inference improvement.

---

## Q1: Is the neural surrogate valid and feasible?

**Valid — yes.** Treating β as additional inputs to the policy network is a standard universal-approximation extension: instead of approximating `φ*(·, ·): K×Z → A` for one fixed β, the network approximates the joint map `φ*(·, ·, ·): K×Z×B → A` over the prior support B. β acts as a context variable; structurally identical to amortized inference networks and meta-learning policy heads. The Maliar–Maliar–Winant (2021) recipe cited in Bayesian.md §3 covers the single-β policy-network case; the (k, z, β) extension is mechanically the same loss with a wider input.

**Feasible in this codebase — yes, modulo the env API change.** The existing PolicyNetwork can be extended trivially; the existing ER and SHAC trainers are structurally compatible; the only real lift is plumbing β as a per-sample tensor through env methods.

## Q2: Build new trainers, or extend existing?

**Decision: new parallel files `src/v2/trainers/er_param.py` and `src/v2/trainers/shac_param.py`.** Existing `er.py` and `shac.py` stay untouched — zero blast radius on production training. The new files share helpers (target-network copy, normalizer fitting, eval callbacks) with the originals via imports from [`src/v2/trainers/core.py`](../../src/v2/trainers/core.py). LR and BRM are out of scope (only ER and SHAC for production).

## Q3: Sequencing

1. Validate the β-conditional surrogate end-to-end in an **independent notebook** mirroring [`docs/01_basic_investment_benchmark.ipynb`](../01_basic_investment_benchmark.ipynb).
2. Only after the surrogate passes the validation gates (below), wire it into Bayesian.md §3 / "Phase B" — i.e., replace the closed-form solver call with a `φ_NN` forward pass inside the MCMC log-target.

---

## Implementation Outline

### Step 1 — New parameterized env class (foundation)

File: `src/v2/environments/parameterized_basic_investment.py` (NEW)

Create `ParameterizedBasicInvestmentEnv` as a sibling of [`BasicInvestmentEnv`](../../src/v2/environments/basic_investment.py), not a modification of it. Existing env stays bit-identical to current behavior — zero risk to production training and the existing Phase A Bayesian pipeline.

Key differences in the new class:
- β is **not** an immutable instance attribute. The constructor still takes nominal `EconomicParams` / `ShockParams` for bounds and calibrated `(r, δ)`, but the structural fields `(α, ρ, σ_ε, φ_quad, φ_prop)` are **expected as per-sample tensors at call time** in a shape-`(batch, 5)` β.
- The nominal anchor used for bounds (k_min, k_max via k*(nominal)) uses the frictionless k* formula — frictions don't enter the bounds calculation, so the existing `EconomicParams`/`ShockParams` are sufficient.
- Adjustment cost is implemented in a self-contained helper inside the new env module: `_friction_cost(k, I, φ_quad, φ_prop) = 0.5·φ_quad·I²/k + φ_prop·|I|`. The existing `EconomicParams` dataclass is **not modified** (user-confirmed).
- `endogenous_transition(s_endo, action, s_exo, beta)`, `exogenous_transition(s_exo, eps, beta)`, `reward(s, a, beta)`, `euler_residual(s, a, s_next, a_next, beta)` all accept the 5-D `beta` tensor.
- `analytical_policy(s, beta)` is only defined when β's friction components are zero (strict check via tolerance). Raises otherwise. For the comparative-statics validation, the notebook explicitly evaluates at φ = 0 β-draws.
- The Euler residual uses the **general tape-based form** (matching [`BasicInvestmentEnv.euler_residual`](../../src/v2/environments/basic_investment.py#L495-L546)) so that friction params enter through autodiff. The frictionless 2-line closed-form residual would be wrong for φ ≠ 0.
- Bound logic (k_min, k_max, z_min, z_max) is shared via composition: instantiate one canonical `BasicInvestmentEnv` internally for bound/grid helpers; override only the β-dependent methods.

Reuse via duplication-where-cheap, composition-where-tested: trajectory generation utilities (`merge_state`, `split_state`, normalization scaffolding) are imported from the existing env; only the β-dependent code is rewritten to take β as a tensor input.

### Step 2 — β-conditional policy network

File: [`src/v2/networks/policy.py`](../../src/v2/networks/policy.py)

Add `ParameterizedPolicyNetwork(PolicyNetwork)`:
- Input dim: `state_dim + 5` where the 5 β-coordinates are `(α, ρ, σ_ε, φ_quad, φ_prop)`.
- Forward: identical hidden stack, just wider input. Reuses existing RunningZScore normalization treating β-components as extra coordinates.
- Add a β-conditional value head if SHAC needs one (it does — V depends on β).

**Why σ_η is excluded from the surrogate's input — and why this is still feasible "train once."** σ_η is observation-noise scale in the Bayesian model; it enters the Kalman filter via the LGSSM's `observation_noise` R-matrix ([`src/v2/estimation/bayesian_basic_investment.py`](../../src/v2/estimation/bayesian_basic_investment.py) lines 158–168). It is **downstream of the policy**: the firm's optimal `k_{t+1}(z; β)` depends only on `(α, ρ, σ_ε, r, δ)` (Bayesian.md eq. line 97). Observation noise is a property of the *econometrician's measurement instrument*, not the firm's optimization. So:
- The surrogate is parameterized over `(k, z, α, ρ, σ_ε)` only and trained once.
- Inside MCMC, σ_η remains a sampled parameter. The log-target evaluates `φ_NN(k, z, α, ρ, σ_ε)` for the policy-relevant pieces and plugs `σ_η` directly into the LGSSM's R-matrix — exactly as in Phase A.
- σ_η is therefore still identified, via its role in the Kalman observation variance, independent of the surrogate. No re-training needed across σ_η.

This separation is the key reason "train once upfront" remains feasible: the surrogate covers only the policy-relevant subset of β, and the Kalman filter handles the remaining parameter (σ_η) analytically.

### Step 3 — β sampler (new sibling, does not touch DataGenerator)

File: `src/v2/estimation/beta_sampler.py` (NEW)

Lives next to the existing Bayesian priors module rather than under [`src/v2/data/`](../../src/v2/data/) — β-from-prior sampling is an inference concern, not a data-pipeline concern. Existing [`DataGenerator`](../../src/v2/data/generator.py) is untouched.

`BetaSampler` samples β **uniformly** over per-coordinate boxes (`DEFAULT_UNIFORM_BOUNDS`). Rationale: BetaSampler is only used by the surrogate trainers and validation notebooks; the Bayesian inference pipeline ([`src/v2/estimation/bayesian_basic_investment.py`](../../src/v2/estimation/bayesian_basic_investment.py)) defines its own `tfd.JointDistributionNamed({"alpha": tfd.Beta(2, 2), …})` independently. The two roles are decoupled, so the surrogate's training distribution can be chosen for *uniform fit quality across the prior support*, not to mimic the inference prior. With Beta(2, 2) the surrogate would be 3–4× less accurate at the validation-sweep tails than at the bulk; uniform avoids that.

`freeze_dims` (e.g., `(3, 4)`) zeros friction dims for nb 08 frictionless validation.

**Reproducibility contract.** β is drawn per minibatch (fresh, not baked into the dataset) but every draw uses a stateless seed derived from `master_seed` via `fold_in_seed(master, trainer_name, "step", step, "beta")`. Same master → bit-identical β sequence. Pinned by `test_per_step_beta_seed_is_deterministic` in [`src/v2/tests/test_er_param.py`](../../src/v2/tests/test_er_param.py).

The new trainers (Step 4 / 5) take a `BetaSampler` instance as an argument; the existing trainers neither import nor see it.

### Step 4 — `train_er_param`

File: `src/v2/trainers/er_param.py` (NEW)

Diff from [`er.py`](../../src/v2/trainers/er.py):
- Each minibatch: sample β from BetaSampler.
- Concatenate β onto the policy input: `a = policy(tf.concat([s, β], -1))`.
- Pass β through to `env.euler_residual(s, a, ..., econ_params=..., shock_params=...)`.
- Loss aggregates Euler residual across (s, z, β) jointly.
- Target policy copy logic from `core.py` (lines 42–93) reused as-is (it tracks `input_dim` from the parent, which now includes β).

### Step 5 — `train_shac_param`

File: `src/v2/trainers/shac_param.py` (NEW)

Diff from [`shac.py`](../../src/v2/trainers/shac.py):
- Each rollout trajectory uses a single β draw held constant for the rollout horizon (a β is *not* time-varying; it's a structural parameter).
- Env transitions and rewards take that β via overrides.
- Both actor and critic networks receive `[s, β]`; the 1-step Bellman target uses the same β at t and t+1.

### Step 6 — Validation notebooks (two, one per regime)

The validation is split into two notebooks so each has a single, transparent purpose:

* [`docs/08_neural_surrogate_validation.ipynb`](../08_neural_surrogate_validation.ipynb) — **frictionless** β-amortization vs closed-form analytical. 3-D effective β via `BetaSampler(mode="uniform", freeze_dims=(3, 4))`. Single-stage training. **`MODE` toggle**: `"SMOKE"` (3×128, ER 4000 / SHAC 800 steps, ~6 min) for fast iteration; `"FULL"` (4×256, ER 8000 / SHAC 1600 steps, ~25 min) for production-quality verification.
* [`docs/09_neural_surrogate_frictional.ipynb`](../09_neural_surrogate_frictional.ipynb) — **full 5-D frictional** baseline for Phase B. Direct single-stage training on the full prior. No analytical reference; validation is held-out Euler residual + ER↔SHAC cross-check + qualitative comparative statics + frictionless-limit info-only.

Both notebooks use:

* **Held-out validation set** of (s_endo, z, β) drawn once at notebook setup, fed to the trainer via `eval_callback` (same pattern as [`docs/01_basic_investment_benchmark.ipynb`](../01_basic_investment_benchmark.ipynb)'s `evaluate_policy_mae`). The callback returns `{"mae_holdout": …}` in nb 08 and `{"euler_residual_holdout": …}` in nb 09 — same plumbing, different metric.
* **Checkpoint history**: `ERConfig(... checkpoint_history=[])` captures policy weights at every eval interval. Post-training, walk the history, pick `argmin(monitor)`, and restore via `restore_selected_snapshot`. Avoids late-training noise — measured 14% / 28% (ER / SHAC SMOKE) and ER FULL diverged to 60× from the best step.
* **Plateau early-stopping** on the same `monitor` metric. `plateau_patience=5` (ER) / 8 (SHAC), `plateau_rel_delta=0.02`. No threshold rule — future-proof for nb 09 where no analytical reference is available to set a target absolute accuracy. The rule fires when the metric stops improving meaningfully, catching both convergence and post-divergence drift.
* **Training-curve diagnostic plot** of `monitor` vs step, marking the chosen best step. Saved to `docs/paper/figures/paramNN-validate/`.
* **Slice plots** with **fixed y-axis = `[0, k_max]`** across all 5 panels (no auto-zoom artifact) and **x-axis = full training-support range** of each swept variable. ρ slice anchors at `z = 1.5·exp(μ)` to avoid the singular point where ρ has no analytical effect. Also saved to the figures dir, with a CSV MAE table.

The earlier 2-stage curriculum design was abandoned: it confused the validation logic (stage 2 retraining overwrites stage 1's frictionless specialization). The clean baselines are easier to interpret and to extend.

Mirror the structure of [`docs/01_basic_investment_benchmark.ipynb`](../01_basic_investment_benchmark.ipynb). Keep both minimal.

| Section | Content |
|---|---|
| 0 | Setup: parameterized env, BetaSampler, anchor point β₀ = prior mean |
| 1 | Train β-conditional ER surrogate (`train_er_param`) |
| 2 | Train β-conditional SHAC surrogate (`train_shac_param`) |
| 3 | **Comparative-statics slices** (the deliverable) |
| 4 | Summary table |

**Section 3 — comparative statics.** One-dimensional slice plots, x-axis = a single parameter, **friction params held at zero** (so the closed-form analytical applies), all other state vars and remaining β components held at anchor values. Same visual idiom as the `k'(z) vs z` and `k'(k) vs k` slices in `01_*.ipynb`, but extended to parameter axes. The closed-form analytical solution (Strebulaev-Whited frictionless) is overlaid as the reference line. This is an **economic sanity check**: at the φ = 0 slice, do the surrogate's comparative statics match the analytical policy?

Concrete plots (each is a single 2-panel figure: ER on left, SHAC on right). All plots fix φ_quad = φ_prop = 0:

1. `k'(α)` vs α, holding `(k, z, ρ, σ_ε) = (k*, E[z], ρ₀, σ_ε₀)` — overlay analytical.
2. `k'(ρ)` vs ρ, holding `(k, z, α, σ_ε) = (k*, E[z], α₀, σ_ε₀)` — overlay analytical.
3. `k'(σ_ε)` vs σ_ε, holding `(k, z, α, ρ) = (k*, E[z], α₀, ρ₀)` — overlay analytical.
4. (For continuity with `01_*.ipynb`:) `k'(z)` vs z at β = β₀ (frictionless slice) — overlay analytical. Confirms the surrogate recovers the single-β behavior at the anchor point.

Each plot reports MAE vs analytical in the legend, mirroring how `01_*.ipynb` annotates correlation and MAE on its slice plots.

**Optional addendum** (Phase A → Phase B bridge, low effort): plot `k'(φ_quad)` vs φ_quad and `k'(φ_prop)` vs φ_prop holding (α, ρ, σ_ε) at anchor. No analytical reference, but the surrogate's curve should be qualitatively sane (k'(I≈0) shrinks as φ_prop rises; inertia region grows as φ_quad rises). Visual-only sanity check.

### Step 7 — Validation gates (must pass before any Bayesian wiring)

1. **MAE of `k'` vs analytical** is small across each 1-D parameter slice (target: same order of magnitude as the single-β ER/SHAC MAE in `01_*.ipynb`).
2. **Visual sanity:** all comparative-statics curves track the analytical reference; no qualitatively wrong shapes (monotonicity, slope sign).

That's it. Tail-β residuals and full Euler-residual diagnostics are deferred to the Phase B plan.

#### Outcome (May 2026)

Two notebooks, each with a single-stage training pass and a transparent baseline.

**Notebook 08 — frictionless validation (gates passed).** 3-D effective β via `BetaSampler(freeze_dims=(3, 4))`. FULL profile: 4 layers × 256 neurons, ER 8000 / SHAC 1600 max steps, plateau early-stopping triggered earlier (ER stopped 3400, SHAC stopped 1300). Total wall time ~9 min on M1 CPU. MAE vs Strebulaev–Whited closed-form:

| slice                | MAE_ER | MAE_SHAC | ER % k range | SHAC % k range |
|----------------------|--------|----------|--------------|----------------|
| α                    | 1.14   | 2.13     | **1.07%**    | 1.99%          |
| ρ (z=1.5 anchor)     | 1.27   | 1.80     | **1.19%**    | 1.68%          |
| σ_ε                  | 1.52   | 1.43     | 1.42%        | **1.34%**      |
| z (anchor β)         | 0.99   | 1.62     | **0.93%**    | 1.52%          |
| k (anchor β)         | 0.91   | 2.13     | **0.86%**    | 1.99%          |

**All slices < 2% of k range. ER wins 4 of 5 slices.** Visual sanity: ER tracks the analytical k-independence in the k slice essentially perfectly (flat at k* ≈ 18); SHAC has mild upward drift at high k. ρ slice anchored at `z = 1.5·exp(μ)` to escape the singular point `z = exp(μ)` where the closed-form is exactly ρ-independent. **β-amortization in the frictionless regime is validated.**

Artifacts: [`docs/paper/figures/paramNN-validate/`](../paper/figures/paramNN-validate/) (`training_curve_full.png`, `slices_full.png`, `slice_mae_full.csv`).

**Notebook 09 — frictional baseline (architectural pass, accuracy not production-ready).** Full 5-D prior. ER 5000 steps, SHAC 1000 steps; same network size; ~11 min total. Diagnostics:

* Held-out Euler-residual (lower is better): ER cross_mean = 0.010, |f1| @p50 = 0.035, @p90 = 0.140. SHAC ~3–5× worse on every metric — expected because ER directly minimizes the Euler residual while SHAC optimizes a Bellman-residual proxy.
* ER ↔ SHAC disagreement on the prior-mean slice is **large** (11–24 units of k). The two methods are *not* converging to the same policy at the frictional anchor.
* ER's policy slices show `k'` **rising with φ_quad and φ_prop** at fixed (k=k\*, z=E[z]) — economically wrong direction (frictions should slow adjustment toward steady state, not push k' further away from k).
* SHAC's policy slices are essentially flat at `k' ≈ 12` across all β-coordinates — collapsed to a near-steady-state policy that ignores the inputs.
* Frictionless-limit MAE vs analytical (info only): ER 4.4%, SHAC 5.6% of k range — much worse than the dedicated frictionless surrogate (nb 08) because the full-5-D training distribution puts essentially zero mass at φ = 0.

**Reading.** The architecture trains, but the unaided 5-D single-stage baseline is not yet production-quality. Both methods have qualitative problems on the frictional axes. Likely contributors to surface in later iterations:

1. **Network capacity / training budget.** 7-D function approximation over wide bounds may require larger networks or longer training. The frictionless 5-D-input case (notebook 08) was clean; the frictional 7-D case is harder.
2. **Reward scale.** SHAC's `reward_scale_override=0.05` was tuned for the frictionless regime; frictional rewards have different magnitude.
3. **ER ↔ SHAC objective mismatch.** ER minimizes Euler residual directly; SHAC's actor-critic loss is an indirect proxy. Disagreement is a signal that one (or both) is on a poor local optimum.

**Conclusion.** Frictionless validation passed; production-grade frictional surrogate needs more work before Phase B Bayesian wiring. Recommended next iteration (separate plan): wider/deeper networks + longer training + per-regime reward-scale calibration; cross-check at fixed β values against PFI/VFI ground truth (instead of an analytical that doesn't exist for frictional).

**Earlier abandoned attempt: two-stage curriculum.** Trained stage 1 on the frictionless slice, then stage 2 on the full 5-D prior with warm-start. Stage 2 inevitably overwrote stage 1's frictionless specialization, contaminating the analytical-comparison signal. Separating into two notebooks removed the confound.

### Step 8 — Deferred: Bayesian Phase B wiring

Out of scope for this plan. Once gates pass, write a separate plan covering:
- Threading the surrogate into `simulate_smm_panel_data` and (for frictional models) into the likelihood path.
- The frictionless-vs-surrogate posterior comparison as a tautological sanity check (same posterior expected within MCMC noise, since policy doesn't enter Phase A likelihood).
- Real test bed: the full basic model with frictions where the surrogate is structurally necessary.

---

## Critical Files to Read or Modify

| File | Role | Change |
|---|---|---|
| `src/v2/environments/parameterized_basic_investment.py` | β-as-tensor env | **NEW** |
| [`src/v2/environments/basic_investment.py`](../../src/v2/environments/basic_investment.py) | existing env | Read only (untouched) |
| [`src/v2/networks/policy.py`](../../src/v2/networks/policy.py) | β-conditional network class | Add subclass only |
| `src/v2/estimation/beta_sampler.py` | prior sampler for β | **NEW** |
| [`src/v2/data/generator.py`](../../src/v2/data/generator.py) | data pipeline | Read only (untouched) |
| `src/v2/trainers/er_param.py` | β-conditional ER | **NEW** |
| `src/v2/trainers/shac_param.py` | β-conditional SHAC | **NEW** |
| [`src/v2/trainers/core.py`](../../src/v2/trainers/core.py) | shared helpers | Read only (reuse via import) |
| `docs/08_neural_surrogate_validation.ipynb` | validation deliverable | **NEW** |
| [`docs/01_basic_investment_benchmark.ipynb`](../01_basic_investment_benchmark.ipynb) | template to mirror | Read only |

Net change to existing production code: **a single subclass added to `policy.py`**. Everything else is new files.

---

## How to Verify End-to-End

1. Unit test: `env.endogenous_transition(k, a, z, beta=…)` returns batched outputs that match per-sample manual computation. Add to `src/v2/tests/`.
2. Unit test: `ParameterizedPolicyNetwork([s, β])` forward + backward through TF autodiff; β-gradient is non-zero where it should be.
3. Run `train_er_param` for 50 epochs on a small `(N=200, horizon=40)` panel with 1000 β samples; verify Euler residual decreases monotonically.
4. Open `docs/08_neural_surrogate_validation.ipynb` end-to-end. Inspect comparative-statics plots against gates 1–2 above.
5. Reproducibility: rerun the notebook with the same `master_seed`; assert bit-identical β samples and final policy parameters.

---

## Notes for the Eventual Phase B Plan

- **σ_η identification (resolved).** σ_η is identified by MCMC via the LGSSM `observation_noise` R-matrix, independent of the surrogate. The MCMC log-target evaluates `φ_NN(k, z, α, ρ, σ_ε)` for the policy-relevant part and plugs σ_η directly into the Kalman filter exactly as in Phase A. No identification loss.
- **σ_ε in policy.** σ_ε enters the closed-form policy via `κ(β)` (Bayesian.md line 97). The surrogate must learn this dependence and so σ_ε is included in the 3-dim β input.
- **Network capacity starting point.** Match `01_*.ipynb`: 2 layers × 32–64 neurons. Scale only if Step 7 gates fail.
- **Frictionless model is a sanity bench.** As noted in Context, the policy does not enter the Phase A Kalman likelihood. So a frictionless-surrogate-MCMC vs frictionless-closed-form-MCMC comparison should give identical posteriors up to MCMC noise. The Phase B plan should treat this as a sanity check, not the real test — the real test is the frictional model where there is no closed form.
