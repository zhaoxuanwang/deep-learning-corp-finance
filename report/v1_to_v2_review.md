# Comprehensive Code Review: v1 → v1.5 → v2

This document provides: (A) a gap analysis of what exists in code but not in the v1 report, (B) an evaluation of the v2 design with resolved decisions, and (C) a concrete migration plan.

---

# Part A: v1.5 Features Not Documented in v1 Report

The current codebase is significantly ahead of what the v1 report describes. Below is every undocumented enhancement.

## A1. Input Normalization System (observation_normalization.py)

**v1 report says:** min-max normalization to [0,1] internally.

**v1.5 code actually supports 5 schemes:**

| Scheme | Formula | Used For (defaults) |
|--------|---------|---------------------|
| `"none"` | identity | — |
| `"minmax"` | (x - min) / span → [0,1] | original v1 approach |
| `"zscore"` | (x - μ) / σ | z (shock), b (debt) |
| `"log"` | log(x) only | — |
| `"log_zscore"` | log(x) → z-score | k (capital, default) |

The defaults in `config.py` are: k → `log_zscore`, z → `zscore` (with internal log via `z_input_space="level"`), b → `zscore`. This is a substantial upgrade from the v1 min-max, fitted from training data statistics rather than bounds.

**Location:** `src/networks/observation_normalization.py` (267 lines), `src/trainers/config.py:67-107`

## A2. Output Head System (output_heads.py)

**v1 report says:** bounded sigmoid for policy, linear for value.

**v1.5 code adds:**
- `"affine_exp"` head: `exp(clip(μ + σ · raw, -20, 20))` — guarantees positive output in log-normal scale, reusing input normalization statistics. This is the **default** for `basic_policy_k` and `risky_policy_k`.
- A validation registry (`ALLOWED_OUTPUT_HEADS`) that restricts which heads can be used for which output channel.

**Location:** `src/networks/output_heads.py` (84 lines)

## A3. Transform Spec System (io_transforms.py)

**v1 report:** not mentioned at all.

**v1.5 code:** A centralized `transform_spec` dictionary captures all normalization parameters, output head configurations, and inference clips in one serializable structure. All `forward_*_levels()` functions use this spec, enabling reproducible inference and checkpoint portability.

Key functions: `build_basic_transform_spec()`, `build_risky_transform_spec()`, `forward_basic_policy_levels()`, `forward_basic_value_levels()`, `forward_risky_policy_levels()`, `forward_risky_value_levels()`, `forward_risky_price_levels()`, and `LevelInferenceHelper` dataclass.

**Location:** `src/trainers/io_transforms.py` (~500 lines)

## A4. Inference Clipping and Diagnostics

**v1 report:** not mentioned.

**v1.5 code:** `InferenceClipConfig` with per-output clip_min/clip_max, dynamic clip tokens ("k_max", "b_max"), `compute_k_clip_diagnostics()` that tracks clipping fraction and pre-clip maximums. `return_preclip` parameter on forward functions.

**Location:** `src/trainers/io_transforms.py:438-449`, `src/trainers/config.py`

## A5. Warm-Start / Curriculum Training (warm_start.py)

**v1 report:** not mentioned.

**v1.5 code:** Full warm-start infrastructure for initializing BR policy networks from a prior LR or ER solution. Includes `extract_policy_model()`, `ensure_policy_compatibility()`, `copy_policy_weights()`, and human-readable source logging.

**Location:** `src/trainers/warm_start.py`

## A6. Huber Loss Functions

**v1 report:** only MSE and AiO cross-product.

**v1.5 code:** `compute_er_loss_huber()`, `compute_br_critic_loss_huber()`, `compute_price_loss_huber()` — robust to outlier residuals with configurable delta.

**Location:** `src/trainers/losses.py`

## A7. Method Registry System

**v1 report:** not mentioned.

**v1.5 code:** `method_names.py` provides canonical name resolution with user-friendly aliases and guards against disabled experimental methods. `method_specs.py` declares dataset key requirements per method. Together they enable safe, extensible method routing.

**Location:** `src/trainers/method_names.py`, `src/trainers/method_specs.py`

## A8. Advanced Configuration Options

**v1 report:** basic hyperparameters only.

**v1.5 code adds 50+ configuration options** including:
- `OptimizerConfig`: gradient clipnorm and clipvalue
- `AnnealingConfig`: exponential or linear temperature schedules
- `EarlyStoppingConfig`: method-specific convergence criteria
- `RiskyDebtConfig`: adaptive_lambda_price, weight_br
- `MethodConfig`: br_normalization modes ("frictionless", "custom", "none"), br_reg_weighting_mode ("fixed", "adaptive")

**Location:** `src/trainers/config.py`

## A9. Offline DDP Path

**v1 report:** mentions DDP exists under `src/ddp` but does not describe integration with NN training.

**v1.5 code:** Full `from_dataset_bundle()` API for metadata-first workflows, data-estimated Markov transition matrices, checkpoint save/load with metadata.json + arrays.npz.

**Location:** `src/ddp/`

## A10. Experimental / Disabled Modules

**v1 report:** not mentioned.

**v1.5 code:** `src/experimental/br_multitask.py` (BR regression with FOC/Envelope regularization — **disabled** due to structural identification failure) and `src/experimental/br_constrained.py` (adaptive Lagrangian — **disabled**). Both excluded from production API with explicit error messages.

**Location:** `src/experimental/`

## A11. Experiments Directory

18 standalone experiment scripts exploring TD3, n-step returns, double-Q, ablation studies, etc. These are research prototypes not integrated into the main pipeline.

**Location:** `experiments/`

---

# Part B: v2 Design Decisions — Resolved

Each subsection records the issue, the agreed solution, rejected alternatives, and any remaining to-dos.

## B1. Input Normalization

**Issue:** v2 proposes stateless symlog for all observations. Concerns raised: (1) symlog does not equalize cross-feature scales, (2) resolution is allocated by natural scale not information content, (3) DreamerV3 analogy is imperfect for small networks.

**Decision:** Offer two normalization options:
- `"symlog"` (default) — stateless, domain-agnostic, follows DreamerV3.
- `"running_zscore"` — exponential moving average of mean/std across batches, the standard in RL (OpenAI Baselines, Stable-Baselines3, CleanRL). Requires state (running μ, σ tensors saved in checkpoint). Stats update during training, freeze at inference.

LayerNorm after the first hidden layer compensates for residual scale differences between features. Concerns (1) and (2) are acknowledged by design and are minor in practice because the network adapts its first-layer weights within a few epochs.

**Rejected alternatives:**
- Per-feature `log_zscore` (v1.5 approach): domain-specific, requires knowing which variables are positive. Violates the domain-agnostic goal.
- Per-batch z-score (compute μ, σ from current mini-batch only): noisy for small batches, not standard practice. Running z-score is strictly better.
- `tanh` compression: saturates for |x| > 3, loses information for large variables.

**To-do:** Update v2 report to remove the imperfect DreamerV3/image-RL analogy and replace with a direct argument about statelessness and LayerNorm compensation.

## B2. Output Head

**Issue:** v2 proposes universal linear output with clip. Concerns raised: (1) zero gradient outside bounds could prevent learning if optimal action is near a boundary, (2) discontinuous gradient at clip boundary may cause SGD oscillation, (3) bond price may suffer near q_max.

**Decision:** Universal linear output + clip with user-specified generous bounds. No domain-specific output heads. The user sets bounds wide enough that the clip almost never activates during training. The zero-gradient region is far from the optimal policy and irrelevant in practice. For MVE actor updates, the DDPG convention applies: pass raw (unclipped) action to Q in the actor loss; Q's gradient handles feasibility implicitly. Clip only appears in reward computation during critic target rollouts. For LR/ER, standard `tf.clip_by_value` is used.

**Rejected alternatives:**
- `bounded_sigmoid` / `softclip`: domain-specific, violates genericity. The vanishing gradient near bounds is a worse problem than the zero gradient outside generous bounds.
- `affine_exp` (v1.5): requires input normalization statistics, couples input and output transforms. Domain-specific.
- Straight-through estimator (`tf.stop_gradient` on clip): provides gradient signal when raw output is outside bounds, but the gradient direction is misleading (it reflects sensitivity at the clipped value, not the raw value). Can push raw further out of bounds. Added complexity with no clear benefit when bounds are generous.

**To-do:** None. Design is final.

## B3. LayerNorm

**Issue:** v1.5 has no hidden-layer normalization. v2 adds LayerNorm after every hidden layer. Concerns: noisy statistics with small (32-neuron) layers; bias removal changes initialization.

**Decision:** Adopt LayerNorm as specified in v2. Use 128 neurons per hidden layer (up from 32) to ensure stable LayerNorm statistics. Dense layers have no bias (absorbed by LN shift parameter). Bias initialization concern is deferred to empirical testing; if problems arise, fix then.

**Rejected alternatives:** None. LayerNorm is well-established and the concerns are minor.

**To-do:** Verify during Phase 1 that LN + 128 neurons matches or exceeds v1.5 accuracy on the basic model.

## B4. Hidden Activation

**Decision:** SiLU (swish) as default. Consistent between v1.5 and v2. No change needed.

## B5. MVE/DDPG Method and Critic Loss

**Issue:** v2 introduces Q(s,a) critic with MVE multi-step targets, replacing V(s) with 1-step targets. Concerns: (1) higher input dimensionality for Q, (2) critic loss formula unspecified, (3) MVE rollout cost, (4) multitask-BR status.

**Decision:**
- Adopt MVE-DDPG as the primary method. Q(s,a) critic with symlog prediction is the default.
- **Critic loss (resolved):** MSE in symlog space. Target computed in level-space, then transformed:
  ```
  Q_target_symlog = symlog(Q_target_level)     # from MVE rollout in levels
  Q_pred_symlog   = critic_network(s, a)       # linear output ≈ symlog(Q)
  L_critic = mean((Q_pred_symlog - stop_gradient(Q_target_symlog))²)
  Q_level = symexp(Q_pred_symlog)              # for actor loss and terminal bootstrap
  ```
  This is scale-invariant and follows DreamerV3. Also addresses the value normalization gap identified in v1.5 (where the value network had no output scaling).
- **Multitask-BR:** Kept as a comparison baseline only, placed in `src/v2/experimental/`. Not a production method.
- Concerns (1) and (3) are acknowledged tradeoffs. Concern (1) is handled by the 128-neuron network. Concern (3) is mitigated by T=1 (pure DDPG) as a fallback if MVE is too slow.

**Rejected alternatives:**
- Raw (un-symlog'd) critic loss: dominated by high-value states, poor scale invariance.
- V(s) critic (v1.5 approach): requires differentiating through reward and dynamics for actor gradient, which fails for non-differentiable rewards. Q(s,a) is strictly more general.

**To-do:** Update v2 report to explicitly specify the critic loss formula above. Discuss MVE computational cost vs 1-step BR.

## B6. Missing Report Sections

**Issue:** v2 report has several unfinished "Section X" references and missing sections.

**Decision:** These are discussion/theory sections that do not block implementation of the basic model. Specifically:
1. "Section X" formal analysis: does not affect implementation.
2. Soft penalty for non-box constraints: skipped for now (no current application needs it).
3. Training algorithms for applications: follows directly from generic framework + model definitions.
4. Results/diagnostics: will be written after implementation.
5. Risky debt training details: deferred entirely. Focus on basic model first; tackle risky debt after v2 basic model is validated.

**To-do:** Write missing sections during/after Phase 3, not before.

## B7. Action Variable Convention

**Issue:** v1 code outputs k' directly. v2 generic framework should use the control variable (investment I) as the action, with capital accumulation built into the transition function.

**Decision:** Policy outputs I (investment), not k'. The transition function encodes k' = (1-δ)k + I. This is consistent with the generic MDP formulation where actions are controls and transitions describe how controls affect states. For the risky debt model (later), actions would be (I, b').

**Clip bounds on I:** Use conservative fixed bounds that are valid for all states:
```python
I_min = -(1 - delta) * k_max    # most negative possible disinvestment
I_max = k_max                    # generous upper bound
```
These are pre-computed by the environment in `action_bounds()` and passed to the network's clip layer. The trainer is agnostic to what these bounds represent. A safety floor `k_next = max((1-δ)k + I, k_min)` is applied inside the transition function as an environment property, not a policy constraint.

**Rejected alternatives:**
- Output k' directly (v1.5 approach): convenient but conflates action with next-period state. Not generic — other models may have actions that are not next-period states.
- State-dependent clip bounds on I (dynamic bounds based on current k): correct but the moving clip boundary complicates learning. Conservative fixed bounds achieve the same result more simply.

**To-do:** None. Design is final.

## B8. Critic Output Head Clarification

**Issue:** v2 architecture summary says "linear identity with clip" universally, but also says critic output is in symlog space. These appear contradictory.

**Decision:** The linear output head is correct for all networks. For the critic, the linear output directly predicts in symlog space (i.e., the training targets are symlog-transformed). `symexp()` is applied externally to recover level-space values for the actor loss and terminal bootstrap. This is not a separate "symlog output head" — it is a linear output where the loss is computed in symlog space.

**To-do:** Clarify this in the v2 report architecture summary.

---

# Part C: Resolved Design Choices from Discussion

## C1. ER Method: Euler Residual Interface

**Decision:** Option A — user supplies `euler_residual(s, a, s_next, a_next) -> Tensor` as an optional method on `MDPEnvironment`. Models with known smooth FOCs override it. Models without (e.g., non-differentiable rewards) leave it as `NotImplementedError`, and the trainer raises a clear error if ER is attempted.

**Rejected alternative:** Automatic FOC derivation via autodiff (Option C). Theoretically elegant but practically fragile: (1) the recursive substitution from ∂V/∂s' to reward partials is model-specific and autodiff doesn't know which identity to apply, (2) non-differentiable components (fixed adjustment cost indicator) produce zero/undefined gradients silently, (3) nested gradient tapes through two policy evaluations are fragile in TensorFlow.

**Naming convention for Euler residual components:** Use descriptive names instead of abbreviations:
- `marginal_cost_of_action` instead of `chi`
- `marginal_benefit_of_action` instead of `m`
- `adjustment_cost_action_deriv` instead of `psi_I`
- `adjustment_cost_state_deriv` instead of `psi_k`
- `production_state_deriv` instead of `pi_k`

## C2. Online vs Offline Training

**Decision:** Default to online training with replay buffer (standard DDPG/TD3). Keep offline mode as an option.

- **Online mode (default):** At each training step, generate transitions using current policy + deterministic seeds from the master seed schedule. Add transitions to a replay buffer. Sample training batches from the buffer. The replay buffer is standard for DDPG/TD3, provides sample efficiency and decorrelation.
- **Offline mode (option):** Pre-generate a fixed dataset, train on it repeatedly. Useful for ablation studies and direct cross-method comparison on identical training data.
- **Comparability across methods:** Achieved through a fixed evaluation dataset (val/test), not through identical training data. Different methods produce different training data in online mode (different policies generate different trajectories), but all are evaluated on the same fixed val/test set.
- **Reproducibility:** Deterministic seeds + deterministic policy → fully deterministic replay buffer contents → identical training run given same seed. No exploration noise needed (see C4).

**Rejected alternative:** Online-only without replay buffer (generate fully fresh data each step, no buffer). Simpler but less sample-efficient and loses the decorrelation benefit. Not standard practice.

## C3. Multitask-BR Location

**Decision:** Place in `src/v2/experimental/`. Even the baseline comparison version should be generic (not model-specific).

## C4. Exploration Noise

**Decision:** Not needed for the basic model. Diversity in training data comes from sampling diverse initial states and shock sequences. For the risky debt model (deferred), exploration near the default boundary is handled by the Gumbel-Sigmoid noise already built into the default probability approximation.

**To-do:** Revisit when implementing the risky debt model.

## C5. Target Network Polyak Rate

**Decision:** Single shared Polyak rate for all target networks (actor and critic). Follows TD3 convention.

## C6. Gradient Clipping

**Decision:** Keep gradient norm clipping as a safety net with max_norm=100 (DreamerV3 default). LayerNorm handles most gradient conditioning; clipping is a last resort.

## C7. Temperature Annealing for Non-Differentiable Rewards (Latent Issue)

**Issue:** The basic model's fixed adjustment cost uses a smooth sigmoid gate controlled by a `temperature` parameter. In v1, temperature is annealed from warm (smooth) to cold (near-hard) during training so gradients flow through the gate early on. In v2, the `MDPEnvironment.reward()` interface accepts `temperature` but no trainer currently passes it — all calls use the default `temperature=1e-6` (near-hard gate).

**Analysis by method:**
- **MVE:** Not affected. MVE never differentiates through `reward()`. Reward enters only inside `stop_gradient` targets (critic rollout). MVE handles non-differentiable rewards by design.
- **ER:** Not affected. ER differentiates through `euler_residual()` (which uses analytical FOC derivatives, not `reward()`). The `reward()` call in `collect_transitions` is outside the gradient tape.
- **LR:** Affected when `cost_fixed > 0`. LR backpropagates through the full T-step rollout including `reward()`. A near-hard gate at `temperature=1e-6` produces zero gradients through the fixed cost indicator, blocking learning of the investment/no-investment boundary.

**Decision:** Defer. The frictionless baseline (`cost_fixed=0`) is unaffected. If MVE outperforms LR and ER (as expected given its design advantages), MVE becomes the production method and LR/ER serve as comparison baselines — in which case this issue is low priority. If LR is needed on fixed-cost scenarios, add a `temperature_schedule` parameter to `LRConfig` and pass it through to `env.reward(s, a, temperature=...)` inside the training loop.

**Risk:** Low. Only affects LR on non-baseline scenarios with fixed costs.

---

# Part D: Migration Plan

## Scope

**v2 initial implementation covers the basic investment model only.** Risky debt model is deferred until the basic model v2 is validated against v1.5 and DDP benchmarks.

Production methods: LR, ER, MVE-DDPG.
Comparison baseline: multitask-BR (in `src/v2/experimental/`).

## Directory Structure

```
src/
├── economy/          # KEEP — shared, model-agnostic
├── ddp/              # KEEP — ground-truth benchmark
├── networks/         # KEEP v1.5 as-is
├── trainers/         # KEEP v1.5 as-is
├── experimental/     # KEEP — v1.5 research prototypes
│
├── v2/
│   ├── __init__.py
│   ├── networks/
│   │   ├── base.py           # symlog → [Dense(no bias) → LN → SiLU]×K → linear+clip
│   │   ├── policy.py         # Generic policy: s → a, with clip
│   │   └── critic.py         # Q(s,a) → symlog-space prediction
│   ├── normalization.py      # symlog/symexp + running_zscore
│   ├── trainers/
│   │   ├── core.py           # Generic training loop, replay buffer, data pipeline
│   │   ├── lr.py             # Lifetime Reward trainer
│   │   ├── er.py             # Euler Residual trainer
│   │   ├── mve.py            # MVE-DDPG trainer (primary method)
│   │   └── config.py         # v2 configuration
│   ├── environments/
│   │   ├── base.py           # Abstract MDPEnvironment interface
│   │   └── basic_investment.py  # Corporate finance basic model
│   ├── experimental/
│   │   └── multitask_br.py   # Generic multitask-BR baseline
│   └── tests/
│       └── ...
```

## Phases

### Phase 0: Preparation (done)
- [x] Commit and tag v1.1 baseline
- [x] Document v1.5 delta (this document, Part A)
- [x] Review v2 design and resolve all ambiguities (this document, Parts B-C)

### Phase 1: Core Infrastructure
1. `normalization.py` — symlog/symexp + running_zscore
2. `networks/base.py` — symlog → [Dense(no bias) → LN → SiLU]×K → linear+clip
3. `networks/policy.py` — generic policy network (s → a)
4. `networks/critic.py` — Q(s,a) critic with symlog-space output
5. `environments/base.py` — abstract MDPEnvironment with state_dim, action_dim, action_bounds, reward, transition, discount, and optional euler_residual

### Phase 2: Training Methods
1. `trainers/core.py` — training loop, replay buffer, online/offline data pipeline, seed schedule, evaluation
2. `trainers/lr.py` — generic LR (trajectory rollout, discounted reward)
3. `trainers/er.py` — generic ER (calls env.euler_residual, AiO cross-product loss)
4. `trainers/mve.py` — MVE-DDPG (Q-critic, multi-step targets, actor via ∂Q/∂a)

### Phase 3: Basic Model Application
1. `environments/basic_investment.py` — wraps src/economy/ functions
2. Run LR, ER, MVE on basic model
3. Compare against v1.5 results and DDP benchmark
4. Acceptance criterion: v2 matches or exceeds v1.5 accuracy

### Phase 4: Validation and Cutover
1. Regression tests against v1.5 test suite
2. Performance benchmarks (wall-clock, convergence speed)
3. Gradual retirement of v1.5 (deprecate, then archive)

## Key Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| symlog + LayerNorm underperforms v1.5 log_zscore | Phase 1 benchmark before proceeding |
| MVE too slow (T-step rollouts) | T=1 (pure DDPG) as fallback |
| Linear+clip boundary issues | Generous bounds; user responsibility |
| 128-neuron networks overfit on small datasets | Monitor val loss; reduce if needed |

## What NOT to Do

1. Do NOT rewrite `src/economy/` or `src/ddp/`.
2. Do NOT delete v1.5 trainers until v2 is fully validated.
3. Do NOT implement risky debt model until basic model v2 is stable.
4. Do NOT add domain-specific output heads (bounded_sigmoid, affine_exp) to v2.
5. Do NOT use autodiff for Euler residuals — use user-supplied functions.
6. Do NOT make v2 checkpoints backward-compatible with v1.

---

# Part E: Implementation Progress and Validation

Status as of v2 basic model baseline validation (BALANCED profile).

## E1. What Has Been Built and Validated

### Core Infrastructure (Phase 1 — Complete)
- `normalization.py` — `RunningZScore` only. Explicit `update()`/`normalize()` API.
- `networks/base.py` — `GenericNetwork`: RunningZScore → [Dense(bias=True) → SiLU] × K. No LayerNorm.
- `networks/policy.py` — Affine-rescaled output: `action = center + sqrt(half_range) * raw`, then clip. Output head uses `Orthogonal(gain=0.01)` for warm start near center.
- `networks/critic.py` — Q(s,a) → level-space output (no symlog).
- `environments/base.py` — Abstract `MDPEnvironment` with `action_bounds()`, `action_scale_reference()`, `action_spec()`, `reward()`, `transition()`, `discount()`, `euler_residual()`.

### Training Methods (Phase 2 — Complete)
- `trainers/lr.py` — BPTT through T-step trajectory. Normalizer updates once with `s0` before GradientTape. Terminal value: `r(s_T, a_T) / (1 - γ)`.
- `trainers/er.py` — Cross-product (AiO) or MSE loss on `euler_residual()`. Target policy with Polyak averaging.
- `trainers/mve.py` — MVE-DDPG with Q(s,a) critic. Multi-step model-based value expansion. Actor gradient via `∂Q/∂a`.

### Basic Model (Phase 3 — Baseline Validated)
- `environments/basic_investment.py` — Self-contained. No `bounds.py` dependency. Explicit multiplier parameters. `_frictionless_kprime()` as single source of truth for k*.
- `_apply_action()` constraint method prevents phantom cash exploit.

### Validation Results (BALANCED profile)

| Method | corr | MSE | Status |
|--------|------|-----|--------|
| **LR** | 1.000 | 23.8 | Excellent — near-perfect fit across full z range |
| **ER** | 1.000 | 4.8 | Excellent — best accuracy of all methods |
| **MVE** | 0.986 | 12465 | Broken — policy learned k' ≈ k (identity) instead of flat optimal |

LR and ER are validated for the basic investment model. MVE needs separate investigation (see E3).

## E2. Current Environment Configuration

```python
BasicInvestmentEnv(
    k_min_mult=0.1,   # k_min = 0.1 × k* ≈ 8.0
    k_max_mult=10.0,  # k_max = 10 × k* ≈ 802
    z_sd_mult=3.0,    # z ∈ [0.53, 1.88] in levels
)
# Derived: k_star ≈ 80.2 (with Jensen's correction)
#          k_ref(z_max) ≈ 349 (frictionless upper envelope)
# Action: I ∈ [-k_max, k_max] = [-802, 802]
# Scale: center = δk* ≈ 12, half_range = k_max ≈ 802
```

## E3. Known Issues Requiring Future Work

1. **MVE divergence.** Q(s,a) single-step DDPG actor update fails. The critic bootstraps on its own errors, causing the actor to learn k' ≈ k (identity). V1.1 experiments showed that V(s) + H-step BPTT through the model works. Consider porting v1's approach to v2.
2. **Right-tail learning speed.** Tighter `k_max_mult` (e.g., 5 instead of 10) would concentrate training samples in the economically relevant region and improve convergence. Heuristic: `k_max_mult ≈ 1.2 × (k_ref / k_star)`.
3. **Temperature annealing for LR with fixed costs.** When `cost_fixed > 0`, the sigmoid gate at `temperature=1e-6` blocks gradients through the fixed cost indicator. Needs `temperature_schedule` in `LRConfig`. Deferred — frictionless baseline unaffected.

---

# Part F: User Preferences and Design Principles

These preferences have been established across multiple iterations. Follow them in all future work.

## F1. Architecture and Code Organization

- **Strict v1/v2 separation.** V2 lives entirely under `src/v2/`. Never import from `src/networks/`, `src/trainers/`, or other v1-specific modules. Shared code (`src/economy/`, `src/ddp/`) is fine.
- **No code duplication.** Single source of truth for each computation. When two functions compute the same thing, keep one and delete the other.
- **No redundant features.** If ablation shows a feature doesn't help, remove it entirely. Don't keep it "just in case." Goal: simple and minimal.
- **Explicit parameters over hidden configs.** Prefer `__init__(k_min_mult=0.1)` over `BoundsConfig(k_min=0.2)` with hardcoded defaults buried in a dataclass.
- **TensorFlow throughout v2.** No numpy in v2 code. Use `tf.exp`, `tf.sqrt`, etc.
- **Readable function names.** English over math notation: `euler_residual()` not `chi()`, `marginal_cost_of_action` not `psi_I`.

## F2. Domain Knowledge Separation (Critical)

- **All domain knowledge lives in `MDPEnvironment` subclasses.** The environment defines bounds, rewards, transitions, Euler residuals, and action scaling references.
- **Trainers, networks, and normalization are fully generic.** They receive bounds, dimensions, and scales as inputs. They never contain economic formulas or model-specific logic.
- **`action_spec()` is the bridge.** Environment bundles all action space info (bounds, center, half_range) into a dict that PolicyNetwork consumes. The network doesn't know what these numbers mean.

## F3. Process Preferences

- **Discuss before implementing.** Major design changes require review and agreement before code is written. Do not jump to implementation.
- **Document rejected ideas.** Record what was tried, what was rejected, and why — so we never revisit dead ends.
- **Back claims with evidence.** Do not present hypotheses as conclusions. When a claim is made about why something fails, either show experimental evidence or explicitly label it as a hypothesis.
- **Understand v1 before proposing v2 changes.** Read v1 code and experiment logs before suggesting new approaches. Many "new ideas" have already been tested.
- **Commit and tag before major changes.** Always preserve a known-good checkpoint before risky refactors.

## F4. Specific Design Preferences (Quick Reference)

| Topic | Preference | Rejected Alternative |
|-------|-----------|---------------------|
| Normalization | RunningZScore only | symlog, LayerNorm, batch z-score, log_zscore |
| Output head | Linear + hard clip | bounded_sigmoid, softclip, affine_exp, straight-through |
| Action variable | I (investment) | k' (next capital) |
| Normalization API | Explicit `update()` + `normalize()` | `__call__(training=True)` |
| Terminal value | `r(s_T, a_T) / (1-γ)` | Domain-specific exact V |
| z sampling | Uniform in levels | Uniform in log-space |
| MVE actor (v1 evidence) | V(s) + H-step BPTT | Q(s,a) + single-step DDPG, TD3 twin critics |
| Framework naming | "generic" | "general" |

---

# Part G: Ablation Results and Key Experimental Findings

These findings are definitive. Do not re-test or re-investigate.

## G1. Normalization Ablation (norm_ablation.py, v2_component_ablation.py)

**Setup:** 5 LR configs + 2 MVE configs, each with symlog vs running_zscore, with/without LayerNorm.

**Results:**

| Config | LR rel_rmse | ER rel_rmse |
|--------|-------------|-------------|
| symlog + LayerNorm (original v2) | 0.857 | 0.804 |
| running_zscore + LayerNorm | 0.245 | 0.217 |
| symlog only (no LN) | 0.493 | 0.318 |
| **running_zscore only (no LN)** | **0.081** | **0.117** |
| v1-gold baseline | 0.272 | — |

**Conclusions:**
1. **LayerNorm is actively harmful** for BPTT-based training. It compounds gradient noise through multi-step rollouts. Responsible for 42% of the v1-v2 performance gap.
2. **symlog is mediocre.** Worse than running z-score across all methods and configurations.
3. **Running z-score alone is optimal.** 3.4x better than v1-gold on LR. No additional normalization layers needed.

## G2. MVE Failure Analysis

**Finding:** Q(s,a) with single-step DDPG actor update diverges on the basic investment model.
- Q-values explode: 0 → 438 in 500 steps.
- All configs produce flat policies (rel_rmse ≈ 0.9).
- TD3 twin critics and delayed actor updates made it worse (v1.1 experiments).
- V1's V(s) + H-step BPTT through model works well.

**Root cause:** Over-estimation of Q compounds through bootstrapping. The basic investment model has a low-dimensional continuous state/action space where function approximation error in Q(s,a) overwhelms the signal. V(s) avoids this by not conditioning on a.

**Implication:** The v2 MVE trainer needs redesign. Port v1's V(s) + H-step BPTT approach, or use MVE only for target computation with V(s) critic.

## G3. Phantom Cash Exploit

**Finding:** When `_apply_action()` clamps `k_next >= k_min`, using raw `I` (instead of effective `I`) in the reward creates an arbitrage. The policy can claim to disinvest more than exists, getting rewarded for the raw negative I while the transition only applies the clamped version.

**Fix:** `_apply_action()` returns both `k_next` (clamped) and `I_effective = k_next - (1-δ)k`. Reward uses `I_effective`. Both `reward()` and `transition()` call `_apply_action()` — single source of truth.

**Lesson:** Any environment with state-dependent constraints must use consistent effective actions in both reward and transition. Never let the reward see a raw action that differs from what the transition actually applies.

## G4. LR Normalizer Contamination

**Finding:** Updating `RunningZScore` inside `GradientTape` during T-step rollout causes divergence at step ~100-150. Seed-independent, data-independent (online and offline show identical failure).

**Root cause:** The normalizer's running mean/std shift between rollout steps, so the same raw state produces different normalized values at step t vs step t+10. This injects non-stationary noise into the gradient computation.

**Fix:** Update normalizer once per training step with initial states `s0`, before entering GradientTape. During rollout, call `normalize()` only (no stat updates).

**Lesson:** Never update adaptive normalization statistics inside a gradient tape that spans multiple time steps.

## G5. Right-Tail Underestimation

**Finding:** Both LR and ER underestimate optimal k' at high z values. Investigated and ruled out four potential causes:
1. **Gradient vanishing from SiLU:** Ruled out — SiLU has non-zero gradient for all inputs, no saturation.
2. **k_max binding:** Ruled out — `clip_by_value` never activates (max raw output << k_max).
3. **Affine scale errors:** Ruled out — raw=9.5 needed for right tail is well within network range.
4. **Uniform sampling bias:** Ruled out — uniform in z-levels actually gives more weight to high z than log-uniform would.

**Root cause:** Insufficient training iterations (FAST_DEBUG profile: 300 LR / 500 ER steps). Euler residuals still decreasing when training stops. Confirmed by BALANCED profile results (MSE dropped from 1454 → 23.8 for LR, 317 → 4.8 for ER).

**Contributing factor:** Wide `k_max_mult=10` wastes training samples on states far from the ergodic distribution. Tighter multipliers would concentrate samples and improve convergence speed.

## G6. Sampling Distribution

**Finding:** Uniform sampling in log(z) space over-represents low-z states and under-represents high-z states (where the right-tail policy targets live).

**Decision:** Changed to uniform sampling in z-levels (raw state space). This is the correct method-agnostic default: the training distribution should cover the state space uniformly, not replicate the ergodic distribution.

---

# Part H: Decision Log — Resolved Technical Questions

Quick reference for questions that have been fully answered. Do not re-investigate.

| Question | Answer | Evidence |
|----------|--------|----------|
| Is LayerNorm beneficial? | No — actively harmful for BPTT | Ablation G1: 42% of v1-v2 gap |
| Is symlog better than running z-score? | No — worse across all methods | Ablation G1: 0.857 vs 0.081 rel_rmse |
| Should we use Q(s,a) or V(s) for MVE? | V(s) + H-step BPTT (v1 approach) | G2: Q(s,a) single-step diverges; v1.1 V(s) works |
| Do TD3 twin critics help? | No — made MVE worse | v1.1 experiments (br_improvements_design.md) |
| Why does the right tail underfit? | Insufficient training steps | G5: BALANCED profile resolves it |
| Is the affine scale causing right-tail error? | No — raw=9.5 is well within range | G5 investigation |
| Should z be sampled in log or level space? | Level space (uniform in raw states) | G6: better coverage at tails |
| Does Jensen's correction matter for k*? | Yes — shifts k* by ~4% (77.2 → 80.2) | Economically correct; propagates to all bounds |
| Should `_apply_action` be shared? | Yes — single source of truth | G3: prevents phantom cash exploit |
| Can normalizer update inside GradientTape? | No — causes divergence | G4: non-stationary noise in gradients |
