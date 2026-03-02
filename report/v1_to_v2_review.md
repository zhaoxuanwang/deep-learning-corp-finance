# Comprehensive Code Review: v1 → v1.5 → v2

This document provides: (A) a gap analysis of what exists in code but not in the v1 report, (B) an evaluation of the v2 design for issues/inconsistencies, and (C) a concrete migration plan with industry best practices.

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

# Part B: Evaluation of v2 Design — Issues and Inconsistencies

## B1. Normalization: symlog vs. Current log_zscore

**v2 proposes:** Uniform `symlog(x) = sign(x) · ln(1 + |x|)` for all observations, no fitted statistics.

**Current v1.5:** Per-feature log_zscore / zscore with fitted μ, σ from training data.

### Assessment

**Strengths of v2 symlog:**
- Stateless — no fitting step, no running statistics, no warm-up issues
- Handles negative values natively (needed for V, Q, returns)
- Domain-agnostic: works without knowing variable ranges
- Follows DreamerV3 which achieved SOTA across diverse environments

**Concerns:**
1. **Scale mismatch across features persists.** symlog compresses large values but does NOT equalize scales across variables. If k ~ [20, 600] and z ~ [0.5, 2.0], then symlog(k) ~ [3.0, 6.4] and symlog(z) ~ [0.4, 1.1]. The 6x ratio persists. v2 relies on LayerNorm after the first hidden layer to fix this, but the first-layer weights still receive gradients at mismatched scales before LayerNorm acts. In v1.5, log_zscore brings everything to mean=0, std=1 before the network sees it — arguably better conditioning for the first layer.

2. **Loss of information for small variables.** For z ~ [0.5, 2.0], symlog ≈ identity (Taylor expansion). This means z values pass through almost unchanged, while k values get log-compressed. The relative resolution allocated to each feature is determined by the accident of their natural scale, not by the information content.

3. **The DreamerV3 analogy is imperfect.** DreamerV3 normalizes observations in image-based RL where pixel values are naturally bounded [0, 255]. Economic state variables have no such natural bounding. DreamerV3 also uses very large networks (millions of parameters) that can compensate for suboptimal input conditioning. The 2-layer 32-neuron networks typical in this project have much less capacity to adapt.

**Recommendation:** symlog is a sound default for generality, but consider offering per-feature overrides (e.g., `"symlog"` default, `"log_zscore"` available) for users who know their variable scales. The v2 report should explicitly discuss the residual scale-ratio issue and explain why LayerNorm compensates.

## B2. Output Head: Linear + Clip vs. Current bounded_sigmoid / affine_exp

**v2 proposes:** Universal linear output with clip for all variables.

**Current v1.5:** bounded_sigmoid (for policy, price), affine_exp (for capital), linear (for value).

### Assessment

**Strengths of v2 linear+clip:**
- Eliminates vanishing gradient near bounds (well-argued in v2 Section on Output Head)
- Uniform treatment — no per-variable head selection
- Follows TD-MPC2 and PPO conventions

**Concerns:**
1. **Zero gradient outside bounds is NOT "benign" in all cases.** v2 claims clipped samples provide "no gradient" but are harmless because interior samples provide the signal. This assumes the network has enough interior samples to learn the boundary region. In economic models with strong mean-reversion, the ergodic distribution concentrates interior to bounds, so boundary states appear rarely in training data. If the network never sees training signal near k_max, it may produce wildly wrong pre-clip values there, and the clip masks the problem during training. At inference time, the clip enforces feasibility but the underlying function approximation is poor.

2. **Gradient is discontinuous at clip boundaries.** The derivative jumps from 1 to 0 at the boundary. This can cause oscillation in SGD when the optimal action is near a bound — the gradient alternates between full and zero as the raw output crosses the clip threshold. Sigmoid at least provides a smooth transition.

3. **For bond price q ∈ [0, q_max], clip may underperform.** The price network's output should approach q_max smoothly for low-risk firms. With clip, the network might overshoot and get clipped, losing learning signal for exactly the states where precision matters most.

**Recommendation:** The v2 argument about gradient saturation is valid and important. A pragmatic middle ground: use linear+clip as the default but offer `bounded_sigmoid` / `softclip` as an option for users who observe boundary-learning issues. The v2 report should add a note about the "discontinuous gradient at boundary" tradeoff.

## B3. LayerNorm: Missing from v1.5, Central to v2

**v2 proposes:** LayerNorm after every hidden layer. Dense layers have no bias (absorbed by LN).

**Current v1.5:** No hidden-layer normalization. Dense layers have bias.

### Assessment

**Strengths:**
- Prevents internal covariate shift
- Enables scale-invariant learning regardless of input preprocessing quality
- Well-established in transformer/RL architectures

**Concerns:**
1. **Interaction with small networks.** LayerNorm with M=32 neurons computes statistics over 32 values. The mean/variance estimates are noisy at this scale. With 2 hidden layers, adding LN doubles the parameter count of the normalization path (32 scale + 32 shift per layer). For 32-unit networks, this is non-negligible overhead.

2. **Bias removal.** When LN absorbs bias, the preceding Dense layer's expressiveness is unchanged (LN shift parameter replaces it), but it changes the initialization story. Standard Xavier/He initialization assumes bias=0; removing bias and relying on LN shift (initialized to 0) should be equivalent, but this deserves testing.

**Recommendation:** LayerNorm is a good addition. Test with 32-unit and 64-unit networks to verify it helps rather than hurts at this scale.

## B4. SiLU Activation

**v2 proposes:** SiLU as default (same as v1.5).

**Assessment:** Consistent. No issue here. The v2 report's argument about dead ReLU neurons with LayerNorm centering is sound.

## B5. MVE/DDPG Method: Q-function vs. V-function

**v2 proposes:** A new Model-Based Value Expansion (MVE) method using Q(s,a) critic instead of V(s) critic, plus TD-k mixture targets.

**Current v1.5:** BR actor-critic uses V(s) critic with 1-step Bellman targets.

### Assessment

**This is the biggest algorithmic change in v2.**

**Strengths:**
- Q(s,a) eliminates the need to differentiate through r(s,a) and f(s,a,ε) for the actor gradient — critical for non-differentiable rewards
- MVE multi-step targets reduce bootstrap bias
- Target networks on both actor and critic (four networks total)
- Well-established in RL literature (DDPG, TD3)

**Concerns:**
1. **Q(s,a) input dimensionality.** For the basic model, Q takes (k, z, k') = 3 inputs instead of V's (k, z) = 2. For risky model, Q takes (k, b, z, k', b') = 5 inputs. Higher-dimensional input spaces require more data/capacity.

2. **v2 mentions symlog for critic output** (`ŷ_φ ≈ symlog(Q)`, recover via symexp). This is the DreamerV3 value prediction approach. But the v2 report does not formally specify the critic loss. Is it MSE in symlog space? `(ŷ - symlog(Q_target))²`? This needs to be specified.

3. **MVE rollout horizon T.** v2 says "moderate T (e.g., 10-25)". But this requires T forward passes through the target policy and dynamics per sample — potentially expensive. The v2 report should specify the computational cost relative to 1-step BR.

4. **The multitask-BR method from v1 (Section 6 in v2).** v2 keeps multitask-BR as Algorithm 6 and then proposes MVE as Algorithm 7. The v2 report correctly identifies that multitask-BR has "fundamental architecture error" but still presents it as a method. Consider whether it should be clearly marked as a baseline/strawman for comparison only, not a recommended method.

**Recommendation:** MVE is a strong methodological advance. The v2 report needs to:
- Specify the critic loss formula explicitly (MSE in symlog space?)
- Specify whether the critic target `Q̂_{i,t}` is computed in level-space and then symlog'd, or computed directly in symlog space
- Discuss computational cost of MVE vs 1-step BR
- Clarify the status of multitask-BR (baseline only?)

## B6. Sections Missing from v2 Report

The v2 report currently ends at line 841 (risky debt network architecture). Several sections referenced in the text are missing:

1. **"Section X" (referenced 7+ times):** Formal treatment of MVE vs multitask-BR, gradient analysis, constraint handling, AiO bias proof, distribution mismatch
2. **Constraint handling section:** Referenced in "Special case: non-box constraints" — soft penalty approach
3. **Training algorithms for applications:** How LR/ER/BR/MVE are applied to basic and risky models (only network architecture is specified)
4. **Results / diagnostics section**
5. **Risky debt training details:** The nested fixed-point for price/default with MVE

These missing sections are critical for implementation. Without them, the v2 spec is incomplete for the risky debt model especially.

## B7. Inconsistency: Action Variable Convention

**v1 report:** Action is investment I = k' - (1-δ)k. The policy maps (k,z) → I, then k' = (1-δ)k + I.

**v2 report:** The general framework says action a ∈ A, but the application section says policy maps (k,z) → k' directly (network output is k', not I). The clip is on k': `clip(raw, k_min, k_max)`.

**Current v1.5 code:** Network outputs k' directly (not I).

**Assessment:** v2 is consistent with current code but inconsistent with v1 report's convention. This should be explicitly noted in v2: "We parameterize the policy to output k' directly, not I."

## B8. Potential Issue: symlog + Linear Output for Value

**v2 says:** Critic raw output ≈ symlog(Q), recovered via symexp.

But the v2 architecture summary says: "linear identity with clip" as the universal output head.

**Inconsistency:** If the output head is linear (identity), then the raw output IS the network's prediction. But v2 also says the raw output is in symlog space. These are contradictory unless there's an implicit "the target labels are transformed to symlog space, and the linear output predicts in that space, then symexp is applied post-network." This needs clarification.

**Recommendation:** The v2 architecture summary should have an exception for critic networks: the linear output head produces symlog-scale predictions, and symexp is applied externally. Or define a "symlog" output head.

---

# Part C: Migration Plan and Industry Best Practices

## C1. Industry Best Practices for Large Refactors

### The "Strangler Fig" Pattern (Recommended)

The most battle-tested approach for large codebase migrations in industry is the **Strangler Fig Pattern** (coined by Martin Fowler). The idea:

1. **Build new modules alongside old ones** in a separate namespace/directory
2. **Route new code paths through the new modules**, leaving old paths untouched
3. **Gradually migrate** callers from old → new
4. **Retire old modules** only after the new ones are fully tested and validated

This is the standard at companies like Google (gradual migration tooling), Stripe (API versioning), and Netflix (canary deployments). It minimizes risk because the old system remains fully functional throughout.

### Applied to Your Case

```
src/
├── economy/          # KEEP — model-agnostic, no changes needed
├── ddp/              # KEEP — offline DDP, no changes needed
├── networks/         # KEEP v1.5 as-is
├── trainers/         # KEEP v1.5 as-is
├── experimental/     # KEEP — research prototypes
│
├── v2/               # NEW — all v2 modules go here
│   ├── __init__.py
│   ├── networks/
│   │   ├── base.py           # symlog + LayerNorm + SiLU + linear+clip
│   │   ├── policy.py         # Generic policy network
│   │   ├── critic.py         # Q(s,a) critic with symlog output
│   │   └── price.py          # Bond price network (risky model)
│   ├── normalization.py      # symlog implementation
│   ├── trainers/
│   │   ├── core.py           # Generic training loop
│   │   ├── lr.py             # LR method (generic)
│   │   ├── er.py             # ER method (generic)
│   │   ├── br.py             # BR actor-critic (generic)
│   │   ├── mve.py            # MVE-DDPG (new in v2)
│   │   └── config.py         # v2 configuration
│   ├── environments/         # Model-specific MDP definitions
│   │   ├── base.py           # Abstract MDP interface
│   │   ├── basic_investment.py
│   │   └── risky_debt.py
│   └── tests/
│       └── ...
```

**Why this is better than a separate repository or branch:**
- v1.5 and v2 coexist, you can compare results at any time
- Shared economy module avoids duplication
- Tests can run both v1.5 and v2 in the same CI
- No merge conflicts since v2 is in its own directory

## C2. Git Worktree Explanation

### What is a git worktree?

Normally, a git repo has one working directory. `git worktree` lets you have **multiple working directories** (each on a different branch) checked out simultaneously:

```bash
# You're on dev/v1-stable in /Users/wangzhaoxuan/Desktop/JPM-TSRL/DL_corp_finance

# Create a second checkout on a new branch
git worktree add ../DL_corp_finance_v2 -b dev/v2-refactor

# Now you have TWO directories:
# /DL_corp_finance/      → dev/v1-stable  (your current code)
# /DL_corp_finance_v2/   → dev/v2-refactor (fresh branch for v2)
```

### Key Properties
- Both directories share the same .git history (no duplication)
- Changes in one worktree don't affect the other
- You can `cd` between them and run tests independently
- Commits in either worktree are visible from both

### Pros
- Complete isolation: v2 branch can't accidentally break v1
- Full git history available in both
- No directory naming conventions needed (it's just a branch)
- Easy to merge v2 back into main when ready

### Cons
- **You lose the ability to compare v1 and v2 side-by-side in the same Python process.** You can't `from src.trainers import BasicTrainerER` and `from src.v2.trainers import GenericER` in the same test.
- **Shared modules (economy/) diverge.** If you fix a bug in economy/ on v1, you need to cherry-pick it to v2, and vice versa. With the strangler fig approach, there's only one copy.
- **Harder to keep in sync.** Over weeks of parallel development, merge conflicts accumulate.

### Verdict

**For your situation, I recommend the Strangler Fig pattern (new `src/v2/` directory) over git worktree.** Here's why:

1. You want to reuse `src/economy/` and `src/ddp/` unchanged — worktree forces duplication
2. You want to compare v1 vs v2 results on the same data — strangler fig makes this trivial
3. You're a solo developer — worktree's isolation benefits are less important than convenience
4. Your v2 changes are additive (new methods, new architecture) not destructive (rewriting existing modules)

Git worktree is better when: (a) multiple developers need to work on different features simultaneously, (b) the refactor touches the same files as the existing code, or (c) you need a "clean room" environment for certification/audit purposes.

## C3. Concrete Migration Plan

### Phase 0: Preparation (Before Writing Any v2 Code)

**0.1. Lock v1.5 baseline**
```bash
git tag v1.5-baseline   # Tag current state
```
Run full test suite, save results as the regression benchmark.

**0.2. Write v1.5 delta documentation**
Document the 11 undocumented features from Part A above (this document serves as that).

**0.3. Complete v2 report gaps**
Fill in the missing "Section X" material (especially critic loss formula, MVE computational cost, constraint handling). The implementation should not start until the spec is complete.

### Phase 1: Core Infrastructure (src/v2/)

**1.1. symlog + symexp utilities**
- Implement and unit-test symlog/symexp transforms
- Benchmark against v1.5 log_zscore on the basic model (compare gradient magnitudes)

**1.2. Generic network architecture**
- Build the symlog → [Dense → LN → SiLU]×K → linear+clip architecture
- Support configurable input_dim, output_dim, n_layers, n_neurons
- Unit test: forward pass, gradient flow, output range

**1.3. MDP environment interface**
```python
class MDPEnvironment(ABC):
    @abstractmethod
    def state_dim(self) -> int: ...
    @abstractmethod
    def action_dim(self) -> int: ...
    @abstractmethod
    def action_bounds(self) -> tuple[Tensor, Tensor]: ...
    @abstractmethod
    def reward(self, s, a) -> Tensor: ...
    @abstractmethod
    def transition(self, s, a, eps) -> Tensor: ...
    @abstractmethod
    def discount(self) -> float: ...
```
Implement `BasicInvestmentEnv` and `RiskyDebtEnv` as concrete subclasses wrapping the existing `src/economy/` functions.

### Phase 2: Training Methods (src/v2/trainers/)

**2.1. LR trainer (generic)**
- Takes an MDPEnvironment, a policy network, and a config
- Rolls out trajectories using env.transition() and env.reward()
- No model-specific code in the trainer

**2.2. ER trainer (generic)**
- Requires the user to supply an Euler residual function (or derive it from the reward/transition automatically via autodiff for differentiable rewards)
- For non-differentiable rewards: skip ER, use MVE instead

**2.3. BR actor-critic trainer (generic)**
- V(s) critic, 1-step Bellman target
- Actor maximizes r(s,a) + γ·V(s')
- This is the v1.5 approach generalized

**2.4. MVE-DDPG trainer (new)**
- Q(s,a) critic with symlog prediction
- Multi-step MVE targets
- TD-k mixture loss
- Actor maximizes Q(s, π(s))

### Phase 3: Corporate Finance Applications (src/v2/environments/)

**3.1. Basic investment model**
- Implement as MDPEnvironment using src/economy/ functions
- Run LR, ER, BR, MVE and compare against v1.5 results and DDP benchmark
- Acceptance criterion: v2 results match or exceed v1.5 accuracy

**3.2. Risky debt model**
- Implement as MDPEnvironment with nested price network
- Run BR and MVE (ER not applicable due to non-closed-form FOC with default)
- Compare against DDP benchmark

### Phase 4: Validation and Cutover

**4.1. Regression tests**
- Run v2 on all test cases from v1.5 test suite
- Compare accuracy metrics (Euler residual, Bellman residual, policy error vs DDP)

**4.2. Performance benchmarks**
- Wall-clock training time: v2 vs v1.5
- Convergence speed: epochs to target accuracy

**4.3. Gradual retirement of v1.5**
- Once v2 passes all tests: update imports in notebooks/pipeline
- Mark v1.5 trainers as deprecated (not deleted)
- After 1+ month with no regressions: archive v1.5 to `src/_legacy/` or delete

### Phase 5: Report Update

**5.1.** Finalize v2 report with complete Section X material
**5.2.** Update all notebooks to use v2 API
**5.3.** Tag v2.0-release

---

## C4. Key Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| v2 symlog + LayerNorm underperforms v1.5 log_zscore on small networks | Phase 1.1 includes head-to-head benchmark before committing |
| MVE is slower than 1-step BR due to T-step rollouts | Benchmark wall-clock per step; consider T=1 (pure DDPG) as fallback |
| Linear+clip output causes learning issues at bounds | Keep bounded_sigmoid as configurable option |
| Missing v2 report sections lead to ambiguous implementation | Phase 0.3 requires completing the spec first |
| Generic MDP interface is over-engineered for 2 models | Keep interface minimal; add complexity only when a third model demands it |
| v1.5 bugs found during v2 development | Fix in v1.5 first (shared economy/ module); v2 inherits the fix automatically |

---

## C5. What NOT to Do

1. **Do NOT rewrite src/economy/.** The economic logic (parameters, bounds, grids, value_scale) is model-agnostic and well-tested. v2 should import from it directly.

2. **Do NOT rewrite src/ddp/.** The DDP solvers are the ground-truth benchmark. Keep them as-is.

3. **Do NOT delete v1.5 trainers prematurely.** They are your working, validated codebase. Retire only after v2 matches or exceeds their accuracy on all test cases.

4. **Do NOT implement v2 without completing the report spec.** The missing "Section X" material covers critical design decisions (critic loss, constraint handling, distribution mismatch). Implementing without a complete spec will lead to rework.

5. **Do NOT try to make v2 trainers backward-compatible with v1 checkpoints.** Clean break on the checkpoint format. Provide a one-time conversion script if needed.
