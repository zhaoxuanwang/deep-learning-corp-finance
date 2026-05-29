# Deep Learning Methods for Corporate Finance

This project solves and estimates dynamic structural corporate finance models with deep learning, classical dynamic programming, linear programming, and finite-difference methods.

It implements the Maliar, Maliar, and Winant (2021) deep learning solvers: Lifetime Reward Maximization (LRM), Euler Residual Minimization (ERM), and Bellman Residual Minimization (BRM). It also adds a Short-Horizon Actor-Critic (SHAC) solver based on Xu et al. (2022), Value and Policy Function Iteration benchmarks, nested VFI for risky debt, Nikolov-Schmid-Steri LP solvers, and a finite-difference CEO-contract solver. For estimation, it provides GMM, SMM, Bayesian EKF/NUTS and RW-MH validation, neural surrogate validation, and empirical-policy indirect inference.

The full methodology, algorithms, and results are in [docs/paper/report.pdf](docs/paper/report.pdf).

## Quick Start

```bash
git clone https://github.com/zhaoxuanwang/deep-learning-corp-finance
cd deep-learning-corp-finance
pip install -r requirements.txt

# Verify installation
python -m pytest -q
```

When using the repository-local virtual environment, prefer:

```bash
.venv/bin/python -m pytest -q
```

## Reproducing the Results

Every figure and table in the report is produced by one of the notebooks in `docs/`. Run them in order. Some later notebooks support profile flags for smoke, baseline, and full runs.

| Notebook | Reproduces | Approx. runtime |
| --- | --- | --- |
| [docs/01_basic_investment_benchmark.ipynb](docs/01_basic_investment_benchmark.ipynb) | Part I, basic investment model. Trains VFI, PFI, LRM, ERM, and SHAC. Reproduces Figures 1 to 5. | 30 min on CPU |
| [docs/02_basic_investment_ablation.ipynb](docs/02_basic_investment_ablation.ipynb) | Original Maliar21 variants and ablation benchmarks. | 20 min |
| [docs/03_risky_debt_vfi_interp.ipynb](docs/03_risky_debt_vfi_interp.ipynb) | Part I, risky debt model. Runs the nested VFI solve. Reproduces Figures 6 to 8. | 5 min |
| [docs/04_gmm_validation.ipynb](docs/04_gmm_validation.ipynb) | Part II, GMM Monte Carlo validation on the basic model. Reproduces Tables 5 to 7. | 10 min |
| [docs/05_smm_validation.ipynb](docs/05_smm_validation.ipynb) | Part II, SMM Monte Carlo validation on the frictionless basic model. Reproduces Tables 8 to 10. | 30 min |
| [docs/06_risky_debt_smm_calibrated.ipynb](docs/06_risky_debt_smm_calibrated.ipynb) | Part II, SMM applied to the Hennessy and Whited (2007) risky debt model. Reproduces Tables 11 and 12. | 40 hr on M1 |
| [docs/07_bayesian_validation.ipynb](docs/07_bayesian_validation.ipynb) | Bayesian basic-investment validation with the closed-form policy, EKF likelihood, NUTS sampling, posterior diagnostics, and coverage checks. | Profile-dependent; SMOKE about 3 min |
| [docs/08a_pretrain_nn_surrogate.ipynb](docs/08a_pretrain_nn_surrogate.ipynb) | Neural surrogate pretraining for the parameterized basic-investment policy. | Cached; retrain about 60 to 90 min CPU |
| [docs/08b_rwmh_three_way_baseline.ipynb](docs/08b_rwmh_three_way_baseline.ipynb) | RW-MH comparison across neural surrogate, closed-form, and diagnostic Bayesian paths. | Profile-dependent |
| [docs/08c_nuts_closedform_validation.ipynb](docs/08c_nuts_closedform_validation.ipynb) | Closed-form NUTS posterior diagnostics, posterior predictive checks, coverage, and sensitivity analysis. | About 7 min CPU_SMOKE; about 27 min CPU_LARGE |
| [docs/09_nikolov_models.ipynb](docs/09_nikolov_models.ipynb) | Nikolov-Schmid-Steri TO, LE, and MH linear-programming model solves. | Profile-dependent |
| [docs/10_nikolov_to_policy_pipeline.ipynb](docs/10_nikolov_to_policy_pipeline.ipynb) | Empirical policy construction and indirect inference for the Nikolov TO model on Hong Kong data. | Profile-dependent |
| [docs/11_nikolov_compustat_cleaning.ipynb](docs/11_nikolov_compustat_cleaning.ipynb) | Compustat cleaning pipeline for Hong Kong listed firms. | Data-dependent |
| [docs/12_nikolov_le_policy_pipeline.ipynb](docs/12_nikolov_le_policy_pipeline.ipynb) | Nikolov LE simulation and policy diagnostics. | Profile-dependent |
| [docs/13_nikolov_mh_policy_pipeline.ipynb](docs/13_nikolov_mh_policy_pipeline.ipynb) | Nikolov MH simulation and policy diagnostics. | Profile-dependent |
| [docs/14_ceo_contract_pipeline.ipynb](docs/14_ceo_contract_pipeline.ipynb) | Marinovic-Varas CEO contract finite-difference solve, SDE simulation, and value reconciliation. | About 10 min |

Each notebook writes its outputs to `outputs/notebooks/<notebook-name>/`. Every run is fully reproducible from a single master seed. See `src/v2/data/rng.py` and `src/v2/utils/seeding.py` for details.

## Design

The codebase keeps three concerns strictly separate:

1. **Environment is the single authority.** Model primitives live under `src/v2/environments/`. Solvers and estimators read the environment contract instead of duplicating economics.
2. **Data simulation is separate from the solver.** Simulation modules generate panels and trajectories; solvers consume model primitives and training data without owning the empirical workflow.
3. **Solvers and estimators are reusable.** VFI, PFI, LRM, ERM, BRM, SHAC, Nikolov LP routines, Bayesian runners, and indirect-inference helpers are organized as reusable components.

Adding a new model means writing the environment or model primitive first, then connecting it to an existing solver or estimator when the contract matches.

## Methods

Solvers for dynamic models are in `src/v2/solvers/` and `src/v2/trainers/`:

- **Value and Policy Function Iteration (VFI / PFI).** Discrete DP benchmark with linear interpolation.
- **Lifetime Reward Maximization (LRM).** BPTT through finite-horizon rollouts with a deterministic-perpetuity terminal correction.
- **Euler Residual Minimization (ERM).** Squared-residual loss with a target policy network.
- **Bellman Residual Minimization (BRM).** Joint policy and value loss. Reproduced for completeness; rejected for production due to convergence to spurious fixed points.
- **Short-Horizon Actor-Critic (SHAC).** Windowed BPTT actor with a one-step DDPG-style critic.
- **Nested VFI.** Outer pricing fixed point combined with an inner Bellman VFI for the endogenous-default risky-debt model.
- **Nikolov LP solvers.** Total-obligation, limited-enforcement, and moral-hazard financing-constraint models solved by sparse linear programming.
- **CEO contract finite-difference solver.** Implicit upwind finite-difference policy iteration for the Marinovic-Varas continuous-time short-termism model.

Structural estimators are in `src/v2/estimation/`:

- **GMM.** Closed-form Euler-equation moments with HAC standard errors.
- **SMM.** Simulation-based moments with two-step optimal weighting and sandwich standard errors.
- **Bayesian EKF/NUTS and RW-MH.** Basic-investment likelihood validation, posterior diagnostics, coverage checks, posterior predictive checks, and prior sensitivity.
- **Neural surrogate validation.** Parameterized basic-investment policy surrogate for amortized Bayesian likelihood evaluation.
- **Empirical-policy indirect inference.** Nikolov empirical policy construction, simulation, and distance-based estimation using cleaned Compustat/Hong Kong data.

## Progress

| Part | Component | Status |
| --- | --- | --- |
| I | Deep learning solvers (LRM, ERM, BRM, SHAC) | Complete |
| I | Discrete DP benchmarks (VFI, PFI, nested VFI) | Complete |
| I | Basic investment and risky debt models | Complete |
| II | GMM and SMM Monte Carlo validation | Complete |
| II | SMM applied to risky debt model | Complete |
| II | Bayesian basic-investment EKF/NUTS and RW-MH validation | Complete |
| II | Neural surrogate pretraining and validation | Complete |
| III | Nikolov TO, LE, and MH LP model pipelines | Complete |
| III | Compustat cleaning and Nikolov TO indirect inference | Complete |
| Bonus | CEO contract finite-difference solver and simulator | Complete |

## Requirements

- Python 3.10 or higher
- TensorFlow 2.16 or higher (use `tensorflow-macos` and `tensorflow-metal` on Apple Silicon)
- TensorFlow Probability
- See [requirements.txt](requirements.txt) for the full list.

## References

Chen, H., Didisheim, A., Scheidegger, S., 2026. Deep Surrogates for Finance: With an Application to Option Pricing. *Journal of Financial Economics* 177, 104222.

Cronqvist, H., Ladika, T., Pazaj, E., Sautner, Z., 2024. Limited Attention to Detail in Financial Markets: Evidence from Reduced-Form and Structural Estimation. *Journal of Financial Economics* 154, 103811. <https://doi.org/10.1016/j.jfineco.2024.103811>

DeAngelo, H., 2022. The Capital Structure Puzzle: What Are We Missing? *Journal of Financial and Quantitative Analysis* 57, 413 to 454. <https://doi.org/10.1017/S002210902100079X>

Duarte, V., Duarte, D., Silva, D.H., 2024. Machine Learning for Continuous-Time Finance. *The Review of Financial Studies* 37, 3217 to 3271.

Fernández-Villaverde, J., 2025. Deep Learning for Solving Economic Models. *NBER working paper*.

Hennessy, C.A., Whited, T.M., 2007. How Costly Is External Financing? Evidence from a Structural Estimation. *The Journal of Finance* 62, 1705 to 1745.

Kase, H., Melosi, L., Rottner, M., 2022. Estimating Nonlinear Heterogeneous Agents Models with Neural Networks. CEPR.

Maliar, L., Maliar, S., Winant, P., 2021. Deep learning for solving dynamic economic models. *Journal of Monetary Economics* 122, 76 to 101. <https://doi.org/10.1016/j.jmoneco.2021.07.004>

Marinovic, I., Varas, F., 2019. CEO Horizon, Optimal Pay Duration, and the Escalation of Short-Termism. *The Journal of Finance* 74, 2011 to 2053.

Nikolov, B., Schmid, L., Steri, R., 2021. The Sources of Financing Constraints. *Journal of Financial Economics* 139, 478 to 501. <https://doi.org/10.1016/j.jfineco.2020.07.018>

Strebulaev, I.A., Whited, T.M., 2012. Dynamic Models and Structural Estimation in Corporate Finance. *Foundations and Trends in Finance*.

Xu, J., Makoviychuk, V., Narang, Y., Ramos, F., Matusik, W., Garg, A., Macklin, M., 2022. Accelerated Policy Learning with Parallel Differentiable Simulation. <https://arxiv.org/abs/2204.07137>
