# Deep Learning Methods for Corporate Finance

This project solves and estimates dynamic structural corporate finance models with deep learning.

It implements three deep learning solvers from Maliar, Maliar, and Winant (2021): Lifetime Reward Maximization (LRM), Euler Residual Minimization (ERM), and Bellman Residual Minimization (BRM). It also adds a new Short-Horizon Actor-Critic (SHAC) solver based on Xu et al. (2022), and includes Value and Policy Function Iteration as classical benchmarks. For estimation, it provides GMM and SMM, validated by Monte Carlo and applied to the Hennessy and Whited (2007) endogenous default model.

The full methodology, algorithms, and results are in [docs/paper/report.pdf](docs/paper/report.pdf).

## Quick Start

```bash
git clone https://github.com/zhaoxuanwang/deep-learning-corp-finance
cd deep-learning-corp-finance
pip install -r requirements.txt

# Verify installation
pytest -q
```

## Reproducing the Results

Every figure and table in the report is produced by one of the notebooks in `docs/`. Run them in order.

| Notebook | Reproduces | Approx. runtime |
| --- | --- | --- |
| [docs/01_basic_investment_benchmark.ipynb](docs/01_basic_investment_benchmark.ipynb) | Part I, basic investment model. Trains VFI, PFI, LRM, ERM, and SHAC. Reproduces Figures 1 to 5. | 30 min on CPU |
| [docs/02_basic_investment_ablation.ipynb](docs/02_basic_investment_ablation.ipynb) | Architecture ablations for input normalization, output head, and hidden activation. | 20 min |
| [docs/03_risky_debt_vfi_interp.ipynb](docs/03_risky_debt_vfi_interp.ipynb) | Part I, risky debt model. Runs the nested VFI solve. Reproduces Figures 6 to 8. | 5 min |
| [docs/04_gmm_validation.ipynb](docs/04_gmm_validation.ipynb) | Part II, GMM Monte Carlo validation on the basic model. Reproduces Tables 5 to 7. | 10 min |
| [docs/05_smm_validation.ipynb](docs/05_smm_validation.ipynb) | Part II, SMM Monte Carlo validation on the frictionless basic model. Reproduces Tables 8 to 10. | 30 min |
| [docs/06_risky_debt_smm_calibrated.ipynb](docs/06_risky_debt_smm_calibrated.ipynb) | Part II, SMM applied to the Hennessy and Whited (2007) risky debt model. Reproduces Tables 11 and 12. | 40 hr on M1 |

Each notebook writes its outputs to `outputs/notebooks/<notebook-name>/`. Every run is fully reproducible from a single master seed. See `src/v2/data/rng.py` for details.

## Design

The codebase keeps three concerns strictly separate:

1. **Environment is the single authority.** All model primitives (state space, action space, reward, transition, parameters) live in one environment class under `src/v2/environments/`. Nothing else owns them.
2. **Data simulation is separate from the solver.** A `DataGenerator` builds the dataset once from the environment. Solvers consume the dataset and never call the simulator during training.
3. **Solvers are generic.** Every solver (VFI, PFI, LRM, ERM, BRM, SHAC) reads the environment through a common interface and works on any model that conforms to it.

Adding a new model means writing one new environment file. The solvers, data pipeline, and utilities do not change.

## Methods

Solvers for dynamic models are in `src/v2/solvers/` and `src/v2/trainers/`:

- **Value and Policy Function Iteration (VFI / PFI).** Discrete DP benchmark with linear interpolation.
- **Lifetime Reward Maximization (LRM).** BPTT through finite-horizon rollouts with a deterministic-perpetuity terminal correction.
- **Euler Residual Minimization (ERM).** Squared-residual loss with a target policy network.
- **Bellman Residual Minimization (BRM).** Joint policy and value loss. Reproduced for completeness; rejected for production due to convergence to spurious fixed points.
- **Short-Horizon Actor-Critic (SHAC).** Windowed BPTT actor with a one-step DDPG-style critic. New method.
- **Nested VFI.** Outer pricing fixed point combined with an inner Bellman VFI for the endogenous default risky debt model.

Structural estimators are in `src/v2/estimation/`:

- **GMM.** Closed-form Euler-equation moments with HAC standard errors.
- **SMM.** Simulation-based moments with two-step optimal weighting and sandwich standard errors.

## Progress

| Part | Component | Status |
| --- | --- | --- |
| I | Deep learning solvers (LRM, ERM, BRM, SHAC) | Complete |
| I | Discrete DP benchmarks (VFI, PFI, nested VFI) | Complete |
| I | Basic investment model | Complete |
| I | Risky debt model | Complete |
| II | GMM and SMM Monte Carlo validation | Complete |
| II | SMM applied to risky debt model | Complete |

## Requirements

- Python 3.10 or higher
- TensorFlow 2.16 or higher (use `tensorflow-macos` and `tensorflow-metal` on Apple Silicon)
- See [requirements.txt](requirements.txt) for the full list.

## References

DeAngelo, H., 2022. The Capital Structure Puzzle: What Are We Missing? *Journal of Financial and Quantitative Analysis* 57, 413 to 454. <https://doi.org/10.1017/S002210902100079X>

Duarte, V., Duarte, D., Silva, D.H., 2024. Machine Learning for Continuous-Time Finance. *The Review of Financial Studies* 37, 3217 to 3271.

Fernández-Villaverde, J., 2025. Deep Learning for Solving Economic Models. *NBER working paper*.

Hennessy, C.A., Whited, T.M., 2007. How Costly Is External Financing? Evidence from a Structural Estimation. *The Journal of Finance* 62, 1705 to 1745.

Maliar, L., Maliar, S., Winant, P., 2021. Deep learning for solving dynamic economic models. *Journal of Monetary Economics* 122, 76 to 101. <https://doi.org/10.1016/j.jmoneco.2021.07.004>

Nikolov, B., Schmid, L., Steri, R., 2021. The Sources of Financing Constraints. *Journal of Financial Economics* 139, 478 to 501. <https://doi.org/10.1016/j.jfineco.2020.07.018>

Strebulaev, I.A., Whited, T.M., 2012. Dynamic Models and Structural Estimation in Corporate Finance. *Foundations and Trends in Finance*.

Xu, J., Makoviychuk, V., Narang, Y., Ramos, F., Matusik, W., Garg, A., Macklin, M., 2022. Accelerated Policy Learning with Parallel Differentiable Simulation. <https://arxiv.org/abs/2204.07137>
