# Deep Learning for Structural Corporate Finance

Neural network methods for solving dynamic structural models in corporate finance. Implements Lifetime Reward (LR), Euler Residual (ER), and Bellman Residual (BR) approaches with benchmarking against discrete dynamic programming.

**Status:** Part I complete (deep learning methods). Part II-III in progress.

## Quick Start

```bash
# Install
git clone https://github.com/zhaoxuanwang/deep-learning-corp-finance
cd deep-learning-corp-finance
pip install -r requirements.txt

# Verify installation
pytest -q

# Run training (Part 1)
jupyter notebook report/01_part1_training.ipynb

# Generate figures from checkpoints
jupyter notebook report/02_part1_results.ipynb
```

## Documentation

| Resource                                            | Description                               |
| --------------------------------------------------- | ----------------------------------------- |
| [Technical Report (PDF)](report/report_brief.pdf)   | Full methodology, algorithms, and results |
| [Training Notebook](report/01_part1_training.ipynb) | Train all models and save checkpoints     |
| [Results Notebook](report/02_part1_results.ipynb)   | Load checkpoints and generate figures     |

## Methods

**Neural Network Approaches:**
- **Lifetime Reward (LR)** - Maximize expected discounted rewards via policy gradient
- **Euler Residual (ER)** - Minimize Euler equation violations
- **Bellman Residual (BR)** - Actor-critic minimizing Bellman equation errors

**Benchmark Solvers:**
- Value Function Iteration (VFI)
- Policy Function Iteration (PFI)

## Project Structure

```
├── report/                      # Documentation and notebooks
│   ├── 01_part1_training.ipynb  # Training pipeline
│   ├── 02_part1_results.ipynb   # Visualization pipeline
│   └── report_brief.pdf         # Technical report
├── src/                         # Core library
│   ├── economy/                 # Economic primitives & data generation
│   ├── networks/                # Neural network architectures
│   ├── trainers/                # Training algorithms (LR, ER, BR)
│   ├── utils/                   # Plotting, analysis, checkpointing
│   └── ddp/                     # Discrete DP solvers (VFI, PFI)
├── tests/                       # Test suite
├── results/                     # Training outputs (gitignored)
│   └── latest/                  # Symlink to most recent run
└── docs/                        # Additional documentation
```

## Requirements

- Python ≥3.10
- TensorFlow ≥2.15
- See [requirements.txt](requirements.txt) for full dependencies

## Progress

| Part | Component                          | Status   |
| ---- | ---------------------------------- | -------- |
| I    | Deep Learning Methods (LR, ER, BR) | Complete |
| I    | Discrete DP Benchmarks (VFI, PFI)  | Complete |
| I    | Basic Investment Model             | Complete |
| I    | Risky Debt Model                   | Complete |
| I    | Unit & Integration Tests           | Complete |
| II   | GMM / SMM Estimation               | Planned  |
| III  | Bayesian Estimation                | Planned  |

## References

Cronqvist, H., Ladika, T., Pazaj, E., Sautner, Z., 2024. Limited attention to detail in financial markets: Evidence from reduced-form and structural estimation. Journal of Financial Economics 154, 103811. https://doi.org/10.1016/j.jfineco.2024.103811

DeAngelo, H., 2022. The Capital Structure Puzzle: What Are We Missing? J. Financ. Quant. Anal. 57, 413–454. https://doi.org/10.1017/S002210902100079X

Maliar, L., Maliar, S., Winant, P., 2021. Deep learning for solving dynamic economic models. Journal of Monetary Economics 122, 76–101. https://doi.org/10.1016/j.jmoneco.2021.07.004

Nikolov, B., Schmid, L., Steri, R., 2021. The Sources of Financing Constraints. Journal of Financial Economics 139, 478–501. https://doi.org/10.1016/j.jfineco.2020.07.018

Strebulaev, I.A., Whited, T.M., 2012. Dynamic Models and Structural Estimation in Corporate Finance. Foundations and Trends in Finance.