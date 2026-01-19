# Deep Learning for Corporate Finance

This repository develops a Python and Tensorflow codebase for solving and estimating structural corporate finance models using deep learning, method of moments, Bayesian estimation methods, and discrete dynamic programming. The project is under active development.

## Goals

- Implementation of deep learning methods for dynamic economic models
  - Lifetime Reward Maximization, Bellman Equation Residual Minimization
  - Complete suite of tests and evaluations
  - Benchmarked against conventional dynamic programming methods
- Quantify the optimal capital structure and leverage level of a firm
- Identify key model parameters
- Extension to other model variants in corporate finance and economics

## Core Modules

- **Model**: core economic logics, shocks, constraints, and simulation
- **Deep learning solver**: train neural networks to approximate policy/value objects with constraint handling and stable optimization
- **Structural estimation**: implement Generalized Method of Moments (GMM) and Simulated Method of Moments (SMM) to identify key model parameters
- **Evaluation**: quantify accuracy, effectiveness, and robustness across different methods
- **Extensions**: Bayesian estimation with TensorFlow Probability and additional model variants

## Requirements
- Python 3.x
- TensorFlow 2.x
- TensorFlow Probability

## Install:
`pip install -r requirements.txt`

## Tests:
`pytest -q`

## Repository structure

```
├── experiments/                 # runnable experiments
├── paper/                       # reports and references
├── src/                         # core library code
│   ├── dnn/                     # deep neural network solvers
│   ├── ddp/                     # discrete dynamic programming solvers
│   └── economy/                 # economic model logic
├── tests/                       # unit and integration tests
├── requirements.txt             # dependencies
└── pytest.ini                   # pytest configuration
```

## References
Cronqvist, H., Ladika, T., Pazaj, E., Sautner, Z., 2024. Limited attention to detail in financial markets: Evidence from reduced-form and structural estimation. Journal of Financial Economics 154, 103811. https://doi.org/10.1016/j.jfineco.2024.103811

DeAngelo, H., 2022. The Capital Structure Puzzle: What Are We Missing? J. Financ. Quant. Anal. 57, 413–454. https://doi.org/10.1017/S002210902100079X

Maliar, L., Maliar, S., Winant, P., 2021. Deep learning for solving dynamic economic models. Journal of Monetary Economics 122, 76–101. https://doi.org/10.1016/j.jmoneco.2021.07.004

Nikolov, B., Schmid, L., Steri, R., 2021. The Sources of Financing Constraints. Journal of Financial Economics 139, 478–501. https://doi.org/10.1016/j.jfineco.2020.07.018

Strebulaev, I.A., Whited, T.M., 2012. Dynamic Models and Structural Estimation in Corporate Finance. Foundations and trends in finance.
