# Common Definitions

**Trainable Parameters**
- Policy net params: $\theta_{\text{policy}}$
- Value net params: $\theta_{\text{value}}$
- Price net params (risky debt only): $\theta_{\text{price}}$

**Notations**
- Current period variable: $x$
- Next period variable: $x'$
- Parameterized function: $\Gamma(\cdot; \theta)$
- DNN trainable weights and biases: $\theta$

**State/Action Variables**
- Exogenous AR(1) shock: $z \gt 0$
- Capital stock: $k\geq0$
- Debt (borrowing): $b\geq 0$

**Shocks**
- $\varepsilon' \sim \mathcal{N}(0,1)$ i.i.d.; AR(1): $\ln z' = \rho \ln z + \sigma \varepsilon'$
- Use two iid draws $(\varepsilon'_{1},\varepsilon'_{2})$ per state for All-in-One (AiO) expectation operator

**Policies and Prices**
- Basic model policy: $(k,z)\mapsto k'$
- Risky-debt model policy: $(k,b,z)\mapsto (k',b')$
- Risky-debt model price: $(k',b',z)\mapsto \tilde r$

**Constraints**
Given pre-computed lower and upper limits for the state variables and for all periods $t$, the training sample is generated from bounded state space below:
- Shock $\log z_t$ is truncated in $m$ standard deviation around stationary mean: 
$$(\log z_{\min},\log z_{\max}) \equiv (\mu - m \sigma_{\log z}, \; \mu + m \sigma_{\log z})$$
- Capital stock $k_t \in (k_{\min}, k_{\max})$
- Risky debt / Borrowing: $b_t \in (0, b_{\max})$ 
- Risky interest rate lower bound: $\tilde r_t \ge r$ where $r>0$ is the risk-free rate
- Risky-debt latent value $\widetilde V(k_t,b_t,z_t)$ can be either positive or negative

**Abbreviations**
For the reminder of the report, abbreviations of the three main methods are commonly used:
- **LR**: Lifetime Reward, e.g., loss  $\widehat{\mathcal{L}}_{\text{LR}}$
- **ER**: Euler Equation Residual, e.g., loss $\widehat{\mathcal{L}}_{\text{ER}}$
- **BR**: Bellman Equation Residual
	- BR-Critic Loss: $\widehat{\mathcal{L}}^{\text{BR}}_{\text{critic}}(\theta_{\text{value}})$
	- BR-Actor Loss: $\widehat{\mathcal{L}}^{\text{BR}}_{\text{actor}}(\theta_{\text{policy}})$

---

# Data Generation

## State Space

**Productivity shocks** 
Given the AR(1) parameters $\mu$, $\sigma$, and $\rho$, the ergodic set of shocks are bounded by:
$$
\log z \in \left[ \mu - m \cdot \sigma_{\log z}, \mu + m \cdot \sigma_{\log z} \right],  
\quad \text{where} \quad 
\sigma_{\log z} = \frac{\sigma}{\sqrt{1-\rho^2}}
$$
 where $m$ is the number of standard deviation around the stationary mean $\mu$. With default value $m=3$ this will covers most of the mass of possible $z$ values.
 
**Capital stock grids** 
The steady state capital stock with no frictional costs is determined by the condition when MPK equals marginal rental cost, $\pi_k \equiv \gamma \cdot z \cdot k^{\gamma-1}  = r + \delta$. Rearrange this equation gives
$$ 
k^*(z) = \left(\frac{z\cdot\gamma}{r+\delta}\right)^{\frac{1}{1-\gamma}}
$$
and I set the "natural" upper and lower limit to be 
$$
k_{\max} = \bar{c} \cdot k^*(z_{\max}) \qquad k_{\min} = \underline{c} \cdot k^*(z_{\min})
$$
with default multiplier $\bar{c}=3$ and $\underline{c}=0.2$ which ensures that the bands are generously wide. This approach recenters the capital grids around the steady state level.

**Debt grids** 
Define the natural borrowing limit to be
$$
b_{\max} = z_{\max}\cdot(k_{\max})^\gamma + k_{\max}
$$
which is the maximum amount that firm can repay debt in the best world $z_{\max}$ and with no liquidation cost (so that firm can sell all capital stock).

##### Implementation
Inputs: 
- Shock parameters: $\mu, \sigma, \rho$ 
- Economic parameters: $\gamma, r, \delta$
- Multipliers: $m, \bar{c}, \underline{c}$
Output (Tuples): 
- Shock (log) state range $[\log z_{\min}, \log z_{\max}] = \left[ \mu - m \cdot \sigma_{\log z}, \, \mu + m \cdot \sigma_{\log z} \right]$
- Capital state range: $[K_{\min}, K_{\max}]$
- Debt state range $[0, B_{\max}]$

## Datasets
Core ideas:
- Use deterministic **RNG seed schedule** to generate datasets and ensure reproducibility
- **Training dataset**: used to train parameters on simulated data
- **Validation dataset**: used repeatedly for model selection / early stopping.
- **Test dataset**: used only once at the end for final reporting

### Training Set
Let $j$ denote the training steps or iterations. The full training dataset is constructed as a stream of batches $\mathcal{B}_n^j$ with $j=1, \dots, J$:
$$
\mathcal{D}_{\text{train}}
= \{ \mathcal{B}^1_n,\dots,\mathcal{B}^J_n \}
$$
where each batch is defined as
$$
\mathcal{B}^j_n = \left\{ \left(k_0, b_0, z_0, ( \varepsilon^{(1)}_1, \varepsilon^{(2)}_1 ), \dots, ( \varepsilon^{(1)}_T, \varepsilon^{(2)}_T) \right) \right\}_{i=1}^n
$$
Note that initial debt $b_0$ is only used in the risky debt model and is ignored in the basic model.

For each tuple $i$ and each period $t$, we take two i.i.d. draws of shocks
$$
\varepsilon_{t,i}^{(1)}\;\overset{i.i.d.}{\sim}\;\mathcal{N}(0,1),\qquad \varepsilon_{t,i}^{(2)}\;\overset{i.i.d.}{\sim}\;\mathcal{N}(0,1),
$$
and draw of a pair initial states from the state space
$$ \begin{align}
k_{0,i} \overset{i.i.d.}{\sim} &\text{Uniform}(k_{\min}, k_{\max}) \\
z_{0,i} \overset{i.i.d.}{\sim} &\text{Uniform}(z_{\min}, z_{\max}) \\
b_{0,i} \overset{i.i.d.}{\sim} &\text{Uniform}(b_{\min}, b_{\max}) \quad \text{(risky debt only)}
\end{align} $$
where the bounds of each state are pre-computed as in previous section.

Key points:
- Inside each batch, every tuple $i$ can be viewed as an i.i.d. firm/agent faced with exogenous initial states and shock rollout over horizon $T$ 
- Shock transition law is AR(1) that maps $(z_t, \varepsilon_t) \to z_{t+1}$
- Given the initial states and the lifetime shock rollout $(k_0, z_0, z_1, \dots, z_{T})$, each firm $i$ aims to find the optimal policies $(k_1, k_2, \dots, k_T)$ that maximize objective
	- In risky debt model, the policies are $(k_t,b_t)_{t=1,\dots,T}$

### Validation Set
The validation set is a fixed dataset simulated from the same DGP:
$$
\mathcal{D}_{\mathrm{val}} = \Big\{(k_{0,i}, b_{0,i}, z_{0,i}, \{\varepsilon_{t,i}^{(1)},\varepsilon_{t,i}^{(2)}\}_{t=1}^T)\Big\}_{i=1}^{N_{\mathrm{val}}}.
$$
where I set the baseline sample size to be 10 times of the training batch size: $N_{val} = 10 n$.

### Test Set
The test set is a fixed dataset simulated from the same DGP:
$$
\mathcal{D}_{\mathrm{test}} = \Big\{(k_{0,i}, b_{0,i}, z_{0,i}, \{\varepsilon_{t,i}^{(1)},\varepsilon_{t,i}^{(2)}\}_{t=1}^T)\Big\}_{i=1}^{N_{\mathrm{test}}}.
$$
where I set the baseline sample size to be 50 times of the training batch size: $N_{test} = 50 n$.

### Main v.s. Fork Path
For each observation $i$, given initial $z_0$ and random draws of lifetime shocks $\{\varepsilon_{t,i}^{(1)}, \varepsilon_{t,i}^{(2)}\}_{t=1}^T$ , I generate the realized main lifetime $z^{(1)}_{t,i}$ path using the first set of $\varepsilon_{t,i}^{(1)}$ over horizon $T$. Then I "fork" the alternative path in each period conditional on the main path, using AR(1) step:
$$
z^{(2)}_{t+1} = (1-\rho)\mu + \rho \log z^{(1)}_{t} + \sigma  \varepsilon_{t,i}^{(2)} 
$$
It is critical to note that $z^{(2)}_{t+1}$ is generated as forks from $z^{(1)}_{t}$ instead of $z^{(2)}_{t}$, so that the rollout of the alternative lifetime shocks $\{z^{(2)}_{t,i}\}^T_{t=1}$ are NOT parallel to the main path $\{z^{(1)}_{t,i}\}^T_{t=1}$.

This nuance is critical because it matches the theoretical transition law in the inter-temporal Euler equation and Bellman equation. This ensures that the ER and BR method can be applied to every period $t$ in the simulated dataset with horizon $T$. However, for the LR method we must use the main path to be consistent with the continuous AR(1) chain.

```
Main (AR1 Chain):  z0 -> z1 -> z2 -> z3 -> ... -> zT
                     \     \     \    ...     \
Fork (1-step AR1):   z1F   z2F   z3F          zTF

Lifetime reward (LR) only use the main path (z0, z1, ..., zT)
Cross-product calculation in ER and BR use (z1 * z1F), ..., (zT * zTF)
```

### Flatten Data for ER and BR

The simulation generates training data by rolling out full trajectories of length $T$ for $N$ i.i.d. firms. While the Lifetime Reward (LR) method requires preserving these trajectories to calculate cumulative lifetime returns, the Euler Residual (ER) and Bellman Residual (BR) methods operate on individual state transitions. To ensure the samples used for Stochastic Gradient Descent (SGD) are independent and identically distributed (i.i.d.), I transform the sequential trajectory data into a randomized experience replay buffer. This is achieved by:

- **Flattening**: The dataset of shape $(N, T)$ is reshaped into a pooled dataset of size $N \times T$. This discards the firm index $i$ and time index $t$, treating every visited state as an independent draw from the ergodic set.

- **Shuffling**: The pooled observations are randomly re-ordered to eliminate serial correlation between consecutive time steps, i.e. apply a random permutation index.

- Each observation in the resulting batch consists of the current state and two next-period realized shocks required for the All-in-One (AiO) estimator: 
$$\text{Obs} = \left( \underbrace{k, b, z}_{\text{Current State}}, \underbrace{z'_1, z'_2}_{\text{Next Shocks}} \right)$$ 
where $z'_1$ is the main shock and $z'_2$ is the forked shock. I also check that after the reshuffling, the training dataset should maintain the identical mean and variance as in the original one.

### Implementation

Input: 
- Master seed pairs: (int32, int32)
- Bounds (tuple) for $k$, $b$, $\log z$ 
- Batch size $n$, Horizon $T$, 
- Sample size for validation $N_{val}$ and test set $N_{test}$ (default values are 10n and 50n)
- Transition laws (AR-1), which is a separate module that compute an AR step

Internally: 
- Called the `RNG.py` to generate seeds for training, validation, and test data
- Use the generated seeds to simulate data
- Use the AR-1 transition law to rollout $z$'s after $z_0$ based on the realized $\varepsilon$'s. There should be two independent shocks.

Output:
- Train dataset A (for LR)
- Train dataset B (identical data points as the LR one, but shuffled for ER and BR)
- Validation dataset
- Test dataset
  
## RNG Seeds
It is important to use pre-registered RNG seeds to create the datasets. Key points:
1. **Reproducibility**: exact reruns reproduce the same training/validation/test data
2. **Common random numbers**: different methods see the _same_ draws and are comparable
3. **No leakage**: train/valid/test use disjoint RNGs

In Tensorflow, I choose to use stateless random functions (e.g., `tf.random.stateless_uniform`) to guarantee the same draws are independent of call order. These functions take a master seed of `shape [2]` (two 32-digit integers) and uniquely map them to a simulated dataset. The key in implementation is to develop an explicit and safe seed schedule. This schedule is described as below.

**Master seed pair**
I first pick and store a master seed
$$
s^{\text{master}} = (m_0,m_1)
$$
where $m_0, m_1$ can be any 32-bit integers.

**Split seeds** 
I allocate disjoint splits by fixed gaps:
$$
\mathbf{s}^{\text{train}} = (m_0+100, m_1),\quad \mathbf{s}^{\text{val}} = (m_0+200, m_1),\quad \mathbf{s}^{\text{test}} = (m_0+300, m_1).
$$
Note that here we can choose any other gaps instead of 100, 200, and 300. We only need to make sure they are unique to each dataset.

**Variable ID** 
Define fixed and permanent variable IDs. In our current model, we have five variables $k_0, z_0, b_0, \epsilon_1, \epsilon_2$ so we can set
- $\text{VarID} (k_0)=1$
- $\text{VarID} (z_0)=2$
- $\text{VarID} (b_0)=3$ (used only in risky debt)
- $\text{VarID} (\epsilon_1)=4$
- $\text{VarID} (\epsilon_2)=5$
and we may assign more to additional states added in future extensions.

**Training seeds** 
Training has iterations/steps $j = 1,\dots,J$. For each step $j$ and variable $x$, define the seed pair as
$$
\mathbf{s}^{\text{train}}_{j,x} = 
\big(m_0+100+\text{VarID}(x),\ m_1+j\big)
$$
where the first seed uniquely identifies training dataset (100) and variable, and the second seed uniquely identifies steps (batches).

Then I use $\mathbf{s}^{\text{train}}_{j,k_0}$ to draw $k_{0,i}$ for all $i=1,\dots,n$ inside a batch $j$. Similarly for the rest variables. The complete seed tuple for training data is then:

$$
\mathcal{S}^{\mathrm{train}}_j = \big( \mathbf{s}^{\mathrm{train}}_{j,k0},\ \mathbf{s}^{\mathrm{train}}_{j,z0},\ \mathbf{s}^{\mathrm{train}}_{j,b0},\ \mathbf{s}^{\mathrm{train}}_{j,\varepsilon1},\ \mathbf{s}^{\mathrm{train}}_{j,\varepsilon2} \big)
$$

**Validation/test seeds** 
Because validation data and test data are single, fixed dataset, we do not need the steps $j$. For each variable, we can simply create 

$$ 
\mathbf{s}^{\text{val}}_{j,x} = 
\big( m_0+200+\text{VarID}(x),\ m_1+0 \big) 
$$

and the full seeds for validation data generation is

$$
\mathcal{S}^{\mathrm{val}} = \big( \mathbf{s}^{\mathrm{val}}_{k0},\ \mathbf{s}^{\mathrm{val}}_{z0},\ \mathbf{s}^{\mathrm{val}}_{b0},\ \mathbf{s}^{\mathrm{val}}_{\varepsilon1},\ \mathbf{s}^{\mathrm{val}}_{\varepsilon2} \big)
$$

Analogously, the seeds to each variable in test data is

$$
\mathbf{s}^{\text{test}}_{j,x} = 
\big(m_0+300+\text{VarID}(x),\ m_1+0\big)
$$

and the full seeds for test data generation is

$$
\mathcal{S}^{\mathrm{test}} = \big( \mathbf{s}^{\mathrm{test}}_{k0},\ \mathbf{s}^{\mathrm{test}}_{z0},\ \mathbf{s}^{\mathrm{test}}_{b0},\ \mathbf{s}^{\mathrm{test}}_{\varepsilon1},\ \mathbf{s}^{\mathrm{test}}_{\varepsilon2} \big).
$$

This seed schedule guarantees that (i) validation and test are fixed and reproducible, and (ii) they are disjoint from training by construction.

**Implementation**
Inputs:
- Master seed pair $(m_0,m_1)$
- Split IDs: 100/200/300
- Variable IDs: $(k_0, z_0, b_0, \epsilon^{(1)}, \epsilon^{(2)} ) \to (1,2,3,4,5)$
- Training steps $J$, batch size $n$, horizon $T$, dataset sizes $N_{\text{val}}, N_{\text{test}}$
Outputs:
- Training data seeds: $\{\mathcal{S}^{\mathrm{train}}_j\}_{j=1}^J$
- Validation data seeds: $\mathcal{S}^{\mathrm{val}}$
- Test data seeds: $\mathcal{S}^{\mathrm{test}}$






---
# Network Architecture

## Basic Model Networks

1. Policy network:

$$k' = \Gamma_{\text{policy}}(k,z;\theta_{\text{policy}})$$

2. Value network (BR only)

$$V(k,z) = \Gamma_{\text{value}}(k,z;\theta_{\text{value}})$$

**Implementation**

Inputs to networks:
- Training data: $(\log k_t, \log z_t)$

Outputs (activations):
- $k' = k_{\min} + (k_{\max} - k_{\min}) \cdot \text{Sigmoid}(\cdot)$
	- Use bounds stored in metadata of the training set
- $V(k,z) = \mathrm{Linear}(\cdot)$

Primitives use levels:
- $k=\exp(\log k)$, $z=\exp(\log z)$
- Use levels in $\pi(k,z)$, $\psi(I,k)$, $e(\cdot)$, etc.
- In this project version: do NOT standardize network inputs/outputs using running mean/std.

## Risky Debt Model Networks

1. Policy network:

$$ (k',b') = \Gamma_{\text{policy}}(k,b,z;\theta_{\text{policy}}) $$

2. Bond pricing network:

$$\tilde{r}(k',b',z) = \Gamma_{\text{price}}(k',b',z;\theta_{\text{price}})$$

3. Continuation (latent) value network:

$$\widetilde{V}(k,b,z) = \Gamma_{\text{value}}(k,b,z;\theta_{\text{value}})$$

**Implementation**

Inputs to networks: Tuple $\left(\log k, \frac{b}{k}, \log z \right)$
- Use normalized ($b/k$) to avoid saturation and improve stability

Outputs (activations):
- Pre-computed bounds for states: $K_{\min}, K_{\max}, B_{\max}$

- $k' = K_{\min} + (K_{\max} - K_{\min}) \cdot \text{Sigmoid}(\cdot)$
	- Ensures $k' \in [K_{\min}, K_{\max}]$

- $b' = B_{\max} \cdot \text{Sigmoid}(\cdot)$
	- Ensures $b' \in [0,B_{\max}]$

- Risky interest rate: $\tilde{r}(k',b',z) = r_{\text{risk-free}} + \mathrm{softplus}(\cdot)$
	- Ensures $\tilde{r} \in [r, \infty]$

- $\widetilde{V}(k,b,z) = \mathrm{linear}(\cdot)$
	- Ensures $\tilde{V}$ can be either positive or negative

## Common Configs

**Structure** 
- Fully Connected Neural Networks (FcNN)
- Baseline: 2 hidden layers with 32 units each
- User can specify number of hidden layers and neurons per layer

**Input convention**
- Networks take log-transformed/normalized inputs $(\log k,\log z)$ or $(\log k,b/k,\log z)$
- Whenever formulas write $(k,b,z)$, these are economic level variables used inside primitives

**Hidden layer activation**
- Default use SiLU (`swish`) to improve stability and avoid saturation

**Others**
- Compute primitives using **levels** recovered from transforms

## Implementation Issues

Several practical issues are common across all methods and need to be handled carefully and accurately.

### Discrete Indicators

In our model, there are two types of indicator functions that need to be handled carefully:

- Inaction region $\mathbb{1}\{ x \neq 0\}$, e.g., fixed capital adjustment cost $\phi_1 \cdot k \cdot \mathbb{1}\{I\neq0\}$

- Regime switch $\mathbb{1}\{x<0\}$ , e.g., equity injection, default on debt

These hard discrete indicators have derivative 0 a.e. and undefined at $I=0$ due to discontinuity. Gradient-based training thus has no useful signal. If we hard-coded the indicator in our reward calculation, this component will has zero gradient through backpropogation, meaning that the network would essentially 'ignore' the inaction region or regime switch that are of interest. This issue does not only affect the ER method that requires existence of derivatives of the cost functions, it can also affect the LR and BR method, or any gradient-based training.

Although it is straightforward to approximate these indicators with smooth functions, we should avoid over-smoothing the discontinuities/kinks because the hard indicators are economically meaningful and supposed to be captured by training. For example, we do not want to erase the inaction region in optimal investment policies due to fixed capital adjustment cost at $I\neq0$.

With these nuances in mind, I adopt Sigmoid smoothing with annealing to replace the discrete indicator with a differentiable function that
- is smooth and provides useful gradient signals during early iterations
- converges to the true discrete indicator as a temperature/anneal parameter $\tau \to 0$ 

Concretely, define sigmoid function $\sigma(x)=1/(1+e^{-x})$ and a tiny tolerance $\epsilon>0$ that is close to zero. By definition, $\sigma(x)\to 1$ when $x\to \infty$ and $\sigma(x)\to 0$ when $x\to -\infty$, and we can use it to approximate binary indicator during training while still ensure differentiability. Let $\tau>0$ denote a temperature parameter that is decreasing during training under an annealing schedule. 



#### Capital adjustment cost

In both the basic and risky debt model, firm pays a fixed capital adjustment cost for any non-zero investment. The fixed investment cost indicator $\mathbb{1}\{ I\neq 0\}$ can be replaced with

$$
\sigma\left(\frac{|I/k|-\epsilon}{\tau}\right) \rightarrow \mathbb{1}\{|I/k|\gt \epsilon\} \quad \text{as} \quad \tau \rightarrow 0, \, |I|\neq \epsilon
$$

where the point $I \neq 0$ is approximated with a small inaction region $[-\epsilon, \epsilon]$ in computation. Normalizing investment by capital is important to avoid saturation.

#### Costly external finance

In ths risky debt model, firm pays a fixed cost for equity injection (external financing) when cash flow $e$ is negative. The equity injection indicator $\mathbb{1}\{e<0\}$ can be replaced with
$$
\sigma\left(-\frac{e/k+\epsilon}{\tau}\right) \rightarrow \mathbb{1}\left\{\frac{e}{k}<-\epsilon \right\} \quad \text{as} \quad \tau \rightarrow 0
$$ 
where the payout is also normzlied by current capital to avoid saturation.

#### Endogenous Default indicator 

In risky debt model, firm choose to default when the continuation value becomes negative, $D=\mathbb{1}\{\widetilde V(z',k',b')<0\}$. In principle, we could use a similar sigmoid $\sigma(-V/\tau)$, but it risks getting stuck in local optima (e.g., the model learns "never default" early on and never explores the alternative). 

Instead, I use Gumbel-Sigmoid that introduces random noise to force the pricing network to explore both default and solvent states around the default boundary. The steps are 

1. Compute the normalized value $\widetilde V/k$ 
2. Clip the norm value between $[-20, 20]$ as additional safety net
3. Draw random noise  $u \sim \text{Uniform}(0,1)$
4. Initialize temperature/anneal $\tau = 1.0$
5. Compute Gumbel-Sigmoid: 
$$ \sigma \left( 
    \frac{- \widetilde V/k + \log(u) - \log(1-u)}{\tau}
\right) \rightarrow \mathbb{1}\left\{\frac{\widetilde V}{k}<0 \right\} \quad \text{as} \quad \tau \rightarrow 0 $$

where the random uniform noise $u \sim \text{Uniform}(0,1)$
- encourages exploration around the default boundary when $\tau$ is large 
- converges to the true indicator as $\tau \rightarrow 0$

Note that the clipping limits and the initial temperature are hyperparameters that can be tuned.

### Annealing Schedule
I build a simple schedule that anneals temperature $\tau$ over iterations:
$$
\tau_j = \tau_0 \cdot d^j
$$
where $\tau_0$ is the initial temperature, $d \in (0,1)$ is the decay rate, and $j$ is the iteration number. 

Given a set $\tau_{\min}>0$, we stop annealing when $\tau_{\min} \leftarrow \tau_j$. Then I hold $\tau_j$ constant at $\tau_{\min}$ to allow the gradients (which become very sharp/spiky at low $\tau$) to fine-tune the decision boundaries (e.g., the exact k where default happens) without the moving target. 

Given the hyperparameter intput $\tau_{\min}$, initial temperature $\tau_0$, and decay $d$, we can compute the number of iterations for decay as 
$$
N_{\text{decay}} = \frac{\log(\tau_{\min}) - \log(\tau_0)}{\log(d)}
$$
The allow for a stablization buffer `anneal_buffer` (default 25%) to compute the final number of iterations for annealing as 
$$
N_{\text{anneal}} = \lceil N_{\text{decay}} \cdot (1+ \text{anneal\_buffer}) \rceil
$$
where $\lceil \cdot \rceil$ denotes the ceiling function. Later, the annealing iteration $N_{\text{anneal}}$ will be used in determining early stopping/convergence.


---
# Basic Model


## Theory (Sec 3.1)

**Variables**
- State: $(k,z)$
- Action/Control: $k'$

**Investment**

$$I = k' - (1 - \delta)k$$

Note that investment can be either positive or negative.

**Payout/Cash flow**

$$e(k,k',z) = \pi(k,z) - \psi(I,k) - I$$

**Production function**

$$\pi(k,z) = z \cdot k^{\gamma} , \quad \gamma \in (0,1)$$

where $\gamma$ is production technology/elasticity parameter.

**Capital adjustment cost**

$$\psi(I,k) = \phi_0 \cdot \frac{I^2}{2k} + \phi_1 \cdot k \cdot \mathbf{1}\{I\neq0\}$$
  
where $\phi_0$ and $\phi_1$ are convex and fixed adjustment cost parameters. $\mathbf{1}\{I\neq0\}$ is a indicator function for investment (or deinvestment).

**Objective**

Firm aims to choose the optimal path of investment $\{k_{t}\}^T_{t=1}$ that maximizes discounted lifetime reward (cash flow):

$$
\max_{k_1, k_2, \dots, k_T} \mathbb{E}_{z_{t+1}|z_t} \left[ \sum^{T-1}_{t=0} \beta^t \cdot e(k_t,k_{t+1},z_t)  \right]
$$

where the discount factor is $\beta = 1/(1+r)$. The expectation operator is taken w.r.t next period shock $z_{t+1}$ conditional on $z_t$. The AR(1) process maps $(z_t, \varepsilon_{t+1}) \to z_{t+1}$.

**Bellman Equation**

$$
V(k,z) = \max_{k'}\{e(k,k',z)+\beta\mathbb{E}[V(k',z')\mid z]\}
$$


## LR Method
Policy Network 
$$k_{t+1}=\Gamma_{\text{policy}}(k_t,z_t;\theta_{\text{policy}})$$

Objective: The objective is to find the optimal parameters $\theta$ that maximize the expected discounted lifetime reward: 
$$\max_{\theta} \quad \mathbb{E}_{k_0, z_0, \{\varepsilon_t\}_{t=1}^{\infty}} \left[ \sum_{t=0}^{\infty}\beta^t e(k_t, k_{t+1}, z_t) \right] $$

I approximate the infinite horizon objective using a finite horizon simulation $T$. To correct for the truncation at period $T$, I append a Terminal Value function (e.g., the steady-state value of capital).

Given a training batch $\mathcal B$ with initial states $(k_{0,i}, z_{0,i})$ and the main AR(1) shock sequences $\{\varepsilon^{(1)}_{t,i} \}_{t=1}^T$, the empirical loss is given by the negative lifetime rewards:

$$ \min_{\theta} \mathcal{L}^{\text{LR}}(\theta) = -\frac{1}{|\mathcal B|}\sum_{i\in \mathcal B} \left( \sum_{t=0}^{T-1} \beta^t e(k_{t,i}, k_{t+1,i}, z^{(1)}_{t,i}) + \beta^T V^{\text{term}}(k_{T,i}, z^{(1)}_{T,i}) \right) $$

There are many ways to determine the terminal value $V^{\text{term}}$. In this project, we use the steady-state value of capital as the terminal value. The idea is
- Assume that with long enough $T$ horizon, $k_T$ has already converged to the steady state capital level
- This implies $k_{SS}\equiv k_T = k_{T+1} = k_{T+2} = \dots$ forever
- The policy is determined by $I_{SS} = \delta k_{SS}$ to maintain steady state
- Terminal value is the infinite sum of discounted cash flow at $(k_{SS},z_T)$

$$
V^{\text{term}}(k_T,z_T) = \sum_{t=T}^{\infty} \beta^t e(k_{SS},k_{SS},z_T)
= \frac{1}{1-\beta} e(k_{SS},k_{SS},z_T)
$$

Note that the terminal value is vanshing with $\beta^T \lt 1$ when $T$ is large and/or discount factor is small.

### Algorithm Summary: LR Method

**Initialization** 
- Initiate economic parameters and the policy network $\Gamma_{\text{policy}}(\theta)$.
- Data Loading: Load a batch from Training Dataset A.

**Differentiable Simulation** 
- Using the current policy $\Gamma(\theta)$ and the loaded shocks, simulate the trajectory for $N$ firms for $T$ periods: 
$$ k_{t+1} = \Gamma_{\text{policy}}(k_t, z^{(1)}_t; \theta) $$

- Keep this operation inside the computation graph (e.g., `tf.GradientTape`) to allow backpropagation through time.
- Since our data generation has deterministically created shock paths, the capital paths are also deterministic given policy $\theta$

**Policy Update**: 
- Compute $\mathcal{L}^{\text{LR}}$ (sum of discounted rewards) and update $\theta_{\text{policy}}$ via Stochastic Gradient Descent (SGD) to minimize the loss.

**Stopping** 
- Repeat the loop above until hitting hard max $N_{\text{iter}}$ (for quick debugging) or when reached the converegnce/stopping criteria for LR (See section below).

**Implementation notes**
- Unlike ER/BR, do not shuffle the time dimension. Retrieve full trajectories of shocks $\varepsilon_{1:T}$ and initial states $(k_0, z_0)$.
- Main Path Only: Use the main shock path ${ z^{(1)}_{t,i} }$ derived from $\varepsilon^{(1)}$. The forked path $\varepsilon^{(2)}$ is not used in LR because we maximize the sum of rewards, not a squared residual, so that we don't need the AiO estimator.
- On-Policy Rollout: The capital sequence $k_{1} \dots k_{T}$ is not loaded from the dataset. It is generated during the training step by applying the current policy $\Gamma_{\text{policy}}$ recursively to the shocks loaded from the dataset. This ensures the loss is differentiable with respect to $\theta$.


## ER Method
**Notation**: Let $F_x$ denote the partial derivative of function $F$ with respect to $x$.

**Optimization Problem**: The Lagrangian is set up as: 
$$\max_{{k_{t+1}}} \mathbb{E} \left[ \sum^{\infty}_{t=0} \beta^t \cdot \big( e(I_t,z_t) - \chi_t \left( k_{t+1} - (1-\delta)k_t - I_t \right) \big) \right]$$ 
where $\chi_t$ is the shadow price of capital (Lagrange multiplier).

**First Order Conditions**:

- Investment: $\chi_t = 1 + \psi_I(I_t,k_t)$
- Next period investment: $\chi_{t+1} = 1 + \psi_I(I_{t+1},k_{t+1})$
- Capital: $\chi_t = \beta \mathbb{E} \left[ \pi_k (z{t+1},k_{t+1}) - \psi_k (I_{t+1}, k_{t+1}) + (1 - \delta) \chi_{t+1} \right]$


Combining these yields the **Euler Equation**: 
$$1+\psi_I(I_t,k_t) = \beta \mathbb{E} \left[ \pi_k(z_{t+1},k_{t+1})-\psi_k(I_{t+1},k_{t+1})+(1-\delta) \big(1+\psi_I(I_{t+1},k_{t+1})\big) \right] $$

**Definitions for Implementation**:

- Current Investment: $I_t = k_{t+1} - (1-\delta)k_t$
- Future Investment: $I_{t+1} = k_{t+2} - (1-\delta)k_{t+1}$
- Marginal Benefit Function $m_\ell$: The RHS integrand is defined as: 
$$ m(k_{t+1}, k_{t+2}, z_{t+1}) \equiv \pi_k(z_{t+1},k_{t+1}) - \psi_k(I_{t+1},k_{t+1}) + (1-\delta)(1+\psi_I(I_{t+1},k_{t+1})) $$
- All-in-One (AiO) Loss: To estimate the squared expectation $\mathbb{E}[f]^2$ unbiasedly, I use two i.i.d. shock realizations $z'_{1}, z'_{2}$ (forks) for the next period. The empirical loss is the cross-product of residuals: 
$$ \mathcal{L}^{\text{ER}} = \frac{1}{n} \sum_{i=1}^n \left( f_{i,1} \times f_{i,2} \right) $$ 
where the realized Euler residual for branch $\ell \in {1, 2}$ is: 
$$ f_{i,\ell} = \big(1 + \psi_I(I_{i}, k_{i})\big) - \beta \cdot m(k'_{i}, k''_{i,\ell}, z'_{i,\ell}), \quad \ell = 1,2$$

### Algorithm Summary: ER Method
**Initialization:**
- Load the training dataset $\mathcal{B}$ (flattened into $N \times T$ transitions and reshuffled). 
- Each observation is a tuple $(k, z, z'_{1}, z'_{2})$
- Initiate current policy $\Gamma_{\text{policy}}(\cdot; \theta_{\text{policy}})$.
- Initiate target policy $\theta^-_{\text{policy}} \leftarrow \theta_{\text{policy}}$ (to stabilize future capital approximations).

**Training Loop:** Repeat until convergence ($|\mathcal{L}^{\text{ER}}| < \epsilon$).
1. **Sample Batch**: Draw a random mini-batch of observations from $\mathcal{B}$.
2. **Current Step (Trainable)**:
- Compute next capital using Current Policy: 
$$k' = \Gamma_{\text{policy}}(k, z; \theta_{\text{policy}})$$

- Compute current investment: 
$$I = k' - (1-\delta)k$$

- Compute LHS marginal cost: 
$$\chi = 1 + \psi_I(I, k)$$

3. **Future Step (Fixed Target)**:
- For both shock forks $\ell \in {1, 2}$, compute the subsequent capital $k''_{\ell}$ using the Target Policy: 
$$ k''_{\ell} = \Gamma_{\text{policy}}(k', z'_{\ell}; \theta^-_{\text{policy}}) $$ 
- Note: Using Target Policy here prevents gradients from flowing into the future choice, stabilizing the Euler inversion.
- Compute future investment for both forks: $I'_{\ell} = k''_{\ell} - (1-\delta)k'$.

4. **Compute Residuals**:
- Calculate the realized marginal benefit $m_{\ell}$ for both $\ell=1$ and $\ell=2$ using $(k', k''_{\ell}, z'_{\ell})$.
- Compute residuals: $f_{\ell} = \chi - \beta \cdot m_{\ell}$.

5. **Update Policy**:
- Compute Empirical Loss $\mathcal{L}^{\text{ER}} = \frac{1}{|\mathcal B|}\sum_{i\in \mathcal B} (f_{i,1} \times f_{i,2})$

- Update $\theta_{\text{policy}}$ using gradient descent to minimize $\mathcal{L}^{\text{ER}}$.
- Polyak Averaging to update target policy: $\theta^-_{\text{policy}} \leftarrow \nu \theta^{-}_{\text{policy}} + (1-\nu) \theta_{\text{policy}}$.

**Stopping** 
- Repeat the loop above until hitting hard max $N_{\text{iter}}$ (for quick debugging) or when reached the converegnce/stopping criteria for ER (See section below).

## BR Method (Actor-Critic)
Unlike LR and ER method, BR method requires two parameterized networks:

**Policy network**
$$k' = \Gamma_{\text{policy}}(k,z;\theta_{\text{policy}})$$

**Value network**
$$V(k,z) = \Gamma_{\text{value}}(k,z;\theta_{\text{value}})$$

**Bellman Residual** is given as
$$V(k,z) - \max_{k'}\{e(k,k',z)+\beta\mathbb{E}_{z'}[V(k',z')]\}$$

We use actor–critic:


1) given policy $\theta_{\text{policy}}$, update $\theta_{\text{value}}$ to fit the Bellman equation;
2) given value $\theta_{\text{value}}$, update $\theta_{\text{policy}}$ to maximize the RHS.

This is a variant of the "direct optimization" BR approach discussed in @Maliar12.

### Critic Update Step

The Bellman RHS target is defined as
$$y_{\ell}=e(k ,\Gamma_{\text{policy}},z)+\beta\,\Gamma_{\text{value}}(\Gamma_{\text{policy}},z'_{\ell};\theta^{-}_{\text{value}}),\qquad \ell\in\{1,2\}$$

where the expectation is dropped given two realized shocks $z'_1, z'_2$ (that were i.i.d in sampling). Our goal is to update our parameterized value network $\theta_{\text{value}}$ to minimize the "square" error from this critic target:
$$
\min_{\theta_{\text{value}}} \left( \Gamma_{\text{value}}(k,z;\theta_{\text{value}})-y_{1}\right)\left( \Gamma_{\text{value}}(k,z;\theta_{\text{value}})-y_{2}\right)
$$
where we use the AiO estimator to consistently estimate the squared expectation. 

This is a classic regression problem. Once the critic target $y_{\ell}$ is computed, it is passed over as a constant (fixed label). It is critical to detach gradients from it when updating $\theta_{\text{value}}$ and also keep the policy network $(\theta_{\text{policy}})$ fixed during the critic update.

**Empirical Loss**

Empirically, the BR-Critic Loss is defined as
$$\widehat{\mathcal{L}}^{\text{BR}}_{\text{critic}}(\theta_{\text{value}})=\frac{1}{N}\sum_{i=1}^N \left(\delta_{i,1} \times \delta_{i,2} \right) $$
where the Bellman residual (error) is
$$\delta_{i,\ell}=\Gamma_{\text{value}}(k_{i},z_{i};\theta_{\text{value}})-y_{i,\ell}, \quad \ell \in \{1,2\}$$
for each observation $i$. For $y_{i,\ell}$, the main shock $z'_{i,1}$ and the forked shock $z'_{i,2}$ are i.i.d. draws.



### Actor update step

For a given fixed policy $\theta_{\text{policy}}$, the previous critic update find the best $\theta_{\text{value}}$ that minimizes the critic loss. However, it is not guaranteed that this $\theta_{\text{policy}}$ is the optimal policy that maximize the Bellman equation. In other words, we need an additional training step to ensures that
$$
\theta_{\text{policy}} = \argmax_{k'=\Gamma(\theta_{\text{policy}})}\{e(k,k',z)+\beta \mathbb{E}_{z'}\left[V(k',z')\right]
$$

This is exactly the "direct optimization" idea described in @Maliar12. 

Empirically, I define the Actor loss as:
$$\widehat{\mathcal{L}}^{\text{BR}}_{\text{actor}}(\theta_{\text{policy}})
=-\frac{1}{N}\sum_{i=1}^N 
\left[e(k'_{i},k_{i},z_{i,1})+\beta\cdot  \Gamma_{\text{value}}(k'_{i},z'_{i,1};\theta_{\text{value}})\right]$$
where it is helpful to clarify that
- Take $\theta_{value}$ as given, update $\theta_{policy}$ of $k'=\Gamma_{\text{policy}}(k,z;\theta_{\text{policy}})$
- I use the main shock $z'_{i,1}$ and sample mean as unbiased estimator
- We do not need the forked shock $z'_{i,2}$ here because this is not a AiO cross-product estimator


### Algorithm Summary: BR Method
**Initialization:**

- Load the training data $\mathcal{D}$ (flattened into $N \times T$ transitions and reshuffled).
- Initiate current networks: Policy $\Gamma_{\text{policy}}(\cdot; \theta_{\text{policy}})$ and Value $\Gamma_{\text{value}}(\cdot; \theta_{\text{value}})$.
- Initiate target networks: $\theta^-_{\text{policy}} \leftarrow \theta_{\text{policy}}$ and $\theta^-_{\text{value}} \leftarrow \theta_{\text{value}}$.
- Training Loop: Repeat A and B for `N_iter` or until stopping criteria in C are met.

#### A. Value function update (Critic)

Objective: Minimize the Bellman residual using the Target Policy to stabilize the learning target.

1. **Sample Batch**: Draw a random mini-batch $\mathcal B$ of observations $(k_i, z_i, z'_{i}, z'_2)$ from $\mathcal{D}$.

2. **Compute Critic Target** 
- Compute next action using the Target Policy: 
$$\widehat{k}'_i = \Gamma_{\text{policy}}(k_i, z_i; \theta^-_{\text{policy}})$$
- Compute next value using the Target Value network for both next period shocks: 
$$V^{\text{next}}_{i,\ell} = \Gamma_{\text{value}}(\widehat{k}'_i, z'_{i,\ell}; \theta^-_{\text{value}}), \quad \ell=1,2$$

- Compute the fixed Bellman target (detach gradients): 
$$y_{i,\ell} = \text{StopGradient} \left\{ e(k_i, \widehat{k}'_i, z_i) + \beta V^{\text{next}}_{i,\ell} \right\}, \quad \ell=1,2$$

3. **Update Critic** 
- Compute current value estimates: $V_{i} = \Gamma_{\text{value}}(k_i, z_i; \theta_{\text{value}})$
- Calculate residuals: $\delta_{i,\ell} = V_{i} - y_{i,\ell}$ for $\ell=1,2$
- Compute AiO Loss for each mini-batch $\mathcal B$: 
$$ \mathcal{L}^{\text{BR}}_{\text{critic}} = \frac{1}{|\mathcal B|} \sum_{i \in \mathcal B} (\delta_{i,1} \times \delta_{i,2}) $$
- Update $\theta_{\text{value}}$ using gradient descent to minimize $\widehat{\mathcal{L}}^\text{BR}_{\text{critic}}$
- Update Target Value: $\theta^-_{\text{value}} \leftarrow \nu \theta^{-}_{\text{value}} + (1-\nu) \theta_{\text{value}}$

Repeat Step A#1-3 by `N_critic_step` times per Actor update.

#### B. Policy update (Actor)
Objective: Maximize the expected value using the Current Policy to allow gradient flow.

1. **Compute Actor Action**
- Compute next action using the Current Policy: 
$$ k'_i = \Gamma_{\text{policy}}(k_i, z_i; \theta_{\text{policy}}) $$

2. **Compute Actor Loss**
- Predict continuation value using the Current Value network (freeze $\theta_{\text{value}}$ and use main shock $z'_1$ only): 
$$V^{\text{proj}}_{i} = \Gamma_{\text{value}}(k'_i, z'_{i,1}; \theta_{\text{value}})$$ 
- Define Loss (negative expected value of Bellman RHS): 
$$ \mathcal{L}^{\text{BR}}_{\text{actor}} = -\frac{1}{|\mathcal B|} \sum_{i \in \mathcal B} \left[ e(k_i, k'_i, z_i) + \beta V^{\text{proj}}_{i} \right] $$

3. **Update Actor**
- Update $\theta_{\text{policy}}$ using gradient descent to minimize $\mathcal{L}^{\text{BR}}_{\text{actor}}$.
- Update Target Policy: $\theta^-_{\text{policy}} \leftarrow \nu \theta^{-}_{\text{policy}} + (1-\nu) \theta_{\text{policy}}$.

**Stopping** 
- Repeat the A-B loop above until hitting hard max $N_{\text{iter}}$ (for quick debugging) or when reached the converegnce/stopping criteria for BR (See section below).

---
## Convergence and Stopping Criteria
The stopping logic is hierarchical. It first enforces a "Gatekeeper" (Annealing) to ensure the problem is economically valid (sharp boundaries) before checking statistical convergence on the **Validation Set** $\mathcal{D}_{val}$.
- The validation set was generated along with the training set using deterministic RNG seeds

### Step A. Gatekeeper for Annealing Schedule Training

Details of the annealing schedule is described in previous section. 

- Inputs: $\tau_0$, $\tau_{\min}$ (e.g., $10^{-5}$), decay_rate, and stablization buffer (25%).
- Calculate $N_{\text{decay}}$ steps required to reach $\tau_{\min}$
- Calculate $N_{\text{stable}}$ (buffer steps at $\tau_{\min}$)
- Output: $N_{\text{anneal}} = N_{\text{decay}} + N_{\text{stable}}$
- Constraint: If current_step < $N_{\text{anneal}}$, ignore all early stopping triggers.

### Step B. Method-Specific Stopping Rules (Post-Annealing) 

These metrics are computed strictly on the Validation Set $\mathcal{D}_{val}$ to avoid overfitting. First, define a `patience_counter` as an integer variable that tracks the number of consecutive validation checks where the method-specific convergence criteria are satisfied.
- Initialize `patience_counter=0`
- Trigger: Only active when current step > `N_anneal`
- Update `patience_counter += 1` if convergence criteria is `True`
- Reset to 0 if convergence criteria is `False`
- Stops when `patience_counter >= PATIENCE_LIMIT` (e.g., 5 consecutive checks)


#### LR Method (Relative Improvement Plateau)
- Goal: Minimize negative lifetime reward (unknown lower bound)
- Metric: Moving Average of the LR on validation set ($\bar L^{LR}$)
- Criteria: Stop if `patience_counter` is exhausted AND relative improvement is negligible: 
$$ \frac{\bar L^{LR}(j) - \bar L^{LR}(j-s)}{|\bar L^{LR}(j-s)|} < \epsilon_{LR} $$
where $s$ is a selected window for improvement evaluation (e.g., 100 steps).


#### ER Method (Zero-Tolerance Plateau)
- Goal: Solve for root where residual is zero.
- Metric: ER loss (mean square error)
$$
L^{ER} < \epsilon_{ER} = 10^{-5}
$$
- Criteria: Stop if `patience_counter` is exhausted AND ER loss is less than tolerance $\epsilon_{ER}$ 
- Note: use log-10 transformation to numericaly stability and easier visualization

#### BR Method (Dual-Condition Convergence)
- Goal: Equilibrium where Value function is accurate AND Policy is optimal.
- Metric 1: Critic Loss (Bellman Residual MSE)
- Metric 2: Actor Objective (Bellman RHS Value)
- Criteria: Stop if `patience_counter` is exhausted AND both conditions are met:
  - Critic Accuracy: Bellman Residual $< \epsilon_{crit}$
  - Policy Stability: Actor Value relative improvement $< \epsilon_{act}$
where relative improvement is computed similar to the LR approach

### API Usage 
For the APIs (e.g., `train_basic_lr`), the training loop is controlled by configurations `max_iter`, `early_stopping`. Example usages:
- Debug/Demo: Set `early_stopping = False`. The loop runs exactly `max_iter` steps. Used to visualize annealing schedules, check gradient flow, and plot preliminary policy surfaces.
- Full Mode: Set `early_stopping = True` and `max_iter = Large_Int` (safety ceiling). The loop terminates dynamically when the Validation Set metrics satisfy method-specific economic criteria.

In practice, I use a "Callback" logic to handle the switch between Debug Mode (run until max_iter) and Full Mode (run until Convergence).

Example callback hook:
- If `step % eval_freq == 0` AND `config.early_stopping`:
    - If `step < N_anneal`: Continue (ignore validation).
    - Else: Compute loss and check if criteria is met.


---
# Risky Debt Model

## Theory (Sec 3.6)

**Variables**
- State: $(k,b,z)$
- Control: $(k',b')$

where $b' \ge 0$ denotes borrowing at endogenous risky interest rate $\tilde{r}$.

**Payout/Cash flow**

$$e(\cdot) = (1-\tau)\pi(k,z) - \psi(I,k) - I +\frac{b'}{1+\tilde r} + \frac{\tau\tilde r\,b'}{(1+\tilde r)(1+r)} - b$$

where $\tau$ is corporate tax rate, $b$ is repayment of last period debt, $b'/(1+\tilde{r})$ is the pricing of a zero-coupon risky bond (debt), and $\frac{\tau\tilde r\,b'}{(1+\tilde r)(1+r)}$ is the tax shield from debt.
  

**External financing cost**

$$\eta(e)=(\eta_0+\eta_1|e|)\mathbf{1}_{e<0}$$

where $\mathbf{1}_{e<0}$ is an indicator function for negative cash flow that triggers costly external financing (e.g., equity injection). 

**Endogenous risky interest rate**

$$\tilde r=\tilde r(z,k',k,b',b)$$

which is determined in equilibrium where the lenders earn zero profit after taking into account the default probablity and expected recovery. 
  

**Latent and actual firm value**

The Bellman equation of latent value is 

$$\widetilde V(k,b,z)=\max_{k',b'}\left\{e(k,b,z;\tilde r)-\eta(e)+\beta\mathbb{E}_{z'}[V(k',b',z')]\right\}$$

and the actual firm value satifies limited liability

$$V(k',b',z')=\max\{ 0, \, \widetilde V(k',b',z') \}$$

which means that shareholders can always choose default and walk away with zeros when the latent value $\widetilde V(k',b',z')<0$.
  
**Bond Pricing**

In equilibrium, the endogenous risky rate, $\tilde{r}$, is determined by the lender's zero-profit condition

$$ b'(1+r)= (1+\tilde r)b'\,\mathbb{E}_{z'|z}[1-D]+\mathbb{E}_{z'|z}[D\cdot R(k',b',z')] $$

where the LHS is the marginal (opportunity) cost of lending and the RHS is the expected marginal return to the risky bond after taking into account default probability and recovery from liquidation.

- If solvent, lender earns $1+\tilde{r}$ that is higher than the risk-free rate
- If defualt, lender recovers $R(k',b',z')$ after liquidation

Note that the risky rate $\tilde r$ is set by lender at the current period given information
- Current states: $k,b,z$
- Firm choice of $k',b'$
- Expectations about $z'$

**Default probability**

Let $D$ denote an indicator for default when firm's latent/continuation value is negative

$$D(z',k',b')=\mathbb{1}\{\widetilde V(z',k',b')<0\},$$
in which case the firm will choose to walk away with actual $V=0$ and liquidate all assets.

**Recovery** under default is defined as

$$R(k',z')=(1-\alpha)\left[(1-\tau)\pi(k',z')+(1-\delta)k'\right]$$

where $\alpha\in[0,1]$ is the deadweight cost applied on the liquidation value of the firm.

### Optimality conditions
To solve for this model, I need to train the neural network to enforce the following conditions:

1. **Bellman equation**:
    $$\widetilde V(k,b,z)=\max_{k',b'}\left\{e(k,b,z;\tilde r)-\eta(e)+\beta\mathbb{E}_{z'}[V(k',b',z')]\right\}$$

    where on the RHS the actual continuation value must be non-negative

    $$V(k',b',z')=\max\{ 0, \, \widetilde V(k',b',z') \}$$

2. **Lender's zero-profit condition** that determines risky rate $\tilde{r}$:

    $$ b'(1+r)= (1+\tilde r)b'\,\mathbb{E}_{z'|z}[1-D]+\mathbb{E}_{z'|z}[D\cdot R(k',b',z')] $$

    where $D=\mathbb{1}\{\widetilde{V}<0\}$ is the default indicator, and $R(k',b',z')$ is the recovery value.

## Networks

Implementation of this model requires at least three neural networks.

**Policy network**

For BR: 
$$ (k',b') = \Gamma_{\text{policy}}(k,b,z;\theta_{\text{policy}})$$

**Pricing network**
$$\tilde{r}(\cdot;\theta_{\text{price}}) = \Gamma_{\text{price}}(k',b',k,b,z;\theta_{\text{price}})$$

**Value network**
$$\widetilde{V}(\cdot;\theta_{\text{value}}) = \Gamma_{\text{value}}(k,b,z;\theta_{\text{value}})$$

**Loop-in-loop problem** 

As described in @Strebulaev12, the key challenge to solve this model is that the latent firm value $\widetilde{V}$ depends on the endogenous risky rate $\tilde{r}$. However, solving for $\tilde{r}$ requires solving the lender's zero-profit condition and particularly on knowing the default probability $\mathbb{E}[D]$ that in turn depends on firm value $\widetilde{V}$. 

The conventional approach to break this circular dependency is to solve for $\tilde{r}$ and $\mathbb{E}[D]$ iteratively via a "inner loop" and a "outer loop". For example, first solving the zero-profit condition to get an estimate of $\tilde{r}$, use it to solve for $\widetilde{V}$, then use the updated $\widetilde{V}$ to update $\tilde{r}$ and repeat until both iteration loop converge. 

Using deep neural networks, this problem is solved more effectively by jointly training three networks.


## Pricing Loss

First let us define an empirical loss for the zero-profit condition. This is essential for training the pricing network and solve fo the endogenous risky rate $\tilde{r}$.

**Smooth default probability**

Since the default indicator $D=\mathbb{1}\{\widetilde{V}<0\}$ is non-differentiable, I approximate the default indicator using a softened, smooth function:

$$p(k',b',z')=\sigma\!\left(-\frac{\widetilde V(k',b',z')}{\tau^D_j}\right) \to D \quad \text{as} \quad \tau^D_j \to 0^+$$

where I built an annealing schedule for $\tau^D_j$ to gradually reduce to zero over $j$ iterations.

**Pricing equation residual**

Given $(z^{(1)}_{t+1,i},(z^{(2)}_{t+1,i})$ and the policy network for $(k',b')$, the residual of the zero-profit condition is computed as
$$f^{(\ell)}_{t,i}
=
b_{t+1,i} (1+r)-
\Big[
    p^{\ell}_{t,i} \cdot R(k_{t+1,i},z^{(\ell)}_{t+1,i}) 
    +(1-p^{\ell}_{t,i}) \cdot b_{t+1,i} (1+\tilde{r}_{t,i})
\Big]$$
where default indicator $p^{\ell}_{t,i}$ and recovery value $R(k_{t+1,i},z^{(\ell)}_{t+1,i})$ can be directly calculated given the realized shocks for each observation $i$ in period $t$.

The endogenous risky rate $\tilde r$ is replaced with a pricing network $\Gamma_{\text{price}}(k',b',z';\theta_{\text{price}})$, so that the residual $f$ becomes a function of trainable parameters $\theta_{\text{price}}$.

The empirical loss function for the zero-profit condition is
$$\widehat{\mathcal{L}}^{\text{price}}
=\frac{1}{N}\sum_{i=1}^N 
\left( \frac{1}{T}\sum_{t=1}^{T-1} f^{(1)}_{t,i}f^{(2)}_{t,i}\right)$$
where I use Monte Carlo sampling mean (across $N$ i.i.d. draws) to approximate the expectation. Then I ensures the residual error is minimized across the $T$ horizon.

---
## BR Method
Similar to the basic model, I implement a critic-actor update algorithm to jointly minimize the Bellman equation residual and the risky bebt pricing equation residual.

Recall that the training dataset has $t=1,\dots,T$ periods and $i=1,\dots,N$ observations (i.i.d. draws of firms). For each $z_{t,i}$, we have two Monte Carlo draws $(\varepsilon^{(1)}_{t+1,i},\varepsilon^{(2)}_{t+1,i})$ and use an AR(1) step to compute $(z^{(1)}_{t+1,i},z^{(2)}_{t+1,i})$. By taking average over $N$ MC draws, we get a consistent estimator for the expected continuation value in the Bellman equation.

### Critic Update

For each observation $i$, the critic target is the RHS of the Bellman equation:
$$y^{(\ell)}_{t}=e(k_t,k_{t+1},b_t,b_{t+1},z_t)-\eta(e)+\beta\cdot \max\{0,\Gamma_{\text{value}}(k_{t+1},b_{t+1},z^{(\ell)}_{t+1};\bar \theta_{\text{value}})\}$$

where as before $\ell = 1, 2$ denotes the two Monte Carlo draws of shocks. We also use the policy network $\Gamma_{\text{policy}}$ to compute $(k_{t+1}, b_{t+1})$. Therefore, this target can be directly computed given
- Economic/production parameters for $e(\cdot)$ and $\eta(\cdot)$
- Current states $(k_t,b_t,z_t)$
- Two realized shocks $(z^{(1)}_{t+1},z^{(2)}_{t+1})$ given $z_t$
- Policy and value networks $\Gamma_{\text{policy}}$ and $\Gamma_{\text{value}}$

To stablized training, I use polyak averaging to update the target value network:
$$
\bar \theta_{\text{value}} \leftarrow \alpha \bar \theta_{\text{value}} + (1-\alpha)\theta_{\text{value}}
$$

The Bellman residual is computed as

$$\delta^{(\ell)}_{t}=\Gamma_{\text{value}}(k_t,b_t,z_t;\theta_{\text{value}})-y^{(\ell)}_{t}$$
where it is important to note that $\bar \theta_{\text{value}}$ is detached from the gradient. I only train $\theta_{\text{value}}$ to update the first term on the RHS and take $y^{(\ell)}_t$ as a constant.

Finally, I compute the residual "square" by taking the cross-product of two residuals for each period $t$, then I take the lifetime mean over $T$ horizon to form the empirical risk:
$$
\widehat{\mathcal{L}}^{\text{BR}}_{\text{critic}}
=\frac{1}{T}\sum_{t=1}^T \left( 
\frac{1}{N}\sum_{i=1}^N \delta^{(1)}_{t,i}\delta^{(2)}_{t,i}
\right)
$$

### Actor Update
The critic update ensures that for given policy $\theta_{\text{policy}}$, the value function is trained to minimize the average lifetime Bellman residuals. However, it is not guaranteed that the policy itself if optimal. Thus in the Actor update step, I train the policy network to maximize the RHS of the Bellman equation, holding the value network constant. The actor empirical risk is given as

$$
\widehat{\mathcal{L}}^{\text{BR}}_{\text{actor}}(\theta_{\text{policy}})
=-\frac{1}{N}\sum_{i=1}^N \frac{1}{T}\sum_{t=0}^{T-1}
\left[
    e(\cdot)-\eta(e)+\beta\cdot \frac{1}{2}\sum_{\ell=1}^2 \max\{0,\Gamma_{\text{value}}(k_{t+1,i},b_{t+1,i},z^{(\ell)}_{t+1,i};\theta_{\text{value}})\}
\right]$$
where we prevent $\theta_{\text{value}}$ from update and only train the policy $\theta_{\text{policy}}$ to maximize the RHS over $N$ draws. Then it is averaged over $T$ period to ensure that the policy is $\argmax_{\theta}$ on average over all periods.

---
> **Warning**: I recommend using BR method to solve the risky debt model. The following sections (LR and ER) are experimental and not implemented.


## LR Method

**Reward (cash flow)**
$$u_t=e(k_t,k_{t+1},b_t,b_{t+1},z_t)-\eta(e(\cdot)).$$

**Networks**

A key feature of this model is that shareholder can choose default and walk away with zero payout when the continuation value is negative. In practice, I define a new neuron output $d_{t+1}\in(0,1)$ to approximate the default probability and train the policy network to "learn" it.

To implement the LR method, we only need two networks: 

1. Policy Networks:
$$
(k_{t+1},b_{t+1}, d_{t+1})=\Gamma_{\text{policy}}(k_t,b_t,z_t;\theta_{\text{policy}})
$$
where the interpretation is: conditional on information at $t$, the firm chooses optimal next-period capital and debt levels, and defaults at the start of next period with probability $d_{t+1}$

2. Pricing Networks:
$$\tilde{r}(k_{t+1},b_{t+1},z_t) = \Gamma_{\text{price}}(k_{t+1},b_{t+1},z_t;\theta_{\text{price}})$$
Note that $\tilde r$ is issued at current period and does NOT depends on the realization of next period $z_{t+1}$.

**Survival Mask**
Define a soft survival indicator $M_t\in[0,1]$ as “probability-weight” that the firm is alive during period $t$.
- Initialization: $M_0=1$.
- Recursion (default happens at start of $t+1$ with probability $d_{t+1}$):
$$M_{t+1}=M_t\,(1-d_{t+1}),\qquad t=0,\dots,T-1.$$
Equivalently,
$$M_t=\prod_{s=0}^{t}(1-d_s),\qquad t\ge 1,\quad d_0=0.$$



**Zero-profit condition**

Given $(z^{(1)}_{t+1,i},z^{(2)}_{t+1,i})$ and the policy network for $(k_{t+1,i},b_{t+1,i},d_{t+1,i})$, the residual of the zero-profit condition is computed as
$$f^{(\ell)}_{t,i}
=
b_{t+1,i} (1+r)-
\Big[
    d_{t+1,i} \cdot R(k_{t+1,i},z^{(\ell)}_{t+1,i}) 
    +(1-d_{t+1,i}) \cdot b_{t+1,i} (1+\tilde{r}_{t,i})
\Big]$$
where default indicator $d_{t+1,i}$ and borrowing $b_{t+1,i}$ are all outputs of the policy network. Recovery value can be computed directly given $k_{t+1,i},z^{(\ell)}_{t+1,i}$.

**Objective**

Let $\mathbf{\theta} \equiv (\theta_{price}, \theta_{policy})$. The objective of training is to find the optimal $\theta$ that jointly minimizes the negative lifetime reward $\mathcal{L}^{\text{LR}}$ and the pricing residual $\mathcal{L}^{\text{price}}$:
$$
\mathcal{L}^{\text{LR}}(\theta) = -\frac{1}{N}\sum_{i=1}^N\sum_{t=0}^{T-1}
\beta^t \cdot M_{t,i} \cdot u_{t,i}^{(1)}
$$
where rewards are computed using the $\ell=1$ main shock path.

$$
\mathcal{L}^{\text{price}}(\theta) = \frac{1}{N}\sum_{i=1}^N 
\left( \frac{1}{T}\sum_{t=0}^{T-1} f^{(1)}_{t,i}f^{(2)}_{t,i}\right)
$$
where the survival mask is computed recursively using neuron output $\{d_{t+1}\}_{t=0}^{T-1}$ from policy network: 
$$
M_{t,i} = \prod_{s=0}^{t} (1-d_{s,i})
$$

These two objectives can be combined into a single empirical risk:
$$
\mathcal{L}(\theta) \equiv \mathcal{L}^{\text{LR}} + \lambda_j (\mathcal{L}^{\text{price}} - \epsilon)
$$
where the Lagrange multiplier $\lambda_j$ is used to enforce the pricing residual to be close to zero by a tolerance $\epsilon$. After each step update of $\theta$ in iteration $j$, the multiplier is updated by
- Detach $\theta$ from $\mathcal{L}_j^{\text{price}}$ to prevent it from being updated
- Compute Polyak averaging $\mathcal{\bar L}_j^{\text{price}}= (1-w) \mathcal{\bar L}_{j-1}^{\text{price}} + w \mathcal{L}_j^{\text{price}}$ with weight $w\in(0,1)$
- Update $\lambda_j \leftarrow \max(0, \lambda_j + \eta_\lambda(\bar L_j^{\text{price}}-\epsilon))$ where $\eta_\lambda$ is a learning rate for the multiplier

The multiplier $\lambda_j \to 0$ when the constraint is slack / satisfied, i.e. the pricing equation residual is small enough $\mathcal{L}_j^{\text{price}} \lt \epsilon$, which ensures that the zero-profit condition holds.

**Training loop summary**
1.	Initialize economic parameters, training hyperparameters, model parameters $\theta$ (policy network $\Gamma_{\text{policy}}$ and pricing network $\Gamma_{\text{price}}$), and Lagrange multiplier(s) $\lambda \ge 0$.
2.	For $j = 1,\dots,N_{\text{iter}}$ during training:
  - Sample a minibatch $B_j$ from the training set
  - Compute $L^{LR}(\theta;B_j)$ (negative lifetime reward) and $L^{\text{price}}(\theta;B_j)$
  - Form the Lagrangian loss: $\mathcal{L}_j(\theta,\lambda)=L^{LR}(\theta;B_j)+\lambda\big(L^{\text{price}}(\theta;B_j)-\epsilon\big)$
  - Update $\theta$: take one optimizer step to minimize $\mathcal{L}_j$ via backpropagation
  - Update $\lambda$ (no backprop): using the detached batch estimate $\mathcal{\bar L}_j^{\text{price}}$ and update 
  $$\lambda \leftarrow \max\!\left(0,\ \lambda+\eta_\lambda(\mathcal{\bar L}_{j}^{\text{price}}-\epsilon)\right)$$
3.	Stop at $N_{\text{iter}}$ (for quick debugging) or when reached the converegnce/stopping criteria for LR (See section below).

## ER Method

Underdevelopment. Coming soon.
