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

This nuance is critical because it matches the theoretical transition law in the inter-temporal Euler equation and Bellman equation. This ensures that the ER and BR method can be applied to every period $t$ in the simulated dataset with horizon $T$. 

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
- Train dataset 
- Validation dataset
- Test dataset (sealed)
- Note: In addition to the component defined above. Each data set should also include the full $z$ rollout using the transition law

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

---
# Theoretical Model Overview

## Basic Model (Sec 3.1)

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

## Risky Debt Model (Sec 3.6)

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

**Endogenous risky rate**

$$\tilde r=\tilde r(z,k',b')$$

which is determined by the equilibrium in debt market (see details below).
  

**Latent and actual value**

The Bellman equation of latent value is 

$$\widetilde V(k,b,z)=\max_{k',b'}\left\{e(k,b,z;\tilde r)-\eta(e)+\beta\mathbb{E}_{z'}[V(k',b',z')]\right\}$$

and the actual firm value satifies limited liability

$$V(k',b',z')=\max\{ 0, \, \widetilde V(k',b',z') \}$$

which means that shareholders can always choose default and walk away with zeros when the latent value $\widetilde V(k',b',z')<0$.
  
**Bond Pricing**

In equilibrium, the **risky rate** is determined by the **Lender's zero-profit condition**

$$ b'(1+r)= (1+\tilde r)b'\,\mathbb{E}_{z'|z}[1-D]+\mathbb{E}_{z'|z}[D\cdot R(k',b',z')] $$

where the LHS is the marginal cost of lending and the RHS is the expected marginal return to the risky bond after taking into account default probability and recovery from liquidation.

Let $D$ denote an indicator for default

$$D(z',k',b')=\mathbb{1}\{\widetilde V(z',k',b')<0\}$$

Then the **default probability** is defined as

$$p^D=\mathbb{E}_{z'|z}[D(z',k',b')]$$

**Recovery** under default is defined as

$$R(k',z')=(1-\alpha)\left[(1-\tau)\pi(k',z')+(1-\delta)k'\right]$$

where $\alpha\in[0,1]$ is the deadweight cost applied on the liquidation value of the firm.

---
# Implementation: Basic Model

## LR Method

**Policy Network**
$$k_{t+1}=\Gamma_{\text{policy}}(k_t,z_t;\theta_{\text{policy}})$$

**Objective**

The objective is to find the optimal $\theta$ that
$$
\max_{\theta} J(\theta_{\text{policy}})=\mathbb{E}\sum_{t=0}^{T-1}\beta^t e(k_t,\Gamma_{\text{policy}},z_t)
$$

or equivalently, 

$$\min_{\theta} -J(\theta_{\text{policy}})$$

**Empirical loss**

Given training dataset with $N$ i.i.d. rollout of firm's life over finite horizon $T$, the empirical objective is

$$\min_{\theta} \widehat{\mathcal{L}}^{\text{LR}}(\theta)=-\frac{1}{N}\sum_{i=1}^N\sum_{t=0}^{T-1}\beta^t e(k_t^{(i)},\Gamma_{\text{policy}}(\theta),z_t^{(i)})$$


**Training loop**

1. Initiate the economic parameters and the neural network hyperparameters
2. Construct parameterized policy $\Gamma(\theta)$ that maps $(k,z)\to k'$
3. Compute $\widehat{\mathcal{L}}^{\text{LR}}$ and update $\theta_{\text{policy}}$ to minimize empirical loss
4. Repeat step #3 until $\widehat{\mathcal{L}}^{\text{LR}}$ is stable (converged)

## ER Method

Notation: Let $F_x$ denote the partial derivative of function $F$ with respect to $x$.

The Lagrangian is set up as
$$
\max_{k_1, k_2, \dots, k_T} \mathbb{E}_{z_{t+1}|z_t} \left[ \sum^{T-1}_{t=0} 
\beta^t \cdot 
\big( e(I_t,z_t) - \chi_t \left( k_{t+1} - (1-\delta)k_t - I_t \right) \big)
\right]
$$
where $\chi_t$ is the Lagrange multiplier, or the shadow price of capital.

Taking derivatives with respect to $I_t$, $I_{t+1}$, and $k_{t+1}$ yields:

$$\begin{gather}
\chi_t = \psi_I(I_t,k_t) + 1 \\
\chi_{t+1} = \psi_I(I_{t+1},k_{t+1}) + 1 \\
\chi_t = \mathbb{E}_{z_{t+1}|z_t} \left[ 
    \beta \left(
        \pi_k (z_{t+1},k_{t+1}) - \psi_k (I_{t+1}, k_{t+1}) + (1 - \delta) \chi_{t+1}
    \right) 
\right]
\end{gather}$$

where 
- cash flow is $e(I_t,z_t) = \pi(z_t,k_t) - \psi(I_t,k_t) - I_t$
- investment is $I_t = k_{t+1} - (1-\delta)k_t$

Given **Policy Network** $k'=\Gamma_{\text{policy}}(k,z;\theta_{\text{policy}})$, investment is
$$
I_t=\Gamma_{\text{policy}}(k_t,z_t;\theta) - (1-\delta) k_t  
$$

**Euler equation** is derived by combining the FOCs:
$$
\mathbb{E}_{z_{t+1}|z_t}\!\left[\beta\Big(
\pi_k(z_{t+1},k_{t+1})-\psi_k(I_{t+1},k_{t+1})+(1-\delta)
\big(1+\psi_I(I_{t+1},k_{t+1})\big)
\Big)\right]=1+\psi_I(I_t,k_t).
$$
   
**Objective**

Define the inner component of the Euler Equation LHS as
$$
m(I_{t+1},k_{t+1}, z_{t+1}) \equiv \pi_k(z_{t+1},k_{t+1})-\psi_k(I_{t+1},k_{t+1})+(1-\delta)\chi_{t+1}
$$

The Euler residual is defined as
$$
f(I_{t+1},I_t,k_{t+1},k_t,z_t;\theta) 
\equiv 1 + \psi_I(I_t,k_t) - \beta\mathbb{E}_{z_{t+1}|z_t}[m(I_{t+1},k_{t+1}, z_{t+1})]
$$

The training objective is to minimize the square of Euler residual:
$$\min_{\theta} J^{\text{ER}}(\theta_{\text{policy}})=\mathbb{E}[f(\theta)]^2$$


**Empirical loss**

Note that $f(\theta)$ itself includes an expectation operator, so we cannot directly square it because $\mathbb{E}[f^2] \neq (\mathbb{E}[f])^2$. Instead, we use the All-in-One (AiO) estimator by taking two i.i.d. shocks $\varepsilon^{(1)},\varepsilon^{(2)}$ and by AR(1) we obtain the realized shocks, $z^{(1)}_{t+1},z^{(2)}_{t+1}$ given $z_t$.

Then the realized Euler residual (error) for each of them is computed as:
$$
f^{\ell}(\cdot,\theta) \equiv 1 + \psi_I(I_t,k_t) - \beta \cdot m(I_{t+1},k_{t+1}, z^{(\ell)}_{t+1}) \quad, \forall \ell=1,2
$$

The one-period empirical loss is the product of the two Euler residuals:

$$\widehat{\mathcal{L}}^{\text{ER}}_t=\frac{1}{N}\sum_{i=1}^N f^{(1)}_{t,i} f^{(2)}_{t,i}$$

Since our dataset is rolled out across finite horizon $T$, I further take the mean Euler residual across all $t$ and across $N$ observations (i.i.d. draws) to obtain the final empirical loss:

$$\widehat{\mathcal{L}}^{\text{ER}}=\frac{1}{N}\sum_{i=1}^N \left( \frac{1}{T}\sum_{t=0}^{T-1} f^{(1)}_{t,i} f^{(2)}_{t,i} \right)$$

where the inner parenthesis is the mean residual over firm $i$'s lifetime, and the outer sum take sample mean over $N$ firms in the training data. This is a consistent estimator for the theoretical loss because my data generator specifically simulate $z^{(2)}_{t+1}$ as a fork of the main $z^{(1)}_{t+1}$ conditional on the same $z_t$ for each period $t$.

**Training loop**
1. Initiate the economic parameters and the neural network hyperparameters
2. Combine parameterized policy $\Gamma(\theta)$ that maps $(k_t,z_t)\to k_{t+1}$
3. Compute implied investment $I_t = \Gamma(k_t,z_t;\theta) - (1-\delta) k_t$
4. Further compute one more step that maps $(k_{t+1} ,z^{\ell}_{t+1})$ to $k_{t+2}$, and compute the implied investment  $I_{t+1}=k_{t+2}-(1-\delta)k_{t+1}$
5. Compute $\widehat{\mathcal{L}}^{\text{ER}}$ and update $\theta_{\text{policy}}$ to minimize empirical loss
6. Repeat previous step until $|\widehat{\mathcal{L}}^{\text{ER}}|\lt \epsilon$ for some tolerance $\epsilon$

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

For each period $t$, the Bellman RHS target is defined as
$$y_{t}^{(\ell)}=e(k_t ,\Gamma_{\text{policy}},z_t)+\beta\,\Gamma_{\text{value}}(\Gamma_{\text{policy}},z^{\ell}_{t+1};\theta^{-}_{\text{value}}),\qquad \ell\in\{1,2\}$$

where the expectation is dropped given two realized shocks $z^{(1)}_{t+1},z^{(2)}_{t+1}$ (that were i.i.d in sampling). Our goal is to update our parameterized value network $\theta_{\text{value}}$ to minimize the "square" error from this critic target (label):
$$
\min_{\theta_{\text{value}}} \left( \Gamma_{\text{value}}(k_t,z_t;\theta_{\text{value}})-y_{t}^{(1)}\right)\left( \Gamma_{\text{value}}(k_t,z_t;\theta_{\text{value}})-y_{t}^{(2)}\right)
$$
where we use the AiO estimator to consistently estimate the squared expectation. 

This is a classic regression problem. Once the critic target $y_{t}^{(\ell)}$ is computed, it is passed over as a constant (fixed label). It is critical to detach gradients from it when updating $\theta_{\text{value}}$ and also keep the policy network $(\theta_{\text{policy}})$ fixed during the critic update.

**Empirical Loss**

Empirically, the BR-Critic Loss is defined as
$$\widehat{\mathcal{L}}^{\text{BR}}_{\text{critic}}(\theta_{\text{value}})=\frac{1}{N}\sum_{i=1}^N \left(
\frac{1}{T} \sum_{t=0}^{T-1} \delta_{t,i}^{(1)} \delta_{t,i}^{(2)}
\right) $$
where the Bellman residual (error) is
$$\delta_{t,i}^{(\ell)}=\Gamma_{\text{value}}(k_{t,i},z_{t,i};\theta_{\text{value}})-y_{t,i}^{(\ell)}$$
for each observation $i$ in period $t$ over finite horizon $T$.

**Critic Training Loop** 

1. Before each critic update step, 
   - Hold constant the current optimal policy $\theta_{\text{policy}}$ that determines $k' =\Gamma_{\text{policy}}$
   - Update $\theta^{-}_{\text{value}} = \nu \theta_{\text{value}} + (1-\nu) \theta^{-}_{\text{value}}$

2. Then proceed the critic update step:
   - Compute $\widehat V^{\text{next}}_{i,\ell}=\Gamma_{\text{value}}(\Gamma_{\text{policy}},z^{\ell}_{t+1};\theta^{-}_{\text{value}})$,
   - Detach gradient flow from RHS: $\widehat V^{\text{next}}_{i,\ell}\leftarrow \mathrm{stopgrad}(\widehat V^{\text{next}}_{i,\ell})$,
   - Compute $y_{i,\ell}=e(\cdot)+\beta\,\widehat V^{\text{next}}_{i,\ell}$ as a constant
   - Compute Bellman error $\delta_{t,i}^{(\ell)}$ in which  $\Gamma_{\text{value}}(k_{t,i},z_{t,i};\theta_{\text{value}})$ remains trainable.
   - Update $\theta_{\text{value}}$ to minimize $\widehat{\mathcal{L}}^\text{BR}_{\text{critic}}$
  
3. Repeat #1 and #2 for `N_critic_step` iterations (normally 5-10), then move to Actor update step

### Actor update step

For a given fixed policy $\theta_{\text{policy}}$, the previous critic update find the best $\theta_{\text{value}}$ that minimizes the critic loss. However, it is not guaranteed that this $\theta_{\text{policy}}$ is the optimal policy that maximize the Bellman equation. In other words, we need an additional training step to ensures that
$$
\theta_{\text{policy}} = \argmax_{k'=\Gamma(\theta_{\text{policy}})}\{e(k,k',z)+\beta \mathbb{E}_{z'}\left[V(k',z')\right]
$$

This is exactly the "direct optimization" idea described in @Maliar12. 

Empirically, I define the Actor loss as:
$$\widehat{\mathcal{L}}^{\text{BR}}_{\text{actor}}(\theta_{\text{policy}})
=-\frac{1}{N}\sum_{i=1}^N 
\frac{1}{T}\sum_{t=0}^{T-1} \left[e(k_{t,i},k_{t+1,i},z_{t,i})+\beta\cdot \frac{1}{2}\sum_{\ell=1}^2 \Gamma_{\text{value}}(k_{t+1,i},z_{t+1,i}^{(\ell)};\theta_{\text{value}})\right].$$
where it is helpful to clarify that
- Using either one of the draws $(z_{t+1,i}^{(1)}, z_{t+1,i}^{(2)})$ is still unbiased but noisier; averaging two draws reduces variance and remains correct.
- Note that this is NOT a AiO cross-product estimator because we are not handling squared expectations on the RHS.

**Actor Training Loop** 
1. Obtain the optimal $\theta_{value}$ fro $\Gamma_{value}$ from the previous critic update loop
2. Compute $k_{t+1}=\Gamma_{\text{policy}}(k_t,z_t;\theta_{\text{policy}})$
3. Compute Actor loss $\widehat{\mathcal{L}}^{\text{BR}}_{\text{actor}}$ using $k_{t+1}$ and $z^{(\ell)}_{t+1}$ with $\ell = 1,2$
4. Update $\theta_{\text{policy}}$ by one step to minimize $\widehat{\mathcal{L}}^{\text{BR}}_{\text{actor}}$ 

### Algorithm 
The complete Actor-Critic training algorithm is summarized as below
1. Initiate the economic parameters and the neural network hyperparameters
2. Initiate the policy network $\Gamma_{\text{policy}}$ and the value network $\Gamma_{\text{value}}$
3. Conduct a Critic update training loop to update $\theta_{value}$ that minimizes $\widehat{\mathcal{L}}^\text{BR}_{\text{critic}}$
4. Conduct an Actor update training loop to update $\theta_{policy}$ that minimizes $\widehat{\mathcal{L}}^\text{BR}_{\text{actor}}$
5. Repeat steps 3 and 4 for `N_iter` iterations, or until
   - Bellman residual is close to zero: $|\widehat{\mathcal{L}}^\text{BR}_{\text{critic}}| \lt \epsilon_{crit}$, and 
   - Optimal policy converged to a fixed point: $|\Delta \widehat{\mathcal{L}}^\text{BR}_{\text{actor}}| \lt \epsilon_{act}$

To summarize, the critic update ensures that for a given policy, the value network satisfied the Bellman equation. Given the optimal value network, the actor update tries to deviate from the current policy to check if there exist a better policy that would maximize the RHS of Bellman equation. Repeating this process until we find the fixed point for both of them.

---
# Implementation: Risky Debt Model

In the risky debt model, there are two optimality conditions that need to be enforced.

The first one is the **Bellman equation**:
$$\widetilde V(k,b,z)=\max_{k',b'}\left\{e(k,b,z;\tilde r)-\eta(e)+\beta\mathbb{E}_{z'}[V(k',b',z')]\right\}$$

where on the RHS the continuation value must be non-negative
$$V(k',b',z')=\max\{ 0, \, \widetilde V(k',b',z') \}$$

The second one is the **Lender's zero-profit condition** that determines $\tilde{r}$:

$$ b'(1+r)= (1+\tilde r)b'\,\mathbb{E}_{z'|z}[1-D]+\mathbb{E}_{z'|z}[D\cdot R(k',b',z')] $$

## Networks
Implementation of this model requires at least two neural networks.

**Policy network**
$$ (k',b') = \Gamma_{\text{policy}}(k,b,z;\theta_{\text{policy}}) $$

**Pricing network**
$$\tilde{r}(k',b',z) = \Gamma_{\text{price}}(k',b',z;\theta_{\text{price}})$$

On top of these, we will need an additional value network to apply the BR method.

**Value network**
$$\widetilde{V}(k,b,z) = \Gamma_{\text{value}}(k,b,z;\theta_{\text{value}})$$

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

## LR Method

**Policy networks**
$$(k_{t+1},b_{t+1})=\Gamma_{\text{policy}}(k_t,b_t,z_t;\theta_{\text{policy}}).$$

**Reward (cash flow)**
$$u_t=e(k_t,k_{t+1},b_t,b_{t+1},z_t)-\eta(e(\cdot)).$$


**Objective**

Let $\mathbf{\theta} \equiv (\theta_{price}, \theta_{policy})$. The objective of training is to find the optimal $\theta$ that jointly minimizes the negative lifetime reward $\mathcal{L}^{\text{LR}}$ and the pricing residual $\mathcal{L}^{\text{price}}$:

$$
\begin{align}
\mathcal{L}^{\text{LR}}(\theta) &= -\frac{1}{N}\sum_{i=1}^N\sum_{t=0}^{T-1}\beta^t u_t^{(i)} \\
\mathcal{L}^{\text{price}}(\theta) &= \frac{1}{N}\sum_{i=1}^N 
\left( \frac{1}{T}\sum_{t=1}^{T-1} f^{(1)}_{t,i}f^{(2)}_{t,i}\right)
\end{align}
$$

These two objectives can be combined into a single empirical risk:
$$
\mathcal{L}(\theta) \equiv \mathcal{L}^{\text{LR}} + \lambda_j (\mathcal{L}^{\text{price}} - \epsilon)
$$
where the Lagrange multiplier $\lambda_j$ is used to enforce the pricing residual to be close to zero by a tolerance $\epsilon$. After each step update of $\theta$ in iteration $j$, the multiplier is updated by
- Detach $\theta$ from $\mathcal{L}_j^{\text{price}}$ to prevent it from being updated
- Compute Polyak averaging $\mathcal{\bar L}_j^{\text{price}}= (1-w) \mathcal{\bar L}_{j-1}^{\text{price}} + w \mathcal{L}_j^{\text{price}}$ with weight $w\in(0,1)$
- Update $\lambda_j \leftarrow \max(0, \lambda_j + \eta_\lambda(\bar L_j^{\text{price}}-\epsilon))$ where $\eta_\lambda$ is a learning rate for the multiplier

The multiplier $\lambda_j \to 0$ when the constraint binds, i.e. the pricing equation residual is small enough $\mathcal{L}_j^{\text{price}} \lt \epsilon$, which ensures that the zero-profit condition holds.

**Training loop summary**
1.	Initialize economic parameters, training hyperparameters, model parameters $\theta$ (policy network $\Gamma_{\text{policy}}$ and pricing network $\Gamma_{\text{price}}$), and Lagrange multiplier(s) $\lambda \ge 0$.
2.	For $j = 1,\dots,N_{\text{iter}}$ during training:
  - Sample a minibatch $B_j$ from the training set
  - Compute $L^{LR}(\theta;B_j)$ (negative lifetime reward) and $L^{\text{price}}(\theta;B_j)$
  - Form the Lagrangian loss: $\mathcal{L}_j(\theta,\lambda)=L^{LR}(\theta;B_j)+\lambda\big(L^{\text{price}}(\theta;B_j)-\epsilon\big)$
  - Update $\theta$: take one optimizer step to minimize $\mathcal{L}_j$ via backpropagation
  - Update $\lambda$ (no backprop): using the detached batch estimate $\mathcal{\bar L}_j^{\text{price}}$ and update 
  $$\lambda \leftarrow \max\!\left(0,\ \lambda+\eta_\lambda(\mathcal{\bar L}_{j}^{\text{price}}-\epsilon)\right)$$
3.	Stop at $N_{\text{iter}}$ or early-stop when evaluation reward plateaus while $L_2$ stays below $\epsilon$; return the best feasible checkpoint

## ER Method

Underdevelopment. Coming soon.

## BR Method

  