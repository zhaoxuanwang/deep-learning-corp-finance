This section reviews the canonical basic model of optimal investment, 

### Definitions

**State Space**:
State variables are capital $k$ and productivity shock $z$, which is stacked into a single state vector:

$$s=\left( k, z\right)^\top$$

with exogenous bounded space $[k_{\min}, k_{\max}]$ and $[z_{\min}, z_{\max}]$.

**Action Space**:
Action variable is investment $a \equiv I = k' -(1-\delta) k $ that can be either positive or negative. The state spaces $k_{\min} \leq k' \leq k_{\max}$ implies the investment is bounded by $[I_{\min}, I_{\max}]$:

$$
\begin{aligned}
I_{\min} &= k_{\min}-(1-\delta)k_{\max}\\
I_{\max} &= k_{\max} - (1-\delta) k_{\min}
\end{aligned}
$$

For example, when $k_{\min} =0$, investment $I\in [-(1-\delta)k_{\max}, k_{\max}]$.


**State Transition Function.** The productivity shocks follow auto-regression (AR-1) process with stationary mean $\mu$, persistent coefficient $\rho$, and variance $\sigma$:
$$\log z' = \mu +\rho \log z + \sigma \epsilon, \quad \epsilon\sim \mathcal{N}(0,1)$$
Capital stock depends on depreciation rate $\delta$ and investment $I$ (action):
$$ k'=(1-\delta)k+I$$
They are stacked into a single vector-valued state transition function:
$$f(s,a,\epsilon) \equiv 
\begin{pmatrix} z' \\ k' \end{pmatrix} = 
\begin{pmatrix}
\exp \left\{ \mu +\rho \log z + \sigma \epsilon \right\}\\
(1-\delta)k+I
\end{pmatrix}$$
**Reward** is defined by the net cash flow $e(k,z,I)$:
$$r(s,a) \equiv e(k,z,I) = \Pi(k,z) - \Psi(I,k) - I$$
where the production function is Cobb-Douglas with parameter $\beta$:
$$\Pi(k,z) = z \cdot k^{\beta}, \quad \beta \in (0,1)$$
and the capital adjustment cost is
$$\Psi(I,k) = \phi_0 \cdot \frac{I^2}{2k} + \phi_1 \cdot k \cdot \mathbf{1}\{I \neq 0\}$$
where $\phi_0$ is the smooth adjustment cost coefficient and $\phi_1$ is the fixed adjustment cost coefficient. The indicator $\mathbf{1}\{I \neq 0\}$ triggers whenever the firm invests or disinvests.

**Objective**: The solution of the model is to find the optimal investment policy that maximizes expected discounted lifetime cash flows:
$$\max_{\{I_{t}\}_{t=0}^{\infty}} \mathbb{E} \left[ \sum_{t=0}^{\infty} \gamma^t \cdot r(k_t, z_t, I_t) \right]$$
where $\gamma = 1/(1+ \bar r)$ is the discount factor and $\bar r$ is the risk-free interest rate. The solution is an optimal parameterized policy function $\pi^*_\theta(s)$ that maps states $(k,z)$ to investment $I$.

### Network Architecture

**Policy Network (All methods)** $\quad k' = \pi_\theta \equiv \pi(k, z;; \theta_{\text{policy}})$

- Input: $(k, z)$ in raw levels, normalized internally
- Hidden layers (default): 2 layers with 32 units each, SiLU activation
- Output: linear identity with clip $[k_{\min},; k_{\max}]$

**Critic Network (MVE only)** $\quad Q_\phi \equiv Q(k, z, k'; \phi)$

- Input: $(k, z, k')$ in raw levels, normalized internally
- Hidden layers (default): 2 layers with 32 units each, SiLU activation
- Raw output: $\hat{y}_\phi \approx \operatorname{symlog}(Q)$, linear identity without clip
- Level-space recovery: $Q_\phi = \operatorname{symexp}(\hat{y}_\phi)$, used in actor objective and terminal bootstrap

**Critic Network (Multitask-BR only)** $\quad V_\phi \equiv V(k, z; \phi)$

- Input: $(k, z)$ in raw levels, normalized internally
- Hidden layers (default): 2 layers with 32 units each, SiLU activation
- Raw output: linear identity $V$ without clip 
- Note: I follow the specification from @maliar21 and @fern26 and do not apply value normalization
