## Unified Mathematical Framework for FAT

TL;DR — A short, consistent mathematical language that maps directly to the code: coefficients = functions, FAT = parametric map on coefficient sequences, $\Pi$ = Chebyshev-only stabilizer, and the approval loss aligns step directions.

### 1. Core objects and notation
- $H$: Hilbert space of scalar functions on the domain (e.g. $L^2([-1,1],w])$).
- $\{\phi_n\}_{n\ge0}$: chosen orthonormal basis in $H$ (in code: Chebyshev polynomials).
- Truncate at order $K$: $V_K=\mathrm{span}\{\phi_0,\dots,\phi_K\}\subset H$. Coefficients $\Theta\in\mathbb{R}^{K+1}$.

Analysis (coefficients):
$$
\operatorname{coeff}_K(f) = (\langle f,\phi_0\rangle_H,\dots,\langle f,\phi_K\rangle_H)^\top \in \mathbb{R}^{K+1}.
$$
Synthesis (reconstruction):
$$
\operatorname{synth}_K(\Theta)(x)=\sum_{n=0}^K \Theta_n\,\phi_n(x).
$$

### 2. Model (one-step autoregression)
Given a context length $L$, define the context
$$
C_t=(\Theta_{t-L+1},\dots,\Theta_t)\in(\mathbb{R}^{K+1})^L.
$$
The FAT model is a parametric map
$$
G_\psi:(\mathbb{R}^{K+1})^L\to\mathbb{R}^{K+1},\qquad \Theta_{t+1}^{\mathrm{raw}}=G_\psi(C_t).
$$
In code the `SpectralHead` provides the linear encoder/decoder between coefficient and embedding space (see `src/FAT/phase2_spectral_head.jl`).

### 3. Stabilizer (Chebyshev-only, non-expansive)
Two projections in the same coefficient/Hilbert space:

- $P_k$: local polynomial anchor (truncate/orthogonally project to order $k$).
- $P_p$: Chebyshev economizer (orthogonal projection onto $V_p$, $p\le k$).

Define
$$
\Pi = P_p\circ P_k.
$$
Since $P_k,P_p$ are orthogonal projections in the same inner-product space, each has operator norm $\le1$, hence
$$
\|\Pi(x)-\Pi(y)\|\le\|x-y\|\quad\forall x,y\in\mathbb{R}^{K+1}.
$$
Practical consequence: applying $\Pi$ cannot amplify errors introduced by $G_\psi$.

### 4. Loss: directional approval (with MSE warmup)
One-step displacements:
$$
\Delta^*_t=\Theta^*_{t+1}-\Theta_t,\qquad\Delta^{\mathrm{pred}}_t=\Theta^{\mathrm{pred}}_{t+1}-\Theta_t.
$$
Unitized directions (numerical floor $\varepsilon>0$):
$$
\nu^*_t=\frac{\Delta^*_t}{\|\Delta^*_t\|+\varepsilon},\qquad\nu^{\mathrm{pred}}_t=\frac{\Delta^{\mathrm{pred}}_t}{\|\Delta^{\\mathrm{pred}}_t\|+\varepsilon}.
$$
Approval (cosine): $A_t=\langle\nu^{\mathrm{pred}}_t,\nu^*_t\rangle\in[-1,1]$. Flow loss:
$$
\mathcal{L}_{\mathrm{flow}}=1-\mathbb{E}_t[A_t].
$$
Blend with MSE via warmup weight $\alpha_t\in[0,1]$:
$$
\mathcal{L}_t=\alpha_t\,\mathcal{L}_{\mathrm{flow}}+(1-\alpha_t)\,\mathrm{MSE}(\Theta^{\mathrm{pred}}_{t+1},\Theta^*_{t+1}).
$$
(See `directional_trajectory_loss` and `approval_alpha_schedule` in `src/FAT/phase4_loss.jl`.)

### 5. Training & generation (compact)
Training objective:
$$
\min_{\psi}\;\mathbb{E}_{\text{data}}[\mathcal{L}_t]+\lambda R(\psi),
$$
optimized with AD (Zygote), AdamW, and gradient clipping (see `src/FAT/phase5_fat_model.jl`).

Generation (one-step):
$$
\Theta_{t+1}=\Pi\big(G_\psi(\Theta_{t-L+1:t})\big),\quad\text{iterate for }t\ge L.
$$

### 6. Intuition for an ML/AI master's student
- Predict coefficients (short vectors) instead of discrete tokens — richer inductive bias for continuous data.
- $\Pi$ projects the raw prediction into a low-order, stable subspace; because it's non-expansive, it prevents stepwise error blow-up.
- The approval loss aligns the *direction* of predicted updates with true updates, making learning robust to scale and focusing on correct dynamics.

### 7. Compact propositions
1. (Non-expansivity) $\|\Pi(x)-\Pi(y)\|\le\|x-y\|$.
2. (Stability bound) For $J=\Pi\circ G_\psi$, $L_J\le L_{G_\psi}$ (operator/Lipschitz bound).

### 8. Quick code map & experiments
- Phase1: basis & helpers — `src/FAT/phase1_hilbert.jl`
- Phase2: spectral head — `src/FAT/phase2_spectral_head.jl`
- Phase3: stabilizer — `src/FAT/phase3_stabilizer.jl`
- Phase4: loss & checks — `src/FAT/phase4_loss.jl` (see `check_loss_gradient`)
- Phase5: model & training — `src/FAT/phase5_fat_model.jl`

Suggested tests: numeric Lipschitz check of $\Pi$, `check_loss_gradient()`, ablation with/without $\Pi$.

---

Saved to `/memories/session/plan.md` (refined). Please confirm where to publish this (README/docs).
