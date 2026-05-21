# Lean 4 Formal Verification of FAT Core Theorems

This directory contains the formalization and machine-checked proofs of the three central mathematical claims of the Functional Autoregressive Transformer (FAT) in Lean 4 + Mathlib.

## Proofs Overview

The proofs are structured into three main modules under `ArchitectProofs/`:

### 1. Proof 1: Stabilizer Non-Expansivity (`ArchitectProofs/Stabilizer.lean`)
We formalized the corrected stabilizer pipeline $\Pi = P_p \circ P_k$ where $0 \le p \le k \le K$. The stabilizer acts entirely in a unified Hilbert/coefficient space $H_w = \text{EuclideanSpace } \mathbb{R}\ (\text{Fin } n)$.
* **Definition**: `truncate k v` zeroes out coefficients corresponding to degree indices $\ge k$.
* **Core Lemmas**:
  * `norm_truncate_le`: Proves $\|P_k(v)\|_2 \le \|v\|_2$ using the sum of squares formulation from `EuclideanSpace.norm_eq`.
  * `truncate_sub`: Proves $P_k(u - v) = P_k(u) - P_k(v)$ via component-wise linearity.
  * `lipschitz_truncate`: Proves $\|P_k(u) - P_k(v)\|_2 \le \|u - v\|_2$.
  * `truncate_comp`: Proves $P_p \circ P_k = P_p$ for $p \le k$ via `omega` tactic logic.
* **Main Theorem**: `lipschitz_stabilizer` proves that the composite stabilizer $\Pi$ is non-expansive:
  $$\|\Pi(u) - \Pi(v)\|_2 \le \|u - v\|_2$$

### 2.  Proof 2: Linear Error Growth (`ArchitectProofs/ErrorBound.lean`)
We formalized the error growth bound under orbit perturbations of a non-expansive map.
* **Setting**: General `NormedAddCommGroup E`.
* **Main Theorem**: `error_growth_bound` proves that for a non-expansive map $f$, a clean orbit $x^{n+1} = f(x^n)$, and a perturbed orbit $y^{n+1} = f(y^n) + e_{n+1}$ with $\|e_{n+1}\| \le \varepsilon_{n+1}$:
  $$\|y^n - x^n\| \le \sum_{i=1}^n \varepsilon_i$$
  The proof is conducted via induction on $n$, using the triangle inequality and map non-expansivity.

### 3. Proof 3: Degeneration to Discrete LMs (`ArchitectProofs/FATGeneralization.lean`)
We proved that when $K = 0$ (represented by a 1-dimensional coefficient space), the stabilizer is the identity map.
* **Main Theorem**: `truncate_one_identity` proves that for $v : \text{EuclideanSpace } \mathbb{R}\ (\text{Fin } 1)$:
  $$\text{truncate } 1\ v = v$$
  This confirms that when the order is 0 (only constant functions are modeled), the stabilizer acts as the identity operator, meaning the system degenerates back to standard discrete language models.

## Technical Setup & Workarounds

1. **Lean Versioning**: Pinned to Lean `v4.16.0` (via `lean-toolchain`) and Mathlib `v4.16.0` (via `lakefile.toml`).
2. **Unix-Portable Compiler Wrapper (`leanc_wrapper.sh`)**:
   On modern macOS versions (16+), `dyld` rejects binaries built by Lean that lack the `SG_READ_ONLY` attribute on `__DATA_CONST` segments. This causes tools like `lake exe cache get` or `lake build` to fail during link/execution steps on macOS.

   To resolve this portably for all Unix-based systems (Linux, macOS, etc.) without altering any files inside `.lake/packages/` (retaining 100% cache compatibility):
   * We created a dynamic compiler wrapper: [leanc_wrapper.sh](file:///Users/souheybboularak/Desktop/Juli/Juli/proofs/leanc_wrapper.sh)
   * The wrapper runs `lean --print-prefix` to locate the active toolchain compiler dynamically.
   * It checks `uname` to detect the OS. It only appends `-Wl,-no_data_const` when compiling on macOS (`Darwin`) and performing a link step. On Linux and other Unix-based systems, it defaults to standard `leanc` invocation.

3. **Building the Proofs**:
   To compile and verify the Lean proofs on any macOS or Linux system, prefix the build command with `LEAN_CC`:
   ```bash
   LEAN_CC=./leanc_wrapper.sh lake build
   ```
