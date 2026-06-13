import ArchitectProofs.Stabilizer
import ArchitectProofs.ErrorBound

open scoped BigOperators

variable {d : ℕ}

/--
Rollout error bound for the corrected FAT stabilizer.

If the clean orbit evolves by the Chebyshev-only stabilizer and the perturbed
orbit evolves by the same stabilizer plus bounded per-step errors, then the
coefficient-space rollout error grows at most linearly with the accumulated
perturbation budget.
-/
theorem stabilizer_rollout_error_bound
    (p k : ℕ)
    (hpk : p ≤ k)
    (x y : ℕ → EuclideanSpace ℝ (Fin d))
    (e : ℕ → EuclideanSpace ℝ (Fin d))
    (epsilon : ℕ → ℝ)
    (hx : ∀ n, x (n + 1) = stabilizer p k (x n))
    (hy : ∀ n, y (n + 1) = stabilizer p k (y n) + e (n + 1))
    (he : ∀ n, ‖e (n + 1)‖ ≤ epsilon (n + 1))
    (h0 : y 0 = x 0) :
    ∀ n, ‖y n - x n‖ ≤ ∑ i in Finset.range n, epsilon (i + 1) := by
  apply error_growth_bound (stabilizer p k)
  · intro a b
    exact lipschitz_stabilizer p k hpk a b
  · exact hx
  · exact hy
  · exact he
  · exact h0
