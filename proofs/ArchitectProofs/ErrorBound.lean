import Mathlib.Analysis.InnerProductSpace.PiL2

open scoped BigOperators

variable {E : Type*} [NormedAddCommGroup E]

theorem error_growth_bound
    (f : E → E)
    (hf : ∀ a b, ‖f a - f b‖ ≤ ‖a - b‖)
    (x y : ℕ → E)
    (e : ℕ → E)
    (epsilon : ℕ → ℝ)
    (hx : ∀ n, x (n + 1) = f (x n))
    (hy : ∀ n, y (n + 1) = f (y n) + e (n + 1))
    (he : ∀ n, ‖e (n + 1)‖ ≤ epsilon (n + 1))
    (h0 : y 0 = x 0) :
    ∀ n, ‖y n - x n‖ ≤ ∑ i in Finset.range n, epsilon (i + 1) := by
  intro n
  induction n with
  | zero =>
    simp [h0]
  | succ n ih =>
    rw [hx n, hy n]
    rw [Finset.sum_range_succ]
    have h_tri : ‖f (y n) + e (n + 1) - f (x n)‖ ≤ ‖f (y n) - f (x n)‖ + ‖e (n + 1)‖ := by
      have : f (y n) + e (n + 1) - f (x n) = (f (y n) - f (x n)) + e (n + 1) := by abel
      rw [this]
      apply norm_add_le
    apply le_trans h_tri
    have h_step : ‖f (y n) - f (x n)‖ + ‖e (n + 1)‖ ≤ ‖y n - x n‖ + epsilon (n + 1) := by
      apply add_le_add (hf (y n) (x n)) (he n)
    apply le_trans h_step
    apply add_le_add_right ih
