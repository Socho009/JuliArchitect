import Mathlib.Analysis.InnerProductSpace.PiL2
import Mathlib.Analysis.SpecialFunctions.Sqrt

open Real
open scoped BigOperators

variable {n : ℕ}

/-- Truncate operator: sets elements beyond index k to 0. -/
def truncate (k : ℕ) (v : EuclideanSpace ℝ (Fin n)) : EuclideanSpace ℝ (Fin n) :=
  (WithLp.equiv 2 (Fin n → ℝ)).symm (fun i => if (i : ℕ) < k then (WithLp.equiv 2 (Fin n → ℝ)) v i else 0)

theorem truncate_apply (k : ℕ) (v : EuclideanSpace ℝ (Fin n)) (i : Fin n) :
    (truncate k v) i = if (i : ℕ) < k then v i else 0 := by
  rfl

theorem norm_truncate_le (k : ℕ) (v : EuclideanSpace ℝ (Fin n)) :
    ‖truncate k v‖ ≤ ‖v‖ := by
  rw [EuclideanSpace.norm_eq (truncate k v), EuclideanSpace.norm_eq v]
  apply Real.sqrt_le_sqrt
  apply Finset.sum_le_sum
  intro i _
  rw [truncate_apply]
  split_ifs with h
  · rfl
  · simp only [norm_zero, zero_pow (by norm_num)]
    positivity

theorem truncate_sub (k : ℕ) (u v : EuclideanSpace ℝ (Fin n)) :
    truncate k (u - v) = truncate k u - truncate k v := by
  ext i
  simp only [truncate_apply, PiLp.sub_apply]
  split_ifs with h
  · rfl
  · simp only [sub_zero]

theorem lipschitz_truncate (k : ℕ) (u v : EuclideanSpace ℝ (Fin n)) :
    ‖truncate k u - truncate k v‖ ≤ ‖u - v‖ := by
  rw [← truncate_sub]
  apply norm_truncate_le

theorem truncate_comp (p k : ℕ) (hpk : p ≤ k) (v : EuclideanSpace ℝ (Fin n)) :
    truncate p (truncate k v) = truncate p v := by
  ext i
  simp only [truncate_apply]
  split_ifs with h1 h2
  · rfl
  · omega
  · rfl
  · rfl

/-- Composite projection operator Π = P_p ∘ P_k -/
def stabilizer (p k : ℕ) (v : EuclideanSpace ℝ (Fin n)) : EuclideanSpace ℝ (Fin n) :=
  truncate p (truncate k v)

theorem lipschitz_stabilizer (p k : ℕ) (hpk : p ≤ k) (u v : EuclideanSpace ℝ (Fin n)) :
    ‖stabilizer p k u - stabilizer p k v‖ ≤ ‖u - v‖ := by
  simp only [stabilizer, truncate_comp p k hpk]
  apply lipschitz_truncate
