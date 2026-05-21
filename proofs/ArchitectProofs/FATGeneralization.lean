import ArchitectProofs.Stabilizer

theorem truncate_one_identity (v : EuclideanSpace ℝ (Fin 1)) :
    truncate 1 v = v := by
  ext i
  rw [truncate_apply, if_pos i.is_lt]
