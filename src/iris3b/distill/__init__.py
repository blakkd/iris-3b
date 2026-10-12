"""Few-step distillation of Iris-3B with Trajectory Distribution Matching (arXiv:2503.06674)."""

from iris3b.distill.config import DistillConfig, DMDConfig, load_distill_config

__all__ = ["DMDConfig", "DistillConfig", "load_distill_config"]
