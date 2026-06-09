"""Gram Critic — amortizing the validation signal into an activation-geometry regularizer."""

from .critic import GramCritic
from .gram import activation_gradient, effective_rank, gram, label_gram
from .harness import evaluate, train
from .models import MLP
from .reg import Regularizer
from .regularizers import GramMatchReg, RuleReg, WeightDecay, make_reg

__all__ = [
    "GramCritic",
    "MLP",
    "train",
    "evaluate",
    "Regularizer",
    "WeightDecay",
    "RuleReg",
    "GramMatchReg",
    "make_reg",
    "gram",
    "label_gram",
    "effective_rank",
    "activation_gradient",
]
