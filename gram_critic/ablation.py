"""How much of the validation signal survives the Gram projection?

Three *oracle* deliveries (all use true labels) differ only in the representation they push
through, isolating what the order/rotation-invariant Gram can carry — each is a ``Regularizer``
built by ``reg.make_reg`` and run on top of weight decay:

  direct         push activations along −g_h          (full activation-space signal)
  gram_gradient  match Gram(h) toward Gram(h − g_h)   (only the rotation-invariant part survives)
  label_gram     match Gram(h) toward the label Gram  (the task's target geometry, Gram-native)

The story we expect: ``direct`` helps most and preserves rank; ``gram_gradient`` helps less (the
invariance floor) but still preserves rank; their gap is the cost of going through the Gram — the
same cost the amortized critic pays.
"""

from __future__ import annotations

from .config import AblationConfig
from .harness import run_condition


def run(cfg: AblationConfig, x_full, y_full, x_test, y_test, device: str):
    results = {}
    for corruption, l2_lambda in zip(cfg.corruptions, cfg.l2_lambdas):
        print(f"  corruption={corruption}  (l2_lambda={l2_lambda}, reg_lambda={cfg.reg_lambda})", flush=True)
        baseline = None
        for method in cfg.methods:
            acc, rank = run_condition(method, cfg=cfg, x_full=x_full, y_full=y_full, x_test=x_test,
                                      y_test=y_test, corruption=corruption, l2_lambda=l2_lambda,
                                      device=device)
            if method == "l2":
                baseline = acc
            delta = "" if method == "l2" else f"  (vs L2 {acc - baseline:+.3f})"
            print(f"    {method:<14}: acc={acc:.3f}  eff_rank={rank:.1f}{delta}", flush=True)
            results[(corruption, method)] = (acc, rank)
    return results
