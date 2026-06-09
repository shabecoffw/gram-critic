# Results

The raw run logs behind the README's tables and figures, so every number is checkable without
re-running the pipeline. Each was produced by a script in [`../experiments/main/`](../experiments/main)
(or the `make` pipeline) on an Apple-MPS machine, 3 seeds unless noted.

| log | backs (README section) | produced by |
| --- | --- | --- |
| `baseline.log` | **Results** — no-regularizer vs tuned-L2 baseline (0.253 / 0.356 @ c=0.9) | `harness.run_condition("l2", l2_lambda=0)` |
| `deploy.log` | Ablations — the Frobenius collapse table (amortized +0.102 @rank1.5) | `make deploy DEPLOY_CONFIG=configs/deploy_c09_frobenius.yaml` |
| `amortized_routing.log` | **Results** — headline log-Euclidean critic (+0.094 @rank25 / +0.031 @rank13) | `experiments/main/amortized_routing.py` |
| `gram_routing.log` | Routing — oracle Frobenius vs log-Euclidean (+0.213→+0.548) | `experiments/main/gram_routing.py` |
| `metric_geometry.log` | Routing / What I learned — full metric sweep (frobenius / power / logeuclid / frob+logdet) | `experiments/main/metric_geometry.py` |
| `logeuclid_control.log` | Ablations — the **control** under the headline metric (random ΔK +0.050 vs critic +0.094 @ c=0.9) | `experiments/main/logeuclid_control.py` |
| `gram_rule_zoo.log` | Heuristic Gram-rule zoo — handcrafted baselines (vicreg +0.158, …) | `experiments/main/gram_rule_zoo.py` |
| `ablation.log` | Ablation — what survives the Gram (direct / gram_gradient / label_gram) | `make ablation` |
| `critic_capacity2.log` | What I learned — the information ceiling (d=32 0.201, d=64 0.216) | `experiments/main/critic_capacity2.py` |
| `critic_depth.log` | What I learned — depth gives only +0.012 | `experiments/main/critic_depth.py` |
| `critic_structured.log` | What I learned / Limitations — the exact structured head underperforms | `experiments/main/critic_structured.py` |
| `anti_collapse.log` | Ablations — anti-collapse restores rank and removes the Frobenius win | `experiments/main/anti_collapse.py` |

Numbers carry ~±0.01 run-to-run spread across the 3 seeds; the README quotes the conservative value
where logs disagree (e.g. the log-Euclidean critic at c=0.9 appears as +0.094 / +0.101 / +0.104).
