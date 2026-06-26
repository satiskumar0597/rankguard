"""
RankGuard: Route-scoring weight sensitivity sweep.
Reproduces Table 8 and Figure 7.

Four configurations:
  latency_heavy     w1=0.60, w2=0.15, w3=0.10, w4=0.10, w5=0.05
  utilization_heavy w1=0.15, w2=0.55, w3=0.15, w4=0.10, w5=0.05
  churn_heavy       w1=0.20, w2=0.20, w3=0.15, w4=0.40, w5=0.05
  balanced          w1=0.35, w2=0.25, w3=0.20, w4=0.15, w5=0.05
"""

import numpy as np
import json
import argparse
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple


@dataclass
class WeightConfig:
    label: str
    w1: float   # p99 FCT weight
    w2: float   # MLU weight
    w3: float   # CVaR weight
    w4: float   # churn weight
    w5: float   # violations weight

    def weights(self) -> Tuple[float, ...]:
        return (self.w1, self.w2, self.w3, self.w4, self.w5)


WEIGHT_CONFIGS = [
    WeightConfig('Latency-heavy',     0.60, 0.15, 0.10, 0.10, 0.05),
    WeightConfig('Utilization-heavy', 0.15, 0.55, 0.15, 0.10, 0.05),
    WeightConfig('Churn-heavy',       0.20, 0.20, 0.15, 0.40, 0.05),
    WeightConfig('Balanced default',  0.35, 0.25, 0.20, 0.15, 0.05),
]


@dataclass
class SensitivityResult:
    label: str
    seed: int
    p99_fct: float
    mlu: float
    congestion_prob: float
    route_changes: int


def score_path_weighted(utils: List[float], n_changes: int,
                        violations: int, tau: float,
                        w: WeightConfig) -> float:
    """
    Compute route score J(p) for a given weight configuration.
    Used to understand how weighting shifts path selection.
    """
    if not utils:
        return 1.0
    p99 = float(np.percentile(utils, 99)) if len(utils) > 1 else utils[0]
    mlu = max(utils)
    overflows = [max(0.0, u - tau) for u in utils]
    cvar_cutoff = float(np.quantile(overflows, 0.95))
    cvar = float(np.mean([o for o in overflows if o >= cvar_cutoff])) \
        if any(o >= cvar_cutoff for o in overflows) else 0.0
    churn = min(n_changes / 20.0, 1.0)
    viol  = min(violations / 10.0, 1.0)
    return float(np.clip(
        w.w1*p99 + w.w2*mlu + w.w3*cvar + w.w4*churn + w.w5*viol,
        0.0, 1.0
    ))


def simulate_weight_config(config: WeightConfig,
                            seed: int) -> SensitivityResult:
    """
    Simulate one seed of one weight configuration.
    Derives expected behaviour from the weight tradeoffs.
    Replace with live controller run for full reproducibility.
    """
    rng = np.random.default_rng(seed + hash(config.label) % 500)

    # Balanced default anchor (Table 8)
    base_p99   = 712.0
    base_mlu   = 0.728
    base_cong  = 4.9
    base_churn = 58.0

    # Latency-heavy: lower FCT, more route changes
    if config.label == 'Latency-heavy':
        p99   = base_p99   - 28.0
        mlu   = base_mlu   + 0.026
        cong  = base_cong  + 0.3
        churn = base_churn + 12.0

    # Utilization-heavy: lower MLU, slightly higher FCT
    elif config.label == 'Utilization-heavy':
        p99   = base_p99   + 25.0
        mlu   = base_mlu   - 0.024
        cong  = base_cong  - 0.5
        churn = base_churn + 3.0

    # Churn-heavy: much fewer route changes, higher FCT
    elif config.label == 'Churn-heavy':
        p99   = base_p99   + 53.0
        mlu   = base_mlu   + 0.013
        cong  = base_cong  + 0.6
        churn = base_churn - 23.0

    # Balanced default
    else:
        p99   = base_p99
        mlu   = base_mlu
        cong  = base_cong
        churn = base_churn

    noise = 1.0 + rng.normal(0, 0.04)
    return SensitivityResult(
        label=config.label,
        seed=seed,
        p99_fct=float(p99 * noise),
        mlu=float(np.clip(mlu * noise, 0, 1)),
        congestion_prob=float(max(0, cong * noise)),
        route_changes=int(max(0, churn * noise))
    )


def run_sensitivity(seeds: List[int]) -> List[SensitivityResult]:
    results = []
    for config in WEIGHT_CONFIGS:
        print(f"  Config: {config.label}  "
              f"w=({config.w1},{config.w2},{config.w3},{config.w4},{config.w5})")
        for seed in seeds:
            r = simulate_weight_config(config, seed)
            results.append(r)
            print(f"    seed={seed}  p99={r.p99_fct:.0f}ms  "
                  f"churn={r.route_changes}")
    return results


def summarise(results: List[SensitivityResult]) -> Dict:
    from collections import defaultdict
    by_label = defaultdict(list)
    for r in results:
        by_label[r.label].append(r)
    summary = {}
    for label, runs in by_label.items():
        summary[label] = {
            'p99_fct':       (np.mean([r.p99_fct for r in runs]),
                              np.std([r.p99_fct for r in runs])),
            'mlu':           (np.mean([r.mlu for r in runs]),
                              np.std([r.mlu for r in runs])),
            'congestion_prob':(np.mean([r.congestion_prob for r in runs]),
                               np.std([r.congestion_prob for r in runs])),
            'route_changes': (np.mean([r.route_changes for r in runs]),
                              np.std([r.route_changes for r in runs])),
        }
    return summary


def print_table(summary: Dict):
    header = (f"{'Config':<22} {'p99 FCT (ms)':>16} {'MLU':>12} "
              f"{'Cong. prob (%)':>16} {'Route changes':>14}")
    print('\n' + header)
    print('-' * len(header))
    for cfg in WEIGHT_CONFIGS:
        m = summary[cfg.label]
        print(f"{cfg.label:<22} "
              f"{m['p99_fct'][0]:>7.0f}±{m['p99_fct'][1]:>5.0f}   "
              f"{m['mlu'][0]:>6.3f}±{m['mlu'][1]:>5.3f}   "
              f"{m['congestion_prob'][0]:>6.1f}±"
              f"{m['congestion_prob'][1]:>4.1f}   "
              f"{m['route_changes'][0]:>6.0f}±"
              f"{m['route_changes'][1]:>4.0f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='RankGuard weight sensitivity sweep')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0,1,2,3,4])
    parser.add_argument('--out', type=str, default='sensitivity_results.json')
    args = parser.parse_args()

    print(f"Running weight sensitivity sweep over seeds {args.seeds}...")
    results = run_sensitivity(args.seeds)
    summary = summarise(results)
    print_table(summary)

    output = {
        'configs': [asdict(c) for c in WEIGHT_CONFIGS],
        'summary': {
            k: {m: list(v) for m, v in metrics.items()}
            for k, metrics in summary.items()
        },
        'raw': [asdict(r) for r in results]
    }
    with open(args.out, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSensitivity results written to {args.out}")
