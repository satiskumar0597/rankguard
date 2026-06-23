"""
RankGuard: Evaluation script.
Computes p99 FCT, MLU, congestion probability, route churn,
and action-rank agreement across policies and seeds.
Implements the statistical comparison (paired Wilcoxon) from the paper.
"""

import numpy as np
import json
import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from scipy import stats


# ── Flow completion time tracker ──────────────────────────────────────────────

@dataclass
class FlowRecord:
    flow_id: str
    start_time: float
    end_time: float = -1.0
    size_bytes: int = 0
    flow_type: str = 'mice'

    @property
    def fct(self) -> float:
        if self.end_time < 0:
            return -1.0
        return self.end_time - self.start_time


class FCTTracker:
    def __init__(self):
        self._flows: Dict[str, FlowRecord] = {}

    def start(self, flow_id: str, t: float, size: int, ftype: str = 'mice'):
        self._flows[flow_id] = FlowRecord(flow_id, t, size_bytes=size,
                                          flow_type=ftype)

    def finish(self, flow_id: str, t: float):
        if flow_id in self._flows:
            self._flows[flow_id].end_time = t

    def fcts(self) -> List[float]:
        return [f.fct for f in self._flows.values() if f.fct >= 0]

    def p99(self) -> float:
        fcts = self.fcts()
        return float(np.percentile(fcts, 99)) if fcts else 0.0

    def p95(self) -> float:
        fcts = self.fcts()
        return float(np.percentile(fcts, 95)) if fcts else 0.0

    def mean(self) -> float:
        fcts = self.fcts()
        return float(np.mean(fcts)) if fcts else 0.0


# ── Link utilization tracker ──────────────────────────────────────────────────

class LinkUtilTracker:
    def __init__(self, tau: float = 0.8):
        self.tau = tau
        self._samples: Dict[str, List[float]] = {}

    def record(self, link_id: str, util: float):
        if link_id not in self._samples:
            self._samples[link_id] = []
        self._samples[link_id].append(float(np.clip(util, 0.0, 1.0)))

    def mlu(self) -> float:
        if not self._samples:
            return 0.0
        return max(max(v) for v in self._samples.values() if v)

    def congestion_probability(self) -> float:
        """Fraction of (link, window) pairs where util > tau."""
        total, congested = 0, 0
        for samples in self._samples.values():
            for u in samples:
                total += 1
                if u > self.tau:
                    congested += 1
        return congested / max(total, 1)

    def cvar_overflow(self, alpha: float = 0.95) -> float:
        all_u = [u for v in self._samples.values() for u in v]
        overflows = [max(0.0, u - self.tau) for u in all_u]
        if not overflows or max(overflows) == 0:
            return 0.0
        cutoff = np.quantile(overflows, alpha)
        tail = [o for o in overflows if o >= cutoff]
        return float(np.mean(tail))


# ── Action-rank agreement ─────────────────────────────────────────────────────

def spearman_rank_agreement(pred_scores: List[float],
                             eval_scores: List[float]) -> float:
    """
    Spearman rank correlation between predictor and evaluator action rankings.
    Returns rho in [-1, 1]; 1.0 = perfect agreement.
    """
    if len(pred_scores) < 2:
        return 1.0
    rho, _ = stats.spearmanr(pred_scores, eval_scores)
    return float(rho)


def top1_agreement(pred_scores: List[float],
                   eval_scores: List[float]) -> float:
    """Fraction of windows where predictor and evaluator agree on top-1 action."""
    if not pred_scores:
        return 0.0
    return float(np.argmin(pred_scores) == np.argmin(eval_scores))


# ── Baseline policies ─────────────────────────────────────────────────────────

class ECMPPolicy:
    """Static ECMP: no rerouting, hash-based path selection."""
    name = 'ECMP'

    def decide(self, flow_id: str, candidates, forecasts, utils) -> str:
        return 'keep'


class ReactiveThresholdPolicy:
    """Reactive: reroute when measured utilization exceeds threshold."""

    def __init__(self, tau: float = 0.8):
        self.tau = tau
        self.name = 'Reactive threshold'

    def decide(self, flow_id: str, candidates, forecasts,
               utils: Dict[str, float]) -> str:
        if any(u > self.tau for u in utils.values()):
            return 'install'
        return 'keep'


class HybridForecastThresholdPolicy:
    """Hybrid: HW-GRU forecast + threshold trigger (strongest baseline)."""

    def __init__(self, tau: float = 0.8, margin: float = 0.05):
        self.tau = tau
        self.margin = margin
        self.name = 'Hybrid forecast-threshold TE'

    def decide(self, flow_id: str, candidates, forecasts: Dict[str, float],
               utils: Dict[str, float]) -> str:
        if any(f > self.tau - self.margin for f in forecasts.values()):
            return 'install'
        return 'keep'


# ── Metrics aggregator ────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    policy: str
    seed: int
    avg_fct: float
    p95_fct: float
    p99_fct: float
    mlu: float
    congestion_prob: float
    cvar_overflow: float
    route_changes: int
    action_rank_agreement: float = 0.0


def aggregate_results(results: List[EvalResult]) -> Dict[str, Dict]:
    """Compute mean ± std across seeds for each policy."""
    from collections import defaultdict
    by_policy = defaultdict(list)
    for r in results:
        by_policy[r.policy].append(r)

    summary = {}
    for policy, runs in by_policy.items():
        summary[policy] = {
            'avg_fct':   (np.mean([r.avg_fct for r in runs]),
                          np.std([r.avg_fct for r in runs])),
            'p95_fct':   (np.mean([r.p95_fct for r in runs]),
                          np.std([r.p95_fct for r in runs])),
            'p99_fct':   (np.mean([r.p99_fct for r in runs]),
                          np.std([r.p99_fct for r in runs])),
            'mlu':       (np.mean([r.mlu for r in runs]),
                          np.std([r.mlu for r in runs])),
            'cong_prob': (np.mean([r.congestion_prob for r in runs]),
                          np.std([r.congestion_prob for r in runs])),
            'route_chg': (np.mean([r.route_changes for r in runs]),
                          np.std([r.route_changes for r in runs])),
        }
    return summary


def paired_wilcoxon(results_a: List[EvalResult],
                    results_b: List[EvalResult],
                    metric: str = 'p99_fct') -> float:
    """
    Paired Wilcoxon signed-rank test between two policies across seeds.
    Returns p-value.
    """
    a_vals = [getattr(r, metric) for r in results_a]
    b_vals = [getattr(r, metric) for r in results_b]
    if len(a_vals) < 2:
        return 1.0
    _, pval = stats.wilcoxon(a_vals, b_vals)
    return float(pval)


# ── Telemetry-loss stress test ────────────────────────────────────────────────

def inject_telemetry_loss(telemetry: Dict[str, float],
                          loss_rate: float,
                          rng: np.random.Generator) -> Dict[str, float | None]:
    """Randomly drop telemetry samples at the given loss rate."""
    return {
        link_id: (None if rng.random() < loss_rate else val)
        for link_id, val in telemetry.items()
    }


def reconstruction_fidelity(original: List[float],
                             repaired: List[float]) -> Dict[str, float]:
    """Compute fidelity metrics between original and repaired series."""
    orig = np.array(original)
    rep = np.array(repaired)
    mae = float(np.mean(np.abs(orig - rep)))
    rmse = float(np.sqrt(np.mean((orig - rep) ** 2)))
    mape = float(np.mean(np.abs((orig - rep) / (orig + 1e-8)))) * 100
    fidelity = max(0.0, 1.0 - mae) * 100
    return {'fidelity_pct': fidelity, 'mae': mae, 'rmse': rmse, 'mape': mape}


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='RankGuard evaluation harness')
    parser.add_argument('--topology', type=str, default='medium',
                        choices=['small', 'medium', 'large'])
    parser.add_argument('--seeds', type=int, nargs='+', default=[0,1,2,3,4])
    parser.add_argument('--duration', type=float, default=1800.0)
    parser.add_argument('--workload', type=str, default='mixed')
    parser.add_argument('--telemetry_loss', type=float, default=0.0)
    parser.add_argument('--out', type=str, default='results.json')
    args = parser.parse_args()

    topo_config = {
        'small':  {'k': 4,  'hosts': 16,  'switches': 20,  'links': 48},
        'medium': {'k': 6,  'hosts': 54,  'switches': 45,  'links': 162},
        'large':  {'k': 8,  'hosts': 128, 'switches': 80,  'links': 384},
    }
    cfg = topo_config[args.topology]
    print(f"Topology: {args.topology} — {cfg['hosts']} hosts, "
          f"{cfg['links']} directed links")
    print(f"Seeds: {args.seeds}, duration: {args.duration}s, "
          f"workload: {args.workload}")

    results = []
    for seed in args.seeds:
        print(f"\n  Seed {seed}...")
        rng = np.random.default_rng(seed)

        # Simulate per-seed metrics (replace with live Mininet run for real eval)
        for policy_name in ['ECMP', 'Reactive threshold',
                            'HW-only forecasting', 'GRU-only forecasting',
                            'Hybrid forecast-threshold TE', 'RankGuard']:
            result = EvalResult(
                policy=policy_name, seed=seed,
                avg_fct=float(rng.normal(180, 20)),
                p95_fct=float(rng.normal(700, 60)),
                p99_fct=float(rng.normal(1100, 130)),
                mlu=float(np.clip(rng.normal(0.85, 0.03), 0, 1)),
                congestion_prob=float(np.clip(rng.normal(12, 2), 0, 100)),
                cvar_overflow=float(np.clip(rng.normal(0.05, 0.01), 0, 1)),
                route_changes=int(rng.integers(50, 90))
            )
            results.append(result)

    summary = aggregate_results(results)

    rankguard_res = [r for r in results if r.policy == 'RankGuard']
    hybrid_res    = [r for r in results if r.policy == 'Hybrid forecast-threshold TE']
    p_val = paired_wilcoxon(rankguard_res, hybrid_res, 'p99_fct')
    print(f"\nPaired Wilcoxon p99 FCT (RankGuard vs Hybrid): p = {p_val:.4f}")

    output = {
        'config': vars(args),
        'topology': cfg,
        'summary': {
            k: {m: list(v) for m, v in metrics.items()}
            for k, metrics in summary.items()
        },
        'wilcoxon_p_value': p_val
    }
    with open(args.out, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults written to {args.out}")


if __name__ == '__main__':
    main()
