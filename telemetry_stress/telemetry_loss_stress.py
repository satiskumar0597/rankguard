"""
RankGuard: Telemetry-loss robustness stress test.
Reproduces Table 9 and Figure 8.

Tests five imputation methods across loss rates 0-30%:
  simple_ma      Simple moving average
  linear_interp  Linear interpolation
  spline_interp  Spline interpolation
  ema            Exponential moving average (RankGuard default for short gaps)
  neighbour      Neighbour-aware fusion (RankGuard default for heavy loss)
"""

import numpy as np
import json
import argparse
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from scipy.interpolate import CubicSpline


# ── Imputation methods ────────────────────────────────────────────────────────

def impute_simple_ma(series: List[Optional[float]],
                     window: int = 5) -> List[float]:
    """Simple moving average imputation."""
    result = list(series)
    observed = [(i, v) for i, v in enumerate(series) if v is not None]
    for i, v in enumerate(result):
        if v is None:
            recent = [val for idx, val in observed if idx < i][-window:]
            result[i] = float(np.mean(recent)) if recent else 0.5
    return result


def impute_linear(series: List[Optional[float]]) -> List[float]:
    """Linear interpolation between observed values."""
    n = len(series)
    result = list(series)
    obs_idx = [i for i, v in enumerate(series) if v is not None]
    obs_val = [series[i] for i in obs_idx]
    if len(obs_idx) < 2:
        return [v if v is not None else 0.5 for v in series]
    for i in range(n):
        if result[i] is None:
            # find surrounding observed points
            lo = max((j for j in obs_idx if j < i), default=obs_idx[0])
            hi = min((j for j in obs_idx if j > i), default=obs_idx[-1])
            if lo == hi:
                result[i] = series[lo]
            else:
                t = (i - lo) / (hi - lo)
                result[i] = series[lo] * (1 - t) + series[hi] * t
    return [float(np.clip(v, 0, 1)) for v in result]


def impute_spline(series: List[Optional[float]]) -> List[float]:
    """Cubic spline interpolation."""
    n = len(series)
    obs_idx = np.array([i for i, v in enumerate(series) if v is not None])
    obs_val = np.array([series[i] for i in obs_idx])
    if len(obs_idx) < 4:
        return impute_linear(series)
    cs = CubicSpline(obs_idx, obs_val, extrapolate=True)
    return [float(np.clip(cs(i), 0, 1)) for i in range(n)]


def impute_ema(series: List[Optional[float]],
               lam: float = 0.7) -> List[float]:
    """
    Exponential moving average imputation.
    RankGuard default for short gaps (equation 3).
    """
    result = []
    smoothed = None
    last_obs = None
    for v in series:
        if v is not None:
            smoothed = lam * v + (1 - lam) * (smoothed if smoothed is not None else v)
            last_obs = v
            result.append(float(np.clip(v, 0, 1)))
        else:
            if smoothed is None:
                result.append(0.5)
            else:
                imputed = lam * last_obs + (1 - lam) * smoothed
                smoothed = imputed
                result.append(float(np.clip(imputed, 0, 1)))
    return result


def impute_neighbour_aware(series: List[Optional[float]],
                            neighbour_series: List[List[Optional[float]]],
                            lam: float = 0.7,
                            neighbour_weight: float = 0.3) -> List[float]:
    """
    Neighbour-aware fusion: combines EMA with neighbouring link evidence.
    RankGuard default for heavy loss (>3 consecutive missing samples).
    """
    ema_repaired = impute_ema(series, lam)
    if not neighbour_series:
        return ema_repaired

    # Average available neighbour values at each timestep
    n = len(series)
    result = []
    for i in range(n):
        if series[i] is not None:
            result.append(float(np.clip(series[i], 0, 1)))
        else:
            neighbour_vals = [
                ns[i] for ns in neighbour_series if ns[i] is not None
            ]
            if neighbour_vals:
                n_avg = float(np.mean(neighbour_vals))
                fused = (1 - neighbour_weight) * ema_repaired[i] + \
                        neighbour_weight * n_avg
            else:
                fused = ema_repaired[i]
            result.append(float(np.clip(fused, 0, 1)))
    return result


# ── Loss injection ────────────────────────────────────────────────────────────

def inject_loss(series: List[float], loss_rate: float,
                rng: np.random.Generator) -> List[Optional[float]]:
    """Randomly null out samples at the given loss rate."""
    return [None if rng.random() < loss_rate else v for v in series]


# ── Fidelity metrics ──────────────────────────────────────────────────────────

def fidelity_metrics(original: List[float],
                     repaired: List[float]) -> Dict[str, float]:
    orig = np.array(original)
    rep  = np.array(repaired)
    mae  = float(np.mean(np.abs(orig - rep)))
    rmse = float(np.sqrt(np.mean((orig - rep) ** 2)))
    mape = float(np.mean(np.abs((orig - rep) / (orig + 1e-8)))) * 100
    fidelity = max(0.0, 1.0 - mae) * 100
    return {'fidelity_pct': fidelity, 'mae': mae, 'rmse': rmse, 'mape': mape}


# ── Interval coverage ─────────────────────────────────────────────────────────

def interval_coverage(original: List[float], repaired: List[float],
                      alpha: float = 0.1) -> float:
    """
    Fraction of original values falling within the conformal interval
    around the repaired value.
    """
    residuals = [abs(o - r) for o, r in zip(original, repaired)]
    radius = float(np.quantile(residuals, 1 - alpha))
    covered = sum(
        abs(o - r) <= radius
        for o, r in zip(original, repaired)
    )
    return covered / len(original)


# ── Main stress test ──────────────────────────────────────────────────────────

LOSS_RATES   = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
METHODS      = ['simple_ma', 'linear_interp', 'spline_interp', 'ema', 'neighbour']
SERIES_LEN   = 1800   # 30-minute run at 1s sampling


def run_stress_test(seeds: List[int]) -> Dict:
    """
    For each method × loss_rate × seed, inject loss and measure fidelity.
    Returns nested dict: method → loss_rate → {mean, std of each metric}.
    """
    from collections import defaultdict
    raw: Dict = defaultdict(lambda: defaultdict(list))

    for seed in seeds:
        rng = np.random.default_rng(seed)

        # Synthetic utilization trace with seasonality + noise
        t = np.arange(SERIES_LEN)
        series = (0.5 + 0.2 * np.sin(2 * np.pi * t / 300) +
                  0.1 * np.sin(2 * np.pi * t / 60) +
                  rng.normal(0, 0.03, SERIES_LEN))
        series = list(np.clip(series, 0.01, 0.99))

        # Synthetic neighbour series for neighbour-aware method
        neighbours = [
            list(np.clip(
                series + rng.normal(0, 0.02, SERIES_LEN), 0.01, 0.99
            ))
            for _ in range(3)
        ]

        for loss_rate in LOSS_RATES:
            corrupted = inject_loss(series, loss_rate, rng)
            corrupted_neighbours = [
                inject_loss(nb, loss_rate * 0.5, rng) for nb in neighbours
            ]

            for method in METHODS:
                if method == 'simple_ma':
                    repaired = impute_simple_ma(corrupted)
                elif method == 'linear_interp':
                    repaired = impute_linear(corrupted)
                elif method == 'spline_interp':
                    repaired = impute_spline(corrupted)
                elif method == 'ema':
                    repaired = impute_ema(corrupted)
                else:  # neighbour
                    repaired = impute_neighbour_aware(
                        corrupted, corrupted_neighbours
                    )

                m = fidelity_metrics(series, repaired)
                m['interval_coverage'] = interval_coverage(series, repaired)
                raw[method][loss_rate].append(m)

    # Aggregate
    summary = {}
    for method in METHODS:
        summary[method] = {}
        for loss_rate in LOSS_RATES:
            runs = raw[method][loss_rate]
            summary[method][loss_rate] = {
                metric: (float(np.mean([r[metric] for r in runs])),
                         float(np.std([r[metric] for r in runs])))
                for metric in runs[0].keys()
            }
    return summary


def print_table9(summary: Dict):
    """Print Table 9: reconstruction accuracy at 10% loss."""
    loss_rate = 0.10
    method_labels = {
        'simple_ma':     'Simple Moving Average',
        'linear_interp': 'Linear Interpolation',
        'spline_interp': 'Spline Interpolation',
        'ema':           'Exponential Moving Average',
        'neighbour':     'Neighbour-aware fusion',
    }
    header = (f"{'Method':<28} {'Fidelity (%)':>14} {'MAE':>10} "
              f"{'RMSE':>10} {'MAPE (%)':>10}")
    print(f"\nTable 9 — reconstruction accuracy at {int(loss_rate*100)}% loss")
    print(header)
    print('-' * len(header))
    for method in METHODS:
        m = summary[method][loss_rate]
        print(f"{method_labels[method]:<28} "
              f"{m['fidelity_pct'][0]:>7.1f}±{m['fidelity_pct'][1]:>4.1f}   "
              f"{m['mae'][0]:>6.3f}±{m['mae'][1]:>4.3f}   "
              f"{m['rmse'][0]:>6.3f}±{m['rmse'][1]:>4.3f}   "
              f"{m['mape'][0]:>6.2f}±{m['mape'][1]:>4.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='RankGuard telemetry-loss stress test')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0,1,2,3,4])
    parser.add_argument('--out',   type=str, default='telemetry_loss_results.json')
    args = parser.parse_args()

    print(f"Running telemetry-loss stress test over seeds {args.seeds}...")
    print(f"Loss rates: {LOSS_RATES}")
    print(f"Methods: {METHODS}\n")

    summary = run_stress_test(args.seeds)
    print_table9(summary)

    # Serialise (convert float keys to strings for JSON)
    serialisable = {
        method: {
            str(lr): {
                metric: list(val)
                for metric, val in metrics.items()
            }
            for lr, metrics in by_rate.items()
        }
        for method, by_rate in summary.items()
    }
    with open(args.out, 'w') as f:
        json.dump(serialisable, f, indent=2)
    print(f"\nTelemetry-loss results written to {args.out}")
