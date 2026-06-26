"""
RankGuard: Plotting scripts for all paper figures.
Generates Figures 3-9 from result JSON files produced by the evaluation scripts.

Usage:
  python plot_figures.py --results_dir ./results --out_dir ./figures

Input files expected in results_dir:
  results_medium.json       (from evaluate.py)
  ablation_results.json     (from run_ablations.py)
  sensitivity_results.json  (from weight_sensitivity.py)
  telemetry_loss_results.json (from telemetry_loss_stress.py)
  drift_results.json        (from drift_recovery.py)
  fidelity_results.json     (from evaluator_fidelity.py)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import json
import argparse
import os
from typing import Dict, List


# ── Style ─────────────────────────────────────────────────────────────────────

POLICY_COLORS = {
    'ECMP':                        '#4C72B0',
    'Reactive threshold':          '#DD8452',
    'HW-only forecasting':         '#55A868',
    'GRU-only forecasting':        '#C44E52',
    'Hybrid forecast-threshold TE':'#8172B2',
    'RankGuard':                   '#937860',
}

POLICY_ORDER = list(POLICY_COLORS.keys())

plt.rcParams.update({
    'font.family':       'sans-serif',
    'font.size':         10,
    'axes.linewidth':    0.6,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'legend.frameon':    False,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
})


def load(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)


# ── Figure 3: seed-level p99 FCT ─────────────────────────────────────────────

def fig3_p99_fct(data: Dict, out_dir: str):
    fig, ax = plt.subplots(figsize=(7, 4))

    raw = data.get('raw', [])
    by_policy = {p: [] for p in POLICY_ORDER}
    for r in raw:
        if r['policy'] in by_policy:
            by_policy[r['policy']].append(r['p99_fct'])

    for i, policy in enumerate(POLICY_ORDER):
        vals = by_policy[policy]
        if not vals:
            continue
        x = np.full(len(vals), i)
        ax.scatter(x, vals, color=POLICY_COLORS[policy],
                   s=40, zorder=3, alpha=0.85)
        ax.errorbar(i, np.mean(vals), yerr=np.std(vals),
                    fmt='_', color='black', capsize=4,
                    linewidth=1.2, markersize=10, zorder=4)

    ax.set_xticks(range(len(POLICY_ORDER)))
    ax.set_xticklabels([p.replace(' ', '\n') for p in POLICY_ORDER],
                       fontsize=8)
    ax.set_ylabel('p99 FCT (ms)')
    ax.set_title('Tail flow completion time across policies')
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    plt.tight_layout()
    path = os.path.join(out_dir, 'fig3_p99_fct.pdf')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ── Figure 4: MLU and congestion probability ──────────────────────────────────

def fig4_mlu_cong(data: Dict, out_dir: str):
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax2 = ax1.twinx()

    summary = data.get('summary', {})
    x = range(len(POLICY_ORDER))
    mlu_means  = [summary.get(p, {}).get('mlu',  [0,0])[0] for p in POLICY_ORDER]
    mlu_stds   = [summary.get(p, {}).get('mlu',  [0,0])[1] for p in POLICY_ORDER]
    cong_means = [summary.get(p, {}).get('cong_prob', [0,0])[0] for p in POLICY_ORDER]
    cong_stds  = [summary.get(p, {}).get('cong_prob', [0,0])[1] for p in POLICY_ORDER]

    ax1.errorbar(x, mlu_means, yerr=mlu_stds, fmt='-o',
                 color='#4C72B0', label='MLU',
                 linewidth=1.4, markersize=5, capsize=3)
    ax2.errorbar(x, cong_means, yerr=cong_stds, fmt='--s',
                 color='#DD8452', label='Congestion probability',
                 linewidth=1.4, markersize=5, capsize=3)

    ax1.set_xticks(list(x))
    ax1.set_xticklabels([p.replace(' ', '\n') for p in POLICY_ORDER], fontsize=8)
    ax1.set_ylabel('Maximum link utilization')
    ax2.set_ylabel('Congestion probability (%)')
    ax1.set_title('Utilization risk across policies')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
    plt.tight_layout()
    path = os.path.join(out_dir, 'fig4_mlu_cong.pdf')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ── Figure 5: counterfactual evaluator fidelity scatter ───────────────────────

def fig5_evaluator_fidelity(data: List[Dict], out_dir: str):
    fig, ax = plt.subplots(figsize=(4.5, 4.5))

    est_all  = [r['est_mlu']  for r in data]
    true_all = [r['true_mlu'] for r in data]

    # Flatten if nested (one record per action per seed)
    ax.scatter(true_all, est_all, alpha=0.55, s=22,
               color='#4C72B0', edgecolors='none')

    lims = [min(true_all + est_all) - 0.02,
            max(true_all + est_all) + 0.02]
    ax.plot(lims, lims, '--', color='gray', linewidth=0.8, label='y = x')
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel('Replayed MLU')
    ax.set_ylabel('Estimated MLU')
    ax.set_title('Counterfactual evaluator fidelity')
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(out_dir, 'fig5_evaluator_fidelity.pdf')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ── Figure 6: post-drift MLU recovery ────────────────────────────────────────

def fig6_drift_recovery(data: Dict, out_dir: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    summary = data['summary']
    drift_policies = [
        'Reactive threshold', 'HW-only forecasting',
        'Hybrid forecast-threshold TE', 'RankGuard'
    ]
    colors = ['#4C72B0', '#55A868', '#8172B2', '#C44E52']

    for policy, color in zip(drift_policies, colors):
        if policy not in summary:
            continue
        m = summary[policy]
        trace_mean = np.array(m['mlu_trace_mean'])
        trace_std  = np.array(m['mlu_trace_std'])
        x = np.arange(len(trace_mean))
        ax.plot(x, trace_mean, color=color, linewidth=1.4, label=policy)
        ax.fill_between(x, trace_mean - trace_std, trace_mean + trace_std,
                        alpha=0.15, color=color)

    ax.axhline(0.80, color='black', linewidth=0.7,
               linestyle=':', label='Recovery target (0.80)')
    ax.set_xlabel('Control windows after congestion shift')
    ax.set_ylabel('Maximum link utilization')
    ax.set_title('Recovery after congestion migration')
    ax.legend(fontsize=8, loc='upper right')
    plt.tight_layout()
    path = os.path.join(out_dir, 'fig6_drift_recovery.pdf')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ── Figure 7: weight sensitivity ─────────────────────────────────────────────

def fig7_weight_sensitivity(data: Dict, out_dir: str):
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax2 = ax1.twinx()

    summary  = data['summary']
    configs  = ['Latency-heavy', 'Utilization-heavy', 'Churn-heavy',
                'Balanced default']
    x        = np.arange(len(configs))
    p99_m    = [summary[c]['p99_fct'][0]       for c in configs]
    p99_s    = [summary[c]['p99_fct'][1]       for c in configs]
    churn_m  = [summary[c]['route_changes'][0] for c in configs]
    churn_s  = [summary[c]['route_changes'][1] for c in configs]

    ax1.errorbar(x, p99_m, yerr=p99_s, fmt='-o',
                 color='#4C72B0', label='p99 FCT',
                 linewidth=1.4, markersize=5, capsize=3)
    ax2.errorbar(x, churn_m, yerr=churn_s, fmt='--s',
                 color='#DD8452', label='Route changes',
                 linewidth=1.4, markersize=5, capsize=3)

    ax1.set_xticks(list(x))
    ax1.set_xticklabels(configs, fontsize=9)
    ax1.set_ylabel('p99 FCT (ms)')
    ax2.set_ylabel('Route changes')
    ax1.set_title('Policy-weight sensitivity')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc='upper center', fontsize=8)
    plt.tight_layout()
    path = os.path.join(out_dir, 'fig7_weight_sensitivity.pdf')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ── Figure 8: telemetry-loss robustness ───────────────────────────────────────

def fig8_telemetry_loss(data: Dict, out_dir: str):
    fig, ax = plt.subplots(figsize=(6, 4))

    loss_rates = sorted([float(k) for k in
                         next(iter(data.values())).keys()])
    x_pct = [lr * 100 for lr in loss_rates]

    # EMA fidelity
    ema_fid = [data['ema'][str(lr)]['fidelity_pct'][0] for lr in loss_rates]
    nb_fid  = [data['neighbour'][str(lr)]['fidelity_pct'][0]
               for lr in loss_rates]
    # Interval coverage (EMA)
    ema_cov = [data['ema'][str(lr)].get('interval_coverage', [0.9,0])[0] * 100
               for lr in loss_rates]

    ax.plot(x_pct, ema_fid, '-o', color='#4C72B0',
            linewidth=1.4, markersize=5, label='EMA fidelity')
    ax.plot(x_pct, nb_fid,  '-s', color='#DD8452',
            linewidth=1.4, markersize=5, label='Neighbour-aware fidelity')
    ax.plot(x_pct, ema_cov, '-^', color='#55A868',
            linewidth=1.4, markersize=5, label='Interval coverage')

    ax.set_xlabel('Injected telemetry loss (%)')
    ax.set_ylabel('Percentage')
    ax.set_title('Telemetry-loss robustness')
    ax.legend(fontsize=8)
    ax.set_ylim(80, 102)
    plt.tight_layout()
    path = os.path.join(out_dir, 'fig8_telemetry_loss.pdf')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ── Figure 9: component ablation ─────────────────────────────────────────────

def fig9_ablation(data: Dict, out_dir: str):
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax2 = ax1.twinx()

    summary  = data['summary']
    variants = [
        'RankGuard full method',
        'No uncertainty calibration',
        'No counterfactual validation',
        'No decision-aware loss',
        'No topology-aware expert',
        'No churn penalty',
    ]
    labels   = ['Full', 'No uncert.', 'No cf. gate',
                'No action\nrank', 'No topology', 'No churn']
    x        = np.arange(len(variants))
    p99_m    = [summary[v]['p99_fct'][0]       for v in variants]
    p99_s    = [summary[v]['p99_fct'][1]       for v in variants]
    churn_m  = [summary[v]['route_changes'][0] for v in variants]
    churn_s  = [summary[v]['route_changes'][1] for v in variants]

    ax1.errorbar(x, p99_m, yerr=p99_s, fmt='-o',
                 color='#4C72B0', label='p99 FCT',
                 linewidth=1.4, markersize=5, capsize=3)
    ax2.errorbar(x, churn_m, yerr=churn_s, fmt='--s',
                 color='#DD8452', label='Route changes',
                 linewidth=1.4, markersize=5, capsize=3)

    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel('p99 FCT (ms)')
    ax2.set_ylabel('Route changes')
    ax1.set_title('Component ablation')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc='upper center', fontsize=8)
    plt.tight_layout()
    path = os.path.join(out_dir, 'fig9_ablation.pdf')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RankGuard figure plotter')
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--out_dir',     type=str, default='./figures')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    def path(fname):
        return os.path.join(args.results_dir, fname)

    print("Generating figures...")

    if os.path.exists(path('results_medium.json')):
        d = load(path('results_medium.json'))
        fig3_p99_fct(d, args.out_dir)
        fig4_mlu_cong(d, args.out_dir)
    else:
        print("  Skipping Fig 3/4: results_medium.json not found")

    if os.path.exists(path('fidelity_results.json')):
        d = load(path('fidelity_results.json'))
        fig5_evaluator_fidelity(d, args.out_dir)
    else:
        print("  Skipping Fig 5: fidelity_results.json not found")

    if os.path.exists(path('drift_results.json')):
        d = load(path('drift_results.json'))
        fig6_drift_recovery(d, args.out_dir)
    else:
        print("  Skipping Fig 6: drift_results.json not found")

    if os.path.exists(path('sensitivity_results.json')):
        d = load(path('sensitivity_results.json'))
        fig7_weight_sensitivity(d, args.out_dir)
    else:
        print("  Skipping Fig 7: sensitivity_results.json not found")

    if os.path.exists(path('telemetry_loss_results.json')):
        d = load(path('telemetry_loss_results.json'))
        fig8_telemetry_loss(d, args.out_dir)
    else:
        print("  Skipping Fig 8: telemetry_loss_results.json not found")

    if os.path.exists(path('ablation_results.json')):
        d = load(path('ablation_results.json'))
        fig9_ablation(d, args.out_dir)
    else:
        print("  Skipping Fig 9: ablation_results.json not found")

    print(f"\nAll available figures saved to {args.out_dir}/")
