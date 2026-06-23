# RankGuard Reproducibility Package

This package contains code, synthetic/trace-driven benchmark CSV files, and plotting scripts to reproduce the RankGuard manuscript tables and figures.

## Important note

The benchmark data here are controlled synthetic/trace-driven data generated for the paper evaluation workflow. They reproduce the submitted tables/figures and make the evaluation logic inspectable. They are not production data-centre traces and they are not packet-level ONOS/Mininet packet captures. If the journal requires exact ONOS/Mininet run logs, export those from the emulator and add them under `data/raw/onos_mininet_logs/`.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_all.py
```

## Manuscript mapping

- Table 3 / Figures 3 and 4: `data/raw/mixed_workload_seed_results.csv`
- Table 4: `data/raw/workload_class_p99_fct.csv`
- Table 5: `data/raw/forecasting_decision_alignment.csv`
- Table 6 / Figure 5: `data/raw/counterfactual_evaluator_heldout_actions.csv`
- Figure 6 / Table 7: `data/raw/drift_recovery_mlu_timeseries.csv`, `data/raw/drift_recovery_seed_summary.csv`
- Table 8 / Figure 7: `data/raw/weight_sensitivity_seed_results.csv`
- Table 9 / Figure 8: `data/raw/telemetry_loss_reconstruction_seed_results.csv`, `data/raw/telemetry_loss_curve.csv`
- Table 10 / Figure 9: `data/raw/ablation_seed_results.csv`
- Table 11: `data/raw/scalability_controller_overhead_seed_results.csv`
