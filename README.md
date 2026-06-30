# RankGuard

**Decision-Aligned Predictive Traffic Engineering for Software-Defined Data-Centre Networks**

Satis Kumar Permal and Teck Chaw Ling  
Universiti Malaya

---

## Overview

RankGuard is a predictive traffic-engineering framework for software-defined data-centre networks. Instead of optimizing traffic-forecasting accuracy alone, RankGuard trains the predictor to preserve the ordering of candidate routing actions. This makes the forecast more directly aligned with the controller decision that follows.

The framework combines:

- Hybrid Holt-Winters and GRU traffic prediction
- Decision-aware action-rank loss
- Conformal uncertainty calibration
- Risk-aware route scoring
- Counterfactual route evaluation
- Telemetry-loss repair and robustness testing

This repository contains the source code, scripts, configuration files, benchmark datasets, trace-driven data, and plotting utilities used to reproduce the paper results.

---

## Repository Structure

```text
rankguard/
├── topology/
│   └── fattree_topo.py                  Fat-tree / Clos topology generator for Mininet
├── workload/
│   └── traffic_generator.py             Mixed workload generator
├── predictor/
│   └── hw_gru_predictor.py              HW-GRU predictor and decision-aware loss
├── controller/
│   └── rankguard_controller.py          Risk-aware scoring and conservative actuation
├── evaluation/
│   └── evaluate.py                      Main evaluation, baselines, metrics, and tests
├── ablation/
│   └── run_ablations.py                 Component-removal experiments
├── sensitivity/
│   └── weight_sensitivity.py            Route-scoring weight sensitivity
├── telemetry_stress/
│   └── telemetry_loss_stress.py          Telemetry-loss robustness experiments
├── drift/
│   └── drift_recovery.py                Post-drift recovery experiments
├── evaluator_fidelity/
│   └── evaluator_fidelity.py            Counterfactual evaluator fidelity check
├── plotting/
│   └── plot_figures.py                  Paper figure generation
├── scripts/
│   ├── run_all.py                       Reproduce summaries, figures, and demo outputs
│   ├── summarize_tables.py              Generate summary CSV files
│   ├── generate_figures.py              Generate manuscript plots
│   └── run_demo_simulation.py           Small trace-driven RankGuard gate demo
├── src/
│   └── rankguard/
│       ├── controller.py                Minimal reusable scoring and decision-gate helpers
│       ├── losses.py                    Action-rank, under-risk, and quantile-loss helpers
│       ├── conformal.py                 Conformal interval helpers
│       └── simulator.py                 Lightweight trace-driven evaluator
├── configs/
│   ├── rankguard_default.json           Default RankGuard parameters
│   └── topologies.json                  Topology metadata
├── data/
│   ├── raw/                             Seed-level CSV files used for manuscript tables
│   ├── processed/                       Generated summary files
│   └── traces/                          Trace-driven utilization traces
├── figures/                             Generated manuscript figures
├── DATA_DICTIONARY.md                   Description of dataset fields
├── requirements.txt                     Python dependencies
└── README.md
```

---

## Data and Reproducibility Scope

The `data/` directory contains the seed-level CSV files and trace-driven utilization traces used to reproduce the manuscript tables and plots.

Important scope note:

- The released data are **controlled trace-driven benchmark data**.
- They are intended for reproducibility of the submitted tables and figures.
- They are **not production data-centre traces**.
- They are **not raw packet-level ONOS/Mininet packet captures**.
- Controller-in-the-loop experiments require Mininet and ONOS; the provided scripts also include simulation paths so reviewers can reproduce the reported outputs without a live SDN testbed.

The main dataset fields are described in `DATA_DICTIONARY.md`.

---

## Requirements

### Python Version

```text
Python >= 3.9
```

### Python Dependencies

Install dependencies using:

```bash
pip install -r requirements.txt
```

The expected `requirements.txt` format is:

```text
numpy>=1.23
pandas>=1.5
matplotlib>=3.6
scipy>=1.10
torch>=2.0
```

If PyTorch is not needed for a lightweight reproduction run, the plotting and CSV-summary scripts can still be executed with only NumPy, Pandas, Matplotlib, and SciPy.

### Optional Dependencies for Live Emulation

The trace-driven simulation mode does not require Mininet or ONOS. They are only required for controller-in-the-loop experiments.

```text
mininet >= 2.3
ONOS controller
```

Useful links:

- Mininet: https://mininet.org/download/
- ONOS: https://wiki.onosproject.org/display/ONOS/Installing+and+Running+ONOS

---

## Quick Start

Clone the repository and install the dependencies:

```bash
git clone https://github.com/satiskumar0597/rankguard.git
cd rankguard

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

To regenerate summaries, figures, and the small trace-driven demo:

```bash
python scripts/run_all.py
```

Generated summary files are written to:

```text
data/processed/
```

Generated figures are written to:

```text
figures/
```

---

## Reproducing the Main Result

The main end-to-end result corresponds to Table 3 in the paper.

### Trace-Driven Simulation

This mode does not require Mininet or ONOS.

```bash
python evaluation/evaluate.py \
    --topology medium \
    --seeds 0 1 2 3 4 \
    --duration 1800 \
    --workload mixed \
    --out results/results_medium.json
```

---

## Controller-in-the-Loop Evaluation

This mode requires Mininet and ONOS.

### 1. Start ONOS

```bash
./bin/onos-service start
```

### 2. Launch the Fat-Tree Topology

```bash
sudo python topology/fattree_topo.py \
    --k 6 \
    --controller 127.0.0.1 \
    --port 6653
```

### 3. Generate Traffic

```bash
python workload/traffic_generator.py \
    --hosts 10.0.0.1 10.0.0.2 \
    --workload mixed \
    --duration 1800 \
    --seed 0 \
    --out flows_seed0.json
```

### 4. Run Evaluation

```bash
python evaluation/evaluate.py \
    --topology medium \
    --seeds 0 1 2 3 4
```

---

## Reproducing All Tables and Figures

Run the following scripts in order. Each script writes a result file that is later consumed by the plotting script.

```bash
mkdir -p results figures

# Table 10 and Figure 9: ablation study
python ablation/run_ablations.py \
    --seeds 0 1 2 3 4 \
    --out results/ablation_results.json

# Table 8 and Figure 7: route-scoring weight sensitivity
python sensitivity/weight_sensitivity.py \
    --seeds 0 1 2 3 4 \
    --out results/sensitivity_results.json

# Table 9 and Figure 8: telemetry-loss robustness
python telemetry_stress/telemetry_loss_stress.py \
    --seeds 0 1 2 3 4 \
    --out results/telemetry_loss_results.json

# Table 7 and Figure 6: post-drift recovery
python drift/drift_recovery.py \
    --seeds 0 1 2 3 4 \
    --out results/drift_results.json

# Table 6 and Figure 5: counterfactual evaluator fidelity
python evaluator_fidelity/evaluator_fidelity.py \
    --seeds 0 1 2 3 4 \
    --out results/fidelity_results.json

# Figures 3 to 9: generate all plots
python plotting/plot_figures.py \
    --results_dir results \
    --out_dir figures
```

The generated figures are written to the `figures/` directory. If a required result file is missing, the plotter skips the corresponding figure instead of terminating.

---

## Dataset-to-Paper Mapping

| Dataset file | Tables / Figures | Description |
| --- | ---: | --- |
| `data/raw/mixed_workload_seed_results.csv` | Table 3, Figures 3 and 4 | Seed-level mixed-workload TE performance |
| `data/raw/workload_class_p99_fct.csv` | Table 4 | Workload-specific p99 FCT results |
| `data/raw/forecasting_decision_alignment.csv` | Table 5 | Forecasting accuracy and action-rank agreement |
| `data/raw/counterfactual_evaluator_heldout_actions.csv` | Figure 5 | Estimated vs replayed MLU for held-out candidate actions |
| `data/processed/table6_counterfactual_fidelity_summary.csv` | Table 6 | Counterfactual evaluator summary statistics |
| `data/raw/drift_recovery_mlu_timeseries.csv` | Figure 6 | MLU trajectory after congestion migration |
| `data/raw/drift_recovery_seed_summary.csv` | Table 7 | Drift recovery summary |
| `data/raw/weight_sensitivity_seed_results.csv` | Table 8, Figure 7 | Route-scoring weight sensitivity |
| `data/raw/telemetry_loss_reconstruction_seed_results.csv` | Table 9 | Telemetry reconstruction accuracy |
| `data/raw/telemetry_loss_curve.csv` | Figure 8 | Telemetry-loss robustness curve |
| `data/raw/ablation_seed_results.csv` | Table 10, Figure 9 | Component ablation results |
| `data/raw/scalability_controller_overhead_seed_results.csv` | Table 11 | Controller overhead under larger topologies |
| `data/traces/synthetic_small_topology_trace_seed*.csv` | Demo / trace-driven replay | Per-link utilization traces used by the lightweight replay demo |

---

## Script-to-Paper Mapping

| Script | Tables | Figures | Description |
| --- | ---: | ---: | --- |
| `evaluation/evaluate.py` | 3, 4, 5, 11 | 3, 4 | End-to-end traffic-engineering performance |
| `evaluator_fidelity/evaluator_fidelity.py` | 6 | 5 | Counterfactual evaluator fidelity |
| `drift/drift_recovery.py` | 7 | 6 | Post-drift recovery after congestion migration |
| `sensitivity/weight_sensitivity.py` | 8 | 7 | Route-scoring weight sensitivity |
| `telemetry_stress/telemetry_loss_stress.py` | 9 | 8 | Telemetry-loss robustness |
| `ablation/run_ablations.py` | 10 | 9 | Component ablation study |
| `scripts/summarize_tables.py` | Summary CSVs | - | Regenerates processed summary files |
| `scripts/generate_figures.py` | - | 3-9 | Regenerates figures from CSV data |
| `scripts/run_demo_simulation.py` | Demo output | - | Runs a lightweight trace-driven decision-gate demo |
| `scripts/run_all.py` | Summary CSVs | 3-9 | Convenience script for local reproducibility |

---

## Topology Configurations

| Name | `k` | Hosts | Switches | Directed Links | Use |
| --- | --: | ----: | -------: | -------------: | --- |
| `small` | 4 | 16 | 20 | 48 | Controller-in-the-loop Mininet experiments |
| `medium` | 6 | 54 | 45 | 162 | Main mixed-workload experiments |
| `large` | 8 | 128 | 80 | 384 | Scalability stress |

---

## Key Hyperparameters

| Parameter | Value | Description |
| --- | ---: | --- |
| `tau` | 0.8 | Congestion threshold as a fraction of link capacity |
| `alpha` | 0.1 | Conformal miscoverage level for 90% prediction intervals |
| `epsilon` | 0.03 | Minimum improvement margin |
| `h` | 2 | Hold-down window count |
| `beta` | 0.5 | Uncertainty penalty weight |
| `K` | 4 | Number of K-shortest-path candidates |
| `w1` to `w5` | 0.35, 0.25, 0.20, 0.15, 0.05 | Default route-scoring weights |

---

## Implemented Equations

| File | Implemented Components |
| --- | --- |
| `predictor/hw_gru_predictor.py` | EMA repair, Holt-Winters forecasting, HW-GRU residual prediction, decision-aware loss, and conformal calibration |
| `controller/rankguard_controller.py` | Route score `J(p)`, conservative decision condition, and Algorithm 1 |
| `src/rankguard/controller.py` | Minimal reusable implementation of the route score and persistence gate |
| `src/rankguard/losses.py` | Pinball loss, under-risk loss, and action-rank loss |
| `src/rankguard/conformal.py` | Sliding-window conformal calibration helpers |
| `src/rankguard/simulator.py` | Lightweight trace-driven candidate-path evaluator |
| `evaluation/evaluate.py` | Wilcoxon signed-rank test, FCT, MLU, congestion probability, and CVaR metrics |
| `ablation/run_ablations.py` | Component-removal variants of the full system |
| `sensitivity/weight_sensitivity.py` | Weight configurations for the route-scoring objective |
| `telemetry_stress/telemetry_loss_stress.py` | EMA repair and neighbour-aware imputation |
| `drift/drift_recovery.py` | Post-drift MLU trajectory and recovery measurement |
| `evaluator_fidelity/evaluator_fidelity.py` | Spearman correlation and top-1 agreement for counterfactual evaluator fidelity |

---

## Baselines

| Policy | Description |
| --- | --- |
| `ECMP` | Static equal-cost multipath without rerouting |
| `Reactive threshold` | Reroutes when measured utilization exceeds the congestion threshold |
| `HW-only forecasting` | Holt-Winters forecast with threshold-based triggering |
| `GRU-only forecasting` | GRU forecast with threshold-based triggering |
| `Hybrid forecast-threshold TE` | HW-GRU forecast with adaptive threshold triggering |
| `RankGuard` | Proposed decision-aligned predictive traffic-engineering framework |

---

## Statistical Testing

All main comparisons use a paired Wilcoxon signed-rank test with `p < 0.05` against the hybrid forecast-threshold baseline.

Each policy is evaluated using the same seed and workload instance. Therefore, paired tests are used instead of independent tests.

---

## Live Runs vs Simulation Runs

All experiment scripts include a `simulate_*` function that approximates the corresponding experiment without requiring Mininet or ONOS.

To run against a live controller, replace the body of the relevant `simulate_*` function with calls to the controller using the topology and workload scripts in `topology/` and `workload/`.

The function signatures and output data structures are kept identical in both modes.

---

## Availability

The code, scripts, configuration files, benchmark data, and plotting utilities used for this study are available in this repository:

```text
https://github.com/satiskumar0597/rankguard
```

---

## License

Add a `LICENSE` file before final archival or journal review. For a simple open-source artifact, the MIT License is usually appropriate for code. If the benchmark data are released only for academic reproducibility, state that clearly in the license or repository notes.

---

## Citation

If you use this code or benchmark package, please cite:

```bibtex
@article{permal2026rankguard,
  title   = {RankGuard: Decision-Aligned Predictive Traffic Engineering for Software-Defined Data-Centre Networks},
  author  = {Permal, Satis Kumar and Ling, Teck Chaw},
  year    = {2026}
}
```
