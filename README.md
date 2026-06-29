# RankGuard

**Decision-Aligned Predictive Traffic Engineering for Software-Defined Data-Centre Networks**

Satis Kumar Permal and Teck Chaw Ling
Universiti Malaya

---

## Overview

RankGuard is a predictive traffic-engineering framework for software-defined data-centre networks. Instead of optimizing forecasting accuracy alone, RankGuard trains the predictor to preserve the ordering of candidate routing actions. This allows the controller to select routing decisions that are better aligned with the final traffic-engineering objective.

The framework combines:

* Hybrid Holt-Winters and GRU traffic prediction
* Decision-aware action-rank loss
* Conformal uncertainty calibration
* Risk-aware route scoring
* Counterfactual route evaluation
* Telemetry-loss repair and robustness testing

---

## Repository Structure

```text
rankguard/
├── topology/
│   └── fattree_topo.py
├── workload/
│   └── traffic_generator.py
├── predictor/
│   └── hw_gru_predictor.py
├── controller/
│   └── rankguard_controller.py
├── evaluation/
│   └── evaluate.py
├── ablation/
│   └── run_ablations.py
├── sensitivity/
│   └── weight_sensitivity.py
├── telemetry_stress/
│   └── telemetry_loss_stress.py
├── drift/
│   └── drift_recovery.py
├── evaluator_fidelity/
│   └── evaluator_fidelity.py
├── plotting/
│   └── plot_figures.py
└── README.md
```

### Directory Description

| Directory             | File                       | Description                                             |
| --------------------- | -------------------------- | ------------------------------------------------------- |
| `topology/`           | `fattree_topo.py`          | Fat-tree / Clos topology for Mininet with `k = 4, 6, 8` |
| `workload/`           | `traffic_generator.py`     | Mixed workload generator with seven traffic classes     |
| `predictor/`          | `hw_gru_predictor.py`      | HW-GRU predictor and action-rank loss                   |
| `controller/`         | `rankguard_controller.py`  | Risk-aware route scoring and counterfactual actuation   |
| `evaluation/`         | `evaluate.py`              | Main evaluation, baselines, metrics, and Wilcoxon test  |
| `ablation/`           | `run_ablations.py`         | Component ablation experiments                          |
| `sensitivity/`        | `weight_sensitivity.py`    | Route-scoring weight sensitivity experiments            |
| `telemetry_stress/`   | `telemetry_loss_stress.py` | Telemetry-loss robustness experiments                   |
| `drift/`              | `drift_recovery.py`        | Post-drift recovery experiments                         |
| `evaluator_fidelity/` | `evaluator_fidelity.py`    | Counterfactual evaluator fidelity experiments           |
| `plotting/`           | `plot_figures.py`          | Generates paper figures from result JSON files          |

---

## Requirements

### Python Dependencies

```text
Python >= 3.9
torch >= 2.0
numpy
scipy
matplotlib >= 3.7
```

Install the Python dependencies:

```bash
pip install torch numpy scipy matplotlib
```

### Optional Dependencies for Live Emulation

The trace-driven simulation mode does not require Mininet or ONOS. They are only required for controller-in-the-loop experiments.

```text
mininet >= 2.3
ONOS controller
```

Installation links:

* Mininet: https://mininet.org/download/
* ONOS: https://wiki.onosproject.org/display/ONOS/Installing+and+Running+ONOS

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

Run the following scripts in order. Each script writes a JSON result file that is later consumed by the plotting script.

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

The generated figures are written as PDF files to the `figures/` directory. If a required JSON file is missing, the plotter skips the corresponding figure instead of terminating.

---

## Script-to-Paper Mapping

| Script                                      |      Tables | Figures | Description                                    |
| ------------------------------------------- | ----------: | ------: | ---------------------------------------------- |
| `evaluation/evaluate.py`                    | 3, 4, 5, 11 |    3, 4 | End-to-end traffic-engineering performance     |
| `evaluator_fidelity/evaluator_fidelity.py`  |           6 |       5 | Counterfactual evaluator fidelity              |
| `drift/drift_recovery.py`                   |           7 |       6 | Post-drift recovery after congestion migration |
| `sensitivity/weight_sensitivity.py`         |           8 |       7 | Route-scoring weight sensitivity               |
| `telemetry_stress/telemetry_loss_stress.py` |           9 |       8 | Telemetry-loss robustness                      |
| `ablation/run_ablations.py`                 |          10 |       9 | Component ablation study                       |

---

## Topology Configurations

| Name     | `k` | Hosts | Switches | Directed Links |
| -------- | --: | ----: | -------: | -------------: |
| `small`  |   4 |    16 |       20 |             48 |
| `medium` |   6 |    54 |       45 |            162 |
| `large`  |   8 |   128 |       80 |            384 |

---

## Key Hyperparameters

| Parameter    |                        Value | Description                                              |
| ------------ | ---------------------------: | -------------------------------------------------------- |
| `tau`        |                          0.8 | Congestion threshold as a fraction of link capacity      |
| `alpha`      |                          0.1 | Conformal miscoverage level for 90% prediction intervals |
| `epsilon`    |                         0.03 | Minimum improvement margin                               |
| `h`          |                            2 | Hold-down window count                                   |
| `beta`       |                          0.5 | Uncertainty penalty weight                               |
| `K`          |                            4 | Number of K-shortest-path candidates                     |
| `w1` to `w5` | 0.35, 0.25, 0.20, 0.15, 0.05 | Default route-scoring weights                            |

---

## Implemented Equations

| File                                        | Implemented Components                                                                                           |
| ------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `predictor/hw_gru_predictor.py`             | EMA repair, Holt-Winters forecasting, HW-GRU residual prediction, decision-aware loss, and conformal calibration |
| `controller/rankguard_controller.py`        | Route score `J(p)`, decision condition, and Algorithm 1                                                          |
| `evaluation/evaluate.py`                    | Wilcoxon signed-rank test, flow completion time, maximum link utilization, and CVaR metrics                      |
| `ablation/run_ablations.py`                 | Component-removal variants of the full system                                                                    |
| `sensitivity/weight_sensitivity.py`         | Weight configurations for the route-scoring objective                                                            |
| `telemetry_stress/telemetry_loss_stress.py` | EMA repair and neighbour-aware imputation                                                                        |
| `drift/drift_recovery.py`                   | Post-drift maximum-link-utilization trajectory and recovery measurement                                          |
| `evaluator_fidelity/evaluator_fidelity.py`  | Spearman correlation and top-1 agreement for counterfactual evaluator fidelity                                   |

---

## Baselines

| Policy                         | Description                                                         |
| ------------------------------ | ------------------------------------------------------------------- |
| `ECMP`                         | Static equal-cost multipath without rerouting                       |
| `Reactive threshold`           | Reroutes when measured utilization exceeds the congestion threshold |
| `HW-only forecasting`          | Holt-Winters forecast with threshold-based triggering               |
| `GRU-only forecasting`         | GRU forecast with threshold-based triggering                        |
| `Hybrid forecast-threshold TE` | HW-GRU forecast with adaptive threshold triggering                  |
| `RankGuard`                    | Proposed decision-aligned predictive traffic-engineering framework  |

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

## Citation

If you use this code, please cite:

```bibtex
@article{permal2026rankguard,
  title   = {RankGuard: Decision-Aligned Predictive Traffic Engineering for Software-Defined Data-Centre Networks},
  author  = {Permal, Satis Kumar and Ling, Teck Chaw},
  year    = {2026}
}
```

---
