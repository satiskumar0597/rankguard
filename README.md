# RankGuard — Code Repository

**RankGuard: Decision-Aligned Predictive Traffic Engineering for Software-Defined Data-Centre Networks**

Satis Kumar Permal, Teck Chaw Ling — Universiti Malaya

---

## Repository structure

```
rankguard/
├── topology/
│   └── fattree_topo.py              Fat-tree / Clos topology for Mininet (k=4,6,8)
├── workload/
│   └── traffic_generator.py         Mixed workload generator (7 traffic classes)
├── predictor/
│   └── hw_gru_predictor.py          HW-GRU predictor + action-rank loss (eqs 4-11)
├── controller/
│   └── rankguard_controller.py      Risk-aware scoring + counterfactual actuation
├── evaluation/
│   └── evaluate.py                  Metrics, baselines, Wilcoxon significance test
├── ablation/
│   └── run_ablations.py             Table 10 + Figure 9
├── sensitivity/
│   └── weight_sensitivity.py        Table 8  + Figure 7
├── telemetry_stress/
│   └── telemetry_loss_stress.py     Table 9  + Figure 8
├── drift/
│   └── drift_recovery.py            Table 7  + Figure 6
├── evaluator_fidelity/
│   └── evaluator_fidelity.py        Table 6  + Figure 5
├── plotting/
│   └── plot_figures.py              Figures 3–9 from result JSON files
└── README.md
```

---

## Requirements

```
Python  >= 3.9
torch   >= 2.0
numpy
scipy
matplotlib >= 3.7
mininet >= 2.3          (for live emulation runs only)
```

Install Python dependencies:

```bash
pip install torch numpy scipy matplotlib
```

Install Mininet: https://mininet.org/download/

ONOS controller: https://wiki.onosproject.org/display/ONOS/Installing+and+Running+ONOS

---

## Reproducing the main result (Table 3)

### Trace-driven simulation — no Mininet required

```bash
python evaluation/evaluate.py \
    --topology medium \
    --seeds 0 1 2 3 4 \
    --duration 1800 \
    --workload mixed \
    --out results/results_medium.json
```

### Controller-in-the-loop — Mininet + ONOS required

**1. Start ONOS:**
```bash
./bin/onos-service start
```

**2. Launch the fat-tree topology:**
```bash
sudo python topology/fattree_topo.py --k 6 --controller 127.0.0.1 --port 6653
```

**3. Generate traffic:**
```bash
python workload/traffic_generator.py \
    --hosts 10.0.0.1 10.0.0.2 ... \
    --workload mixed \
    --duration 1800 \
    --seed 0 \
    --out flows_seed0.json
```

**4. Run evaluation:**
```bash
python evaluation/evaluate.py --topology medium --seeds 0 1 2 3 4
```

---

## Reproducing all tables and figures

Run the scripts below in order. Each writes a JSON file consumed by the plotter.

```bash
mkdir -p results figures

# Table 10 + Figure 9 — ablation study
python ablation/run_ablations.py \
    --seeds 0 1 2 3 4 --out results/ablation_results.json

# Table 8 + Figure 7 — weight sensitivity
python sensitivity/weight_sensitivity.py \
    --seeds 0 1 2 3 4 --out results/sensitivity_results.json

# Table 9 + Figure 8 — telemetry-loss robustness
python telemetry_stress/telemetry_loss_stress.py \
    --seeds 0 1 2 3 4 --out results/telemetry_loss_results.json

# Table 7 + Figure 6 — post-drift recovery
python drift/drift_recovery.py \
    --seeds 0 1 2 3 4 --out results/drift_results.json

# Table 6 + Figure 5 — counterfactual evaluator fidelity
python evaluator_fidelity/evaluator_fidelity.py \
    --seeds 0 1 2 3 4 --out results/fidelity_results.json

# Figures 3–9 — generate all plots
python plotting/plot_figures.py \
    --results_dir results --out_dir figures
```

Figures are written as PDF to `figures/`. The plotter skips any figure
whose JSON file is missing rather than crashing.

---

## Script-to-paper mapping

| Script | Table | Figure | Description |
|--------|-------|--------|-------------|
| `evaluation/evaluate.py` | 3, 4, 5, 11 | 3, 4 | End-to-end TE performance |
| `evaluator_fidelity/evaluator_fidelity.py` | 6 | 5 | Counterfactual evaluator fidelity |
| `drift/drift_recovery.py` | 7 | 6 | Post-drift recovery after congestion migration |
| `sensitivity/weight_sensitivity.py` | 8 | 7 | Route-scoring weight sensitivity |
| `telemetry_stress/telemetry_loss_stress.py` | 9 | 8 | Telemetry-loss robustness |
| `ablation/run_ablations.py` | 10 | 9 | Component ablation study |

---

## Topology configurations

| Name | k | Hosts | Switches | Directed links |
|------|---|-------|----------|----------------|
| small | 4 | 16 | 20 | 48 |
| medium | 6 | 54 | 45 | 162 |
| large | 8 | 128 | 80 | 384 |

---

## Key hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| τ | 0.8 | Congestion threshold (fraction of link capacity) |
| α | 0.1 | Conformal miscoverage level (90% intervals) |
| ε | 0.03 | Minimum improvement margin (3%) |
| h | 2 | Hold-down window count |
| β | 0.5 | Uncertainty penalty weight |
| K | 4 | K-shortest-path candidates |
| w1–w5 | 0.35, 0.25, 0.20, 0.15, 0.05 | Route score weights (balanced default) |

---

## Equations implemented

| File | Equations |
|------|-----------|
| `predictor/hw_gru_predictor.py` | (3) EMA repair, (4)–(7) Holt-Winters, (8) HW-GRU residual, (9)–(11) decision-aware loss, (12)–(14) conformal calibration |
| `controller/rankguard_controller.py` | (1)(15) route score J(p), (2) decision condition, Algorithm 1 |
| `evaluation/evaluate.py` | Wilcoxon signed-rank test, FCT / MLU / CVaR metrics |
| `ablation/run_ablations.py` | Component-removal variants of the full system |
| `sensitivity/weight_sensitivity.py` | Weight configurations for eq. (1) |
| `telemetry_stress/telemetry_loss_stress.py` | EMA (eq. 3) and neighbour-aware imputation |
| `drift/drift_recovery.py` | Post-drift MLU trajectory and recovery measurement |
| `evaluator_fidelity/evaluator_fidelity.py` | Spearman ρ and top-1 agreement for counterfactual evaluator |

---

## Baselines

| Policy | Description |
|--------|-------------|
| ECMP | Static equal-cost multipath, no rerouting |
| Reactive threshold | Reroute when measured utilization exceeds τ |
| HW-only forecasting | Holt-Winters forecast + threshold trigger |
| GRU-only forecasting | GRU forecast + threshold trigger |
| Hybrid forecast-threshold TE | HW-GRU forecast + adaptive threshold (strongest baseline) |
| RankGuard | This work |

---

## Statistical test

All main comparisons use a paired Wilcoxon signed-rank test (p < 0.05) against
the hybrid forecast-threshold baseline. Each policy is run on the same seed and
workload instance, so paired rather than independent tests are used.

---

## Notes on live vs simulation runs

All scripts contain a `simulate_*` function that approximates the
corresponding experiment without requiring Mininet or ONOS. To run
against a live controller, replace the body of that function with
calls to the actual controller using the topology and workload scripts
in `topology/` and `workload/`. Function signatures and output data
structures are identical in both modes.

---

## Citation

If you use this code, please cite:

```
Permal, S. K. and Ling, T. C. (2026).
RankGuard: Decision-Aligned Predictive Traffic Engineering
for Software-Defined Data-Centre Networks.
```
