# RankGuard — Code Repository

**RankGuard: Decision-Aligned Predictive Traffic Engineering for Software-Defined Data-Centre Networks**

Satis Kumar Permal, Teck Chaw Ling — Universiti Malaya

---

## Repository structure

```
rankguard/
├── topology/
│   └── fattree_topo.py          Fat-tree / Clos topology for Mininet (k=4,6,8)
├── workload/
│   └── traffic_generator.py     Mixed workload generator (7 traffic classes)
├── predictor/
│   └── hw_gru_predictor.py      HW-GRU predictor + action-rank loss (eqs 4-11)
├── controller/
│   └── rankguard_controller.py  Risk-aware scoring + counterfactual actuation
├── evaluation/
│   └── evaluate.py              Metrics, baselines, Wilcoxon significance test
└── README.md
```

---

## Requirements

```
Python >= 3.9
torch >= 2.0
numpy
scipy
mininet >= 2.3          (for live emulation runs)
```

Install Python dependencies:
```bash
pip install torch numpy scipy
```

Install Mininet following https://mininet.org/download/

ONOS controller: https://wiki.onosproject.org/display/ONOS/Installing+and+Running+ONOS

---

## Reproducing the main result (Table 3)

### Trace-driven simulation (no Mininet required)

```bash
python evaluation/evaluate.py \
    --topology medium \
    --seeds 0 1 2 3 4 \
    --duration 1800 \
    --workload mixed \
    --out results_medium.json
```

### Controller-in-the-loop (Mininet + ONOS required)

1. Start ONOS:
```bash
./bin/onos-service start
```

2. Launch the fat-tree topology:
```bash
sudo python topology/fattree_topo.py --k 6 --controller 127.0.0.1 --port 6653
```

3. Generate traffic in Mininet CLI:
```bash
python workload/traffic_generator.py \
    --hosts 10.0.0.1 10.0.0.2 ... \
    --workload mixed \
    --duration 1800 \
    --seed 0 \
    --out flows_seed0.json
```

4. Run evaluation:
```bash
python evaluation/evaluate.py --topology medium --seeds 0 1 2 3 4
```

---

## Topology configurations

| Name   | k | Hosts | Switches | Directed links |
|--------|---|-------|----------|----------------|
| small  | 4 | 16    | 20       | 48             |
| medium | 6 | 54    | 45       | 162            |
| large  | 8 | 128   | 80       | 384            |

---

## Key hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| τ         | 0.8   | Congestion threshold (fraction of link capacity) |
| α         | 0.1   | Conformal miscoverage level (90% intervals) |
| ε         | 0.03  | Minimum improvement margin (3%) |
| h         | 2     | Hold-down window count |
| β         | 0.5   | Uncertainty penalty weight |
| K         | 4     | K-shortest-path candidates |
| w1–w5     | 0.35, 0.25, 0.20, 0.15, 0.05 | Route score weights |

---

## Equations implemented

| File | Equations |
|------|-----------|
| `hw_gru_predictor.py` | (3) EMA repair, (4)-(7) Holt-Winters, (8) HW-GRU, (9)-(11) decision-aware loss, (12)-(14) conformal calibration |
| `rankguard_controller.py` | (1)(15) route score, (2) decision condition, Algorithm 1 |
| `evaluation/evaluate.py` | Wilcoxon signed-rank test, FCT/MLU/CVaR metrics |

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

## Citation

If you use this code, please cite:

```
Permal, S. K. and Ling, T. C. (2026).
RankGuard: Decision-Aligned Predictive Traffic Engineering
for Software-Defined Data-Centre Networks.
```
