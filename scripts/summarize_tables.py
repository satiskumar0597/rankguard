
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/"data/raw"; OUT=ROOT/"data/processed"; OUT.mkdir(exist_ok=True)
for file, group, cols in [
("mixed_workload_seed_results.csv",["policy"],["avg_fct_ms","p95_fct_ms","p99_fct_ms","mlu","congestion_probability_pct","route_changes"]),
("forecasting_decision_alignment.csv",["model"],["mae_pp","rmse_pp","mape_pct","coverage_pct","action_rank_agreement_pct"]),
("weight_sensitivity_seed_results.csv",["policy_weighting"],["p99_fct_ms","mlu","congestion_probability_pct","route_changes"]),
("ablation_seed_results.csv",["variant"],["p99_fct_ms","mlu","congestion_probability_pct","route_changes"]),
("scalability_controller_overhead_seed_results.csv",["hosts","switches","directed_links"],["median_decision_time_ms","p99_decision_time_ms"])]:
    df=pd.read_csv(RAW/file)
    df.groupby(group)[cols].agg(["mean","std"]).to_csv(OUT/("summary_"+file))
print("Summary tables written to", OUT)
