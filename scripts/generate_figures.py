
from pathlib import Path
import pandas as pd, numpy as np, matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]; RAW=ROOT/"data/raw"; FIG=ROOT/"figures"; FIG.mkdir(exist_ok=True)
BLUE="#1f77b4"; ORANGE="#ff7f0e"

def fig3():
    df=pd.read_csv(RAW/"mixed_workload_seed_results.csv")
    order=["ECMP","Reactive threshold","HW-only forecasting","GRU-only forecasting","Hybrid forecast-threshold TE","RankGuard"]; labels=["ECMP","Reactive","HW-only","GRU-only","Hybrid","RankGuard"]
    fig,ax=plt.subplots(figsize=(8,4.5))
    for i,p in enumerate(order):
        y=df[df.policy==p].p99_fct_ms.to_numpy()
        ax.scatter(np.full(len(y),i)+np.linspace(-.05,.05,len(y)),y,alpha=.65)
        ax.errorbar(i,y.mean(),yerr=y.std(ddof=1),color="black",marker="o",capsize=4)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(labels,rotation=25,ha="right"); ax.set_ylabel("p99 FCT (ms)"); ax.set_title("Tail flow completion time across policies"); ax.grid(axis="y",alpha=.25); fig.tight_layout(); fig.savefig(FIG/"p99_fct_comparison.png",dpi=300)

def dual_axis(csv, group_col, order, labels, left_col, right_col, title, left_label, right_label, out):
    df=pd.read_csv(RAW/csv); x=np.arange(len(order)); fig,ax1=plt.subplots(figsize=(8,4.5)); ax2=ax1.twinx()
    lm=[df[df[group_col]==p][left_col].mean() for p in order]; ls=[df[df[group_col]==p][left_col].std(ddof=1) for p in order]
    rm=[df[df[group_col]==p][right_col].mean() for p in order]; rs=[df[df[group_col]==p][right_col].std(ddof=1) for p in order]
    l1=ax1.errorbar(x,lm,yerr=ls,color=BLUE,marker="o",capsize=4,label=left_label)
    l2=ax2.errorbar(x,rm,yerr=rs,color=ORANGE,marker="s",linestyle="--",capsize=4,label=right_label)
    ax1.set_xticks(x); ax1.set_xticklabels(labels,rotation=25,ha="right")
    ax1.set_ylabel(left_label,color=BLUE); ax2.set_ylabel(right_label,color=ORANGE)
    ax1.tick_params(axis="y",colors=BLUE); ax2.tick_params(axis="y",colors=ORANGE)
    ax1.set_title(title); ax1.grid(axis="y",alpha=.25); ax1.legend([l1,l2],[left_label,right_label],loc="upper right")
    fig.tight_layout(); fig.savefig(FIG/out,dpi=300)

def fig5():
    df=pd.read_csv(RAW/"counterfactual_evaluator_heldout_actions.csv"); fig,ax=plt.subplots(figsize=(5,4.5))
    ax.scatter(df.replayed_mlu,df.estimated_mlu,alpha=.65,s=18); ax.plot([.5,1],[.5,1],"k--",lw=1)
    ax.set_xlabel("Replayed MLU"); ax.set_ylabel("Estimated MLU"); ax.set_title("Counterfactual evaluator fidelity"); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(FIG/"twin_fidelity.png",dpi=300)

def fig6():
    df=pd.read_csv(RAW/"drift_recovery_mlu_timeseries.csv"); fig,ax=plt.subplots(figsize=(7,4.5))
    for p in ["Reactive","HW-only","Hybrid","RankGuard"]:
        g=df[df.policy==p].groupby("control_window_after_shift").mlu; m=g.mean(); s=g.std()
        ax.plot(m.index,m.values,label=p); ax.fill_between(m.index,m.values-s.values,m.values+s.values,alpha=.12)
    ax.set_xlabel("Control windows after congestion shift"); ax.set_ylabel("Maximum link utilization"); ax.set_title("Recovery after congestion migration"); ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(FIG/"drift_recovery.png",dpi=300)

def fig8():
    df=pd.read_csv(RAW/"telemetry_loss_curve.csv"); fig,ax=plt.subplots(figsize=(7,4.5))
    ax.plot(df.loss_pct,df.ema_fidelity_pct,marker="o",label="EMA fidelity"); ax.plot(df.loss_pct,df.neighbour_aware_fidelity_pct,marker="s",label="Neighbour-aware fidelity"); ax.plot(df.loss_pct,df.interval_coverage_pct,marker="^",label="Interval coverage")
    ax.set_xlabel("Injected telemetry loss (%)"); ax.set_ylabel("Percentage"); ax.set_title("Telemetry-loss robustness"); ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(FIG/"telemetry_loss_robustness.png",dpi=300)

if __name__=="__main__":
    fig3()
    dual_axis("mixed_workload_seed_results.csv","policy",["ECMP","Reactive threshold","HW-only forecasting","GRU-only forecasting","Hybrid forecast-threshold TE","RankGuard"],["ECMP","Reactive","HW-only","GRU-only","Hybrid","RankGuard"],"mlu","congestion_probability_pct","Utilization risk across policies","Maximum link utilization","Congestion probability (%)","congestion_risk_comparison.png")
    fig5(); fig6()
    dual_axis("weight_sensitivity_seed_results.csv","policy_weighting",["Latency-heavy","Utilization-heavy","Churn-heavy","Balanced default"],["Latency-heavy","Utilization-heavy","Churn-heavy","Balanced"],"p99_fct_ms","route_changes","Policy-weight sensitivity","p99 FCT (ms)","Route changes","weight_sensitivity.png")
    fig8()
    dual_axis("ablation_seed_results.csv","variant",["RankGuard full method","No uncertainty calibration","No counterfactual validation","No decision-aware loss","No topology-aware expert","No churn penalty"],["Full","No uncert.","No cf. gate","No action\nrank","No topology","No churn"],"p99_fct_ms","route_changes","Component ablation","p99 FCT (ms)","Route changes","ablation_modules.png")
    print("Figures written to", FIG)
