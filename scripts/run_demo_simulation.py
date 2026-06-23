
from pathlib import Path
import sys, pandas as pd, numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from rankguard.controller import decision_gate, PersistenceGate
from rankguard.simulator import evaluate_candidate_path
trace=pd.read_csv(ROOT/"data/traces/synthetic_small_topology_trace_seed1.csv")
gate=PersistenceGate(h=2); events=[]
for t in range(0,600,2):
    current=trace[trace.time_s==t].utilization.to_numpy()
    candidate=current.copy(); candidate[np.argsort(candidate)[-5:]]*=0.86
    current_score=evaluate_candidate_path(current,0); candidate_score=evaluate_candidate_path(candidate,2)
    uncertainty=0.025+0.04*float(current.max()>0.8)
    cond=decision_gate(candidate_score,current_score,uncertainty)
    events.append({"time_s":t,"current_score":current_score,"candidate_score":candidate_score,"uncertainty":uncertainty,"gate_condition":cond,"action":gate.update(cond),"current_mlu":current.max(),"candidate_mlu":candidate.max()})
out=ROOT/"data/processed/demo_gate_events.csv"; pd.DataFrame(events).to_csv(out,index=False); print("Wrote",out)
