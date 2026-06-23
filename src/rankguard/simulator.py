
import numpy as np
from .controller import risk_score

def overflow_cvar(utilizations, tau=0.8, alpha=0.95):
    overflow=np.maximum(0,np.asarray(utilizations)-tau)
    cutoff=np.quantile(overflow,alpha)
    tail=overflow[overflow>=cutoff]
    return float(tail.mean()) if tail.size else 0.0

def evaluate_candidate_path(link_utils, delta_rules=1, violations=0, tau=0.8):
    link_utils=np.asarray(link_utils)
    mlu=float(link_utils.max())
    cvar=overflow_cvar(link_utils,tau=tau)
    p99_proxy=float(1.0/max(1e-3,1.0-min(0.99,mlu)))
    return risk_score(min(1,p99_proxy/10), min(1,mlu), min(1,cvar/0.2), min(1,delta_rules/10), min(1,violations/5))
