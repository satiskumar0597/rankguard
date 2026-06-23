
from dataclasses import dataclass

@dataclass
class ScoreWeights:
    w_fct: float = 0.35
    w_mlu: float = 0.25
    w_cvar: float = 0.20
    w_rules: float = 0.15
    w_violations: float = 0.05

def risk_score(p99_fct_norm, mlu_norm, cvar_norm, delta_rules_norm, violations_norm, weights=ScoreWeights()):
    return (weights.w_fct*p99_fct_norm + weights.w_mlu*mlu_norm + weights.w_cvar*cvar_norm +
            weights.w_rules*delta_rules_norm + weights.w_violations*violations_norm)

def decision_gate(candidate_score, current_score, candidate_uncertainty, beta=0.5, actuation_cost=0.01, epsilon=0.03):
    return candidate_score + beta*candidate_uncertainty + actuation_cost < current_score - epsilon

class PersistenceGate:
    def __init__(self, h=2):
        self.h=h
        self.counter=0
    def update(self, condition_satisfied):
        if condition_satisfied:
            self.counter += 1
            if self.counter >= self.h:
                self.counter = 0
                return "install"
            return "defer"
        self.counter = 0
        return "keep"
