
import numpy as np

def pinball_loss(y_true, y_pred, q):
    err=np.asarray(y_true)-np.asarray(y_pred)
    return float(np.mean(np.maximum(q*err,(q-1)*err)))

def under_risk_loss(u_true, u_pred, tau=0.8, delta=0.05):
    u_true=np.asarray(u_true); u_pred=np.asarray(u_pred)
    return float(np.mean(np.maximum(0,u_true-u_pred)*np.maximum(0,u_true-tau+delta)))

def action_rank_loss(predicted_scores, evaluator_preferences, margin=0.05):
    scores=np.asarray(predicted_scores)
    losses=[]
    for i,j,s in evaluator_preferences:
        losses.append(max(0.0, margin - s*(scores[j]-scores[i])))
    return float(np.mean(losses)) if losses else 0.0
