"""
RankGuard: Risk-aware route scoring and counterfactual-validated actuation.
Implements equations (1), (2), (15) and Algorithm 1 from the paper.
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


# ── Telemetry repair ─────────────────────────────────────────────────────────

class MissingMetricsHandler:
    """
    EMA-based telemetry repair for short gaps (≤3 samples).
    Equation (3) in the paper.
    """

    MAX_SHORT_GAP = 3

    def __init__(self, lambda_ema: float = 0.7):
        self.lam = lambda_ema
        self._last: Dict[str, float] = {}
        self._smoothed: Dict[str, float] = {}
        self._gap_count: Dict[str, int] = {}
        self._imputed_count: Dict[str, int] = {}

    def repair(self, link_id: str,
               observed: Optional[float]) -> Tuple[float, bool]:
        """
        Returns (repaired_value, was_imputed).
        Records imputation frequency for confidence tracking.
        """
        if observed is not None:
            self._gap_count[link_id] = 0
            prev_smooth = self._smoothed.get(link_id, observed)
            self._smoothed[link_id] = (
                self.lam * observed + (1 - self.lam) * prev_smooth
            )
            self._last[link_id] = observed
            return observed, False

        # Missing sample
        self._gap_count[link_id] = self._gap_count.get(link_id, 0) + 1
        self._imputed_count[link_id] = self._imputed_count.get(link_id, 0) + 1

        last = self._last.get(link_id, 0.5)
        smooth = self._smoothed.get(link_id, last)
        repaired = self.lam * last + (1 - self.lam) * smooth
        self._smoothed[link_id] = repaired
        return float(np.clip(repaired, 0.0, 1.0)), True

    def confidence(self, link_id: str, window: int = 100) -> float:
        """Confidence in [0,1]: lower when many recent samples were imputed."""
        imputed = self._imputed_count.get(link_id, 0)
        return max(0.0, 1.0 - imputed / max(window, 1))


# ── Route scoring ─────────────────────────────────────────────────────────────

@dataclass
class CandidatePath:
    path_id: str
    links: List[str]
    predicted_utils: Dict[str, float]   # link_id → predicted utilization
    interval_widths: Dict[str, float]   # link_id → 2 * q^(1-alpha)
    n_rule_changes: int = 0
    class_violations: int = 0


def cvar_overflow(utils: List[float], threshold: float,
                  alpha: float = 0.95) -> float:
    """
    CVaR_alpha of (u - tau)^+ across links on the path.
    Expected overflow magnitude in worst (1-alpha) fraction.
    Equation (1), CVaR term.
    """
    overflows = [max(0.0, u - threshold) for u in utils]
    if not overflows:
        return 0.0
    cutoff = np.quantile(overflows, alpha)
    tail = [o for o in overflows if o >= cutoff]
    return float(np.mean(tail)) if tail else 0.0


def score_path(path: CandidatePath, tau: float = 0.8,
               w1: float = 0.35, w2: float = 0.25,
               w3: float = 0.20, w4: float = 0.15,
               w5: float = 0.05) -> float:
    """
    Risk-aware route score J(p). Equations (1) and (15).
    Lower score = better path.
    All components normalised to [0,1] before weighting.
    """
    utils = list(path.predicted_utils.values())
    if not utils:
        return 1.0

    # p99 FCT proxy: max predicted utilization on path
    p99_fct = float(np.percentile(utils, 99)) if len(utils) > 1 else utils[0]

    # MLU
    mlu = max(utils)

    # CVaR tail risk
    cvar = cvar_overflow(utils, tau)

    # Rule churn (normalised against a reasonable max of 20 changes)
    churn = min(path.n_rule_changes / 20.0, 1.0)

    # Class-specific violations (normalised against max 10)
    violations = min(path.class_violations / 10.0, 1.0)

    score = (w1 * p99_fct +
             w2 * mlu +
             w3 * cvar +
             w4 * churn +
             w5 * violations)
    return float(np.clip(score, 0.0, 1.0))


# ── Counterfactual evaluator ──────────────────────────────────────────────────

class CounterfactualEvaluator:
    """
    Lightweight replay-based counterfactual evaluator (digital twin).
    Estimates the effect of a candidate path change on MLU, tail latency,
    and rule churn without packet-level simulation.
    """

    def __init__(self, tau: float = 0.8):
        self.tau = tau
        self._trace_cache: Dict[str, np.ndarray] = {}

    def load_traces(self, link_id: str, trace: np.ndarray):
        """Register a historical utilization trace for a link."""
        self._trace_cache[link_id] = trace

    def estimate(self, path: CandidatePath,
                 current_utils: Dict[str, float]) -> Dict[str, float]:
        """
        Estimate post-reroute outcome for a candidate path.
        Returns dict with mlu, p99_fct_proxy, cvar, churn_cost.
        """
        # Shift load from current path to candidate path (simplified model)
        adjusted = {}
        for link_id, pred_u in path.predicted_utils.items():
            # Use trace variance as uncertainty proxy
            if link_id in self._trace_cache:
                noise = float(np.std(self._trace_cache[link_id][-20:]))
            else:
                noise = 0.05
            adjusted[link_id] = float(np.clip(pred_u + noise * 0.5, 0.0, 1.0))

        utils = list(adjusted.values())
        return {
            'mlu': max(utils) if utils else 1.0,
            'p99_fct_proxy': float(np.percentile(utils, 99))
                             if len(utils) > 1 else (utils[0] if utils else 1.0),
            'cvar': cvar_overflow(utils, self.tau),
            'churn_cost': path.n_rule_changes * 0.01
        }

    def rank_candidates(self,
                        candidates: List[CandidatePath],
                        current_utils: Dict[str, float]
                        ) -> List[Tuple[CandidatePath, float]]:
        """
        Score and rank candidate paths. Returns list sorted best→worst.
        """
        scored = []
        for path in candidates:
            est = self.estimate(path, current_utils)
            twin_path = CandidatePath(
                path_id=path.path_id,
                links=path.links,
                predicted_utils={k: est['mlu'] for k in path.links},
                interval_widths=path.interval_widths,
                n_rule_changes=path.n_rule_changes,
                class_violations=path.class_violations
            )
            j = score_path(twin_path)
            scored.append((path, j))
        scored.sort(key=lambda x: x[1])
        return scored


# ── Main RankGuard controller ─────────────────────────────────────────────────

class RankGuardController:
    """
    Per-window control loop. Algorithm 1 in the paper.

    At each window:
      1. Collect + repair telemetry.
      2. Forecast utilization intervals.
      3. Generate candidate paths for at-risk links.
      4. Score candidates via counterfactual evaluator.
      5. Apply reroute only if decision condition holds for h windows.
    """

    def __init__(self,
                 predictor,           # RegimeAdaptivePredictor instance
                 evaluator: CounterfactualEvaluator,
                 tau: float = 0.8,
                 beta: float = 0.5,   # uncertainty penalty weight
                 epsilon: float = 0.03,  # minimum improvement margin (3%)
                 h: int = 2,          # hold-down window count
                 w1: float = 0.35, w2: float = 0.25,
                 w3: float = 0.20, w4: float = 0.15, w5: float = 0.05):

        self.predictor = predictor
        self.evaluator = evaluator
        self.tau = tau
        self.beta = beta
        self.epsilon = epsilon
        self.h = h
        self.weights = (w1, w2, w3, w4, w5)

        self.mm_handler = MissingMetricsHandler()
        self._pending: Dict[str, int] = {}   # flow_id → consecutive-window count
        self._current_paths: Dict[str, CandidatePath] = {}
        self._decision_log: List[Dict] = []

    def _actuation_cost(self, path: CandidatePath) -> float:
        """Estimated actuation cost: proportional to rule changes."""
        return path.n_rule_changes * 0.005

    def step(self, window_t: int,
             raw_telemetry: Dict[str, Optional[float]],
             candidate_paths: Dict[str, List[CandidatePath]]
             ) -> Dict[str, str]:
        """
        One control-loop iteration. Algorithm 1.

        Args:
            window_t: current window index
            raw_telemetry: {link_id: utilization | None if missing}
            candidate_paths: {flow_id: [CandidatePath, ...]}

        Returns:
            decisions: {flow_id: 'install'|'defer'|'keep'}
        """
        # Step 1-2: repair telemetry
        repaired = {}
        confidences = {}
        for link_id, obs in raw_telemetry.items():
            val, imputed = self.mm_handler.repair(link_id, obs)
            repaired[link_id] = val
            confidences[link_id] = self.mm_handler.confidence(link_id)

        # Step 3: update predictor observations
        link_ids = list(repaired.keys())
        utils_arr = np.array([repaired[l] for l in link_ids])
        self.predictor.observe(utils_arr)

        # Step 4: forecast
        forecasts = {}
        intervals = {}
        for i, link_id in enumerate(link_ids):
            pred, interval = self.predictor.predict(i)
            forecasts[link_id] = pred
            intervals[link_id] = interval
            self.predictor.update_calibration(i, repaired[link_id], pred)

        # Step 5-9: evaluate candidates and apply decision rule
        decisions = {}
        for flow_id, candidates in candidate_paths.items():
            current = self._current_paths.get(flow_id)

            # Attach forecasts to candidates
            for cand in candidates:
                for link_id in cand.links:
                    if link_id in forecasts:
                        cand.predicted_utils[link_id] = forecasts[link_id]
                        cand.interval_widths[link_id] = (
                            intervals[link_id][1] - intervals[link_id][0]
                        )

            ranked = self.evaluator.rank_candidates(candidates, repaired)
            if not ranked:
                decisions[flow_id] = 'keep'
                continue

            best_path, best_score = ranked[0]

            # Score current path
            if current is not None:
                for link_id in current.links:
                    if link_id in forecasts:
                        current.predicted_utils[link_id] = forecasts[link_id]
                j_current = score_path(current, self.tau, *self.weights)
            else:
                j_current = 1.0

            # Uncertainty of best candidate
            sigma = float(np.mean(list(best_path.interval_widths.values()))
                          if best_path.interval_widths else 0.1)
            c_act = self._actuation_cost(best_path)

            # Decision condition: eq. (2)
            lhs = best_score + self.beta * sigma + c_act
            rhs = j_current - self.epsilon

            if lhs < rhs:
                self._pending[flow_id] = self._pending.get(flow_id, 0) + 1
            else:
                self._pending[flow_id] = 0

            # Hold-down check: condition must hold for h consecutive windows
            if self._pending.get(flow_id, 0) >= self.h:
                decisions[flow_id] = 'install'
                self._current_paths[flow_id] = best_path
                self._pending[flow_id] = 0
            elif lhs < rhs:
                decisions[flow_id] = 'defer'
            else:
                decisions[flow_id] = 'keep'

            self._decision_log.append({
                'window': window_t,
                'flow_id': flow_id,
                'decision': decisions[flow_id],
                'j_best': best_score,
                'j_current': j_current,
                'sigma': sigma,
                'consecutive': self._pending.get(flow_id, 0)
            })

        return decisions

    def decision_log(self) -> List[Dict]:
        return list(self._decision_log)
