"""
RankGuard: Decision-aligned HW-GRU predictor with action-rank loss.
Implements equations (4)-(11) from the paper.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


# ── Holt-Winters component ───────────────────────────────────────────────────

class HoltWinters:
    """
    Additive Holt-Winters with level, trend, and seasonality.
    Equations (4)-(7) in the paper.
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.1,
                 gamma: float = 0.2, season_len: int = 60):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.season_len = season_len
        self.L = None   # level
        self.T = None   # trend
        self.S = []     # seasonal indices

    def fit(self, series: np.ndarray):
        p = self.season_len
        if len(series) < 2 * p:
            series = np.pad(series, (2 * p - len(series), 0), mode='edge')

        self.L = np.mean(series[:p])
        self.T = (np.mean(series[p:2*p]) - np.mean(series[:p])) / p
        self.S = list(series[:p] - self.L)
        self._history = list(series)

        for i in range(p, len(series)):
            y = series[i]
            s_prev = self.S[i % p]
            L_prev, T_prev = self.L, self.T
            self.L = self.alpha * (y - s_prev) + (1 - self.alpha) * (L_prev + T_prev)
            self.T = self.beta * (self.L - L_prev) + (1 - self.beta) * T_prev
            self.S[i % p] = self.gamma * (y - self.L) + (1 - self.gamma) * s_prev

    def predict(self, h: int = 1) -> np.ndarray:
        """Forecast h steps ahead. Returns array of length h."""
        preds = []
        p = self.season_len
        for m in range(1, h + 1):
            s_idx = (len(self._history) + m - 1) % p
            yhat = self.L + m * self.T + self.S[s_idx]
            preds.append(float(np.clip(yhat, 0.0, 1.0)))
        return np.array(preds)

    def update(self, new_obs: float):
        """Online update with a new observation."""
        p = self.season_len
        idx = len(self._history) % p
        s_prev = self.S[idx]
        L_prev, T_prev = self.L, self.T
        self.L = self.alpha * (new_obs - s_prev) + (1 - self.alpha) * (L_prev + T_prev)
        self.T = self.beta * (self.L - L_prev) + (1 - self.beta) * T_prev
        self.S[idx] = self.gamma * (new_obs - self.L) + (1 - self.gamma) * s_prev
        self._history.append(new_obs)


# ── GRU residual model ───────────────────────────────────────────────────────

class GRUResidual(nn.Module):
    """
    GRU that predicts the HW residual r_t = y_t - yhat_HW_t.
    Input: window of recent residuals + topology features.
    Output: next-step residual correction delta_{t+1}.
    Equation (8) in the paper.
    """

    def __init__(self, input_dim: int = 1, hidden_dim: int = 32,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, input_dim) → (batch, 1)"""
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


# ── Decision-aware loss ──────────────────────────────────────────────────────

class DecisionAwareLoss(nn.Module):
    """
    Combined loss: pinball + under-risk + action-rank.
    Equation (9) in the paper.
    """

    def __init__(self, quantile_lo: float = 0.05, quantile_hi: float = 0.95,
                 lambda_risk: float = 0.5, eta_rank: float = 1.0,
                 congestion_threshold: float = 0.8, delta: float = 0.05,
                 margin_gamma: float = 0.1):
        super().__init__()
        self.q_lo = quantile_lo
        self.q_hi = quantile_hi
        self.lambda_risk = lambda_risk
        self.eta_rank = eta_rank
        self.tau = congestion_threshold
        self.delta = delta
        self.gamma = margin_gamma

    def pinball(self, pred: torch.Tensor, target: torch.Tensor,
                q: float) -> torch.Tensor:
        err = target - pred
        return torch.mean(torch.where(err >= 0, q * err, (q - 1) * err))

    def under_risk(self, pred: torch.Tensor,
                   target: torch.Tensor) -> torch.Tensor:
        """
        Equation (10): extra penalty for underprediction near threshold tau.
        """
        under = torch.clamp(target - pred, min=0.0)
        near_sat = torch.clamp(target - self.tau + self.delta, min=0.0)
        return torch.mean(under * near_sat)

    def action_rank(self, pred_scores: torch.Tensor,
                    evaluator_scores: torch.Tensor) -> torch.Tensor:
        """
        Equation (11): pairwise ranking loss over candidate actions.
        pred_scores: (n_actions,) route scores from predictor
        evaluator_scores: (n_actions,) route scores from counterfactual evaluator
        """
        n = pred_scores.shape[0]
        if n < 2:
            return torch.tensor(0.0, requires_grad=True)

        loss = torch.tensor(0.0)
        count = 0
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # evaluator says action i is preferred over j
                s_psi = float(evaluator_scores[i] < evaluator_scores[j])
                if s_psi > 0:
                    violation = torch.clamp(
                        self.gamma - (pred_scores[j] - pred_scores[i]),
                        min=0.0
                    )
                    loss = loss + violation
                    count += 1
        return loss / max(count, 1)

    def forward(self, pred_lo: torch.Tensor, pred_hi: torch.Tensor,
                target: torch.Tensor,
                pred_scores: Optional[torch.Tensor] = None,
                evaluator_scores: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, Dict[str, float]]:

        l_q = self.pinball(pred_lo, target, self.q_lo) + \
              self.pinball(pred_hi, target, self.q_hi)
        l_ur = self.under_risk(pred_lo, target)

        l_ar = torch.tensor(0.0)
        if pred_scores is not None and evaluator_scores is not None:
            l_ar = self.action_rank(pred_scores, evaluator_scores)

        total = l_q + self.lambda_risk * l_ur + self.eta_rank * l_ar
        return total, {
            'pinball': l_q.item(),
            'under_risk': l_ur.item(),
            'action_rank': l_ar.item()
        }


# ── Conformal calibration ────────────────────────────────────────────────────

class ConformalCalibrator:
    """
    Sliding-window conformal calibration for per-link prediction intervals.
    Equations (12)-(14) in the paper.
    """

    def __init__(self, window: int = 100, alpha: float = 0.1):
        self.window = window
        self.alpha = alpha          # miscoverage level → 1-alpha = 90% intervals
        self._residuals: List[float] = []

    def update(self, observed: float, predicted: float):
        r = abs(observed - predicted)
        self._residuals.append(r)
        if len(self._residuals) > self.window:
            self._residuals.pop(0)

    def radius(self) -> float:
        """Empirical (1-alpha) quantile of recent residuals."""
        if not self._residuals:
            return 0.1
        return float(np.quantile(self._residuals, 1 - self.alpha))

    def interval(self, pred: float) -> Tuple[float, float]:
        r = self.radius()
        return (max(0.0, pred - r), min(1.0, pred + r))


# ── Regime-adaptive expert combiner ──────────────────────────────────────────

class RegimeAdaptivePredictor:
    """
    Combines seasonal-residual, topology-aware, and burst experts.
    Expert weights updated after each window using residual + action-rank error.
    """

    def __init__(self, n_links: int, history_len: int = 60,
                 season_len: int = 60, device: str = 'cpu'):
        self.n_links = n_links
        self.history_len = history_len
        self.device = device

        # One HW + GRU per link
        self.hw = [HoltWinters(season_len=season_len) for _ in range(n_links)]
        self.gru = GRUResidual(input_dim=1).to(device)

        # Topology-aware GRU (takes link + neighbour features)
        self.topo_gru = GRUResidual(input_dim=4).to(device)

        # Burst expert (compact)
        self.burst_gru = GRUResidual(input_dim=3).to(device)

        # Expert weights (seasonal-residual, topology, burst)
        self.weights = np.array([1/3, 1/3, 1/3])

        self.calibrators = [ConformalCalibrator() for _ in range(n_links)]
        self._history = [[] for _ in range(n_links)]

    def observe(self, link_utils: np.ndarray):
        """Record a new telemetry observation for all links."""
        for i, u in enumerate(link_utils):
            self._history[i].append(float(u))
            if len(self._history[i]) > self.history_len:
                self._history[i].pop(0)
            if len(self._history[i]) >= 10:
                self.hw[i].fit(np.array(self._history[i]))

    def predict(self, link_idx: int) -> Tuple[float, Tuple[float, float]]:
        """
        Predict next-window utilization for link_idx.
        Returns (point_estimate, (lower_bound, upper_bound)).
        """
        hist = np.array(self._history[link_idx])
        if len(hist) < 10:
            mid = float(hist[-1]) if len(hist) > 0 else 0.5
            return mid, (max(0.0, mid - 0.1), min(1.0, mid + 0.1))

        # Expert 1: HW-GRU seasonal-residual
        hw_pred = self.hw[link_idx].predict(1)[0]
        residuals = hist - np.array([
            self.hw[link_idx].predict(1)[0]
            for _ in range(len(hist))
        ])
        res_tensor = torch.tensor(
            residuals[-self.history_len:].reshape(1, -1, 1),
            dtype=torch.float32
        ).to(self.device)
        with torch.no_grad():
            delta = self.gru(res_tensor).item()
        pred_sr = np.clip(hw_pred + delta, 0.0, 1.0)

        # Expert 2: topology-aware (simplified — uses own history only here)
        t_in = torch.tensor(
            hist[-self.history_len:].reshape(1, -1, 1).repeat(4, axis=2),
            dtype=torch.float32
        ).to(self.device)
        with torch.no_grad():
            pred_topo = float(np.clip(self.topo_gru(t_in).item(), 0.0, 1.0))

        # Expert 3: burst (uses gradient + variance features)
        grad = np.gradient(hist[-20:]) if len(hist) >= 20 else np.zeros(3)
        burst_feat = np.stack([
            hist[-min(20, len(hist)):],
            np.abs(np.gradient(hist[-min(20, len(hist)):])),
            np.full(min(20, len(hist)), np.var(hist[-min(20, len(hist)):]))
        ], axis=1)
        b_in = torch.tensor(
            burst_feat[-self.history_len:].reshape(1, -1, 3),
            dtype=torch.float32
        ).to(self.device)
        with torch.no_grad():
            pred_burst = float(np.clip(self.burst_gru(b_in).item(), 0.0, 1.0))

        preds = np.array([pred_sr, pred_topo, pred_burst])
        point = float(np.dot(self.weights, preds))
        point = np.clip(point, 0.0, 1.0)

        cal = self.calibrators[link_idx]
        interval = cal.interval(point)
        return point, interval

    def update_weights(self, residual_errors: np.ndarray,
                       rank_errors: np.ndarray):
        """
        Reweight experts based on recent residual + action-rank error.
        residual_errors: (3,) — one per expert
        rank_errors: (3,) — action-rank disagreement per expert
        """
        combined = residual_errors + rank_errors
        inv = 1.0 / (combined + 1e-6)
        self.weights = inv / inv.sum()

    def update_calibration(self, link_idx: int,
                           observed: float, predicted: float):
        self.calibrators[link_idx].update(observed, predicted)
