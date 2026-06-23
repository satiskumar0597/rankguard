
import numpy as np

def nonconformity_scores(y_true, y_pred):
    return np.abs(np.asarray(y_true)-np.asarray(y_pred))

def conformal_radius(recent_residuals, alpha=0.1):
    return float(np.quantile(np.asarray(recent_residuals), 1-alpha))

def prediction_interval(point_forecast, radius):
    return point_forecast-radius, point_forecast+radius
