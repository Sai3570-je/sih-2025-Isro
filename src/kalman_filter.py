"""
Kalman Filter implementation for GNSS error prediction
8D state: [X, X_dot, Y, Y_dot, Z, Z_dot, Clock, Clock_dot]
"""
import numpy as np
import logging

logger = logging.getLogger(__name__)

def build_state_matrices(dt=900.0):
    """
    Build state transition matrix F and observation matrix H
    
    Args:
        dt: Time step in seconds (default 900s = 15 minutes)
    
    Returns:
        tuple: (F, H)
    """
    # 8D state: [X, X_dot, Y, Y_dot, Z, Z_dot, Clock, Clock_dot]
    F = np.eye(8)
    
    # Position dynamics: X_next = X + dt * X_dot
    F[0, 1] = dt  # X += dt * X_dot
    F[2, 3] = dt  # Y += dt * Y_dot
    F[4, 5] = dt  # Z += dt * Z_dot
    F[6, 7] = dt  # Clock += dt * Clock_dot
    
    # Observation matrix: we observe [X, Y, Z, Clock]
    H = np.zeros((4, 8))
    H[0, 0] = 1  # Observe X
    H[1, 2] = 1  # Observe Y
    H[2, 4] = 1  # Observe Z
    H[3, 6] = 1  # Observe Clock
    
    return F, H

def initialize_covariance_matrices(position_var=100.0, velocity_var=1.0, 
                                   clock_var=1e4, clock_drift_var=10.0,
                                   meas_var_pos=1.0, meas_var_clock=100.0):
    """
    Initialize process noise Q and measurement noise R
    
    Returns:
        tuple: (Q, R, P0)
    """
    # Process noise Q (8x8)
    Q = np.diag([
        position_var, velocity_var,  # X, X_dot
        position_var, velocity_var,  # Y, Y_dot
        position_var, velocity_var,  # Z, Z_dot
        clock_var, clock_drift_var   # Clock, Clock_dot
    ])
    
    # Measurement noise R (4x4)
    R = np.diag([
        meas_var_pos,    # X measurement
        meas_var_pos,    # Y measurement
        meas_var_pos,    # Z measurement
        meas_var_clock   # Clock measurement
    ])
    
    # Initial state covariance P0 (8x8)
    P0 = np.diag([
        position_var, velocity_var,
        position_var, velocity_var,
        position_var, velocity_var,
        clock_var, clock_drift_var
    ])
    
    return Q, R, P0

def estimate_measurement_noise(df_train):
    """
    Estimate measurement noise R from training data variance
    
    Args:
        df_train: Training DataFrame
    
    Returns:
        R: 4x4 measurement noise covariance matrix
    """
    target_cols = ['X_Error', 'Y_Error', 'Z_Error', 'Clock_Error']
    
    variances = []
    for col in target_cols:
        if col in df_train.columns:
            var = df_train[col].dropna().var()
            variances.append(var if var > 0 else 1.0)
        else:
            variances.append(1.0)
    
    R = np.diag(variances)
    logger.info(f"Estimated R diagonal: {variances}")
    
    return R

class KalmanFilter8D:
    """8-dimensional Kalman Filter for GNSS errors"""
    
    def __init__(self, dt=900.0):
        """
        Initialize Kalman Filter
        
        Args:
            dt: Time step in seconds
        """
        self.dt = dt
        self.F, self.H = build_state_matrices(dt)
        self.Q, self.R, self.P = initialize_covariance_matrices()
        
        # State: [X, X_dot, Y, Y_dot, Z, Z_dot, Clock, Clock_dot]
        self.x = np.zeros((8, 1))
        
        # Innovation tracking
        self.innovations = []
        self.residuals = []
    
    def initialize_state(self, x_obs, y_obs, z_obs, clock_obs):
        """Initialize state from first observation"""
        self.x = np.array([
            [x_obs], [0],      # X, X_dot
            [y_obs], [0],      # Y, Y_dot
            [z_obs], [0],      # Z, Z_dot
            [clock_obs], [0]   # Clock, Clock_dot
        ])
    
    def predict(self):
        """Kalman predict step"""
        # Predict state: x = F * x
        self.x = self.F @ self.x
        
        # Predict covariance: P = F * P * F' + Q
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return self.x.copy(), self.P.copy()
    
    def update(self, z):
        """
        Kalman update step
        
        Args:
            z: Measurement vector [X, Y, Z, Clock] (4x1)
        """
        z = np.array(z).reshape(4, 1)
        
        # Innovation: y = z - H * x
        y = z - self.H @ self.x
        self.innovations.append(y.flatten())
        
        # Innovation covariance: S = H * P * H' + R
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain: K = P * H' * inv(S)
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state: x = x + K * y
        self.x = self.x + K @ y
        
        # Update covariance: P = (I - K * H) * P
        I = np.eye(8)
        self.P = (I - K @ self.H) @ self.P
        
        # Store residual
        residual = z - self.H @ self.x
        self.residuals.append(residual.flatten())
        
        return self.x.copy()
    
    def get_observation(self):
        """Extract observable components from state"""
        return self.H @ self.x  # Returns [X, Y, Z, Clock]

def run_kalman_forward(df_sat, Q, R, x0, P0, F, H):
    """
    Run Kalman filter forward on satellite data
    
    Args:
        df_sat: Satellite DataFrame
        Q: Process noise covariance
        R: Measurement noise covariance
        x0: Initial state (8x1)
        P0: Initial covariance (8x8)
        F: State transition matrix
        H: Observation matrix
    
    Returns:
        dict: {
            'filtered_states': list of states,
            'predicted_obs': list of predictions,
            'innovations': list of innovations,
            'final_state': final x,
            'final_cov': final P
        }
    """
    kf = KalmanFilter8D()
    kf.F = F
    kf.H = H
    kf.Q = Q
    kf.R = R
    kf.x = x0.copy()
    kf.P = P0.copy()
    
    filtered_states = []
    predicted_obs = []
    timestamps = []
    
    target_cols = ['X_Error', 'Y_Error', 'Z_Error', 'Clock_Error']
    
    for idx, row in df_sat.iterrows():
        # Predict
        x_pred, P_pred = kf.predict()
        
        # Get predicted observation
        y_pred = kf.get_observation()
        predicted_obs.append(y_pred.flatten())
        timestamps.append(row['timestamp'])
        
        # Update if measurement available
        if all(col in row and not pd.isna(row[col]) for col in target_cols):
            z = np.array([row[col] for col in target_cols])
            kf.update(z)
        
        filtered_states.append(kf.x.copy())
    
    return {
        'filtered_states': filtered_states,
        'predicted_obs': np.array(predicted_obs),
        'innovations': np.array(kf.innovations),
        'residuals': np.array(kf.residuals),
        'final_state': kf.x,
        'final_cov': kf.P,
        'timestamps': timestamps
    }

def forecast_day8(last_state, last_P, steps, F, Q, H):
    """
    Forecast Day 8 using pure prediction (no updates)
    
    Args:
        last_state: Final state from training (8x1)
        last_P: Final covariance from training (8x8)
        steps: Number of forecast steps
        F: State transition matrix
        Q: Process noise
        H: Observation matrix
    
    Returns:
        numpy array: Predicted observations (steps x 4)
    """
    kf = KalmanFilter8D()
    kf.F = F
    kf.H = H
    kf.Q = Q
    kf.x = last_state.copy()
    kf.P = last_P.copy()
    
    predictions = []
    
    for _ in range(steps):
        # Predict only (no update)
        kf.predict()
        
        # Get predicted observation
        y_pred = kf.get_observation()
        predictions.append(y_pred.flatten())
    
    return np.array(predictions)

import pandas as pd  # Add this for the run_kalman_forward function
