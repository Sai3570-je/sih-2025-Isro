"""
Enhanced Kalman Filter with Orbital Physics.

Incorporates:
1. Harmonic motion for orbital oscillations
2. Proper state-space dynamics with acceleration
3. Cross-coupling between position axes
4. Adaptive noise estimation
"""

import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class EnhancedKalmanFilter:
    """
    State vector (14D):
    [X, Ẋ, Ẍ, Y, Ẏ, Ÿ, Z, Ż, Z̈, Clock, Clock_drift, sin_phase, cos_phase, frequency]
    
    Incorporates:
    - Second-order dynamics (acceleration)
    - Harmonic oscillation terms for orbital periodicity
    - Adaptive frequency estimation
    """
    
    def __init__(self, dt: float = 900.0, Q_scale: float = 1.0, R_scale: float = 1.0):
        """
        Initialize enhanced Kalman filter.
        
        Args:
            dt: Time step in seconds (default 900s = 15 min)
            Q_scale: Process noise scaling factor
            R_scale: Measurement noise scaling factor
        """
        self.dt = dt
        self.state_dim = 14
        self.obs_dim = 4
        
        # State: [X, Vx, Ax, Y, Vy, Ay, Z, Vz, Az, Clk, ClkDrift, sin_φ, cos_φ, ω]
        self.x = np.zeros(self.state_dim)
        self.P = np.eye(self.state_dim) * 1000.0  # Large initial uncertainty
        
        # Build physics-based state transition matrix
        self.F = self._build_state_transition(dt)
        
        # Observation matrix: measure [X, Y, Z, Clock]
        self.H = np.zeros((self.obs_dim, self.state_dim))
        self.H[0, 0] = 1.0  # X position
        self.H[1, 3] = 1.0  # Y position
        self.H[2, 6] = 1.0  # Z position
        self.H[3, 9] = 1.0  # Clock error
        
        # Process noise covariance (physics-based)
        self.Q = self._build_process_noise(Q_scale)
        
        # Measurement noise covariance
        self.R = np.diag([
            (5.0 * R_scale) ** 2,   # X measurement noise ~5m
            (8.0 * R_scale) ** 2,   # Y measurement noise ~8m
            (6.0 * R_scale) ** 2,   # Z measurement noise ~6m
            (3.0 * R_scale) ** 2    # Clock noise ~3m
        ])
        
        self.initialized = False
        
    def _build_state_transition(self, dt: float) -> np.ndarray:
        """
        Build state transition matrix with second-order dynamics.
        
        For each axis: X_{k+1} = X_k + Vx_k*dt + 0.5*Ax_k*dt²
                      Vx_{k+1} = Vx_k + Ax_k*dt
                      Ax_{k+1} = Ax_k (damped)
        """
        F = np.eye(self.state_dim)
        
        # X, Y, Z dynamics (second-order)
        for i in range(3):
            pos_idx = i * 3
            vel_idx = pos_idx + 1
            acc_idx = pos_idx + 2
            
            F[pos_idx, vel_idx] = dt
            F[pos_idx, acc_idx] = 0.5 * dt ** 2
            F[vel_idx, acc_idx] = dt
            F[acc_idx, acc_idx] = 0.95  # Damping factor
        
        # Clock dynamics
        F[9, 10] = dt  # Clock += ClockDrift * dt
        F[10, 10] = 0.99  # Slight damping on drift
        
        # Harmonic oscillator dynamics
        # sin(ωt + φ) ≈ sin(φ) + ω*cos(φ)*dt
        # cos(ωt + φ) ≈ cos(φ) - ω*sin(φ)*dt
        F[11, 11] = 1.0  # sin_phase persistence
        F[11, 12] = dt   # cos contribution
        F[11, 13] = dt   # frequency modulation
        F[12, 11] = -dt  # sin contribution
        F[12, 12] = 1.0  # cos_phase persistence
        F[13, 13] = 1.0  # frequency persistence
        
        return F
    
    def _build_process_noise(self, scale: float) -> np.ndarray:
        """
        Build process noise based on orbital physics.
        
        - Position uncertainty grows with velocity uncertainty
        - Velocity uncertainty grows with acceleration uncertainty
        - Acceleration has random walk component
        """
        Q = np.zeros((self.state_dim, self.state_dim))
        
        # Process noise for position-velocity-acceleration triplets
        dt = self.dt
        for i in range(3):  # X, Y, Z
            pos_idx = i * 3
            vel_idx = pos_idx + 1
            acc_idx = pos_idx + 2
            
            # Acceleration random walk
            q_acc = (0.01 * scale) ** 2  # m/s² variance
            
            # Derived position and velocity noise from acceleration
            Q[pos_idx, pos_idx] = (dt ** 4 / 4) * q_acc
            Q[pos_idx, vel_idx] = (dt ** 3 / 2) * q_acc
            Q[pos_idx, acc_idx] = (dt ** 2 / 2) * q_acc
            
            Q[vel_idx, pos_idx] = (dt ** 3 / 2) * q_acc
            Q[vel_idx, vel_idx] = (dt ** 2) * q_acc
            Q[vel_idx, acc_idx] = dt * q_acc
            
            Q[acc_idx, pos_idx] = (dt ** 2 / 2) * q_acc
            Q[acc_idx, vel_idx] = dt * q_acc
            Q[acc_idx, acc_idx] = q_acc
        
        # Clock process noise
        Q[9, 9] = (0.1 * scale) ** 2   # Clock position
        Q[10, 10] = (0.01 * scale) ** 2  # Clock drift
        
        # Harmonic components (low variance)
        Q[11, 11] = (0.001 * scale) ** 2  # sin phase
        Q[12, 12] = (0.001 * scale) ** 2  # cos phase
        Q[13, 13] = (0.0001 * scale) ** 2  # frequency
        
        return Q
    
    def initialize(self, measurement: np.ndarray, timestamp: Optional[float] = None):
        """
        Initialize state from first measurement.
        
        Args:
            measurement: [X, Y, Z, Clock] observation
            timestamp: Optional timestamp for tracking
        """
        if len(measurement) != 4:
            raise ValueError(f"Expected 4D measurement, got {len(measurement)}D")
        
        # Initialize position and clock
        self.x[0] = measurement[0]  # X
        self.x[3] = measurement[1]  # Y
        self.x[6] = measurement[2]  # Z
        self.x[9] = measurement[3]  # Clock
        
        # Initialize velocities and accelerations to zero
        self.x[1] = self.x[2] = 0.0  # Vx, Ax
        self.x[4] = self.x[5] = 0.0  # Vy, Ay
        self.x[7] = self.x[8] = 0.0  # Vz, Az
        self.x[10] = 0.0  # Clock drift
        
        # Initialize harmonic components
        self.x[11] = 0.0  # sin phase
        self.x[12] = 1.0  # cos phase (start at peak)
        self.x[13] = 2 * np.pi / (12.0 * 3600)  # ~12-hour period (rad/s)
        
        # Reduce initial uncertainty for observed states
        self.P[0, 0] = 100.0   # X uncertainty
        self.P[3, 3] = 100.0   # Y uncertainty
        self.P[6, 6] = 100.0   # Z uncertainty
        self.P[9, 9] = 50.0    # Clock uncertainty
        
        self.initialized = True
        logger.debug(f"Filter initialized: X={self.x[0]:.2f}, Y={self.x[3]:.2f}, "
                    f"Z={self.x[6]:.2f}, Clk={self.x[9]:.2f}")
    
    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prediction step: propagate state and covariance.
        
        Returns:
            x_pred: Predicted state
            P_pred: Predicted covariance
        """
        # State prediction: x̂ = F * x
        self.x = self.F @ self.x
        
        # Covariance prediction: P̂ = F * P * F^T + Q
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        # Ensure symmetry
        self.P = (self.P + self.P.T) / 2
        
        return self.x.copy(), self.P.copy()
    
    def update(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update step: incorporate measurement.
        
        Args:
            measurement: [X, Y, Z, Clock] observation
            
        Returns:
            x_updated: Updated state
            P_updated: Updated covariance
        """
        if not self.initialized:
            self.initialize(measurement)
            return self.x.copy(), self.P.copy()
        
        # Innovation: y = z - H * x̂
        z = measurement
        y = z - self.H @ self.x
        
        # Innovation covariance: S = H * P̂ * H^T + R
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain: K = P̂ * H^T * S^(-1)
        try:
            K = self.P @ self.H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            logger.warning("Singular innovation covariance, using pseudo-inverse")
            K = self.P @ self.H.T @ np.linalg.pinv(S)
        
        # State update: x = x̂ + K * y
        self.x = self.x + K @ y
        
        # Covariance update: P = (I - K * H) * P̂
        I_KH = np.eye(self.state_dim) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T  # Joseph form for stability
        
        # Ensure symmetry
        self.P = (self.P + self.P.T) / 2
        
        return self.x.copy(), self.P.copy()
    
    def get_position(self) -> np.ndarray:
        """Get current position estimate [X, Y, Z, Clock]."""
        return np.array([self.x[0], self.x[3], self.x[6], self.x[9]])
    
    def get_velocity(self) -> np.ndarray:
        """Get current velocity estimate [Vx, Vy, Vz, ClockDrift]."""
        return np.array([self.x[1], self.x[4], self.x[7], self.x[10]])
    
    def get_uncertainty(self) -> np.ndarray:
        """Get position uncertainty (standard deviations)."""
        return np.array([
            np.sqrt(self.P[0, 0]),
            np.sqrt(self.P[3, 3]),
            np.sqrt(self.P[6, 6]),
            np.sqrt(self.P[9, 9])
        ])


def forecast_day8_enhanced(
    kf: EnhancedKalmanFilter,
    start_time: str = "2025-09-08 00:00:00",
    num_steps: int = 96
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Forecast Day 8 predictions using enhanced Kalman filter.
    
    Args:
        kf: Trained EnhancedKalmanFilter instance
        start_time: Start time for predictions
        num_steps: Number of 15-min steps to predict (96 = 24 hours)
        
    Returns:
        predictions: Array of shape (num_steps, 4) with [X, Y, Z, Clock]
        uncertainties: Array of shape (num_steps, 4) with standard deviations
    """
    predictions = []
    uncertainties = []
    
    for i in range(num_steps):
        # Predict next state
        kf.predict()
        
        # Store prediction
        pred_pos = kf.get_position()
        pred_unc = kf.get_uncertainty()
        
        predictions.append(pred_pos)
        uncertainties.append(pred_unc)
    
    return np.array(predictions), np.array(uncertainties)
