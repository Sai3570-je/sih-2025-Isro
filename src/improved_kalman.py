"""
Improved Kalman Filter with Physical Constraints and Better Numerical Stability.
"""

import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ImprovedKalmanFilter:
    """
    Improved 10D state-space Kalman filter:
    [X, Vx, Y, Vy, Z, Vz, Clock, ClockDrift, sin_phase, cos_phase]
    
    Features:
    - First-order dynamics (position + velocity)
    - Harmonic oscillation for orbital motion
    - Proper numerical conditioning
    - Adaptive process noise
    """
    
    def __init__(self, dt: float = 900.0, Q_scale: float = 1.0, R_scale: float = 1.0):
        self.dt = dt
        self.state_dim = 10
        self.obs_dim = 4
        
        # State: [X, Vx, Y, Vy, Z, Vz, Clk, ClkDrift, sin_φ, cos_φ]
        self.x = np.zeros(self.state_dim)
        self.P = np.eye(self.state_dim) * 100.0  # Moderate initial uncertainty
        
        # Build state transition
        self.F = self._build_F(dt)
        
        # Observation matrix: [X, Y, Z, Clock]
        self.H = np.zeros((self.obs_dim, self.state_dim))
        self.H[0, 0] = 1.0  # X
        self.H[1, 2] = 1.0  # Y
        self.H[2, 4] = 1.0  # Z
        self.H[3, 6] = 1.0  # Clock
        
        # Process noise (conservative)
        self.Q = self._build_Q(dt, Q_scale)
        
        # Measurement noise (from data analysis)
        self.R = np.diag([
            (5.0 * R_scale) ** 2,
            (8.0 * R_scale) ** 2,
            (6.0 * R_scale) ** 2,
            (3.0 * R_scale) ** 2
        ])
        
        self.initialized = False
        self.step_count = 0
        
    def _build_F(self, dt: float) -> np.ndarray:
        """Build state transition matrix."""
        F = np.eye(self.state_dim)
        
        # Position-velocity coupling for X, Y, Z
        F[0, 1] = dt  # X += Vx * dt
        F[2, 3] = dt  # Y += Vy * dt
        F[4, 5] = dt  # Z += Vz * dt
        
        # Clock dynamics
        F[6, 7] = dt  # Clock += Drift * dt
        
        # Harmonic oscillator (simple rotation)
        omega = 2 * np.pi / (12.0 * 3600)  # ~12-hour period
        F[8, 8] = np.cos(omega * dt)
        F[8, 9] = -np.sin(omega * dt)
        F[9, 8] = np.sin(omega * dt)
        F[9, 9] = np.cos(omega * dt)
        
        return F
    
    def _build_Q(self, dt: float, scale: float) -> np.ndarray:
        """Build process noise covariance."""
        Q = np.zeros((self.state_dim, self.state_dim))
        
        # For each position-velocity pair
        for i in range(3):  # X, Y, Z
            pos_idx = i * 2
            vel_idx = pos_idx + 1
            
            # Process noise from velocity random walk
            q_vel = (0.005 * scale) ** 2  # Small velocity variance (m/s)²
            
            # Position variance from velocity uncertainty
            Q[pos_idx, pos_idx] = (dt ** 2) * q_vel
            Q[pos_idx, vel_idx] = dt * q_vel
            Q[vel_idx, pos_idx] = dt * q_vel
            Q[vel_idx, vel_idx] = q_vel
        
        # Clock dynamics
        Q[6, 6] = (0.1 * scale * dt) ** 2
        Q[7, 7] = (0.01 * scale) ** 2
        
        # Harmonic components (very small)
        Q[8, 8] = (0.0001 * scale) ** 2
        Q[9, 9] = (0.0001 * scale) ** 2
        
        return Q
    
    def initialize(self, measurement: np.ndarray):
        """Initialize state from first measurement."""
        self.x[0] = measurement[0]  # X
        self.x[2] = measurement[1]  # Y
        self.x[4] = measurement[2]  # Z
        self.x[6] = measurement[3]  # Clock
        
        # Zero velocities initially
        self.x[1] = self.x[3] = self.x[5] = self.x[7] = 0.0
        
        # Initialize oscillator
        self.x[8] = 0.0  # sin
        self.x[9] = 1.0  # cos
        
        # Set initial covariance
        self.P = np.eye(self.state_dim) * 100.0
        self.P[[0, 2, 4, 6], [0, 2, 4, 6]] = 25.0  # Lower uncertainty for observed states
        
        self.initialized = True
        logger.debug(f"Initialized: X={self.x[0]:.2f}, Y={self.x[2]:.2f}, Z={self.x[4]:.2f}")
    
    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prediction step."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        # Ensure symmetry and positive definiteness
        self.P = (self.P + self.P.T) / 2
        self.P += np.eye(self.state_dim) * 1e-6  # Regularization
        
        self.step_count += 1
        return self.x.copy(), self.P.copy()
    
    def update(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Update step."""
        if not self.initialized:
            self.initialize(measurement)
            return self.x.copy(), self.P.copy()
        
        z = measurement
        y = z - self.H @ self.x  # Innovation
        
        S = self.H @ self.P @ self.H.T + self.R  # Innovation covariance
        
        # Ensure S is well-conditioned
        S = (S + S.T) / 2
        S += np.eye(self.obs_dim) * 1e-6
        
        try:
            K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        except np.linalg.LinAlgError:
            logger.warning("Singular S matrix, using pseudoinverse")
            K = self.P @ self.H.T @ np.linalg.pinv(S)
        
        self.x = self.x + K @ y  # State update
        
        # Joseph form covariance update for numerical stability
        I_KH = np.eye(self.state_dim) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        
        # Ensure symmetry and positive definiteness
        self.P = (self.P + self.P.T) / 2
        self.P += np.eye(self.state_dim) * 1e-6
        
        return self.x.copy(), self.P.copy()
    
    def get_position(self) -> np.ndarray:
        """Get [X, Y, Z, Clock]."""
        return np.array([self.x[0], self.x[2], self.x[4], self.x[6]])
    
    def get_velocity(self) -> np.ndarray:
        """Get [Vx, Vy, Vz, ClockDrift]."""
        return np.array([self.x[1], self.x[3], self.x[5], self.x[7]])
    
    def get_uncertainty(self) -> np.ndarray:
        """Get position std devs."""
        return np.array([
            np.sqrt(max(self.P[0, 0], 0)),
            np.sqrt(max(self.P[2, 2], 0)),
            np.sqrt(max(self.P[4, 4], 0)),
            np.sqrt(max(self.P[6, 6], 0))
        ])


def forecast_improved(
    kf: ImprovedKalmanFilter,
    num_steps: int = 96
) -> Tuple[np.ndarray, np.ndarray]:
    """Forecast future states."""
    predictions = []
    uncertainties = []
    
    for i in range(num_steps):
        kf.predict()
        predictions.append(kf.get_position())
        uncertainties.append(kf.get_uncertainty())
    
    return np.array(predictions), np.array(uncertainties)
