import numpy as np


class KalmanFilter:
    def __init__(self, w0: np.ndarray, p: float, q: float, r: float):
        """
        Initialize the Kalman Filter.
        Params:
            w0: np.ndarray: Initial state estimate.
            p: float: Initial estimate covariance.
            q: float: Process noise covariance.
            r: float: Measurement noise covariance.
        Returns:
            None
        """
        self.w0 = w0
        self.n = 2
        self.a = np.eye(self.n)
        self.p = np.eye(self.n) * p
        self.q = np.eye(self.n) * q
        self.r = r
        self.i = np.eye(self.n)


    def predict(self, x_t, y_t) -> np.ndarray:
        """
        Perform a single prediction and update step of the Kalman Filter.
        Params:
            x_t: float: The current input measurement.
            y_t: float: The current output measurement.
        Returns:
            np.ndarray: The updated state estimate.
        """
        # Current measurement matrix
        c_t = np.array([[1, x_t]])

        # Prediction
        w_pred = self.a @ self.w0
        p_pred = self.a @ self.p @ self.a.T + self.q

        # Update step
        k = p_pred @ c_t.T @ np.linalg.inv(c_t @ p_pred @ c_t.T + self.r)
        w_upd = w_pred + k @ (y_t - c_t @ w_pred)
        p_upd = (self.i - k @ c_t) @ p_pred

        # Update for next iteration
        self.w0 = w_upd
        self.p = p_upd

        return w_upd


    def coef(self):
        """
        Get the current coefficients of the Kalman Filter.
        Returns:
            np.ndarray: The current state estimate.
        """
        return self.w0
