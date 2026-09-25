import numpy as np
import scipy.stats as stats

class DynamicConnectivityGenerator:
    """
    Generates dynamic sliding-window functional connectivity matrices (N x N)
    from ROI time series signals (T x N).
    """
    def __init__(self, window_length=30, stride=10, metric="pearson", apply_fisher_z=True):
        self.window_length = window_length
        self.stride = stride
        self.metric = metric
        self.apply_fisher_z = apply_fisher_z

    def _compute_connectivity(self, window_data):
        """
        Computes N x N connectivity matrix for a single time window (W x N).
        """
        W, N = window_data.shape
        # Handle zero variance ROIs safely
        stds = np.std(window_data, axis=0, keepdims=True)
        stds[stds == 0] = 1e-8
        norm_data = (window_data - np.mean(window_data, axis=0, keepdims=True)) / stds
        
        if self.metric == "pearson":
            corr = (norm_data.T @ norm_data) / (W - 1)
        elif self.metric == "covariance":
            corr = np.cov(window_data, rowvar=False)
        else:
            raise ValueError(f"Unknown connectivity metric: {self.metric}")
            
        # Clean NaNs and infs
        corr = np.nan_to_num(corr, nan=0.0, posinf=1.0, neginf=-1.0)
        np.fill_diagonal(corr, 1.0)
        corr = np.clip(corr, -0.9999, 0.9999)
        
        if self.apply_fisher_z:
            # Fisher z-transformation: z = 0.5 * ln((1 + r) / (1 - r))
            corr = np.arctanh(corr)
            np.fill_diagonal(corr, 0.0)  # zero out self-loops after Fisher-z
            
        return corr.astype(np.float32)

    def generate_dynamic_connectivity(self, time_series):
        """
        Input: time_series array of shape (T, N) where N=116.
        Output: connectivity sequence array of shape (T_windows, N, N).
        """
        T, N = time_series.shape
        if T < self.window_length:
            # Pad time series if scan is shorter than window_length
            padding = np.tile(time_series[-1:], (self.window_length - T, 1))
            time_series = np.vstack([time_series, padding])
            T, N = time_series.shape
            
        windows = []
        start = 0
        while start + self.window_length <= T:
            window = time_series[start : start + self.window_length, :]
            conn_matrix = self._compute_connectivity(window)
            windows.append(conn_matrix)
            start += self.stride
            
        if len(windows) == 0:
            conn_matrix = self._compute_connectivity(time_series)
            windows.append(conn_matrix)
            
        return np.array(windows, dtype=np.float32)

    def generate_static_connectivity(self, time_series):
        """
        Computes single static full-scan connectivity matrix (1, N, N).
        """
        conn = self._compute_connectivity(time_series)
        return np.expand_dims(conn, axis=0).astype(np.float32)

if __name__ == "__main__":
    # Test with dummy ROI time series
    dummy_ts = np.random.randn(150, 116)
    gen = DynamicConnectivityGenerator(window_length=30, stride=10)
    dyn_conn = gen.generate_dynamic_connectivity(dummy_ts)
    stat_conn = gen.generate_static_connectivity(dummy_ts)
    print(f"Input time series shape: {dummy_ts.shape}")
    print(f"Dynamic connectivity shape (T_windows x N x N): {dyn_conn.shape}")
    print(f"Static connectivity shape (1 x N x N): {stat_conn.shape}")
