import numpy as np
import math


class OneEuroFilter:
    
    def __init__(self, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None
        
    def _smoothing_factor(self, cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)
    
    def _exponential_smoothing(self, x, x_prev, alpha):
        return alpha * x + (1.0 - alpha) * x_prev
    
    def filter(self, x, t):
        if self.t_prev is None:
            self.x_prev = x
            self.dx_prev = 0.0
            self.t_prev = t
            return x
        
        dt = t - self.t_prev
        if dt <= 0:
            return self.x_prev
        
        dx = (x - self.x_prev) / dt
        alpha_d = self._smoothing_factor(self.d_cutoff, dt)
        dx_hat = self._exponential_smoothing(dx, self.dx_prev, alpha_d)
        
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        alpha = self._smoothing_factor(cutoff, dt)
        x_hat = self._exponential_smoothing(x, self.x_prev, alpha)
        
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        
        return x_hat
    
    def reset(self):
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None


class PoseFilter:
    
    def __init__(self, num_landmarks=33, num_coords=3, min_cutoff=1.0, beta=0.007):
        self.num_landmarks = num_landmarks
        self.num_coords = num_coords
        self.filters = []
        for _ in range(num_landmarks):
            landmark_filters = []
            for _ in range(num_coords):
                landmark_filters.append(OneEuroFilter(min_cutoff=min_cutoff, beta=beta))
            self.filters.append(landmark_filters)
    
    def filter(self, landmarks, timestamp):
        if landmarks is None:
            return None
        filtered = np.zeros_like(landmarks[:, :3])
        for i in range(min(self.num_landmarks, len(landmarks))):
            for j in range(self.num_coords):
                filtered[i, j] = self.filters[i][j].filter(landmarks[i, j], timestamp)
        return filtered
    
    def reset(self):
        for landmark_filters in self.filters:
            for f in landmark_filters:
                f.reset()


class KalmanFilter1D:
    
    def __init__(self, process_noise=0.1, measurement_noise=0.5):
        self.q = process_noise
        self.r = measurement_noise
        self.x = 0.0
        self.p = 1.0
        self.initialized = False
        
    def update(self, measurement):
        if not self.initialized:
            self.x = measurement
            self.initialized = True
            return self.x
        
        self.p = self.p + self.q
        
        k = self.p / (self.p + self.r)
        self.x = self.x + k * (measurement - self.x)
        self.p = (1 - k) * self.p
        
        return self.x
    
    def reset(self):
        self.x = 0.0
        self.p = 1.0
        self.initialized = False
