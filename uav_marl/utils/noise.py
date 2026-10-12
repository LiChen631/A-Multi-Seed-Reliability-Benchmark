"""Exploration noise (Ornstein-Uhlenbeck process)."""
import numpy as np

class OUNoise:
    def __init__(self, action_dim, sigma=0.1, theta=0.15, dt=0.5):
        self.action_dim = action_dim
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.state = None

    def reset(self):
        self.state = np.zeros(self.action_dim)

    def sample(self):
        if self.state is None:
            self.reset()
        x = self.state
        dx = -self.theta * x * self.dt + self.sigma * np.sqrt(self.dt) * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state.copy()
