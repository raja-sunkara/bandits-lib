import numpy as np


class ThompsonSampling:
    """Thompson Sampling with Beta-Bernoulli conjugate prior."""

    def __init__(self, n_arms: int, alpha0: float = 1.0, beta0: float = 1.0):
        self.n_arms = n_arms
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.alpha = np.full(n_arms, alpha0)
        self.beta = np.full(n_arms, beta0)

    def select(self) -> int:
        samples = self.rng.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def update(self, arm: int, reward: float) -> None:
        self.alpha[arm] += reward
        self.beta[arm] += 1 - reward

    def reset(self) -> None:
        self.alpha[:] = self.alpha0
        self.beta[:] = self.beta0

    @property
    def rng(self):
        if not hasattr(self, "_rng"):
            self._rng = np.random.default_rng()
        return self._rng
