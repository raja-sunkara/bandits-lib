import numpy as np


class BernoulliBandit:
    """Multi-armed bandit where each arm yields Bernoulli(p) rewards."""

    def __init__(self, means, seed=None):
        self.means = np.array(means)
        self.n_arms = len(means)
        self.best_arm = int(np.argmax(self.means))
        self.best_mean = float(self.means[self.best_arm])
        self.rng = np.random.default_rng(seed)

    def pull(self, arm: int) -> float:
        return float(self.rng.random() < self.means[arm])

    def regret(self, arm: int) -> float:
        return self.best_mean - self.means[arm]

    def reset(self, seed=None):
        self.rng = np.random.default_rng(seed)
