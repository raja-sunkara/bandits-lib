import numpy as np


class MOSS:
    """MOSS: Minimax Optimal Strategy in the Stochastic case (Audibert & Bubeck, 2009).

    Requires horizon T and number of arms K upfront.
    Index = mean + sqrt(max(0, log(T / (K * n))) / n)
    """

    def __init__(self, n_arms: int, T: int):
        self.n_arms = n_arms
        self.T = T
        self.counts = np.zeros(n_arms, dtype=int)
        self.values = np.zeros(n_arms)
        self.t = 0

    def select(self) -> int:
        unpulled = np.where(self.counts == 0)[0]
        if len(unpulled) > 0:
            return int(unpulled[0])
        log_term = np.log(np.maximum(self.T / (self.n_arms * self.counts), 1.0))
        indices = self.values + np.sqrt(log_term / self.counts)
        return int(np.argmax(indices))

    def update(self, arm: int, reward: float) -> None:
        self.t += 1
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n

    def reset(self) -> None:
        self.counts[:] = 0
        self.values[:] = 0
        self.t = 0
