import numpy as np


class UCB:
    """UCB1 algorithm (Auer et al., 2002)."""

    def __init__(self, n_arms: int, c: float = 2.0):
        self.n_arms = n_arms
        self.c = c
        self.counts = np.zeros(n_arms, dtype=int)
        self.values = np.zeros(n_arms)
        self.t = 0

    def select(self) -> int:
        # Pull each arm once before using UCB scores
        unpulled = np.where(self.counts == 0)[0]
        if len(unpulled) > 0:
            return int(unpulled[0])
        ucb_scores = self.values + self.c * np.sqrt(np.log(self.t) / self.counts)
        return int(np.argmax(ucb_scores))

    def update(self, arm: int, reward: float) -> None:
        self.t += 1
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n

    def reset(self) -> None:
        self.counts[:] = 0
        self.values[:] = 0
        self.t = 0
