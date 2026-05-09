import numpy as np


class EpsilonGreedy:
    """Epsilon-greedy with optional annealing schedule.

    If decay is set, epsilon_t = min(1, epsilon * n_arms / (d^2 * t))
    where d is the expected gap (default 1.0 gives the standard annealing rate).
    """

    def __init__(self, n_arms: int, epsilon: float = 0.1, decay: bool = False, seed=None):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.decay = decay
        self.counts = np.zeros(n_arms, dtype=int)
        self.values = np.zeros(n_arms)
        self.t = 0
        self.rng = np.random.default_rng(seed)

    def _current_epsilon(self) -> float:
        if not self.decay or self.t == 0:
            return self.epsilon
        return min(1.0, self.epsilon * self.n_arms / self.t)

    def select(self) -> int:
        if self.rng.random() < self._current_epsilon():
            return int(self.rng.integers(self.n_arms))
        return int(np.argmax(self.values))

    def update(self, arm: int, reward: float) -> None:
        self.t += 1
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n

    def reset(self) -> None:
        self.counts[:] = 0
        self.values[:] = 0
        self.t = 0
