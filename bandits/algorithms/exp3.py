import numpy as np


class EXP3:
    """EXP3 algorithm for adversarial bandits (Auer et al., 2002).

    Works for Bernoulli rewards but makes no stochastic assumption.
    Uses importance-weighted updates with learning rate gamma.
    If gamma is None, uses the theoretically optimal rate for horizon T.
    """

    def __init__(self, n_arms: int, gamma: float | None = None, T: int | None = None, seed=None):
        self.n_arms = n_arms
        if gamma is not None:
            self.gamma = gamma
        elif T is not None:
            self.gamma = min(1.0, np.sqrt(n_arms * np.log(n_arms) / ((np.e - 1) * T)))
        else:
            self.gamma = 0.1
        self.weights = np.ones(n_arms)
        self.t = 0
        self.rng = np.random.default_rng(seed)

    def _probabilities(self) -> np.ndarray:
        w = self.weights
        return (1 - self.gamma) * w / w.sum() + self.gamma / self.n_arms

    def select(self) -> int:
        probs = self._probabilities()
        return int(self.rng.choice(self.n_arms, p=probs))

    def update(self, arm: int, reward: float) -> None:
        self.t += 1
        probs = self._probabilities()
        estimated_reward = reward / probs[arm]
        self.weights[arm] *= np.exp(self.gamma * estimated_reward / self.n_arms)

    def reset(self) -> None:
        self.weights[:] = 1.0
        self.t = 0
