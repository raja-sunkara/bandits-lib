import numpy as np


def _kl_bernoulli(p: float, q: float) -> float:
    p = np.clip(p, 1e-10, 1 - 1e-10)
    q = np.clip(q, 1e-10, 1 - 1e-10)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


def _kl_ucb_index(p_hat: float, n: int, t: int, c: float = 0.0, tol: float = 1e-6) -> float:
    """Binary search for largest q s.t. KL(p_hat, q) <= (log(t) + c*log(log(t))) / n."""
    threshold = (np.log(t) + c * np.log(max(np.log(t), 1))) / n
    lo, hi = p_hat, 1.0 - 1e-10
    for _ in range(50):
        mid = (lo + hi) / 2
        if _kl_bernoulli(p_hat, mid) < threshold:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return lo


class KLUCB:
    """KL-UCB algorithm (Garivier & Cappé, 2011) — asymptotically optimal for Bernoulli bandits."""

    def __init__(self, n_arms: int, c: float = 0.0):
        self.n_arms = n_arms
        self.c = c
        self.counts = np.zeros(n_arms, dtype=int)
        self.values = np.zeros(n_arms)
        self.t = 0

    def select(self) -> int:
        unpulled = np.where(self.counts == 0)[0]
        if len(unpulled) > 0:
            return int(unpulled[0])
        indices = [
            _kl_ucb_index(self.values[a], self.counts[a], self.t, self.c)
            for a in range(self.n_arms)
        ]
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
