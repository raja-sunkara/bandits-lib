import numpy as np
from scipy.stats import beta as beta_dist


class BayesUCB:
    """Bayes-UCB with Beta-Bernoulli model (Kaufmann et al., 2012).

    Index = (1 - 1/t) quantile of the posterior Beta distribution.
    """

    def __init__(self, n_arms: int, alpha0: float = 1.0, beta0: float = 1.0, c: float = 0.0):
        self.n_arms = n_arms
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.c = c  # exploration bonus exponent
        self.alpha = np.full(n_arms, alpha0)
        self.beta = np.full(n_arms, beta0)
        self.t = 0

    def select(self) -> int:
        self.t += 1
        quantile = 1.0 - 1.0 / (self.t * (self.c + 1))
        indices = beta_dist.ppf(quantile, self.alpha, self.beta)
        return int(np.argmax(indices))

    def update(self, arm: int, reward: float) -> None:
        self.alpha[arm] += reward
        self.beta[arm] += 1 - reward

    def reset(self) -> None:
        self.alpha[:] = self.alpha0
        self.beta[:] = self.beta0
        self.t = 0
