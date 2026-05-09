"""Benchmark all algorithms on a Bernoulli bandit and print cumulative regret."""
import numpy as np
from bandits.environments import BernoulliBandit
from bandits.algorithms import UCB, ThompsonSampling, KLUCB, BayesUCB, EpsilonGreedy, EXP3, MOSS

MEANS = [0.1, 0.3, 0.5, 0.7, 0.9]
T = 5000
N_RUNS = 50

agents = {
    "UCB1":             lambda: UCB(len(MEANS)),
    "KL-UCB":           lambda: KLUCB(len(MEANS)),
    "Thompson":         lambda: ThompsonSampling(len(MEANS)),
    "BayesUCB":         lambda: BayesUCB(len(MEANS)),
    "EpsilonGreedy":    lambda: EpsilonGreedy(len(MEANS), epsilon=0.1, decay=True),
    "EXP3":             lambda: EXP3(len(MEANS), T=T),
    "MOSS":             lambda: MOSS(len(MEANS), T=T),
}

print(f"{'Algorithm':<16} {'Mean Regret':>12} {'Std':>8}")
print("-" * 38)

for name, make_agent in agents.items():
    regrets = []
    for run in range(N_RUNS):
        env = BernoulliBandit(MEANS, seed=run)
        agent = make_agent()
        total = 0.0
        for _ in range(T):
            arm = agent.select()
            reward = env.pull(arm)
            agent.update(arm, reward)
            total += env.regret(arm)
        regrets.append(total)
    print(f"{name:<16} {np.mean(regrets):>12.1f} {np.std(regrets):>8.1f}")
