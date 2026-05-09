"""Compare all algorithms on a single Bernoulli bandit over 500 rounds."""
import numpy as np
import matplotlib.pyplot as plt

from bandits.environments import BernoulliBandit
from bandits.algorithms import UCB, ThompsonSampling, KLUCB, BayesUCB, EpsilonGreedy, EXP3, MOSS

MEANS = [0.2, 0.5, 0.8]
T = 500
N_RUNS = 200  # average over runs to smooth curves

agents = {
    "UCB1":          lambda: UCB(len(MEANS)),
    "KL-UCB":        lambda: KLUCB(len(MEANS)),
    "Thompson":      lambda: ThompsonSampling(len(MEANS)),
    "BayesUCB":      lambda: BayesUCB(len(MEANS)),
    "Eps-Greedy":    lambda: EpsilonGreedy(len(MEANS), epsilon=0.1, decay=True),
    "EXP3":          lambda: EXP3(len(MEANS), T=T),
    "MOSS":          lambda: MOSS(len(MEANS), T=T),
}

# Collect cumulative regret trajectories
results = {name: np.zeros((N_RUNS, T)) for name in agents}

for name, make_agent in agents.items():
    for run in range(N_RUNS):
        env = BernoulliBandit(MEANS, seed=run)
        agent = make_agent()
        cum_regret = 0.0
        for t in range(T):
            arm = agent.select()
            reward = env.pull(arm)
            agent.update(arm, reward)
            cum_regret += env.regret(arm)
            results[name][run, t] = cum_regret

# Plot
fig, ax = plt.subplots(figsize=(9, 5))

# Error bars at evenly spaced ticks, not every round (keeps it readable)
error_ticks = np.linspace(0, T - 1, 10, dtype=int)

colors = plt.cm.tab10.colors
for i, (name, runs) in enumerate(results.items()):
    mean = runs.mean(axis=0)
    std = runs.std(axis=0)
    ax.plot(mean, label=name, color=colors[i], linewidth=2)
    ax.errorbar(
        error_ticks, mean[error_ticks], yerr=std[error_ticks],
        fmt="none", color=colors[i], capsize=4, capthick=1.5, elinewidth=1.5,
    )

ax.set_xlabel("Round", fontsize=12)
ax.set_ylabel("Cumulative Regret", fontsize=12)
ax.set_title(f"Bernoulli Bandit — means={MEANS}  (avg over {N_RUNS} runs)", fontsize=13)
ax.legend(loc="upper left", fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()

out = "examples/regret_comparison.png"
plt.savefig(out, dpi=150)
print(f"Saved → {out}")
plt.show()
