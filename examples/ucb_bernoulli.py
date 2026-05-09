from bandits.environments import BernoulliBandit
from bandits.algorithms import UCB

env = BernoulliBandit(means=[0.2, 0.5, 0.8], seed=42)
agent = UCB(n_arms=3)

T = 1000
total_regret = 0.0

for t in range(T):
    arm = agent.select()
    reward = env.pull(arm)
    agent.update(arm, reward)
    total_regret += env.regret(arm)

print(f"Cumulative regret after {T} steps: {total_regret:.2f}")
print(f"Arm pull counts: {agent.counts}")
print(f"Estimated means:  {agent.values.round(3)}")
print(f"True means:       {env.means}")
