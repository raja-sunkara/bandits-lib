# bandits-lib

A modular, research-grade Python library for bandit algorithms, environments, and benchmark problems.

## Installation

```bash
pip install bandits-lib
```

Or from source:

```bash
git clone https://github.com/YOUR_USERNAME/bandits-lib.git
cd bandits-lib
pip install -e ".[dev]"
```

## Structure

```
bandits/
├── algorithms/     # UCB, Thompson Sampling, EXP3, LinUCB, ...
├── environments/   # Bernoulli, Gaussian, contextual, adversarial, ...
└── problems/       # Best arm identification, pure exploration, ...
```

## Quick Start

```python
from bandits.environments import BernoulliBandit
from bandits.algorithms import UCB

env = BernoulliBandit(means=[0.2, 0.5, 0.8])
agent = UCB(n_arms=3)

for t in range(1000):
    arm = agent.select()
    reward = env.pull(arm)
    agent.update(arm, reward)
```

## Contributing

PRs welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT — free for personal and academic use.  
For commercial use, see [LICENSE](LICENSE).

## Support

If this library is useful to your research or product, consider [sponsoring](https://github.com/sponsors/YOUR_USERNAME).
