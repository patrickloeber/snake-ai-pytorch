# Simple Implementations of Reinforcement Learning Algorithms for Snake Game with PyTorch and Pygame

This repository, an extension of the excellent tutorial by Patrick Loeber ([patrickloeber/snake-ai-pytorch](https://github.com/patrickloeber/snake-ai-pytorch)), significantly expands the original project by incorporating a variety of classic reinforcement learning (RL) algorithms. Originally featuring only Deep Q-Networks (DQN), this fork now offers an array of sophisticated RL strategies applied to the classic Snake game, all implemented using PyTorch and Pygame.

This codebase favors explicit and straightforward logic over more abstract, complex structures, to facilitate learning and comprehension more on RL algorithms themselves.

## Implementations:
- **MC Exploring Starts**: Utilizes Monte Carlo methods with function approximation.
- **SARSA**: Applies SARSA (State-Action-Reward-State-Action) with function approximation.
- **REINFORCE**: Integrates the REINFORCE algorithm.
- **AC (Actor-Critic)**: Implements the Actor-Critic method. (Not converging, still WIP)
- **A2C (Advantage Actor-Critic)**: Features the Advantage Actor-Critic approach. (Not converging, still WIP)
- *More algorithms are planned to be added in future updates.*

## Getting Started:
1. **To Train Agents**: Simply run the respective algorithm script, like `python agent_DQN.py`, to begin training the AI agent. The same process applies to other algorithms.
2. **To Play Yourself**: If you'd like to play the Snake game manually, run `python snake_game_human.py`.

Stay tuned for further updates and enhancements to this growing project!