# SnakeAI
AI program learning how to play a PyGame replication of the classic Snake Game.

**Version:** 1.0.0  
**Release Date:** August 2023

## Overview

SnakeAI is a project that combines the classic Snake Game with artificial intelligence. The goal of this project is to teach an AI program how to play a Python-based replication of the Snake Game created using the PyGame library. The game incorporates a unique feature: the game speed increases in proportion with the Snake's length, providing an additional challenge as the Snake grows longer. To achieve autonomous Snake gameplay, a PyTorch-based AI agent has been implemented, utilizing a neural network trained through reinforcement learning to make decisions during each game.

## Features

- **Classic Snake Game:** Enjoy a replication of the classic Snake Game using Python and PyGame, with an added twist of increasing game speed as the Snake grows longer.

- **Autonomous Snake Gameplay:** Experience an AI agent that learns how to play the game by itself, making strategic decisions to maximize its score.

- **Reinforcement Learning:** The AI agent's decision-making is powered by a PyTorch-based neural network, which learns and improves its performance over time through reinforcement learning.

- **Performance Visualization:** The project includes graphical representations of the AI agent's performance, showcasing its score progression across multiple games using Matplotlib.

## Installation

To run this project locally, follow these steps:

1. Clone this GitHub repository to your local machine.

   ```bash
   git clone https://github.com/your-username/snake-ai.git
   ```

2. Navigate to the project directory.

   ```bash
   cd snake-ai
   ```

3. Install the required Python packages using pip.

   ```bash
   pip install -r requirements.txt
   ```

4. Run the Agent program.

   ```bash
   python agent.py
   ```

You can now watch the AI agent play the Snake Game and analyze its performance.

## Usage

1. Launch the agent.py program, and the AI agent will begin playing the Snake Game autonomously. Or if you want to play Snake by yourself, launch the snake_game.py file.

2. Observe the Snake's movements and its strategy in collecting food while avoiding collisions with itself.

3. The Matplotlib graphs will display the AI agent's score progression over multiple game sessions, demonstrating its learning and improvement.

## Feedback and Contributions

We welcome feedback, bug reports, and contributions to this project. If you encounter any issues or have suggestions for improvements, please open an issue on this GitHub repository. Feel free to submit pull requests with enhancements or fixes as well.

Enjoy watching and experimenting with SnakeAI as it learns to master the Snake Game autonomously! If you have any questions or need further assistance, please don't hesitate to contact us.
