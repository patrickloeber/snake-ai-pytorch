import torch
import random
import numpy as np
from collections import deque
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from game import SnakeGameAI, Direction, Point
from models.model_REINFORCE_net import PolicyNetwork, ReinforceTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.0001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = PolicyNetwork(11, 256, 3)
        # file_name = './model/model.pth'
        # self.model.load_state_dict(torch.load(file_name))
        self.trainer = ReinforceTrainer(self.model, lr=LR)


    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int)

    def remember(self, episode):
        self.memory.append(episode)  # Store the full episode

    def train_long_memory(self):
        # Here we train over episodes stored in memory
        if len(self.memory) > 0:
            episodes = list(self.memory)  # Get all episodes
            self.memory.clear()
            self.trainer.train(episodes)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float)
        probabilities = self.model(state).detach().numpy()  # Get probabilities from the model
        action = np.random.choice(len(probabilities), p=probabilities)  # Sample an action
        final_move = [0] * len(probabilities)
        final_move[action] = 1  # One-hot encode the sampled action

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        episode = []  # Store state, action, reward for the entire episode
        # get old state
        state_old = agent.get_state(game)
        final_move = [0, 0, 0]
        final_move[random.randint(0, 2)] = 1
        reward, done, score = game.play_step(final_move)
        episode.append((state_old, final_move, reward))  # Append the initial state, action, reward

        while not done:
            state_old = agent.get_state(game)
            final_move = agent.get_action(state_old)
            reward, done, score = game.play_step(final_move)
            episode.append((state_old, final_move, reward))  # Append each step's data
            # agent.train_long_memory()

            if done:
                # Train with the episode's data and reset the game
                agent.remember(episode)
                game.reset()
                agent.n_games += 1
                if len(agent.memory) % 10 == 0:
                    agent.train_long_memory()

                if score > record:
                    record = score
                    agent.model.save()

                print('Game', agent.n_games, 'Score', score, 'Record:', record)

                plot_scores.append(score)
                total_score += score
                mean_score = total_score / agent.n_games
                plot_mean_scores.append(mean_score)
                plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    train()
