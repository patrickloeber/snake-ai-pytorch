import torch
import torch.nn as nn
import random
import numpy as np
from collections import deque
from zz_game import zz_game_ai, Direction, Point
from zz_model import Linear_QNet, Trainer
from zz_helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000

class Agent:

    def __init__(self):
        self.gamma = 0.9
        self.epsilon = 0
        self.n_games = 0
        self.memory = deque(maxlen=MAX_MEMORY) # popleft() if larger
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = Trainer(self.model)


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

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self, memory):
        self.n_games += 1
        if len(memory) > BATCH_SIZE:
            minibatch = random.sample(memory, BATCH_SIZE) # list of tuples
        else:
            minibatch = memory
        
        states, actions, rewards, next_states, dones = zip(*minibatch)
        states = torch.tensor(states, dtype=torch.float) #[1, ... , 0]
        actions = torch.tensor(actions, dtype=torch.long) # [1, 0, 0]
        rewards = torch.tensor(rewards, dtype=torch.float) # int
        next_states = torch.tensor(next_states, dtype=torch.float) #[True, ... , False]
        targets = rewards.clone()

        for idx, done in enumerate(dones):
            if not done:
                targets[idx] = rewards[idx] + self.gamma * torch.max(self.model(next_states[idx]))                
        
        locations = [[x] for x in torch.argmax(actions, dim=1).numpy()]
        locations = torch.tensor(locations)
        
        preds = self.model(states).gather(1, locations)
        preds = preds.squeeze(1)
        
        self.trainer.train_step(targets, preds, False)

    def train_short_memory(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        target = reward
        if not done:
            target = reward + self.gamma * torch.max(self.model(next_state))
        pred = self.model(state)
        
        target_f = pred.clone()
        target_f[torch.argmax(action).item()] = target
                
        self.trainer.train_step(target_f, pred, True)

    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] += 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] += 1
        return final_move


def train():

    plot_scores = []
    plot_mean_scores =[]
    total_score = 0
    record = 0
    agent = Agent()
    game = zz_game_ai()
    while True:
        #get old state
        state_old = agent.get_state(game)
        
        final_move = agent.get_action(state_old)

        #perform new move and get new state
        reward, done, score = game.frame_step(final_move)
        state_new = agent.get_state(game)
    
        #train short memory base on the new action and state
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        
        # store the new data into a long term memory
        agent.remember(state_old, final_move, reward, state_new, done)

        if done == True:
            # One game is over, train on the memory and plot the result.
            game.reset()
            agent.train_long_memory(agent.memory)
            
            if score > record:
                record = score
                agent.model.save()
                
            print('Game', agent.n_games, ', Score:', score, ', Record:', record)
            
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)



if __name__ == '__main__':
    train()