import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os

class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.softmax(self.linear2(x), dim=-1)  # Using softmax to get a probability distribution
        return x

    def save(self, file_name='model.pth'):
        pass
        # model_folder_path = './model'
        # if not os.path.exists(model_folder_path):
        #     os.makedirs(model_folder_path)

        # file_name = os.path.join(model_folder_path, file_name)
        # torch.save(self.state_dict(), file_name)


class ReinforceTrainer:
    def __init__(self, model, lr):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.gamma = 0.9

    def train(self, episodes):
        for episode in episodes:
            states, actions, rewards = zip(*episode)

            # Convert lists to PyTorch tensors
            states = torch.tensor(states, dtype=torch.float)
            actions = torch.tensor(actions, dtype=torch.long)
            rewards = torch.tensor(rewards, dtype=torch.float)

            # Calculate discounted rewards
            G = 0
            discounted_rewards = []
            for reward in reversed(rewards):  # reverse buffer r
                G = reward + self.gamma * G
                discounted_rewards.insert(0, G)
            discounted_rewards = torch.tensor(discounted_rewards)  # B * 1

            # Normalize discounted rewards
            # discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-6)

            # Get the log probabilities of the taken actions
            log_probs = torch.log(self.model(states))
            # print(log_probs.shape)
            selected_log_probs = log_probs[torch.arange(len(log_probs)), torch.argmax(actions, 1)]  # B * 1
            # print(selected_log_probs.shape)
            # print(selected_log_probs)
            # print(torch.argmax(actions, 1))
            # print(discounted_rewards.shape)

            # Calculate policy gradient loss
            policy_gradient = -selected_log_probs * discounted_rewards  # B * 1
            loss = policy_gradient.sum()

            # Perform a single optimization step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
