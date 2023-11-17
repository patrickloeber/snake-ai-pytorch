import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os


class ValueNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)  # Using softmax to get a probability distribution
        return x

    def save(self, file_name='model.pth'):
        pass
        # model_folder_path = './model'
        # if not os.path.exists(model_folder_path):
        #     os.makedirs(model_folder_path)

        # file_name = os.path.join(model_folder_path, file_name)
        # torch.save(self.state_dict(), file_name)


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


class ACTrainer:
    def __init__(self, value_model, policy_model, lr):
        self.policy_model = policy_model
        self.value_model = value_model
        self.policy_optimizer = optim.Adam(policy_model.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(value_model.parameters(), lr=lr)
        self.gamma = 0.9

    def train_step(self, state, action, reward, next_state, next_action, done):

        # Convert lists to PyTorch tensors
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        action = torch.tensor(action, dtype=torch.long).unsqueeze(0)
        reward = torch.tensor(reward, dtype=torch.float).unsqueeze(0)
        next_state = torch.tensor(next_state, dtype=torch.float).unsqueeze(0)
        next_action = torch.tensor(next_action, dtype=torch.long).unsqueeze(0)

        # Get value of next state
        pred_q_value = self.value_model(state)[torch.arange(len(state)), torch.argmax(action, 1)].unsqueeze(-1)  # 1 * 1
        with torch.no_grad():
            target_q_value = reward + self.gamma * self.value_model(next_state)[torch.arange(len(next_state)), torch.argmax(next_action, 1)]  # 1 * 1

        # print(pred_q_value.shape)
        # print(pred_q_value.shape)
        value_loss = nn.MSELoss()(target_q_value, pred_q_value)
        value_loss = value_loss.sum()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # Calculate policy gradient loss
        log_probs = torch.log(self.policy_model(state))
        selected_log_probs = log_probs[torch.arange(len(log_probs)), torch.argmax(action, 1)].unsqueeze(-1)  # 1 * 1

        with torch.no_grad():
            pred_q_value = self.value_model(state)[torch.arange(len(state)), torch.argmax(action, 1)]  # 1 * 1
        policy_loss = -selected_log_probs * pred_q_value  # B * 1
        policy_loss = policy_loss.sum()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
