import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


class Linear_VNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        pass
        # model_folder_path = './model'
        # if not os.path.exists(model_folder_path):
        #     os.makedirs(model_folder_path)

        # file_name = os.path.join(model_folder_path, file_name)
        # torch.save(self.state_dict(), file_name)


class SARSATrainer:
    def __init__(self, model, lr):
        self.lr = lr
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.gamma = 0.9

    def train(self, state_old, final_move, reward, state_new, final_move_new, done):
        self.train_step(state_old, final_move, reward, state_new, final_move_new, done)

    def train_step(self, state_old, action_old, reward, state_new, action_new, done):
        state_old = torch.tensor(state_old, dtype=torch.float)
        action_old = torch.tensor(action_old, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.long)
        state_new = torch.tensor(state_new, dtype=torch.float)
        action_new = torch.tensor(action_new, dtype=torch.long)

        state_old = torch.unsqueeze(state_old, 0)
        action_old = torch.unsqueeze(action_old, 0)
        reward = torch.unsqueeze(reward, 0)
        state_new = torch.unsqueeze(state_new, 0)
        action_new = torch.unsqueeze(action_new, 0)

        pred_old = self.model(state_old)
        with torch.no_grad():
            pred_new = self.model(state_new)
        # print(pred.shape)
        # print(action.shape)
        # print(action)
        value = pred_old[:, torch.argmax(action_old)]
        target = reward + self.gamma * pred_new[:, torch.argmax(action_new)]
        # value = pred.gather(1, action.view(-1, 1)).squeeze(1)
        # print(value.shape)
        # print(Gt.shape)

        self.optimizer.zero_grad()
        loss = self.criterion(value, target)
        loss.backward()
        self.optimizer.step()
