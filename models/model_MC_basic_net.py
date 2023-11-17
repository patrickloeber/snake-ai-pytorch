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


class MCTrainer:
    def __init__(self, model, lr):
        self.lr = lr
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train(self, episodes):
        for episode in episodes:
            states, actions, rewards = zip(*episode)
            states = torch.tensor(states, dtype=torch.float)
            actions = torch.tensor(actions, dtype=torch.long)
            returns = self.compute_returns(rewards)

            for _, (state, action, Gt) in enumerate(zip(states[-1:], actions[-1:], returns[-1:])):
                self.train_step(state, action, Gt)

    def train_step(self, state, action, Gt):
        state = torch.unsqueeze(state, 0)
        action = torch.unsqueeze(action, 0)
        Gt = torch.unsqueeze(Gt, 0)
        # Gt = torch.tensor([Gt], dtype=torch.float)

        pred = self.model(state)
        # print(pred.shape)
        # print(action.shape)
        # print(action)
        value = pred[:, torch.argmax(action)]
        # value = pred.gather(1, action.view(-1, 1)).squeeze(1)
        # print(value.shape)
        # print(Gt.shape)

        self.optimizer.zero_grad()
        loss = self.criterion(value, Gt)
        loss.backward()
        self.optimizer.step()

    def compute_returns(self, rewards, gamma=0.9):
        R = 0
        returns = []
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        # Normalize for numerical stability
        returns = torch.tensor(returns)
        # returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        return returns
