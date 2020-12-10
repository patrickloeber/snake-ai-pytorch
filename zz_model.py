import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        #x = self.linear1(x)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    
    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
            
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class Trainer:
    
    def __init__(self, model):
        self.lr = 0.001
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        
    def train_step(self, target, pred, zero_grad):
        if zero_grad:
            self.optimizer.zero_grad()
            
        loss = self.criterion(target, pred)
        loss.backward()
        
        self.optimizer.step()