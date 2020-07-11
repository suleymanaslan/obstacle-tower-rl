import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, input_size, action_size, feature_size):
        super(QNetwork, self).__init__()
        net_size = [input_size,
                    feature_size,
                    feature_size,
                    action_size]
        self.fc1 = nn.Linear(net_size[0], net_size[1])
        self.fc2 = nn.Linear(net_size[1], net_size[2])
        self.fc3 = nn.Linear(net_size[2], net_size[3])
    
    def forward(self, state):
        out = F.relu(self.fc1(state))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        
        return out
