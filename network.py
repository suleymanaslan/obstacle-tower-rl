import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingDoubleDQN(nn.Module):
    def __init__(self, input_size, action_size, feature_size):
        super(DuelingDoubleDQN, self).__init__()
        net_size = [input_size,
                    feature_size,
                    feature_size,
                    action_size]
        self.fc1 = nn.Linear(net_size[0], net_size[1])
        self.fc_v = nn.Linear(net_size[1], 1)
        self.fc_a = nn.Linear(net_size[2], net_size[3])
    
    def forward(self, state):
        out = F.relu(self.fc1(state))
        v = self.fc_v(out)
        a = self.fc_a(out)
        
        return v, a
