import torch
import torch.nn as nn
import torch.optim as optim


class Model():
    def __init__(self, q_network, lr, batch_size):
        self.q_network = q_network
        self.lr = lr
        self.batch_size = batch_size
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()
        self.device = torch.device("cuda:0")
        
        self.q_network.to(self.device)
    
    def train(self, train_batch, gamma):
        batch_indices, batch_state, batch_new_state, batch_action, batch_reward, batch_done = train_batch
        
        self.optimizer.zero_grad()
        
        net_q = self.q_network.forward(batch_state)[batch_indices, batch_action]
        net_new_q = self.q_network.forward(batch_new_state)
        net_new_q[batch_done] = 0.0
        
        target_q = batch_reward * gamma + torch.max(net_new_q, dim=1)[0]
        
        loss = self.loss(target_q, net_q).to(self.device)
        loss.backward()
        
        self.optimizer.step()
