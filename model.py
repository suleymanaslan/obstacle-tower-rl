import torch
import torch.nn as nn
import torch.optim as optim


class Model():
    def __init__(self, online_network, target_network, lr, batch_size, target_update):
        self.online_network = online_network
        self.target_network = target_network
        self.lr = lr
        self.batch_size = batch_size
        self.target_update = target_update
        
        self.optimizer = optim.Adam(self.online_network.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()
        self.device = torch.device("cuda:0")
        
        self.online_network.to(self.device)
        self.target_network.to(self.device)
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()
    
    def train(self, train_batch, gamma, update_target_network):
        batch_indices, batch_state, batch_new_state, batch_action, batch_reward, batch_done = train_batch
        
        self.optimizer.zero_grad()
        
        if update_target_network:
            self.target_network.load_state_dict(self.online_network.state_dict())
        
        v_online, a_online = self.online_network.forward(batch_state)
        v_target, a_target = self.target_network.forward(batch_new_state)
        v_next, a_next = self.online_network.forward(batch_new_state)
        
        q_pred = torch.add(v_online, (a_online - a_online.mean(dim=1, keepdim=True)))[batch_indices, batch_action]
        q_next = torch.add(v_target, (a_target - a_target.mean(dim=1, keepdim=True)))
        q_next[batch_done] = 0.0
        q_eval = torch.add(v_next, (a_next - a_next.mean(dim=1, keepdim=True)))
        
        max_actions = torch.argmax(q_eval, dim=1)
        q_target = batch_reward + gamma * q_next[batch_indices, max_actions]
        
        loss = self.loss(q_target, q_pred).to(self.device)
        loss.backward()
        self.optimizer.step()
