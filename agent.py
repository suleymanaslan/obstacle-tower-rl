# adapted from https://github.com/Kaixhin/Rainbow

import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from network import DQN


class Agent():
    def __init__(self, env, atoms, V_min, V_max, batch_size, multi_step, discount, norm_clip, lr, adam_eps, hidden_size, noisy_std):
        self.device = torch.device("cuda:0")
        self.action_size = len(env.action_space)
        self.atoms = atoms
        self.Vmin = V_min
        self.Vmax = V_max
        self.support = torch.linspace(self.Vmin, self.Vmax, self.atoms).to(self.device)
        self.delta_z = (self.Vmax - self.Vmin) / (self.atoms - 1)
        self.batch_size = batch_size
        self.n = multi_step
        self.discount = discount
        self.norm_clip = norm_clip
        
        self.online_net = DQN(self.atoms, self.action_size, env.window, hidden_size=256, noisy_std=0.1).to(self.device)
        self.online_net.train()
        
        self.target_net = DQN(self.atoms, self.action_size, env.window, hidden_size=256, noisy_std=0.1).to(self.device)
        self.update_target_net()
        self.target_net.train()
        
        for param in self.target_net.parameters():
            param.requires_grad = False
        
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr, eps=adam_eps)
    
    def train(self):
        self.online_net.train()
    
    def eval(self):
        self.online_net.eval()
    
    def update_target_net(self):
        self.target_net.load_state_dict(self.online_net.state_dict())
    
    def reset_noise(self):
        self.online_net.reset_noise()
    
    def act(self, observation):
        with torch.no_grad():
            return (self.online_net(observation.unsqueeze(0)) * self.support).sum(2).argmax(1).item()
    
    def act_e_greedy(self, state, epsilon=0.001):
        return np.random.randint(0, self.action_size) if np.random.random() < epsilon else self.act(state)
    
    def learn(self, mem):
        idxs, states, actions, returns, next_states, nonterminals, weights = mem.sample(self.batch_size)
        
        log_ps = self.online_net(states, use_log_softmax=True)
        log_ps_a = log_ps[range(self.batch_size), actions]
        
        with torch.no_grad():
            pns = self.online_net(next_states)
            dns = self.support.expand_as(pns) * pns
            argmax_indices_ns = dns.sum(2).argmax(1)
            self.target_net.reset_noise()
            pns = self.target_net(next_states)
            pns_a = pns[range(self.batch_size), argmax_indices_ns]
            
            Tz = returns.unsqueeze(1) + nonterminals * (self.discount ** self.n) * self.support.unsqueeze(0)
            Tz = Tz.clamp(min=self.Vmin, max=self.Vmax)
            b = (Tz - self.Vmin) / self.delta_z
            l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.atoms - 1)) * (l == u)] += 1
            
            m = states.new_zeros(self.batch_size, self.atoms)
            offset = torch.linspace(0, ((self.batch_size - 1) * self.atoms), self.batch_size).unsqueeze(1).expand(self.batch_size, self.atoms).to(actions)
            m.view(-1).index_add_(0, (l + offset).view(-1), (pns_a * (u.float() - b)).view(-1))
            m.view(-1).index_add_(0, (u + offset).view(-1), (pns_a * (b - l.float())).view(-1))
        
        loss = -torch.sum(m * log_ps_a, 1)
        self.online_net.zero_grad()
        (weights * loss).mean().backward()
        clip_grad_norm_(self.online_net.parameters(), self.norm_clip)
        self.optimizer.step()
        
        mem.update_priorities(idxs, loss.detach().cpu().numpy())
