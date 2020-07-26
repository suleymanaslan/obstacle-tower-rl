# adapted from https://github.com/Kaixhin/Rainbow

import torch
import numpy as np
from collections import namedtuple


Transition = namedtuple('Transition', ('timestep', 'state', 'action', 'reward', 'nonterminal'))
blank_trans = Transition(0, torch.zeros(84, 84, dtype=torch.uint8), None, 0, False)


class SegmentTree():
    def __init__(self, size):
        self.index = 0
        self.size = size
        self.full = False
        self.sum_tree = np.zeros((2 * size - 1, ), dtype=np.float32)
        self.data = np.array([None] * size)
        self.max = 1
    
    def _propagate(self, index, value):
        parent = (index - 1) // 2
        left, right = 2 * parent + 1, 2 * parent + 2
        self.sum_tree[parent] = self.sum_tree[left] + self.sum_tree[right]
        if parent != 0:
            self._propagate(parent, value)
    
    def update(self, index, value):
        self.sum_tree[index] = value
        self._propagate(index, value)
        self.max = max(value, self.max)
    
    def append(self, data, value):
        self.data[self.index] = data
        self.update(self.index + self.size - 1, value)
        self.index = (self.index + 1) % self.size
        self.full = self.full or self.index == 0
        self.max = max(value, self.max)
        
    def total(self):
        return self.sum_tree[0]
    
    def _retrieve(self, index, value):
        left, right = 2 * index + 1, 2 * index + 2
        if left >= len(self.sum_tree):
            return index
        elif value <= self.sum_tree[left]:
            return self._retrieve(left, value)
        else:
            return self._retrieve(right, value - self.sum_tree[left])
    
    def find(self, value):
        index = self._retrieve(0, value)
        data_index = index - self.size + 1
        return (self.sum_tree[index], data_index, index)
    
    def get(self, data_index):
        return self.data[data_index % self.size]


class ReplayMemory():
    def __init__(self, capacity, history_length, discount, multi_step, priority_weight, priority_exponent):
        self.device = torch.device("cuda:0")
        self.capacity = capacity
        self.history = history_length
        self.discount = discount
        self.n = multi_step
        self.priority_weight = priority_weight
        self.priority_exponent = priority_exponent
        self.t = 0
        self.transitions = SegmentTree(capacity)
        
    def append(self, state, action, reward, terminal):
        state = state[-1].mul(255).to(dtype=torch.uint8, device=torch.device("cpu"))
        self.transitions.append(Transition(self.t, state, action, reward, not terminal), self.transitions.max)
        self.t = 0 if terminal else self.t + 1
        
    def _get_transition(self, idx):
        transition = np.array([None] * (self.history + self.n))
        transition[self.history - 1] = self.transitions.get(idx)
        for t in range(self.history - 2, -1, -1):
            if transition[t + 1].timestep == 0:
                transition[t] = blank_trans
            else:
                transition[t] = self.transitions.get(idx - self.history + 1 + t)
        for t in range(self.history, self.history + self.n):
            if transition[t - 1].nonterminal:
                transition[t] = self.transitions.get(idx - self.history + 1 + t)
            else:
                transition[t] = blank_trans
        return transition
        
    def _get_sample_from_segment(self, segment, i):
        valid = False
        while not valid:
            sample = np.random.uniform(i * segment, (i + 1) * segment)
            prob, idx, tree_idx = self.transitions.find(sample)
            if (self.transitions.index - idx) % self.capacity > self.n and (idx - self.transitions.index) % self.capacity >= self.history and prob != 0:
                valid = True
        
        transition = self._get_transition(idx)
        state = torch.stack([trans.state for trans in transition[:self.history]]).to(device=self.device).to(dtype=torch.float32).div_(255)
        next_state = torch.stack([trans.state for trans in transition[self.n:self.n + self.history]]).to(device=self.device).to(dtype=torch.float32).div_(255)
        action = torch.tensor([transition[self.history - 1].action], dtype=torch.int64, device=self.device)
        R = torch.tensor([sum(self.discount ** n * transition[self.history + n - 1].reward for n in range(self.n))], dtype=torch.float32, device=self.device)
        nonterminal = torch.tensor([transition[self.history + self.n - 1].nonterminal], dtype=torch.float32, device=self.device)
        
        return prob, idx, tree_idx, state, action, R, next_state, nonterminal
    
    def sample(self, batch_size):
        p_total = self.transitions.total()
        segment = p_total / batch_size
        batch = [self._get_sample_from_segment(segment, i) for i in range(batch_size)]
        probs, idxs, tree_idxs, states, actions, returns, next_states, nonterminals = zip(*batch)
        states, next_states, = torch.stack(states), torch.stack(next_states)
        actions, returns, nonterminals = torch.cat(actions), torch.cat(returns), torch.stack(nonterminals)
        probs = np.array(probs, dtype=np.float32) / p_total
        capacity = self.capacity if self.transitions.full else self.transitions.index
        weights = (capacity * probs) ** -self.priority_weight
        weights = torch.tensor(weights / weights.max(), dtype=torch.float32, device=self.device)
        return tree_idxs, states, actions, returns, next_states, nonterminals, weights
    
    def update_priorities(self, idxs, priorities):
        priorities = np.power(priorities, self.priority_exponent)
        [self.transitions.update(idx, priority) for idx, priority in zip(idxs, priorities)]


class SimpleReplayMemory():
    def __init__(self, capacity):
        self.device = torch.device("cuda:0")
        self.capacity = capacity
        self.counter = 0
        self.state_memory = np.zeros((self.capacity, 1, 84, 84), dtype=np.float32)
        self.action_memory = np.zeros(self.capacity, dtype=np.int32)
        self.reward_memory = np.zeros(self.capacity, dtype=np.float32)
        self.done_memory = np.zeros(self.capacity, dtype=np.bool)
        self.new_state_memory = np.zeros((self.capacity, 1, 84, 84), dtype=np.float32)
    
    def append(self, state, action, reward, done, new_state):
        memory_ix = self.counter % self.capacity
        self.state_memory[memory_ix] = state
        self.action_memory[memory_ix] = action
        self.reward_memory[memory_ix] = reward
        self.done_memory[memory_ix] = done
        self.new_state_memory[memory_ix] = new_state
        self.counter += 1
        
    def sample(self, batch_size):
        available_memory = min(self.counter, self.capacity)
        batch_ix = np.random.choice(available_memory, batch_size, replace=False)
        batch_indices = np.arange(batch_size, dtype=np.int32)
        
        batch_state = torch.tensor(self.state_memory[batch_ix]).to(self.device)
        batch_action = self.action_memory[batch_ix]
        batch_reward = torch.tensor(self.reward_memory[batch_ix]).to(self.device)
        batch_done = torch.tensor(self.done_memory[batch_ix]).to(self.device)
        batch_new_state = torch.tensor(self.new_state_memory[batch_ix]).to(self.device)
        
        train_batch = [batch_indices, batch_state, batch_action, batch_reward, batch_done, batch_new_state]
        
        return train_batch
