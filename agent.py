import numpy as np
import torch
import gym


class Agent():
    def __init__(self, model, input_size, action_size, gamma, epsilon, min_epsilon, epsilon_decay, memory_size):
        self.gamma = gamma
        self.epsilon = epsilon
        self.action_space = [i for i in range(action_size)]
        self.memory_size = memory_size
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.memory_counter = 0
        
        self.model = model
        
        self.state_memory = np.zeros((self.memory_size, input_size), dtype=np.float32)
        self.new_state_memory = np.zeros((self.memory_size, input_size), dtype=np.float32)
        self.action_memory = np.zeros(self.memory_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.memory_size, dtype=np.float32)
        self.done_memory = np.zeros(self.memory_size, dtype=np.bool)
    
    def save_transition(self, state, new_state, action, reward, done):
        memory_ix = self.memory_counter % self.memory_size
        
        self.state_memory[memory_ix] = state
        self.new_state_memory[memory_ix] = new_state
        self.action_memory[memory_ix] = action
        self.reward_memory[memory_ix] = reward
        self.done_memory[memory_ix] = done
        
        self.memory_counter += 1
    
    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = torch.tensor([observation]).to(self.model.device)
            _, advantage = self.model.online_network.forward(state)
            action = torch.argmax(advantage).item()
        else:
            action = np.random.choice(self.action_space)
        
        return action
    
    def train(self, update_target_network):
        if self.memory_counter < self.model.batch_size:
            return
        
        available_memory = min(self.memory_counter, self.memory_size)
        batch_ix = np.random.choice(available_memory, self.model.batch_size, replace=False)
        batch_indices = np.arange(self.model.batch_size, dtype=np.int32)
        
        batch_state = torch.tensor(self.state_memory[batch_ix]).to(self.model.device)
        batch_new_state = torch.tensor(self.new_state_memory[batch_ix]).to(self.model.device)
        batch_action = self.action_memory[batch_ix]
        batch_reward = torch.tensor(self.reward_memory[batch_ix]).to(self.model.device)
        batch_done = torch.tensor(self.done_memory[batch_ix]).to(self.model.device)
        
        train_batch = [batch_indices, batch_state, batch_new_state, batch_action, batch_reward, batch_done]
        
        self.model.train(train_batch, self.gamma, update_target_network)
        
        self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon > self.min_epsilon else self.min_epsilon
    
    def start_training(self, episodes, window_size):
        env = gym.make("LunarLander-v2")
        
        scores = []
        steps = 0
        for episode_ix in range(episodes):
            score = 0
            done = False
            observation = env.reset()
            while not done:
                action = self.choose_action(observation)
                new_observation, reward, done, info = env.step(action)
                score += reward
                steps += 1
                self.save_transition(observation, new_observation, action, reward, done)
                self.train(update_target_network=(steps % self.model.target_update == 0))
                observation = new_observation
            scores.append(score)
            print(f"Episode:{episode_ix+1},\tScore:{score:.2f},\tAvg. Score:{np.mean(scores[-window_size:]):.2f},\tEpsilon:{self.epsilon:.2f}")
        avg_scores = [np.mean(scores[max(0, i-window_size):i+1]) for i in range(len(scores))]
            
        return scores, avg_scores
