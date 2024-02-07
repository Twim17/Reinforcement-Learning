from dense import DenseModel
import numpy as np
import torch
import torch.distributions as td
import torch.nn as nn

class ActionModel(DenseModel):
    def __init__(self, in_size, out_size, hidden_size, dist, epsilon=0.4, expl_decay=7000.0, min_epsilon=0.05, action_size=4, device='cuda'):
        super().__init__(in_size, out_size, hidden_size, dist)
        #('ciao sono l action model, la mia in_size e: ', in_size)
        self.epsilon = epsilon
        self.expl_decay = expl_decay
        self.min_epsilon = min_epsilon
        self.action_size = action_size

    
    def forward(self, model_state):
        #print('model_state.shape', model_state.shape)
        #print('in_size', self.in_size)
        action_dist = self.get_action_dist(model_state)
        action = action_dist.sample()
        action = action + action_dist.probs - action_dist.probs.detach() #algorithm 1
        # print(action)
        # print(action_dist)
        return action, action_dist

    def get_action_dist(self, modelstate):
        logits = self.dense(modelstate)
        return torch.distributions.OneHotCategorical(logits=logits)         #forse da provare straight through

    def explore(self, action: torch.Tensor, itr: int):
        epsilon = self.epsilon
        epsilon = epsilon - itr/self.expl_decay
        epsilon = max(self.min_epsilon, epsilon)  
        #print('-'*10, 'explore', '-'*10)    
        #print('epsilon', epsilon)
        #print('action', action)
        if np.random.uniform(0, 1) < epsilon:
            index = torch.randint(0, self.action_size, action.shape[:-1], device=action.device)
            # print('action.shape', action.shape)
            # print('action.shape[:-1]', action.shape[:-1])
            # print('index', index)
            action = torch.zeros_like(action)
            #print('action2', action)
            action[:, index] = 1
            #print('action3', action)
        return action