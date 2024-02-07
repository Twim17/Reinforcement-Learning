import random
import numpy as np
import gym
import time
from gym import spaces
import os
import pickle
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler

class VanillaFeatureEncoder:
    def __init__(self, env):
        self.env = env
        
    def encode(self, state):
        return state
    
    @property
    def size(self): 
        return self.env.observation_space.shape[0]

class RBFFeatureEncoder:
    def __init__(self, env): # modify
        self.env = env
        self.n_components = 100
        self.rbf_randomstate = 1
        self.scaler = StandardScaler()
        self.obs_samples = np.array([env.observation_space.sample() for x in range(10000)])
        self.scaler.fit(self.obs_samples)
        self.rbfs = FeatureUnion([
            ("rbf1",RBFSampler(gamma=3.0,n_components=self.n_components,random_state=self.rbf_randomstate)),
            ("rbf2",RBFSampler(gamma=2.0,n_components=self.n_components,random_state=self.rbf_randomstate)),
            ("rbf3",RBFSampler(gamma=1.0,n_components=self.n_components,random_state=self.rbf_randomstate)),
            ("rbf4",RBFSampler(gamma=0.5,n_components=self.n_components,random_state=self.rbf_randomstate))
        ])
        self.rbfs.fit(self.scaler.transform(self.obs_samples))
        
    def encode(self, state): # modify
        # encoding function
        state = np.reshape(state, (1,-1))
        scaled = self.scaler.transform(state)
        state = self.rbfs.transform(scaled)

        return state[0]
    
    @property
    def size(self): # modify
        # return the correct size of the observation
        return self.n_components*4
        #return self.env.observation_space.shape[0]

class TDLambda_LVFA:
    def __init__(self, env, feature_encoder_cls=RBFFeatureEncoder,
        alpha=0.00751, alpha_decay=1, 
        gamma=0.9999, epsilon=0.4, epsilon_decay=0.99, lambda_=0.9): # modify if you want
        self.env = env
        self.feature_encoder = feature_encoder_cls(env)
        self.shape = (self.env.action_space.n, self.feature_encoder.size)
        self.weights = np.random.random(self.shape)
        self.traces = np.zeros(self.shape)
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.lambda_ = lambda_
        
    def Q(self, feats):
        # print("w",self.weights.shape)
        # print("f1",feats.shape)
        feats = feats.reshape(-1,1)
        # print("f2",feats.shape)
        return self.weights@feats
    
    def update_transition(self, s, action, s_prime, reward, done): # modify
        #print(self.weights.shape)
        s_feats = self.feature_encoder.encode(s)
        s_prime_feats = self.feature_encoder.encode(s_prime)
        action_prime = self.epsilon_greedy(s_prime)
        delta = reward
        if not done:
            delta += self.gamma * self.Q(s_prime_feats)[action_prime]
        
        delta -= self.Q(s_feats)[action]
        self.traces = self.traces * self.lambda_ * self.gamma
        self.traces[action] += s_feats
        self.weights[action] += self.alpha * (delta * self.traces[action])
        #print(self.traces)


        
    def update_alpha_epsilon(self): # modify
        self.epsilon = max(0.05, self.epsilon*self.epsilon_decay)
        self.alpha = max(0.005, self.alpha*self.alpha_decay)
        #pass
        
        
    def policy(self, state): # do not touch
        state_feats = self.feature_encoder.encode(state)
        return self.Q(state_feats).argmax()
    
    def epsilon_greedy(self, state, epsilon=None):  # modify
        if epsilon is None: epsilon = self.epsilon
        if random.random()<epsilon:
            return self.env.action_space.sample()
        return self.policy(state)
       
        
    def train(self, n_episodes=200, max_steps_per_episode=200): # do not touch
        for episode in range(n_episodes):
            done = False
            s, _ = self.env.reset()
            self.traces = np.zeros(self.shape)
            for i in range(max_steps_per_episode):
                
                action = self.epsilon_greedy(s)
                s_prime, reward, done, _, _ = self.env.step(action)
                self.update_transition(s, action, s_prime, reward, done)

                s = s_prime
                
                if done: break
                
            self.update_alpha_epsilon()

            if episode % 20 == 0:
                print(episode, self.evaluate(), self.epsilon, self.alpha)
                
    def evaluate(self, env=None, n_episodes=10, max_steps_per_episode=200): # do not touch
        if env is None:
            env = self.env
            
        rewards = []
        for episode in range(n_episodes):
            total_reward = 0
            done = False
            s, _ = env.reset()
            for i in range(max_steps_per_episode):
                action = self.policy(s)
                
                s_prime, reward, done, _, _ = env.step(action)
                
                total_reward += reward
                s = s_prime
                if done: break
            
            rewards.append(total_reward)
            
        return np.mean(rewards)

    def save(self, fname):
        with open(fname, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, fname):
        return pickle.load(open(fname,'rb'))
