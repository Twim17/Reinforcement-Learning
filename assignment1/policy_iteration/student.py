import random

import numpy as np
import gym
import time
from gym import spaces
import os

def value_iteration(env):
    gamma=0.99
    iters=100

    #initialize values
    values = np.zeros((env.num_states))
    best_actions = np.zeros((env.num_states), dtype=int)
    STATES = np.zeros((env.num_states, 2), dtype=np.uint8)
    REWARDS = env.reward_probabilities()
    i = 0
    for r in range(env.height):
        for c in range(env.width):
            state = np.array([r, c], dtype=np.uint8)
            STATES[i] = state
            i += 1

    for i in range(iters):
        v_old = values.copy()
        for s in range(env.num_states):
            state = STATES[s]

            if (state == env.end_state).all() or i >= env.max_steps:
                continue # if we reach the termination condition, we cannot perform any action
                
            max_va = -np.inf
            best_a = 0
            for a in range(env.num_actions):
                next_state_prob = env.transition_probabilities(state, a).flatten()
                
                va = (next_state_prob*(REWARDS + gamma*v_old)).sum()

                if va > max_va:
                    max_va = va
                    best_a = a
            values[s] = max_va
            best_actions[s] = best_a

    return best_actions.reshape((env.height, env.width))

def policy_evaluation(env, policy_i, STATES, values):
    gamma=0.99
    iters=100

    #initialize values
    values = values
    policy_i = policy_i
    STATES = STATES
    REWARDS = env.reward_probabilities()

    
    v_old = values.copy()
    for s in range(env.num_states):
        state = STATES[s]

        next_state_prob = env.transition_probabilities(state, policy_i[s]).flatten()
        values[s] = (next_state_prob*(REWARDS + gamma*v_old)).sum()

    return values


def policy_improvement(env, policy_i, STATES, values, old_max_va, gamma=0.99):
    REWARDS = env.reward_probabilities()
    optimal = True
    for s in range(env.num_states):
        state = STATES[s]
        
        max_va = old_max_va.copy()
        old_a = policy_i[s]
        best_a = 0
        for a in range(env.num_actions):
            next_state_prob = env.transition_probabilities(state, a).flatten()

            va = (next_state_prob*(REWARDS + gamma*values)).sum()

            if va > max_va[s]:
                max_va[s] = va
                best_a = a
        policy_i[s] = best_a
        if old_a != policy_i[s]:
            optimal = False
        
    return policy_i, max_va, optimal

def policy_iteration(env, gamma=0.99, iters=1000):
    policy_i = np.zeros(env.num_states, dtype=np.int)
    values_i = np.zeros((env.num_states))
    old_max_va = -np.inf * np.ones((env.num_states))
    STATES = np.zeros((env.num_states, 2), dtype=np.uint8)
    REWARDS = env.reward_probabilities()
    
    i = 0
    for r in range(env.height):
        for c in range(env.width):
            state = np.array([r, c], dtype=np.uint8)
            STATES[i] = state
            i += 1
    for i in range(iters):
        values_i = policy_evaluation(env, policy_i, STATES, values_i)
        #print("values",(i,values_i))
        policy_i, old_max_va, optimal = policy_improvement(env, policy_i, STATES, values_i, old_max_va)
        if optimal == True: #Policy doesn't change for any state
             break
        #print("policy",(i,policy_i))

    return policy_i.reshape(env.height, env.width)