import minatar
import gym
import numpy as np
from torchvision import transforms
import torch

class GymMinAtar(gym.Env):
    def __init__(self, env_name = 'breakout', action_repeat=1, display_time = 50):
        self.env_name = env_name
        self.action_repeat = action_repeat
        self.display_time = display_time
        self.env = minatar.Environment(env_name) 
        self.minimal_actions = self.env.minimal_action_set()
        h,w,c = self.env.state_shape()
        self.env.action_space = gym.spaces.Discrete(len(self.minimal_actions))
        self.act_shape = self.env.action_space.n
        self.env.action_space.sample = self.sample_action
        self.observation_space = gym.spaces.MultiBinary((c,h,w))
    
    def reset(self):
        self.env.reset()
        return self.env.state().transpose(2, 0, 1)
    
    def step(self, action):
        '''index is the action id, considering only the set of minimal actions'''
        # print("action",action)
        if type(action) == int:
            index = action
        else:
            index = np.argmax(action).astype(int)
        action = index
        action = self.minimal_actions[action]

        r, terminal = self.env.act(action)
        total_reward = r
        current_step = 1
        while current_step < self.action_repeat and not terminal:
            r, terminal = self.env.act(action)
            total_reward += r
            current_step += 1
            self.game_over = terminal
        # print("primo",self.env.state().shape)
        # print("secondo",self.env.state().transpose(2, 0, 1).shape)
        return self.env.state().transpose(2, 0, 1), r, terminal, {}

    def seed(self, seed=69420777):
        self.env = minatar.Environment(self.env_name, random_seed=seed)

    def sample_action(self):
        actions = self.act_shape
        index = np.random.randint(0, actions)
        return index

    def render(self, mode='human'):
        if mode == 'rgb_array':
            return self.env.state()
        elif mode == 'human':
            self.env.display_state(self.display_time)

    # def close(self):
    #     if self.env.visualized:
    #         self.env.close_display()
    #     return 0


# class OneHotAction(gym.Wrapper):
#     def __init__(self, env):
#         assert isinstance(env.action_space, gym.spaces.Discrete), "This wrapper only works with discrete action space"
#         shape = (env.action_space.n,)
#         env.action_space = gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
#         env.action_space.sample = self._sample_action
#         super(OneHotAction, self).__init__(env)
    
#     def step(self, action):
#         index = np.argmax(action).astype(int)
#         reference = np.zeros_like(action)
#         reference[index] = 1
#         return self.env.step(index)

#     def reset(self):
#         return self.env.reset()
    
#     def _sample_action(self):
#         actions = self.env.action_space.shape[0]
#         index = np.random.randint(0, actions)
#         reference = np.zeros(actions, dtype=np.float32)
#         reference[index] = 1.0
#         return reference

class GymAtari(gym.Env):
    def __init__(self, env_name = 'Breakout-v0', action_repeat=1, render_mode = 'rgb_array'): #render_mode = 'rgb_array'
        self.env_name = env_name
        self.env = gym.make(env_name, render_mode=render_mode)
        self.act_shape = self.env.action_space.n
        self.obs_shape = self.env.observation_space.shape
        self.resize = transforms.Resize((64,64))
        self.grayscale = transforms.Grayscale()
        self.action_repeat = action_repeat
        
        # self.action_space = gym.spaces.Discrete(len(self.minimal_actions))
        # self.observation_space = gym.spaces.MultiBinary((c,h,w))
    
    def reset(self):
        state = self.env.reset()
        return self.process_img(state) #mette channels davanti c,h,w
    
    def step(self, action):
        '''index is the action id, considering only the set of minimal actions'''
        if type(action) == int:
            index = action
        else:
            index = np.argmax(action).astype(int)
        # print("action",action)
        # print("index",index)
        action = index

        state, reward, terminal, info = self.env.step(action)
        total_reward = reward
        current_step = 1
        while current_step < self.action_repeat and not terminal:
            state, reward, terminal, info = self.env.step(action)
            total_reward += reward
            current_step += 1
            self.game_over = terminal

        return self.process_img(state), reward, terminal, info

    def process_img(self, state):
        # print("primo",state.shape)
        # print("secondo",state.transpose(2,0,1).shape)
        state = torch.from_numpy(state.transpose(2,0,1)).type(dtype=torch.float32)
        state = self.grayscale(state)
        self.state = self.resize(state)
        return self.state
    

