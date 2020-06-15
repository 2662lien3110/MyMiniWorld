import gym
import math
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from csv import writer, reader

from gym_miniworld.wrappers import *
from cdqn_model_res import DQN
from rpmBaseline import rpm

import logging
logging.basicConfig(level=logging.DEBUG)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#np.random.seed(1)
#random.seed(1)
#torch.manual_seed(1)
#torch.cuda.manual_seed(1)
#torch.cuda.manual_seed_all(1)
#torch.backends.cudnn.benchmark = False
#torch.backends.cudnn.deterministic = True

start_time = time.time()

def time_limit(time_out):
    global start_time
    end_time = time.time()
    #print(end_time-start_time)
    if (end_time - start_time > time_out):
        return True
    else:
        return False

class Agent(object):

    def __init__(self, **kwargs):
        self.lr = 3e-4
        print("updated three")
        self.batch_size = 64
        self.gamma = 0.999
        self.epsilon = 0.85
        print(self.epsilon)
        self.Vmin = -25
        self.Vmax = 25
        self.atoms = 80
        self.actions = 3
        self.policy = DQN(9, self.actions, self.atoms)
        self.target = DQN(9, self.actions, self.atoms)
        self.reward = []
        self.updata_time = 0
        self.memory = rpm(25000)
        self.target.load_state_dict(self.policy.state_dict())
        self.optimizer_policy = optim.Adam(self.policy.parameters(), lr = self.lr)
        self.support = torch.linspace(self.Vmin, self.Vmax, self.atoms)

    def get_action(self, state, test=True):
        if test:
            epsilon = 0.1
        else:
            epsilon = self.epsilon
        if random.random() < epsilon:
            return random.randint(0, self.actions-1)
        with torch.no_grad():
            self.eval()
            state = state.to(dtype=torch.float)#, device=device)
            state = state.reshape([1] + list(state.shape))
            tmp   = (self.policy(state) * self.support).sum(2).max(1)[1]
        return (int (tmp))

    def updata_target(self):
        #print("update target")
        self.target.load_state_dict(self.policy.state_dict())
        #print("update target")

    def save_model(self):
        torch.save(self.policy.state_dict(),'DQN.pkl')
        
    def save_model_test(self):
        torch.save(self.policy.state_dict(),'DQNTest.pkl')

    def load_model(self):
        self.policy.load_state_dict(torch.load('DQNTest.pkl'))

    # def updata_device(self, device=torch.device("cuda")):
    #     self.policy = self.policy.to(device=device)
    #     self.target = self.target.to(device=device)

    def train(self):
        self.policy.train()
        self.target.train()

    def eval(self):
        self.policy.eval()
        self.target.eval()

    def step(self, step, env, m_obs, test=False):#, m_inv, test=False):
        TD_step = 2
        _reward = 0
        print(step)
        frame = 0
        done = False
        m_reward = [0 for _ in range(10)]
        m_action = [torch.tensor([0]) for _ in range(10)]
        state = [state_to(m_obs[-3:]) for _ in range(10)]
        while frame < step:
            action_num = self.get_action(state[-1], test)
            obs, rew, done, info, t = envstep(env, action_num)
            #print(rew)
            _reward += rew
            frame += t

            for i in range(9):
                m_obs[i] = m_obs[i+1]
                #m_inv[i] = m_inv[i+1]
                state[i] = state[i+1]
                m_reward[i] = m_reward[i+1]
                m_action[i] = m_action[i+1]


            if not done :
                m_obs[-1] = np2torch(obs)
                #m_inv[-1] = obs['inventory']
                state[-1] = state_to(m_obs[-3:])
                m_reward[-1] = rew
                m_action[-1] = torch.tensor([action_num])

            if not test:
                reward, gam = 0.0, 1.0
                for i in range(TD_step):
                    reward += gam * m_reward[i-TD_step]
                    gam *= self.gamma
                reward = torch.tensor([reward])
                _done = torch.tensor([0.0])
                gam = torch.tensor([gam])
                important = reward > 0.0
                if frame >= TD_step and reward < 2.1:
                    self.memory.push([state[-TD_step-1], m_action[-TD_step], state[-1], reward, _done, gam], important)

            if done and not test:
                for i in range(TD_step-1):
                    reward, gam = 0.0, 1.0
                    for k in range(TD_step-i-1):
                        reward += gam * m_reward[i-TD_step+1+k]
                        gam *= self.gamma
                    reward = torch.tensor([reward])
                    _done = torch.tensor([1.0])
                    gam = torch.tensor([gam])
                    important = reward > 0
                    self.memory.push([state[-TD_step+i], m_action[-TD_step+i+1], state[-1], reward, _done, gam], important)
                env.reset()


        if not test:
            return _reward, frame

        return _reward, done


def np2torch(s):
    state = torch.from_numpy(s.copy())
    return state.to(dtype=torch.float)

def state_to(pov):
    state = torch.cat(pov, 2)
    state = state.permute(2, 0, 1)
    return state#.to(torch.device('cpu'))

def envstep(env, action_num):
    reward = 0
    action = action_num
    for i in range(4):
        obs, rew, done, info = env.step(action)
        #env.render('human')
        reward += rew
        if done: #or action_num == 3 or action_num == 4:
            return obs, reward, done, info, i+1
    return obs, reward, done, info, 4

def write_episode(loss, _rew, Q):
    with open('BSLResults.csv', 'a', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow([_rew, loss, Q])

def write_start():
    with open('BSLResults.csv', 'a', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(["START"])


def train(episode):

    #print('Start train')
    env = gym.make('MiniWorld-OneRoom-v0')
    
    agent1 = Agent()  # treechop
    #agent1.updata_device()
    #agent1.load_model()
    write_start()
    max_reward = 0
    env.max_episode_steps = 1000
    sum_episodes = episode
    all_frame = 0
    rew_all = []
    for i_episode in range(sum_episodes):
        env.seed(i_episode)
        obs = env.reset()
        #env.render('human')
        done = False

        m_obs = [np2torch(obs) for _ in range(10)]
        #m_inv = [obs['inventory'] for _ in range(10)]
        _reward = 0
        frame = 0
        _reward, frame = agent1.step(1000, env, m_obs)#, m_inv)
        Q = 0
        loss = 0
        write_episode(loss, _reward, Q)
        
        # writer.add_scalar('validate/Q-value', Q, i_episode)
        # writer.add_scalar('validate/Q-loss', loss, i_episode)
        # writer.add_scalar('validate/total_reward', _reward, i_episode)
        # writer.add_scalar('validate/step', all_frame, i_episode)
        if i_episode > sum_episodes :
            break

    # reset rpm
    agent1.memory.clear()
    agent1.save_model()
    env.close()

if __name__ == '__main__':
    train(50)
