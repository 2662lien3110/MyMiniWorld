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
        self.batch_size = 64
        self.gamma = 0.999
        self.epsilon = 0.1
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

    def get_action(self, state, test=False):
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
        print("update target")

    def updata_epsilon(self, rew):
        self.reward.append(rew)
        print("update epsilon")
        if len(self.reward) > 25:#100:
            self.epsilon = 0.1
        elif len(self.reward) > 15: #60:
            self.epsilon = 0.2
        elif np.sum(self.reward) > 0:
            self.epsilon = max(0.4, self.epsilon * 0.8)

    def projection_distribution(self, next_state, reward, done, gam):
        with torch.no_grad():
            batch_size = next_state.size(0)
            delta_z = float(self.Vmax - self.Vmin) / (self.atoms - 1)
            support = torch.linspace(self.Vmin, self.Vmax, self.atoms).to(device)

            next_dist   = self.target(next_state).to(device) * support
            next_action = next_dist.sum(2).max(1)[1]
            next_action = next_action.unsqueeze(1).unsqueeze(1).expand(next_dist.size(0), 1, next_dist.size(2)).to(device)
            print("projection distribution")
            #DoubleDQN
            next_dist   = (self.policy(next_state).to(device)
            next_dist = next_dist.gather(1, next_action).to(device).squeeze(1)).to(device)

            reward  = reward.expand_as(next_dist).to(device)
            done    = done.expand_as(next_dist).to(device)
            gam     = gam.expand_as(next_dist).to(device)
            support = support.unsqueeze(0).expand_as(next_dist).to(device)

            Tz = reward + (1 - done) * gam * support
            Tz = Tz.clamp(self.Vmin, self.Vmax)
            b  = (Tz - self.Vmin) / delta_z
            l  = b.floor().long()
            u  = b.ceil().long()
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.atoms - 1)) * (l == u)] += 1

            offset = torch.linspace(0, (batch_size - 1) * self.atoms, batch_size).long()\
                    .unsqueeze(1).expand(batch_size, self.atoms)
            offset = offset.to(device)

            proj_dist = torch.zeros(next_dist.size()).to(device)

            proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
            proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))

            #print(next_dist.sum(1))
            #print(proj_dist.sum(1))
            #input()
            return proj_dist


    def learn(self):

        self.train()
        _loss = 0
        _Q_pred = 0

        if len(self.memory) < self.batch_size:
            return _loss, _Q_pred
        print("Learn")
        state_batch, action_batch, next_state_batch, reward_batch, done_batch, gam_batch = self.memory.sample(self.batch_size)

        action_batch = action_batch.unsqueeze(1).expand(action_batch.size(0), 1, self.atoms)
        action_batch = action_batch.to(device)
        dist_pred    = self.policy(state_batch).to(device)
        dist_pred = dist_pred.gather(1, action_batch).to(device).squeeze(1)
        dist_pred = dist_pred.to(device)
        dist_true    = self.projection_distribution(next_state_batch, reward_batch, done_batch, gam_batch)

        dist_pred.data.clamp_(0.001, 0.999)
        loss = - (dist_true * dist_pred.log()).sum(1).mean()
        loss=loss.to(device)
        print("loss", loss)
        self.optimizer_policy.zero_grad()
        loss.backward()
        self.optimizer_policy.step()

        with torch.no_grad():
            _loss = float(loss)
            _Q_pred = float((dist_pred * torch.linspace(self.Vmin, self.Vmax, self.atoms).to(device)).sum(1).mean())
        return _loss, _Q_pred

    def train_data(self, time):
        loss = []
        Q = []
        print("train data")
        for i in range(time):
            _loss, _Q = self.learn()
            loss.append(_loss)
            Q.append(_Q)
            self.updata_time += 1
            if self.updata_time % 1000 == 0:
                self.updata_target()
        return np.mean(loss), np.mean(Q)

    def save_model(self):
        torch.save(self.policy.state_dict(),'DQN.pkl')

    def load_model(self, path):
        self.policy.load_state_dict(torch.load('DQN.pkl'))

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
                important = reward > 0.001
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
                    important = frame < 17900
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
        if done or action_num == 3 or action_num == 4:
            return obs, reward, done, info, i+1
    return obs, reward, done, info, 4

def write_episode(loss, _rew):
    with open('BSL-EpisodeResults.csv', 'a', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow([_rew, loss])

def write_start():
    with open('BSL-EpisodeResults.csv', 'a', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(["START"])


def train(episode):

    print('Start train')
    env = gym.make('MiniWorld-OneRoom-v0')

    agent1 = Agent()  # treechop
    #agent1.updata_device()
    write_start()
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

        all_frame += frame
        if all_frame > 200:
            print("time")
            time = frame // 200
            print(time)
        else :
            time = 0
        loss, Q = agent1.train_data(time)
        write_episode(loss, _reward)
        agent1.updata_epsilon(_reward)
        rew_all.append(_reward)

        print('epi %d all frame %d frame %5d Q %2.5f loss %2.5f reward %3d (%3.3f)'%\
                (i_episode, all_frame, frame, Q, loss, _reward, np.mean(rew_all[-50:])))
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
    train(700)
