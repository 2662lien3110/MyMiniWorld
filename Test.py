import gym
import torch
import math
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from csv import writer, reader

import torch.optim as optim
import torch.distributions.categorical as categorical
from gym_miniworld.wrappers import *
from A2CNN2 import *
from rpm import rpm


class Agent(object):
    def __init__(self, **kwargs):
        self.lr_act = 5e-3
        self.lr_crit = 0
        self.batch_size = 64
        self.atoms = 80
        self.actions = 3
        self.channels = 9
        self.gamma = 0.99
        self.tryNum = 14
        self.lambdaEntrop = 0.5
        self.testR = 0
        self.lambdaCrit = 0.5
        self.weightDecay = False
        self.actor = CNNBase(self.channels, self.actions, self.atoms)
        self.optimizer_actor = optim.RMSprop(self.actor.parameters(), lr=self.lr_act, alpha= 0.99, eps=1e-5)#, weight_decay=self.weightDecay)
        self.memory = rpm(250000)
        self.maxReward = 0
        self.minFrame = 0
        self.AveRew = 0
        self.bestEps = 0
        self.ModUpdate = 0
        self.maxSteps = 360

    def get_action(self, state):
        with torch.no_grad():
            self.eval()
            state = state.to(dtype=torch.float)
            state = state.reshape([1] + list(state.shape))
            a, val = self.actor.act(state)
            #print(a)
            #tmp   = a.max(1)
            #tmp1 = tmp[1]
            #print("Test", tmp)
            #randi = random.randrange(0,10)
            #if (randi>=9):
                #print("val", val)
                #print("act", a)
            #print(dist)
            #print("Max", tmp1)
            #pred = tmp[0].type(torch.FloatTensor)
            #mask = 1 / 10000
            #pred = pred / 10
            #pred = pred + mask
            #print(pred.type())
            #tmp[0].to(dtype=torch.float)
            #log = torch.log(pred)
            log = 0.99
        return int (a), val, log


    def train(self):
        self.actor.train()

    def eval(self):
        self.actor.eval()

    def save_model(self, path):
        torch.save(self.actor.state_dict(), path + 'DQNTest.pkl')
        self.memory.save_ipt(path)

    def load_model(self):
        self.actor.load_state_dict(torch.load('DQNTest.pkl'))
        # self.memory.load_ipt(path)

    def train_data(self, reps, frame):
        aloss = []
        closs = []
        for i in range(reps):
            #print("update: ", i)
            a_loss, c_loss = self.learn(frame)
            aloss.append(a_loss)
            closs.append(c_loss)
        return np.mean(aloss), np.mean(closs)

    def test(self, max_episode_steps, env, m_obs, i_episode):
        rew_all = 0
        for i in range(i_episode):
            print("TESTING")
            frame = 0
            rew = 0
            m_reward = [0 for _ in range(10)]
            m_action = [torch.FloatTensor([0]) for _ in range(10)]
            m_value = [torch.FloatTensor([0]) for _ in range(10)]
            m_log = [torch.FloatTensor([0]) for _ in range(10)]
            state = [state_to(m_obs[-3:]) for _ in range(10)]  # the last 3 items
            # print("state: ", type(state), len(state))
            _reward = []
            done = False
            frame = 0
            batch_frame = 0

            while frame<max_episode_steps:
                action_num, value, log = self.get_action(state[-1])
                s_1, r, done, info, t = envstep(env, action_num)
                frame += t
                rew += r
                for i in range(9):
                    m_obs[i] = m_obs[i + 1]
                    state[i] = state[i + 1]
                    m_reward[i] = m_reward[i + 1]
                    m_action[i] = m_action[i + 1]
                    m_value[i] = m_value[i + 1]
                    m_log[i] = m_log[i + 1]

                m_obs[-1] = np2torch(s_1)
                state[-1] = state_to(m_obs[-3:])
                m_reward[-1] = torch.FloatTensor([r])
                m_action[-1] = torch.FloatTensor([action_num])
                m_value[-1] = torch.FloatTensor([value])
                m_log[-1] = torch.FloatTensor([log])
                #reward = torch.tensor(sum(rew))

                if done:
                    env.reset()
                    rew_all += rew


                if frame == (max_episode_steps-1):
                    #print_testresults(rew, frame)
                    env.reset()
                    print(rew)

            write_episode(rew, frame, 0)

            if rew_all>self.testR:
                self.testR = rew_all
                #self.save_model('train' + str(self.tryNum) + '/' + 'test/')


def np2torch(s):
    state = torch.from_numpy(s.copy())
    state.to(dtype=torch.float)
    #state = state.reshape([1] + list(state.shape))
    return state#, device=device)

def state_to(pov):
    state = torch.cat(pov, 2) #concatenates given sequence of tensors in given dimension
    state = state.permute(2, 0, 1) #permute dimensions of tensor
    return state#.to(torch.device('cpu'))

def do_print(loss, aclos, critlos, entropy):
    print('loss %2.7f acloss %2.7f critloss %2.7f entropy %2.7f' % \
          (loss, aclos, critlos, entropy))


def envstep(env, action_num):
    reward = 0
    #print(action)
    obs, rew, done, info = env.step(action_num)
    env.render('human', view='top')
    #rew = -0.01
    if rew>0:
        print("REWARD")
        #rew = torch.LongTensor(rew)
    return obs, rew, done, info, 1

def plotGraph(episodes, codeName, rew_all, Plotrew_all, list_lr, list_ac_loss, i_episode, entropy):
    plt.figure()
    plt.plot(episodes, rew_all, 'r--', episodes, list_lr, 'b.')
    plt.savefig('/home/annalien/Pictures/Skripsie graphs/AC/Try15/' + str(codeName) + 'A2C2Episode' + str(i_episode) + '.png')
    plt.close()
    plt.figure()
    plt.plot(episodes, Plotrew_all, 'r--', episodes, list_ac_loss, 'b--')
    plt.savefig('/home/annalien/Pictures/Skripsie graphs/AC/Try15/' + str(codeName) + 'A2C2Loss' + str(i_episode) + '.png')
    plt.close()

def read():
    with open ('BSLResults.csv', 'r') as f:
        Reader1 = reader(f, delimiter=',')
        Rows = list(Reader1)
        Tot_rows = len(Rows)
    return Tot_rows

def write(Agent1, cdName, AveRew, sum_episodes, tot_frame):
    with open('BSLResults.csv', 'a', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow([str(cdName), "AC2", str(Agent1.lr_act), str(Agent1.lr_crit),
                             str(Agent1.gamma),str(Agent1.lambdaCrit),str(Agent1.lambdaEntrop),
                             str(Agent1.weightDecay), str(Agent1.maxReward),
                             str(Agent1.minFrame), str(Agent1.bestEps),
                             str(AveRew), str(tot_frame), str(Agent1.maxSteps), str(sum_episodes),
                             str(Agent1.ModUpdate), str(Agent1.batch_size), str(Agent1.channels)])
def print_testresults(rew, frame):
    with open('BSLResults.csv', 'a', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(["TEST", rew, frame])

def write_episode(_rew, frame, entropy):
    with open('BSLResults.csv', 'a', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow([_rew, frame, entropy])



def train(episode, env):

    Agent1 = Agent()
    Agent1.load_model() #gym-miniworld/scripts/train14/test (another copy)
    sum_episodes = episode
    rew_all = []
    Plotrew_all = []
    codeName = read()
    list_lr = []
    list_ac_loss = []
    list_crit_loss = []
    tot_rew = 0
    tot_frame = 0
    obs = env.reset()
    # env.render('human')
    m_obs = [np2torch(obs) for _ in range(10)]
    Agent1.test(env.max_episode_steps, env, m_obs, sum_episodes)

    #Agent1.save_model('train' + str(Agent1.tryNum) + '/test/')
    write(Agent1, codeName, sum_episodes, tot_frame)



if __name__ == '__main__':
    print("Make environment")
    env = gym.make('MiniWorld-OneRoom-v0')
    #env = RGBImgPartialObsWrapper(env)
    #env = ImgObsWrapper(env)
    env.render('human', view='top')
    env.framerate = 5
    done = False
    obs = env.reset()
    env.seed(1000)
    #print(obs.shape())
    env.max_episode_steps =1000
    train(50, env)
