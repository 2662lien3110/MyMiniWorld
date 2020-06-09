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
from A2CNN3 import *
from rpm import rpm

import logging
logging.basicConfig(level=logging.DEBUG)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Agent(object):
    def __init__(self, **kwargs):
        self.lr_act = 0.0038
        print(self.lr_act)
        self.lr_crit = 0
        self.batch_size = 64
        self.atoms = 80
        self.actions = 3
        self.channels = 9
        self.gamma = 0.65
        self.lambdaEntrop = 0.99
        print(self.lambdaEntrop)
        self.lambdaCrit = 0.41667
        self.weightDecay = False
        self.actor = CNNBase(self.channels, self.actions, self.atoms)
        self.optimizer_actor = optim.RMSprop(self.actor.parameters(), lr= self.lr_act, alpha=0.88, eps=1e-5)#, alpha= 0.99, eps=1e-5)#, weight_decay=self.weightDecay)
        self.memory = rpm(250000)
        self.maxReward = 0
        self.minFrame = 0
        self.AveRew = 0
        self.bestEps = 0
        self.ModUpdate = 0
        self.Good = False
        self.maxSteps = 360

    def get_action(self, state):
        with torch.no_grad():
            self.eval()
            state = state.to(dtype=torch.float, device=device)
            state = state.reshape([1] + list(state.shape))
            a, val = self.actor.act(state)

            log = 0.99
        return int (a), val, log

    def learn(self, frame):

        self.train()
        #ac_loss = 0
        _actor_loss = 0
        _critic_loss = 0
        Qval = 0
        Qvals = []
        #state_batch, action_batch, next_state_batch, reward_batch, log_batch, value_batch = self.memory.sample_spec(frame)
        state_batch, action_batch, next_state_batch, reward_batch, log_batch, value_batch, done_batch = self.memory.sample(frame)
        state_batch = state_batch.to(dtype=torch.float, device=device)
        action_batch = action_batch.to(dtype=torch.float, device=device)
        reward_batch = reward_batch.to(dtype=torch.float, device=device)
        next_state_batch = next_state_batch.to(dtype=torch.float, device=device)
        done_batch = done_batch.to(dtype=torch.float, device=device)
        #print(next_state_batch.size()) #[12,3,60,80]
        #print("Log", log_batch.size()) #[12,1]

        #print(action_batch)
        vals, logs, entropy = self.actor.evaluate_actions(state_batch, action_batch)
        vals = vals.to(dtype=torch.float, device=device)
        entropy = entropy.to(dtype=torch.float, device=device)
        new_vals, _, _ = self.actor.evaluate_actions(next_state_batch, action_batch)
        new_vals = new_vals.to(dtype=torch.float, device=device)
        advantages = (reward_batch + (1-done_batch)*self.gamma*new_vals- vals).to(device)
        critic_loss = advantages.pow(2).mean()
        actor_loss = -(advantages.detach() * logs).mean()
        loss = (actor_loss+critic_loss*self.lambdaCrit -self.lambdaEntrop*entropy).to(device)
        #print(loss)
        self.optimizer_actor.zero_grad()

        # Calculate gradients
        loss.backward()
        #ac_loss.backward()
        # Apply gradients
        self.optimizer_actor.step()

        with torch.no_grad():
            #ac_loss = float(ac_loss)
            _actor_loss = float(actor_loss)
            _critic_loss = float(critic_loss)

        return loss, actor_loss, critic_loss, entropy

    def train(self):
        self.actor.train()

    def eval(self):
        self.actor.eval()

    def save_model(self):
        torch.save(self.actor.state_dict(),'A2C.pkl')
        #self.memory.save_ipt(path)

    def load_model(self, path):
        self.actor.load_state_dict(torch.load(path + 'A2C.pkl'))
        # self.memory.load_ipt(path)


    def step(self, steps, env, m_obs, i_episode):
        #print("steps", steps) #250
        m_reward = [0 for _ in range(10)]
        m_action = [torch.FloatTensor([0]) for _ in range(10)]
        m_value = [torch.FloatTensor([0]) for _ in range(10)]
        m_log = [torch.FloatTensor([0]) for _ in range(10)]
        m_done = [torch.FloatTensor([0]) for _ in range(10)]
        state = [state_to(m_obs[-3:]) for _ in range(10)]  # the last 3 items
        #print("state: ", type(state), len(state))
        _reward =[]
        done = False
        frame = 0
        batch_frame = 0
        while frame<steps:
            #print(self.Good)
            # Get actions and convert to numpy array
            #print(state[-1].size()) #[3, 60, 80]
            action_num, value, log = self.get_action(state[-1])
            s_1, r, done, info, t = envstep(env, action_num)
            frame += t
            batch_frame += t
            _reward.append(r)
            #disc_reward = 0
            # print('reward %2.5f' % \
            #       ( _reward))
            for i in range(9):
                m_obs[i] = m_obs[i + 1]
                state[i] = state[i + 1]
                m_reward[i] = m_reward[i + 1]
                m_action[i] = m_action[i + 1]
                m_value[i] = m_value[i+1]
                m_log[i] = m_log[i+1]
                m_done[i] = m_done[i + 1]


            m_obs[-1] = np2torch(s_1)
            state[-1] = state_to(m_obs[-3:])
            m_reward[-1] = torch.FloatTensor([r])
            m_action[-1] = torch.FloatTensor([action_num])
            m_value[-1] = torch.FloatTensor([value])
            m_log[-1] = torch.FloatTensor([log])
            m_done[-1] = torch.FloatTensor([done])
            reward = torch.tensor(sum(_reward))
            # print(reward)
            important = r > 0
            reward = sum(_reward)
            if important and not self.Good:
                for i in reversed(range(2,3)):
                    gam = pow(self.gamma, i)
                    rew = torch.FloatTensor([gam*m_reward[-1]])
                    #print(str(-1-i))
                    self.memory.push([state[-1-i], m_action[-i], state[-i], rew, m_log[-i], m_value[-i], m_done[-i]],
                                 important)
                    #vf important = r>5
            self.memory.push([state[-2], m_action[-1], state[-1], m_reward[-1], m_log[-1], m_value[-1], m_done[-1]],
                                 important)
            #print(batch_frame)
            if batch_frame == self.batch_size:
                #print("Update time")
                loss, aclos, critlos, entropy = self.learn(batch_frame)
                batch_frame = 1
                #do_print(loss, aclos, critlos, entropy)
                # if ((entropy < 0.25) and (reward > 1)) and i_episode>5:
                #     #print("TEST", reward, entropy)
                #     #self.save_model('train/test/')
                #     #self.test(m_obs, m_reward, m_log, m_value, m_action, state)
                #     #self.Good = True
                # else:
                #     self.Good = False

            # If done, batch data
            if done:
                obs = env.reset()
            if frame == steps:

                loss, aclos, critlos, entropy = self.learn(batch_frame)
                do_print(loss, aclos, critlos, entropy)

                #obs = env.reset()

        return reward, frame, loss, entropy

def np2torch(s):
    state = torch.from_numpy(s.copy())
    state.to(dtype=torch.float)
    #state = state.reshape([1] + list(state.shape))
    return state.to(dtype=torch.float, device=device)#, device=device)

def state_to(pov):
    state = torch.cat(pov, 2) #concatenates given sequence of tensors in given dimension
    state = state.permute(2, 0, 1) #permute dimensions of tensor
    return state.to(dtype=torch.float, device=device)#.to(torch.device('cpu'))

def do_print(loss, aclos, critlos, entropy):
    print('loss %2.7f acloss %2.7f critloss %2.7f entropy %2.7f' % \
          (loss, aclos, critlos, entropy))


def envstep(env, action_num):
    reward = 0
    #print(action)
    obs, rew, done, info = env.step(action_num)
    #env.render('human')
    #rew = -0.01
    if rew>0:
        print("REWARD")
        #rew = torch.LongTensor(rew)
    if done:
        done = 1
    else:
        done = 0
    return obs, rew, done, info, 1

def plotGraph(episodes, codeName, rew_all, Plotrew_all, list_lr, list_ac_loss, i_episode, entropy):
    plt.figure()
    plt.plot(episodes, rew_all, 'r--', episodes, list_lr, 'b.')
    plt.savefig('/home/anna/gym-miniworld/scripts/' + str(codeName) + 'A2C2Episode' + str(i_episode) + '.png')
    plt.close()
    plt.figure()
    plt.plot(episodes, Plotrew_all, 'r--', episodes, list_ac_loss, 'b--')
    plt.savefig('/home/anna/gym-miniworld/scripts/' + str(codeName) + 'A2C2Loss' + str(i_episode) + '.png')
    plt.close()

def read():
    with open ('A2CResults.csv', 'r') as f:
        Reader1 = reader(f, delimiter=',')
        Rows = list(Reader1)
        Tot_rows = len(Rows)
    return Tot_rows

def write(Agent1, cdName, AveRew, sum_episodes, tot_frame):
    with open('A2CResults.csv', 'a', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow([str(cdName), "AC2", str(Agent1.lr_act), str(Agent1.lr_crit),
                             str(Agent1.gamma),str(Agent1.lambdaCrit),str(Agent1.lambdaEntrop),
                             str(Agent1.weightDecay), str(Agent1.maxReward),
                             str(Agent1.minFrame), str(Agent1.bestEps),
                             str(AveRew), str(tot_frame), str(Agent1.maxSteps), str(sum_episodes),
                             str(Agent1.ModUpdate), str(Agent1.batch_size), str(Agent1.channels)])

def write_episode(_rew, frame, entropy):
    with open('A2C-EpisodeResults.csv', 'a', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow([_rew, entropy])

def write_start():
    with open('A2C-EpisodeResults.csv', 'a', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(["START"])

def train(episode, env):

    Agent1 = Agent()
    Agent1.actor= Agent1.actor.to(device = device)
    write_start()
    #Agent1.load_model('train' + str(Agent1.tryNum) + '/')
    sum_episodes = episode
    rew_all = []
    Plotrew_all = []
    codeName = read()
    list_lr = []
    list_ac_loss = []
    list_crit_loss = []
    tot_rew = 0
    tot_frame = 0
    for i_episode in range(sum_episodes):
        print("episode: ", i_episode)
        eps = i_episode
        obs = env.reset()
        #env.render('human')
        m_obs = [np2torch(obs) for _ in range(10)]
        _reward = []
        #Agent1.memory.load_ipt('train' + str(Agent1.tryNum) + '/')
        _rew, frame, ac_loss, entropy= Agent1.step(env.max_episode_steps, env, m_obs, i_episode)
        write_episode(_rew, frame, entropy)
        list_ac_loss.append(ac_loss)
        #list_crit_loss.append(crit_loss)
        list_lr.append(Agent1.lr_act)
        rew_all.append(_rew)
        tot_rew += _rew

        if _rew >Agent1.maxReward:
            Agent1.maxReward = _rew
            Agent1.minFrame = frame
            Agent1.bestEps = i_episode
            if entropy < 0.7:
                Agent1.save_model()
                if _rew>= 14:
                    Agent1.lr_act = 1e-7
        tot_frame += frame
        Plottot_rew = _rew - 1
        Plotrew_all.append(Plottot_rew)

        if (i_episode % 100 == 0) and (i_episode != 0) or (i_episode == episode-1):
            episodes = range(0, i_episode+1)
            plotGraph(episodes, codeName, rew_all, Plotrew_all, list_lr, list_ac_loss, i_episode, entropy )
            AveRew = tot_rew / (eps+1)
            #Agent1.save_model('train/')
            #write(Agent1, codeName, AveRew, sum_episodes, tot_frame)

        print('epi %d frame %5d loss %2.5f entropy %2.5f reward %2.5f'%\
               (i_episode, frame, ac_loss, entropy,  _rew))


    AveRew = tot_rew / eps
    #Agent1.save_model('train/')
    write(Agent1, codeName, AveRew, sum_episodes, tot_frame)



if __name__ == '__main__':
    print("Make environment")
    env = gym.make('MiniWorld-OneRoom-v0')
    #env = RGBImgPartialObsWrapper(env)
    #env = ImgObsWrapper(env)
    #env.render('human')
    #env.framerate = 5
    done = False
    obs = env.reset()
    #a = float(sys.argv[1])
    env.seed(1000)
    #print(obs.shape())
    env.max_episode_steps =1000
    train(700, env)
