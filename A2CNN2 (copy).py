import torch
import torch.nn as nn
import torch.nn.functional as F
from distributions import Categorical as Cat
import numpy as np
from utils import init, init_normc_

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class CNNBase(nn.Module):
    def __init__(self,  chan, outputs, atoms):
        super(CNNBase, self).__init__()#recurrent, hidden_size, hidden_size)
        self.actions = outputs
        self.chan = chan
        self.atoms = atoms
        self.new_size = 1
        self.dist = Cat(self.atoms, self.actions)
        init_ = lambda m: init(m, nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               nn.init.calculate_gain('relu'))

        # For 80x60 input
        self.main = nn.Sequential(
            init_(nn.Conv2d(chan, 32, kernel_size=5, stride=2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            Print(),
            init_(nn.Conv2d(32, 32, kernel_size=5, stride=2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            init_(nn.Conv2d(32, 32, kernel_size=4, stride=2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            Print(),
            Flatten(),

            init_(nn.Linear(1568, self.atoms)),
            Print(),
            nn.ReLU()
        )

        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0))

        self.critic_linear = init_(nn.Linear(self.atoms, 1))

        self.train()


    def forward(self, inputs):
        #print(inputs.size())

        x = inputs / 255.0
        #print(x.size())

        x = self.main(x)
        print("x")
        print(x.size())


        return x, self.critic_linear(x)

    def act(self, input):
        x, val = self.forward(input)
        #print(x)
        #print("dist")
        dist = self.dist(x)
        #print(x)
        action = dist.sample()
        # print("Act", action)
        return action, val

    def evaluate_actions(self, input, actions):
        action_probs, vals = self.forward(input)
        #print(action_probs.size())
        dist = self.dist(action_probs)
        #print(dist)
        #print(actions.size())
        #dist = self.dist(action_probs)
        #print(actions)
        #logs = torch.gather(dist,1, actions)
        action_logs = dist.log_probs(actions)
        dist_entropy = dist.entropy().mean()

        return vals, action_logs, dist_entropy


class Print(nn.Module):
    def __init__(self):
        super(Print, self).__init__()

    def forward(self, x):
        print('layer input:', x.shape)
        #print(x)
        return x



# class Critic(nn.Module):
#     def __init__(self, chan, outputs, atoms):
#         super(Critic, self).__init__()
#         self.atoms = atoms
#         self.actions = outputs
#         self.in_planes = chan
#         #self.state = 0
#         #self.gamma = gamma
#
#         self.conv1 = nn.Conv2d(self.in_planes, 64, kernel_size=4, stride=2, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.conv2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(512)
#
#
#         # self.fc = nn.Linear(512, self.atoms * self.actions)
#         #
#         #
#         # self.fc0 = nn.Linear(2048,512)
#         self.model = nn.Sequential(nn.Linear(256,128), nn.ReLU(),
#                                 nn.Linear(128,1), nn.ReLU())
#         self.model.apply(self.weights_init_normal)
#         #self.fc2 = nn.Linear(517, 256)
#         #self.fc3 = nn.Linear(256, 1)
#         #self.fc4 = nn.Linear(1,1)
#
#     def weights_init_normal(self, m):
#         #print(m)
#         classname = m.__class__.__name__
#         if classname.find('Linear') != -1:
#             y = m.in_features
#             self.wg = m.weight.data.normal_(0.0,1/np.sqrt(y))
#             m.bias.data.fill_(0)
#             #print(m.weight)
#
#     def forward(self, x):
#         #actions = actor.forward(self.state)
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = layers(x, 64, 128, 3, 2, 1)
#         x = layers(x, 128, 256, 3, 2, 1)
#         x = F.avg_pool2d(x, 4)
#         x = x.view(x.size(0), -1)
#         #print("x", x.size())
#
#         x = self.model(x)
#         wgr = self.wg.view(1,-1)
#         wg0 = nn.Linear(128,1)
#         wgr = wg0(wgr)
#         #x = self.fc4(x)
#         x =  F.linear(x, weight= wgr)
#         return x