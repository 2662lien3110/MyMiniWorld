# from collections import deque
import numpy as np
import random
import torch
import pickle as pickle

class rpm(object):
    # replay memory
    def __init__(self, buffer_size, rcsize=40):
        self.buffer_size = buffer_size
        self.buffer = []
        self.index = 0
        self.ipt_buffer = []
        self.ipt_index = 0

        self.recent = []
        self.rec_index = 0
        self.recent_size = rcsize

    def push_ipt(self, obj):
        if len(self.ipt_buffer) == self.buffer_size:
            self.ipt_buffer[self.ipt_index] = obj
        else:
            self.ipt_buffer.append(obj)
        self.ipt_index = (self.ipt_index + 1) % self.buffer_size
        if self.ipt_index > 6000:
            self.clear_some_ipt()
        #print(self.ipt_index)

    def clear(self):
        self.buffer = []
        self.index = 0
        self.ipt_buffer = []
        self.ipt_index = 0
        self.clear_recent()

    def clear_recent(self):
        self.recent = []
        self.rec_index = 0

    def clear_some(self):
        for i in range(500):
            j = i + 499
            self.buffer[i] = self.buffer[j]
        self.index = 501

    def clear_some_ipt(self):
        for i in range(3000):
            j = i + 1499
            self.ipt_buffer[i] = self.ipt_buffer[j]
        self.ipt_index = 3001



    def push_recent(self, obj):
        if len(self.recent) == self.recent_size:
            self.recent[self.rec_index] = obj
        else:
            self.recent.append(obj)
        self.rec_index = (self.rec_index + 1) % self.recent_size


    def push(self, obj, important):
        if len(self.buffer) == self.buffer_size:
            self.buffer[self.index] = obj
        else:
            self.buffer.append(obj)
        self.index = (self.index + 1) % self.buffer_size
        #print("Index: ", self.index)
        if self.index > 1000:
            self.clear_some()
        self.push_recent(obj)
        #print("important: ", important)
        if important:
            for tmp in self.recent:
                self.push_ipt(tmp)
            self.clear_recent()

    def sample(self, batch_size, only_state=False):# device=torch.device("cuda"), only_state=False):
        #print("sample")
        if len(self.ipt_buffer) < 8:
            batch = random.sample(self.ipt_buffer, len(self.ipt_buffer))
        else:
            batch = random.sample(self.ipt_buffer, 8)
        batch += random.sample(self.buffer, batch_size - len(batch))

        if only_state:
            res = torch.stack(tuple(item[3] for item in batch), dim=0)
            return res#.to(device)
        else:
            item_count = 6
            res = []
            #print("start stack")
            for i in range(6):
                k = torch.stack(tuple(item[i] for item in batch), dim=0)
                #if i == 0 or i == 2:
                #    k = k.to(dtype=torch.float)
                res.append(k)#.to(device))
            return res[0], res[1], res[2], res[3], res[4], res[5]

    def __len__(self):
        return len(self.buffer)

class rpm_meta(object):
    # replay memory
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []
        self.index = 0
        self.ipt_buffer = []
        self.ipt_index = 0

    def push_ipt(self, obj):
        if len(self.ipt_buffer) == self.buffer_size:
            self.ipt_buffer[self.ipt_index] = obj
        else:
            self.ipt_buffer.append(obj)
        self.ipt_index = (self.ipt_index + 1) % self.buffer_size

    def push(self, obj, important=False):
        if len(self.buffer) == self.buffer_size:
            self.buffer[self.index] = obj
        else:
            self.buffer.append(obj)
        self.index = (self.index + 1) % self.buffer_size

        if important:
            self.push_ipt(obj)

    def sample(self, batch_size, only_state=False): #device=torch.device("cuda"), only_state=False):
        if len(self.ipt_buffer) < 10:
            batch = random.sample(self.ipt_buffer, len(self.ipt_buffer))
        else:
            batch = random.sample(self.ipt_buffer, 10)
        batch += random.sample(self.buffer, batch_size-len(batch))

        if only_state:
            res = torch.stack(tuple(item[3] for item in batch), dim=0)
            return res#.to(device)
        else:
            item_count = 6
            res = []
            for i in range(6):
                k = torch.stack(tuple(item[i] for item in batch), dim=0)
                if i == 0 or i == 2:
                    k = k.to(dtype=torch.float)
                res.append(k)#.to(device))
            return res[0], res[1], res[2], res[3], res[4], res[5]

    def __len__(self):
        return len(self.buffer)


