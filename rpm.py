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
        self.long_buffer = []
        self.index = 0
        self.ipt_buffer = []
        self.ipt_index = 0
        #print("PRM2")
        self.recent = []
        self.rec_index = 0
        self.recent_size = rcsize

    def push_ipt(self, obj):
        if len(self.ipt_buffer) == self.buffer_size:
            self.ipt_buffer[self.ipt_index] = obj
        else:
            self.ipt_buffer.append(obj)
        self.ipt_index = (self.ipt_index + 1)
        #print(self.ipt_index)

    def clear(self):
        self.buffer = []
        self.index = 0
        #self.ipt_buffer = []
        #self.ipt_index = 0
        self.clear_recent()
        #print("Clear")

    def clear_recent(self):
        self.recent = []
        self.rec_index = 0

    def clear_ipt(self):
        self.ipt_buffer = []
        #self.ipt_index = 0

    def clear_long(self):
        self.long_buffer = []

    def save_ipt(self, path):
        torch.save(self.buffer, path + 'file.pt')

    def load_ipt(self, path):
        self.buffer = torch.load(path + 'file.pt')

    def clear_some(self):
        temp = []
        temp1 = []
        # for i in range(1,4):
        #     j = i + (len(self.buffer)-4)
        #     #print(j)
        #     temp.append(self.buffer[j])
        # self.index = 2
        # self.clear()
        # for i in range(len(temp)-1):
        #     self.buffer.append(temp[i])
        if len(self.buffer)>22000:
            print(len(self.buffer))
            for i in range(1,1999):
                j = i * 11
                temp1.append(self.buffer[j])
            #self.clear_long()
            self.clear()
            for i in range(len(temp1) - 1):
                self.buffer.append(temp1[i])
            #print(len(self.buffer))


    def clear_some_ipt(self):
        temp = []
        for i in range(999):
            j = i * 2
            temp.append(self.ipt_buffer[j])
        self.clear_ipt()
        for i in range(len(temp) - 1):
            self.ipt_buffer.append(temp[i])
        self.ipt_index = 101
        #print(len(self.ipt_buffer))



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
            #print(len(self.buffer))
            #self.long_buffer.append(obj)
        self.index = (self.index + 1) % self.buffer_size
        # print("Index: ", self.index)
        # if self.index > 200:
        #    self.clear_some()
        self.push_recent(obj)

        #print("important: ", important)
        if important:
            #self.ipt_buffer.append(obj)
            #self.ipt_index = self.ipt_index + 1
            if len(self.recent) > 20:
                for i in reversed(range(1, 20)):
                    tmp = self.recent[-i]
                    self.push_ipt(tmp)
            else:
                for tmp in self.recent:
                    self.push_ipt(tmp)
            if (len(self.ipt_buffer)) > 2000:
                self.clear_some_ipt()
            #print(len(self.ipt_buffer))
            self.clear_recent()

    def sample(self, batch_size, only_state=False):# device=torch.device("cuda"), only_state=False):
        #print("sample")
        batch = self.buffer[-2:]#long_buffer[-1:]
        if len(self.buffer) < 63:
            batch += random.sample(self.buffer, len(self.buffer))
        else:
            batch += random.sample(self.buffer, 63)
        #batch += random.sample(self.buffer, 200)
        #print(len(batch))
        #batch += self.ipt_buffer

        if len(self.ipt_buffer) < 65:
            batch += random.sample(self.ipt_buffer, len(self.ipt_buffer))
        else:
            batch += random.sample(self.ipt_buffer, 65)

        self.clear_some()
        self.clear_recent()
        #print(len(batch))
        if only_state:
            res = torch.stack(tuple(item[3] for item in batch), dim=0)
            return res#.to(device)
        else:
            item_count = 7
            res = []
            #print("start stack")
            for i in range(7):
                k = torch.stack(tuple(item[i] for item in batch), dim=0)
                #if i == 0 or i == 2:
                #    k = k.to(dtype=torch.float)
                res.append(k)#.to(device))
            return res[0], res[1], res[2], res[3], res[4], res[5], res[6]
    def __len__(self):
        return len(self.buffer)



