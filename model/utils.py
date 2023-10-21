from ctypes.wintypes import PINT
import re
import numpy as np
import tensorflow as tf
from numba import jit
from collections import defaultdict

def observation_and_action_constraint_splitter(obs):
    return obs['observations'], obs['action_mask']


class Buffer:
    def __init__(self, batch_size):
        self.batch_size = batch_size
    
        self.states = []
        self.feats = []
        self.masks = []
        self.log_probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []
        

    def get(self):
        buffer_size = len(self.actions)
        batch_start = np.arange(0, buffer_size, self.batch_size)
        indices = np.arange(buffer_size, dtype=np.int64)
        # np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        # states_dict = defaultdict(list)
        # for state in self.states:
        #     for key, value in state.items():
        #         states_dict[key].append(value)
                

        return np.array(self.states),\
            np.array(self.masks),\
            np.array(self.feats),\
            np.array(self.actions),\
            np.array(self.log_probs),\
            np.array(self.vals),\
            np.array(self.rewards),\
            np.array(self.dones),\
            batches
    
    
    def store(self, state, mask, feat, action, log_probs, vals, reward, done):
        self.states.append(state)
        self.feats.append(feat)
        self.masks.append(mask)
        self.actions.append(action)
        self.log_probs.append(log_probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)
    
    
    def clear(self):
        self.states = []
        self.feats = []
        self.masks = []
        self.log_probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []
    

class InstanceLoader:
    
    def __init__(self, name=None):
        super
        assert isinstance(name, str)
        self.name = name
        self.data_parser()
        self.static_data()

    def reload(self, name=None):
        if name:
           self.name = name
        self.data_parser()
        self.static_data()
         

    @property
    def num_jobs(self):
        return self._n
    
    @property
    def cycle_time(self):
        return self._C
    
    @property
    def processing_time(self):
        return self._p
    
    @property
    def precedence(self):
        return self._idle
    
    def data_parser(self):
        with open(self.name, 'r') as f:
            no_break = "".join(line for line in f if not line.isspace())
            split_text = re.split("<.*>+\n", no_break)

            self._n = int(split_text[1].split('\n')[0])
            self._C = int(split_text[2].split('\n')[0])

            task = split_text[4].split('\n')[0:-1]
            self._p = np.array([int(e.split(' ')[1]) for e in task], dtype=int)

            pr = split_text[5].split('\n')[0:-1]
            self.edges = np.array([[int(x)-1 for x in e.split(',')] for e in pr])
            
            self.G = np.zeros((self._n, self._n), dtype=int)
            for i,j in zip(self.edges[:,0], self.edges[:,1]):
                self.G[i,j] = 1
    
    
    
    def static_data(self):
        def BFS(start, graph):
            tasks = []
            visited = np.zeros(self._n, dtype=bool)
            q = [start]
            visited[q[0]] = True
            while q:
                i = q.pop(0)
                tasks.append(i)
                for j in graph[i]:
                    if not visited[j]:
                        q.append(j)
                        visited[j] = True
            return tasks[1:]
        
        # GUB = np.min([self._n, 
        #               np.floor(self._p.sum() / (self._C + 1 - self._p.max())) + 1, 
        #               np.floor(2 * self._p.sum() / (self._C + 1)) + 1])
        
        self.IP = [np.array([], dtype=int) for _ in range(self._n)]
        self.IF = [np.array([], dtype=int) for _ in range(self._n)]
        P = [[] for _ in range(self._n)]
        self.F = [[] for _ in range(self._n)]
        
        for i in range(len(self.edges)):
            pre = np.array(self.edges[i][0])
            fol = np.array(self.edges[i][1])
            self.IP[fol] = np.append(self.IP[fol], pre).astype(int)
            self.IF[pre] = np.append(self.IF[pre], fol).astype(int)
            
        self.F_bar = np.zeros(self._n, dtype=int)
        self.IF_bar = np.array([len(x) for x in self.IF])
        self.IP_bar = np.array([len(x) for x in self.IP])
        self.P_bar = np.zeros(self._n, dtype=int)
        self._idle = np.zeros(self._n, dtype=int)
        self.pw = np.array(self._p, dtype=int)
        # ic = np.where(np.logical_and(self.IF_bar<1, self.IP_bar<1))[0]

            
        # for c in ic:
        #     self.G[c,np.where(self.IF_bar<1)[0]]=1
        for i in range(self._n):
            pres = BFS(i, self.IP)
            P[i] = pres
            self._idle[i] += self._p[pres].sum()
            self.P_bar[i] += len(pres)
            fols = BFS(i, self.IF)
            self.F[i] = fols
            self.F_bar[i] += len(fols)
            self.pw[i] += self._p[fols].sum()
                

        
        GUB = np.min([self._n, 
                      np.floor(self._p.sum() / (self._C + 1 - self._p.max())) + 1, 
                      np.floor(2 * self._p.sum() / (self._C + 1)) + 1])
        
        self.L = (GUB + 1 - np.ceil(np.array(self.pw / self._C)))
        
        self.tdL = self._p / self.L
        
class ALBSolver:
    
    def __init__(self, instance):
        self.instance = InstanceLoader(name=instance)
        