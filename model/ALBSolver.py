import os
import re
import numpy as np
from natsort import natsorted
from numpy.core.numeric import indices
import pandas as pd
from tqdm import tqdm
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import bellman_ford
import random


class ALBSolver:
    def __init__(self, instance, step=1):
        self.instance = instance
        self.step = step
        self.dec = []
        
    def data_parser(self):
        with open(self.instance, 'r') as f:
            no_break = "".join(line for line in f if not line.isspace())
            split_text = re.split("<.*>+\n", no_break)

            self.n = int(split_text[1].split('\n')[0])
            self.C = int(split_text[2].split('\n')[0])
            # strength = split_text[3].split('\n')[0].replace(',', ".")
            # strength = float(strength)
            tsk = split_text[4].split('\n')[0:-1]
            self.t = np.array([int(e.split(' ')[1]) for e in tsk], dtype=int)

            pr = split_text[5].split('\n')[0:-1]
            self.edges = np.array([[int(x)-1 for x in e.split(',')] for e in pr])
            
            self.G = np.zeros((self.n, self.n), dtype=int)
            for i, j in self.edges:
                self.G[i,j] = 1


    def static_data(self):
        def BFS(start, graph):
            tasks = []
            visited = np.zeros(self.n, dtype=bool)
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
        
        GUB = np.min([self.n, 
                      np.floor(self.t.sum() / (self.C + 1 - self.t.max())) + 1, 
                      np.floor(2 * self.t.sum() / (self.C + 1)) + 1])
        
        self.IP = [np.array([], dtype=int) for _ in range(self.n)]
        self.IF = [np.array([], dtype=int) for _ in range(self.n)]
        self.P = [[] for _ in range(self.n)]
        self.F = [[] for _ in range(self.n)]
        
        for i in range(len(self.edges)):
            pre = np.array(self.edges[i][0])
            fol = np.array(self.edges[i][1])
            self.IP[fol] = np.append(self.IP[fol], pre).astype(int)
            self.IF[pre] = np.append(self.IF[pre], fol).astype(int)
            
        self.F_bar = np.zeros(self.n, dtype=int)
        # P_bar = np.zeros(self.n, dtype=int)
        self.idle = np.zeros(self.n, dtype=int)
        self.pw = np.array(self.t, dtype=int)
        
        for i in range(self.n):
            pres = BFS(i, self.IP)
            self.P[i] = pres
            self.idle[i] += self.t[pres].sum()
            # P_bar[i] += len(pres)
            fols = BFS(i, self.IF)
            self.F[i] = fols
            self.F_bar[i] += len(fols)
            self.pw[i] += self.t[fols].sum()
            
        E = np.ceil(np.array(self.idle + self.t) / self.C)
        self.L = (GUB + 1 - np.ceil(np.array(self.pw / self.C)))
        self.S = self.L - E + 1
    
    
    
    def feature_extract(self):
        def DFSUtil(start, visited, stack, graph, rev_graph):
            if not visited[start]  and len(rev_graph[start]) <= 1:
                stack.append(start)
                visited[start] = True
                for nxt in graph[start]:
                    if not visited[nxt]  and len(graph[nxt]) <= 1:
                            DFSUtil(nxt, visited, stack, graph, rev_graph)


        def DFS(start, visited, graph, rev_graph):

            stack = []
            DFSUtil(start, visited, stack, graph, rev_graph)

            return stack
        
        
        rem = np.where(self.t>0)[0]
        size = len(rem)
        
        s1 = np.sum([len(v)==0 for v in self.IP])
        f = np.sum([len(v)==0 for v in self.IF])
            
        IF_bar = np.array([len(x) for x in self.IF])
        IP_bar = np.array([len(x) for x in self.IP])

        ED = np.sum(IF_bar) + s1 if s1 > 1 else np.sum(IF_bar)
        EC = np.sum(IF_bar) + f if f > 1 else np.sum(IF_bar)

        OS = 2*self.F_bar.sum() / max(1, (size * (size - 1)))
        # density = len(self.edges) / (self.n * (self.n - 1))
        AIP = np.sum(IP_bar) / size
        div = 1 - (np.sum(IF_bar) + s1 - size)/ ED
        conv = 1 - (np.sum(IF_bar) + f - size)/ EC

        deg = np.array([IP_bar[i]+IF_bar[i] for i in range(self.n)])
        
        dist = np.zeros((self.n + 1, self.n + 1),dtype=int)
        dist[0,1:] = -1
        dist[1:,1:] = -self.G
        dist = csr_matrix(dist)
        LP = bellman_ford(csgraph=dist, directed=True, indices=0)
        LP = -LP[1:]
        tpK = np.mean(np.unique(LP, return_counts=True)[1])
        
        search_bn = np.where((IF_bar >= 2) & (IP_bar >= 2))[0]
        bottlenecks = []
        for bn in search_bn:
            if (np.sum(IF_bar[self.IP[bn]] == 1) >= 2 and 
                np.sum(IP_bar[self.IF[bn]] == 1) >= 2):
                bottlenecks.append(bn)
        avg_BN_deg = np.mean(deg[bottlenecks]) if deg[bottlenecks].size > 0 else 0
        deg_thresh = 4 if size < 50 else 8
        
        search_ch = np.where((IF_bar<=1)&(IP_bar<=1))[0]
        res = []
        vis_f = np.zeros(self.n, dtype=bool)
        vis_p = np.zeros(self.n, dtype=bool)

        for i in search_ch:
            prec = DFS(i, vis_p, self.IP, self.IF)
            folw = DFS(i, vis_f, self.IF, self.IP)

            if IF_bar[self.IP[i]].sum() <= 1 and IP_bar[self.IF[i]].sum() <= 1:
                res.append(list(set(prec) | set(folw)))
            else:
                res.append(prec)
                res.append(folw)
                
        chains = [x for x in res if len(x) > 1]
        CH_len = [len(x) for x in chains]
        avg_CH_len = np.mean(CH_len) if len(CH_len)>0 else 0
        
        features = [ # self.n, # instance size
                    int(avg_BN_deg >= deg_thresh), # structure BN
                    int(sum(CH_len)/size >= 0.40), # structure CH
                    int((sum(CH_len)/size < 0.40) & (avg_BN_deg < deg_thresh)), # MIXED
                    OS, # order strength
                    AIP, # average number of immediate predecessors
                    div, # degeree of divergence
                    conv, # degeree of convergence
                    s1/size, # share of tasks without predecessors
                    int(LP[-1]), # number of stages
                    tpK, # average number of tasks per stage
                    len(bottlenecks)/size, # share of bottleneck tasks
                    avg_BN_deg, # average degree of bottleneck tasks
                    sum(CH_len)/size, # share of chain tasks
                    avg_CH_len, # average chain length
                    self.t.sum()/self.C, # tSum/C
                    self.t.min()/self.C, # tMin/C
                    self.t.max()/self.C, # tMax/C
                    self.t.std()/self.C, # tStd/C
                    ]
        

            
        return np.nan_to_num(features)
    
    def decision(self, features):
        from joblib import load
        model = load('new_model.joblib')
        
        return model.predict([features])[0].astype(bool)
    
    def single_rule(self):
        self.static_data()
        available_tasks = np.where(self.idle==0)[0]
        soln = []
        load_sum = []

        while self.t.any():
            station = []
            station_load = 0
            i = available_tasks[np.nanargmax(self.pw[available_tasks])]
            station.append(i)
            station_load += self.t[i]
            self.idle[self.F[i]] -= self.t[i]
            
            self.t[i] = 0
            self.idle[i] = -1
            self.G[i,:] = 0
            available_tasks = np.where(self.idle==0)[0]
            
            while station_load <= self.C:
                to_add = available_tasks[np.argsort(self.pw[available_tasks])][::-1]        
                loadQ = self.t[to_add]+station_load
                q = to_add[loadQ<=self.C]
                if q.size > 0:
                    j = q[0]
                    station.append(j)
                    station_load += self.t[j]
                    self.idle[self.F[j]] -= self.t[j]
                    self.t[j] = 0
                    self.idle[j] = -1
                    self.G[i,:] = 0
                    available_tasks = np.where(self.idle==0)[0]
                else:
                    break
            if station != []:
                soln.append(station)
                load_sum.append(station_load)
        return soln
    
    def rules(self, tasks):
        return np.array([self.t[tasks],
                        self.t[tasks]/self.L[tasks],
                        self.t[tasks]/self.S[tasks],
                        self.L[tasks],
                        self.pw[tasks],
                        self.F_bar[tasks]
                        ])

    
    def ml_rule_comp(self, output='solution'):
        cnt = 0
        self.static_data()
        available_tasks = np.where(self.idle==0)[0]
        soln = []
        load_sum = []
        iters = 9999
        subs = []
        
        while self.t.any():
            if iters >= self.step:
                cnt += 1
                iters = 0
                features = self.feature_extract()
                decisions = self.decision(features=features)
                sub_name = os.path.split(self.instance)[1][:-4] + '_s' + str(cnt)
                random.seed(42)
                idx = random.choice(range(np.sum(decisions)))
                # self.dec.append(idx)
                
                subs.append(np.concatenate(([sub_name], features, decisions.astype(int))))
            
            station = []
            station_load = 0
            i = available_tasks[np.argmax(self.rules(available_tasks)[decisions], axis=1)[idx]]
            station.append(i)
            station_load += self.t[i]
            self.idle[self.F[i]] -= self.t[i]
            self.F_bar[self.P[i]] -= 1
            self.t[i] = 0
            self.idle[i] = -1
            self.G[i,:] = 0
            for folw in self.IF[i]:
                self.IP[folw] = self.IP[folw][self.IP[folw]!=i]

            for pred in self.IP[i]:
                self.IF[folw] = self.IF[pred][self.IF[pred]!=i]
                        
            
            available_tasks = np.where(self.idle==0)[0]
            
            while station_load <= self.C:
                
                to_add = available_tasks[np.argsort(self.rules(available_tasks)[decisions], axis=1)[idx][::-1]]     
                loadQ = self.t[to_add]+station_load
                q = to_add[loadQ<=self.C]

                
                if q.size > 0:
                    j = q[0]
                    # print(j, q, self.idle[q], to_add, available_tasks)
                    station.append(j)
                    station_load += self.t[j]
                    self.idle[self.F[j]] -= self.t[j]
                    self.F_bar[self.P[j]] -= 1
                    self.t[j] = 0
                    self.idle[j] = -1
                    self.G[j,:] = 0
                    for folw in self.IF[j]:
                        self.IP[folw] = self.IP[folw][self.IP[folw]!=j]

                    for pred in self.IP[j]:
                        self.IF[folw] = self.IF[pred][self.IF[pred]!=j]
                        
                    available_tasks = np.where(self.idle==0)[0]

                else:
                    break
            if station != []:
                soln.append(station)
                iters += 1
                load_sum.append(station_load)
        
        if output == 'solution':
            return soln
        elif output == 'decision':
            return self.dec
        elif output == 'subinstance':
            return subs
