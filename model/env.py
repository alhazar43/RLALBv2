
import gym
import numpy as np
from gym import spaces
from .utils import InstanceLoader


class ALBEnv(gym.Env):

    def __init__(self, instance, reset_after_done=False, record_soln=False):
        super().__init__()
        self.instance_name = instance
        self.instance = InstanceLoader(self.instance_name)
        self.diag = np.diag_indices(self.instance.num_jobs)
        self.prec = np.array(self.instance.precedence)
        self.job_time = np.array(self.instance.processing_time)
        self.fols = np.array(self.instance.IF_bar)
        self.pres = np.array(self.instance.tdL)
        self.graph = np.array(self.instance.G)
        self.pw = np.array(self.instance.pw)
        
        
        self.observation_space = spaces.Dict(
            precedence=spaces.Box(low=-self.prec, high=self.prec, dtype=np.int32),
            job_time=spaces.Box(low=-self.job_time, high=self.job_time, dtype=np.int32),
            prec_mask=spaces.Box(low=np.zeros(self.instance.num_jobs), high=np.ones(self.instance.num_jobs), dtype=np.bool_),
            graph=spaces.Box(shape=(self.instance.num_jobs, self.instance.num_jobs), low=0, high=1),
            station_load=spaces.Discrete(1),
            num_station=spaces.Discrete(1)
        )
        
        self.station_memory = []
        self.solution_memory = []
        self.action_space = spaces.Discrete(self.instance.num_jobs)
        
        self.state = {
            'precedence': np.array(self.prec, dtype=np.int32),
            'job_time': np.array(self.job_time, dtype=np.int32),
            'prec_mask': np.array(self.prec==0, dtype=np.bool_),
            'successor' : np.array(self.fols, dtype=np.int32),
            'predcessor' : np.array(self.pres, dtype=np.int32),
            'graph': np.array(self.graph, dtype=int),
            'station_load': 0,
            'num_station': 1
        }            
        self.record_soln = record_soln
        
        
        self.episode_done = False
        self.reset_after_done = reset_after_done


    def get_mask(self):
        return np.array(self.state[:-1]==0, dtype=bool)
    

    def step(self, action):
        if self.reset_after_done and self.episode_done:
            self.reset()
        
        reward = 0.0
        job_cntr = 0
        if self.state['precedence'][action]==0:
            self.state['precedence'][action] = -1
            # self.state['job_time'][action] = -1

            self.state['precedence'][self.instance.F[action]] -= self.job_time[action]
            self.state['prec_mask'][action] = False
            self.state['graph'][action,:] = 0
            reward +=   self.job_time[action]*max(1, self.instance.IF_bar[action])
            if self.state['station_load'] + self.job_time[action] <= self.instance.cycle_time:
            # if self.state['precedence'][-1] + self.job_time[action] <= self.instance.cycle_time:
                self.state['station_load'] += self.job_time[action]
                job_cntr += 1
                
                if self.record_soln and not self.episode_done:
                    self.station_memory.append(action)
            else:
                job_cntr = 1
                self.state['num_station'] += 1
                reward -= 100 * (self.instance.cycle_time - self.state['station_load'])
                self.state['station_load'] = self.job_time[action]
                
                if self.record_soln:
                    self.solution_memory.append(self.station_memory)
                    self.station_memory = [action]
        prec_mask = np.array(self.state['precedence']==0, dtype=np.bool_)
        cycle_mask = 1000-self.state['station_load']-self.state['job_time']>=0
        final_mask = np.logical_and(prec_mask, cycle_mask)
        self.state['prec_mask'] = final_mask if final_mask.sum() > 0 else prec_mask
        self.state['graph'][self.diag] = 1
        
        if self.state['prec_mask'].sum() == 0:
            self.solution_memory.append(self.station_memory)
            self.episode_done = True
            reward -= 100 * self.state['num_station']
            return self.state, reward, self.episode_done, {}
            
        return self.state, reward, self.episode_done, {}
    
    def reset(self, new_instance=None):
        if new_instance:
            self.instance = InstanceLoader(new_instance)
        else:
            self.instance = InstanceLoader(self.instance_name)
    

        self.prec = np.array(self.instance.precedence)
        self.job_time = np.array(self.instance.processing_time)
        self.fols = np.array(self.instance.IF_bar)
        self.pres = np.array(self.instance.tdL)
        self.graph = np.array(self.instance.G)
        
        self.state = {
            'precedence': np.array(self.prec, dtype=np.int32),
            'job_time': np.array(self.job_time, dtype=np.int32),
            'prec_mask': np.array(self.prec==0, dtype=np.bool_),
            'successor' : np.array(self.fols, dtype=np.int32),
            'predcessor' : np.array(self.pres, dtype=np.int32),
            'graph': np.array(self.graph, dtype=int),
            'station_load': 0,
            'num_station': 1
        }
        self.episode_done = False
        self.station_memory = []
        self.solution_memory = []
        return self.state
        
        
    
        
   

        