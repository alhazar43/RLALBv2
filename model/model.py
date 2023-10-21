from argparse import Action
from collections import deque
from itertools import cycle
from matplotlib.style import available
import numpy as np
import random
from env import ALBEnv
from tqdm import tqdm
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


class DQNAgent():
    def __init__(self, env_id, episodes=1000, gamma=1.0, epsilon=0.9, epsilon_min=0.01, e_decay=0.99, alpha=0.01, a_decay=0.01, max_step=1000):
        self.env = ALBEnv(env_id)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.e_decay = e_decay
        self.alpha = alpha
        self.a_decay = a_decay
        self.episodes = episodes
        self.memory = deque(maxlen=1000)
        self.max_step = max_step
        
        self.model = self.build_model()
        
    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=1000, activation='relu'))
        model.add(Dense(48, activation='relu'))
        model.add(Dense(1000, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.alpha, decay=self.a_decay))
        
        
        return model
    
    def preprocess_state(self, state):
        return np.reshape(state, [1,1000])
    
    def record(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def state_action(self, state, train=True):
        mask = np.array(state[0]==0)[0]
        # mask = np.reshape(mask, -1)
        if train and (np.random.rand() <= self.epsilon and mask.sum()>0):
            p = np.zeros(self.env.n, dtype='float64')
            p[mask] = 1/np.sum(mask)
            p /= p.sum()
            return np.random.choice(self.env.n, p=np.nan_to_num(p))

        return np.argmax(self.model.predict(state))
    
    def replay(self):
        x_batch, y_batch = [], []
        mini_batch = random.sample(
            self.memory, min(len(self.memory), 1000))
        for state, action, reward, next_state, done in mini_batch:

            y_target = self.model.predict(state)

            y_target[0][action] = reward if done else reward + self.gamma * np.max(self.model.predict(next_state)[0])


            x_batch.append(state[0])
            y_batch.append(y_target[0])
        
        self.model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=1)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.e_decay
        
    def run(self,train=True):
        for e in tqdm(range(self.episodes)):
            state = self.preprocess_state(self.env.reset())
            done = False
            cycle = 0
            station_count = 1
            scores = []
            station = []
            for i in range(self.max_step):
                action = self.state_action(state, train=train)
                # print(state[action], action)
                # print(state[0])
                if state[0][action] == 0:
                    cycle += self.env.t[action]
                    self.env.t[action] = 0
                    next_state, reward, done, _ = self.env.step(action)
                    next_state = self.preprocess_state(next_state)
                    if cycle > self.env.C:
                        station_count += 1
                        cycle = 0
                        reward += (cycle-self.env.t[action])/self.env.C
                    if done:
                        reward += -station_count
                    self.record(state, action, reward, next_state, done)
                    state = next_state
            if train:
                    self.replay()
            if done:
                print(station_count)
                
            else:
                print('Failed to construct a solution', station_count)
            
            scores.append(station_count)
        print(np.mean(scores), np.min(scores))