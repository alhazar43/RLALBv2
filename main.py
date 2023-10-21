import numpy as np
from model.agent import PPOAgent
from model.env import ALBEnv
from tqdm.auto import tqdm
import os
import matplotlib.pyplot as plt


import numpy as np
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)


n = 20
# idx = np.random.choice(list(range(1,526)))
idx = 17
permute = False
if n!=50 and not permute:
    instance_name = "instance_n={}_{}".format(str(n), str(idx))
else:
    pi = np.random.choice(list(range(2,9)))
    instance_name = "instance_n={}_{}p{}".format(str(n), str(idx), str(pi))
    
instance = 'data/otto2014/n={}/'.format(str(n)) + instance_name + '.txt'

soln = []
best_soln = []
reward_history = []

N = 5
batch_size = 10
n_epochs = 8
episode = 100
alpha = 0.003
agent = PPOAgent(batch_size=batch_size, alpha=alpha, n_epochs=n_epochs)


out_dir = os.path.join(os.path.dirname(__file__), "output")

if not os.path.isdir(out_dir):
    os.makedirs(out_dir)

fig_name = os.path.normpath(os.path.join(out_dir, instance_name + "_training_{}.pdf".format(str(episode))))
csv_name = os.path.normpath(os.path.join(out_dir, instance_name + "_training_{}.csv".format(str(episode))))
txt_name = os.path.normpath(os.path.join(out_dir, instance_name + "_training_{}.txt".format(str(episode))))



env = ALBEnv(instance, record_soln=True)

learn_iters = 0
global_step = 0
pbar = tqdm(range(episode))
temp = 999

import warnings
warnings.filterwarnings('ignore')


for i in pbar:
    observation = env.reset()
    done = False
    score = 0
    ix = 0
    while not env.episode_done:
        
        action, prob, val = agent.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        global_step += 1
        score += reward
        agent.store_transition(np.array(observation['graph']), 
                               np.array(observation['prec_mask']),  
                               [10*observation['job_time'],
                                observation['successor'],
                                observation['predcessor']], 
                               action,
                               prob, 
                               val, 
                               reward, 
                               done)
        
        if global_step % N == 0 or env.episode_done:
            agent.update_policy(*agent.buffer.get())
            learn_iters += 1

        observation = observation_
 
        if global_step % (batch_size) == 0:
            agent.buffer.clear()
            
    reward_history.append(score)

    
    pbar.set_description("n = %d" % observation_['num_station'])

    nbs = observation_['num_station']
    if nbs < temp:
        temp = nbs
        best_soln.append(env.solution_memory)
        agent.save_models(model="_"+instance_name)
        
        
    soln.append(nbs)
    # solnm.append(env.solution_memory)
    # print(observation_['num_station'], env.solution_memory)
    
np.savetxt(csv_name, np.asarray(soln, dtype=np.int32), delimiter=",")

lj = np.arange(env.instance.num_jobs)
rs = []
random_rew = []
for i in range(len(soln)):
    step = env.reset()
    av = step['precedence']==0
    score = 0
    while av.sum()>0:
        action = np.random.choice(lj[av])
        step, r, done, _ = env.step(action)
        # av = np.where(step['precedence']==0)[0]
        score += r
        av = step['precedence']==0
        bv = 1000-step['station_load']-step['job_time']>0
        av = np.logical_and(av, bv) if np.logical_and(av, bv).sum()>0 else av
    rs.append(step['num_station'])
    random_rew.append(score)
    
hs = []
heuristic_rew = []
for i in range(len(soln)):
    step = env.reset()
    av = step['precedence']==0
    score = 0
    while av.sum()>0:
        action = lj[av][np.nanargmax(env.instance.tdL[av])]
        step, r, done, _ = env.step(action)
        # av = np.where(step['precedence']==0)[0]
        score += r
        av = step['precedence']==0
        bv = 1000-step['station_load']-step['job_time']>0
        av = np.logical_and(av, bv) if np.logical_and(av, bv).sum()>0 else av
    hs.append(step['num_station'])
    heuristic_rew.append(score)
    
    
fig, ax = plt.subplots(figsize=(8, 6))
x = list(range(1,episode+1))
ax.plot(x, reward_history, alpha=0.8, label='RL')
ax.plot(x, random_rew, alpha=0.8, label='Random Search')
ax.plot(x, heuristic_rew, alpha=0.8, label='PRM_MaxTdL')
# vals = ax.get_yticks()
# ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 
plt.legend(loc='best')
plt.xlabel('Episodes')
plt.ylabel('Reward') 
fig.savefig(fig_name, bbox_inches='tight', 
               transparent=True,
               pad_inches=0)

import csv
with open(txt_name, 'w', newline='') as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerows(best_soln[-1])