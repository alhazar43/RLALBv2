
import numpy as np

from agent import PPOAgent
from env import ALBEnv
from tqdm.auto import tqdm
import os
import random
import matplotlib.pyplot as plt


import numpy as np
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)
import warnings
warnings.filterwarnings('ignore')


from numba import cuda 
device = cuda.get_current_device()
device.reset()

n = 100


idxs = [1,
4,
6,
8,
9,
11,
12]

N = 5
batch_size = 10
n_epochs = 5
episode = 50
alpha = 0.003
agent = PPOAgent(batch_size=batch_size, alpha=alpha, n_epochs=n_epochs)

out_dir = './output'




if not os.path.isdir(out_dir):
    os.makedirs(out_dir)

csv2_name = out_dir + "/n={}_testing.csv".format(str(n))
print(csv2_name)
for idx in idxs:

    permute = False
    if n==50 and permute:
        pi = np.random.choice(list(range(2,9)))
        instance_name = "instance_n={}_{}p{}".format(str(n), str(idx), str(pi))
        
    else:
        instance_name = "instance_n={}_{}".format(str(n), str(idx))
        
        
    instance = './data/n={}/'.format(str(n)) + instance_name + '.txt'

    soln = []
    best_soln = []
    reward_history = []


    fig_name = out_dir + "/" + instance_name + "_training_rwd_{}.pdf".format(str(episode))
    fig2_name = out_dir + "/" + instance_name + "_training_soln_{}.pdf".format(str(episode))
    csv_name = out_dir + "/" + instance_name + "_training_{}.csv".format(str(episode))
    
    txt_name = out_dir + "/" + instance_name + "_training_{}.txt".format(str(episode))



    env = ALBEnv(instance, record_soln=True)

    learn_iters = 0
    global_step = 0
    pbar = tqdm(range(episode))
    temp = 999

    for i in pbar:
        agent.buffer.clear()
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
                                [10*observation['job_time'], observation['successor']], 
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

        
        pbar.set_description(instance_name + ": n = %d" % observation_['num_station'])

        nbs = observation_['num_station']
        if nbs < temp:
            temp = nbs
            best_soln.append(env.solution_memory)
            # agent.save_models(model="_"+instance_name)
            
            
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
    ax.plot(x, reward_history, alpha=0.5, label='RL')
    ax.plot(x, random_rew, alpha=0.5, label='Random Search')
    ax.plot(x, heuristic_rew, alpha=0.5, label='PRM_MaxTdL')
    # vals = ax.get_yticks()
    # ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 
    plt.legend(loc='best')
    plt.xlabel('Episodes')
    plt.ylabel('Reward') 
    fig.savefig(fig_name, bbox_inches='tight', 
                transparent=True,
                pad_inches=0)
    
    
    fig, ax = plt.subplots(figsize=(8, 6))
    x = list(range(1,episode+1))
    ax.plot(x, soln, alpha=0.5, label='RL')
    ax.plot(x, rs, alpha=0.5, label='Random Search')
    ax.plot(x, hs, alpha=0.5, label='PRM_MaxTdL')
    # vals = ax.get_yticks()
    # ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 
    plt.legend(loc='best')
    plt.xlabel('Episodes')
    plt.ylabel('Num. of stations') 
    fig.savefig(fig2_name, bbox_inches='tight', 
                transparent=True,
                pad_inches=0)

    import csv
    with open(txt_name, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerows(best_soln[-1])

agent.save_models(model="_n=20")


all_soln = []
for i in tqdm(range(1, 526)):
    inst = './data/n={}/instance_n={}_{}.txt'.format(str(n), str(n), str(i))
    env = ALBEnv(inst, record_soln=True)
    observation = env.reset()
    done = False
    score = 0
    ix = 0
    while not env.episode_done:
        
        action, prob, val = agent.choose_action(observation)
        
        observation_, reward, done, info = env.step(action)
        

        observation = observation_
    all_soln.append(observation_['num_station'])
    
np.savetxt(csv2_name, np.asarray(all_soln, dtype=np.int32), delimiter=",")
 