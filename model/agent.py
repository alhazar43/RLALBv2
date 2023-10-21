import numpy as np

# TF
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
from tf_agents.distributions.masked import MaskedCategorical

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical

from .utils import Buffer
from .networks import ActorNet, CriticNet, GAT

class PPOAgent:
    def __init__(self, attention_params=[32,2], gamma=0.99, alpha=3e-4,
                 gae_lambda=0.95, policy_clip=0.1, batch_size=64,
                 n_epochs=10, model_dir='./models/'):
        assert len(attention_params)>=2, "Specify hidden layer size and number of heads"
        self.attention_params = attention_params

        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.model_dir = model_dir

        self.actor = GAT(1,hidden_units=self.attention_params[0], num_heads=self.attention_params[1])
        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic = CriticNet()
        self.critic.compile(optimizer=Adam(learning_rate=alpha))
        self.buffer = Buffer(batch_size)

    def store_transition(self, state, mask, feat, action, probs, vals, reward, done):
        self.buffer.store(state, mask, feat, action, probs, vals, reward, done)

    def save_models(self, model):
        print('... saving models ...')
        self.actor.save_weights(self.model_dir + 'actor' + str(model) + '.h5')
        self.critic.save_weights(self.model_dir + 'critic' + str(model) + '.h5')

    def load_models(self, model):
        print('... loading models ...')
        self.actor = keras.models.load_model(self.model_dir + 'actor' + str(model))
        self.critic = keras.models.load_model(self.model_dir + 'critic' + str(model))

    def choose_action(self, observation, use_mask=True):
        
        state = observation['graph']
        edges = tf.where(state)
        
        
        feat = tf.transpose(tf.convert_to_tensor([observation['job_time'], observation['successor']]))
        logits = tf.transpose(self.actor((feat, edges)))
        
        
        if use_mask:
            mask = tf.convert_to_tensor([observation['prec_mask']])
            dist = MaskedCategorical(logits=logits, mask=mask)
        else:
            dist = tfp.distributions.Categorical(logits)

        
        action = dist.sample()
        # print(mask.shape[1])
        log_prob = dist.log_prob(action)
        
        # log_prob = dist.log_prob(action)
        value = self.critic(tf.convert_to_tensor([state]))
        action = action.numpy()[0]
        value = value.numpy()[0]
        log_prob = log_prob.numpy()
        
        return action, log_prob, value
    

    
    @tf.function
    def update_policy(self, 
                      state_arr, 
                      mask_arr, 
                      feat_arr, 
                      action_arr, 
                      old_prob_arr,
                      vals_arr, 
                      reward_arr, 
                      dones_arr, 
                      batches):
        values = tf.cast(vals_arr, dtype=tf.float32)
        reward_arr = tf.cast(reward_arr, dtype=tf.float32)
        
        for _ in tf.range(self.n_epochs):
            # state_arr, mask_arr, feat_arr, action_arr, old_prob_arr,\
            #     vals_arr, reward_arr, dones_arr, batches = self.buffer.get()

            
            # advantage = np.zeros(len(reward_arr), dtype=np.float32)
            advantage = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
            
            gamma = tf.cast(self.gamma, dtype=tf.float32)
            gae_lambda = tf.cast(self.gae_lambda, dtype=tf.float32)

            for t in tf.range(len(reward_arr)-1):
                discount = tf.constant(1.0, dtype=tf.float32)
                a_t = tf.constant([0.0], dtype=tf.float32)
                for k in tf.range(t, len(reward_arr)-1):
                    # A_t^{pi_k} advantage estimates
                    a_t += discount * (reward_arr[k] + gamma * values[k+1] * tf.cast(1-int(dones_arr[k]), dtype=tf.float32) - values[k])
                    discount *= gamma*gae_lambda
                # advantage[t] = a_t
                advantage = advantage.write(t, a_t)
            advantage = advantage.stack()

            for batch in batches:

                with tf.GradientTape(persistent=True) as tape:
                    states = tf.gather(state_arr, batch)
                    masks = tf.gather(mask_arr, batch)
                    feats = tf.gather(feat_arr, batch)
                    actions = tf.gather(action_arr, batch)        
                    old_probs = tf.gather(old_prob_arr, batch)
                    new_probs = tf.TensorArray(dtype=tf.float32, size=len(batch), dynamic_size=True)
                    for i in tf.range(len(batch)):
                        edge = tf.where(states[batch[i]])
                        feat = tf.transpose(feats[batch[i]])
                        print(tf.shape(feat), len(batch))
                        logits = tf.transpose(self.actor((feat, edge)))
                        # tf.print(logits)
                        dist = MaskedCategorical(logits=logits, mask=tf.convert_to_tensor([masks[batch[i]]]))
                        
                        action = tf.squeeze(actions[batch[i]])
                        new_probs = new_probs.write(i, dist.log_prob(dist.sample()))

                        
                    new_probs = new_probs.stack()
                    # entropy = -tf.math.reduce_mean(tf.clip_by_value(tf.math.exp(new_probs), 1e-5, 1) * new_probs)
                    # prob_ratio = tf.math.divide_no_nan(tf.clip_by_value(tf.math.exp(new_probs), 0, 1),tf.clip_by_value(tf.math.exp(old_probs), 0, 1))
                    # prob_ratio = tf.clip_by_value(prob_ratio, 1e-3, 1e+3)
                    # tf.print(prob_ratio)
                    prob_ratio = tf.math.exp(new_probs-old_probs)
                    weighted_probs = tf.gather(advantage, batch) * (prob_ratio + 1e-5)
                    
                    clipped_probs = tf.clip_by_value(prob_ratio, 1-self.policy_clip, 1+self.policy_clip)
                    weighted_clipped_probs = clipped_probs * tf.gather(advantage, batch)

                    
                    actor_loss = -tf.math.minimum(weighted_probs, weighted_clipped_probs)
                    
                    actor_loss = tf.math.reduce_mean(actor_loss)

                    # if tf.greater(actor_loss, 0):
                    #     actor_loss = -actor_loss
                    
                    critic_value = self.critic(tf.convert_to_tensor(states))
                    critic_value = tf.squeeze(critic_value, 1)
                    returns = tf.gather(advantage, batch) + tf.gather(values, batch)
                    critic_loss = tf.math.reduce_mean(tf.math.pow(returns-critic_value, 2))
                    critic_loss = keras.losses.MSE(critic_value, returns)
                    
                
                actor_params = self.actor.trainable_weights
                

                actor_grads = tape.gradient(actor_loss, actor_params)
                
                critic_params = self.critic.trainable_variables
                critic_grads = tape.gradient(critic_loss, critic_params)
                
                self.actor.optimizer.apply_gradients(zip(actor_grads, actor_params))
                self.critic.optimizer.apply_gradients(zip(critic_grads, critic_params))
        
        