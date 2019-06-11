#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v2.pkl Humanoid-v2 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import os
import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
import datetime
import config
from tqdm import tqdm

class Expert():
    
    def __init__(self, config):
        self.config = config
        self.max_timesteps = self.config.environment.max_timesteps
        
    def load_expert_policy(self, envname):
        assert envname
        
        self.envname = envname
        
        print('loading expert policy : ', envname)
        policy_file = os.path.join('experts', envname + '.pkl')
        policy_fn = load_policy.load_policy(policy_file)
        return policy_fn

    def generate_rollouts_all(self):
        for envname, num_rollouts in zip(self.config.environment.envlist, 
                                         self.config.environment.rolloutlist):
            policy_fn = self.load_expert_policy(envname)
            self.generate_rollout(policy_fn, num_rollouts)
 
    def generate_rollout(self, policy_fn, num_rollouts, render=False):

        assert policy_fn
        
        with tf.Session():
            tf_util.initialize()
    
            env = gym.make(self.envname)
            obs = env.reset()
            obs_dim = env.observation_space.shape[0]
            act_dim = env.action_space.shape[0]
            
            print (obs_dim,act_dim)

            max_steps = self.max_timesteps or env.spec.timestep_limit
    
            returns = []
            observations = []
            actions = []
            for i in tqdm(range(num_rollouts), desc="Rollout"):
                # print('Rollout ', i)
                obs = env.reset()
                done = False
                totalr = 0.
                steps = 0
                
                while not done:
                    action = policy_fn(obs[None,:])
                    observations.append(obs)
                    actions.append(action)
                    obs, r, done, _ = env.step(action)
                    totalr += r
                    steps += 1
                    if render: env.render()
                    if steps >= max_steps: break
                
                returns.append(totalr)
    
            print('returns', returns)
            print('mean return', np.mean(returns))
            print('std of return', np.std(returns))
            
        expert_data = {'observations': np.array(observations),
                       'actions': np.array(actions),
                       'returns': np.array(returns)}

        with open(os.path.join('expert_data', self.envname + '.pkl'), 'wb') as f:
            pickle.dump(expert_data, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    begin_time = datetime.datetime.now()
    
    cfg = config.Config()
    bc = Expert(cfg)
    bc.generate_rollouts_all()
   
    end_time = datetime.datetime.now()          
    print()
    print("Exiting...", end_time)
    print("Running Time", end_time - begin_time)
