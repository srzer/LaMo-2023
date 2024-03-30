import os
import glob
import time
from datetime import datetime

import torch
import numpy as np

import gym

from PPO import PPO
import pickle


#################################### Testing ###################################
def test():
    print("============================================================================================")

    ################## hyperparameters ##################

    # env_name = "CartPole-v1"
    # has_continuous_action_space = False
    # max_ep_len = 400
    # action_std = None

    # env_name = "LunarLander-v2"
    # has_continuous_action_space = False
    # max_ep_len = 300
    # action_std = None

    # env_name = "BipedalWalker-v2"
    # has_continuous_action_space = True
    # max_ep_len = 1500           # max timesteps in one episode
    # action_std = 0.1            # set same std for action distribution which was used while saving

    # env_name = "ant_dir-0"
    has_continuous_action_space = True
    max_ep_len = 1000           # max timesteps in one episode
    action_std = 0.1            # set same std for action distribution which was used while saving

    render = False              # render environment on screen
    frame_delay = 0             # if required; add delay b/w frames

    total_test_episodes = 50   # total num of testing episodes

    K_epochs = 80               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor

    lr_actor = 0.0003           # learning rate for actor
    lr_critic = 0.001           # learning rate for critic

    #####################################################
    
    config_save_path = os.path.join('../config')
    
    env_name = "reacher2d"
    from envs.reacher_2d import Reacher2dEnv
    env = Reacher2dEnv()
    max_ep_len = 100

    # state space dimension
    state_dim = env.observation_space.shape[0]

    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

    # preTrained weights directory

    random_seed = 0             #### set this to load a particular checkpoint trained on random seed
    run_num_pretrained = 0      #### set this to load a particular checkpoint num

    directory = "PPO_preTrained" + '/' + env_name + '/'
    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("loading network from : " + checkpoint_path)

    ppo_agent.load(checkpoint_path)

    print("--------------------------------------------------------------------------------------------")

    test_running_reward = 0

    data_list = []
    
    ep = 1
    while ep < total_test_episodes:
        ep_reward = 0
        state = env.reset()

        data_dict = {'observations': [], 'next_observations': [], 'actions': [], 'rewards': [], 'terminals': []}
        
        for t in range(1, max_ep_len+1):
            data_dict['observations'].append(state)
            
            action = ppo_agent.select_action(state)
            state, reward, done, _ = env.step(action)
            ep_reward += reward

            if render:
                env.render()
                time.sleep(frame_delay)

            if done:
                break
            
            data_dict['next_observations'].append(np.array(state))
            data_dict['actions'].append(np.array(action))
            data_dict['rewards'].append(np.array([reward]))
            data_dict['terminals'].append(np.array([done]))
        
        data_dict = {key: np.array(value) for key, value in data_dict.items()}
        
        # clear buffer
        ppo_agent.buffer.clear()

        print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))
        if ep_reward > 20: 
            test_running_reward +=  ep_reward
            data_list.append(data_dict)
            ep += 1
        ep_reward = 0

    env.close()

    # with open(env_name+".pkl", "wb") as fo:
    #     pickle.dump(data_list, fo)
    
    print("============================================================================================")

    avg_test_reward = test_running_reward / total_test_episodes
    avg_test_reward = round(avg_test_reward, 2)
    print("average test reward : " + str(avg_test_reward))

    print("============================================================================================")


if __name__ == '__main__':

    test()
