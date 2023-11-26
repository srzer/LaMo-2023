import gym
import numpy as np

import pickle

import random

env_names = ["hopper", "walker2d", "halfcheetah"]
dataset_types = ["medium"]
suffix = "yoursuffix"

for env_name in env_names:
    for dataset_type in dataset_types:
        ratios = [0.1, 0.01, 0.005]
        for ratio in ratios:
            path = env_name+"-"+dataset_type+"-v2.pkl"
            with open(path, "rb") as fo:
                data_list = pickle.load(fo)
            fo.close()
            print(env_name+str(ratio))
            print("before:", len(data_list))
            data_list = random.sample(data_list, int(len(data_list)*ratio))
            new_path = env_name + "-" + dataset_type + "-" + str(ratio) + "-" + suffix + "-v2.pkl"
            with open(new_path, "wb") as fo:
                pickle.dump(data_list, fo)
            print("after:", len(data_list))