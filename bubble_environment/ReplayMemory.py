#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
module introduction

Authors: shipeng
Date:    2020/6/27 下午1:20
"""
import collections
import random
import numpy as np


class ReplayMemory(object):
    """
    DQN's experience replay
    """
    def __init__(self, max_size):
        """
        static deque
        :param max_size:
        """
        self.buffer = collections.deque(maxlen=max_size)

    def append(self, exp):
        """
        add a new experience
        :return:
        """
        self.buffer.append(exp)

    def sample(self, batch_size):
        """
        simple n experience from deque
        :param batch_size:
        :return:
        """
        simple_batch = random.sample(self.buffer, batch_size)

        # one experience is combined with obs, action, reward, next_obs and done_flag
        obs_list, action_list, reward_list, next_obs_list, done_list = [], [], [], [], []

        for item in simple_batch:
            obs, action, reward, next_obs, done = item

            obs_list.append(obs)
            action_list.append(action)
            reward_list.append(reward)
            next_obs_list.append(next_obs)
            done_list.append(done)

        return np.array(obs_list).astype('float32'), np.array(action_list).astype('float32'), \
               np.array(reward_list).astype('float32'), np.array(next_obs_list).astype('float32'), \
               np.array(done_list).astype('float32')

    def __len__(self):
        """

        :return:
        """
        return len(self.buffer)



