#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#
"""
Agent for interacting with the environment
 
Authors: shipeng
Date:    2020/6/22 下午10:51
"""
import parl
import paddle.fluid as fluid
from parl import layers
import numpy as np


class Agent(parl.Agent):
    """
    response for interact with environment
    """
    def __init__(self, alg, obs_dim, act_dim, greedy, greedy_decrement=0.00005, min_greedy=0.05):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        super(Agent, self).__init__(alg)
        self.greedy = greedy
        self.update_target_model_steps = 500
        self.global_step = 0
        self.greed_decrement = greedy_decrement
        self.min_greedy = min_greedy

    def build_program(self):
        """
        define and bind input data
        :return:
        """
        self.pred_program = fluid.Program()
        self.learn_program = fluid.Program()

        with fluid.program_guard(self.pred_program):
            obs = layers.data(name='obs', shape=[self.obs_dim], dtype='float32')
            self.value = self.algorithm.predict(obs)

        with fluid.program_guard(self.learn_program):
            obs = layers.data(name='obs', shape=[self.obs_dim], dtype='float32')
            action = layers.data(name='action', shape=[1], dtype='int32')
            reward = layers.data(name='reward', shape=[], dtype='float32')
            next_obs = layers.data(name='next_obs', shape=[self.obs_dim], dtype='float32')
            done = layers.data(name='done', shape=[], dtype='bool')
            self.cost = self.algorithm.learn(obs, action, reward, next_obs, done)

    def predict(self, obs):
        """
        predict the best action
        :param obs:
        :return:
        """
        obs = np.expand_dims(obs, axis=0)
        predict_Q = self.fluid_executor.run(self.pred_program,
                                            feed={'obs': obs.astype('float32')},
                                            fetch_list=[self.value])[0]
        predict_Q = np.squeeze(predict_Q, axis=0)
        return np.argmax(predict_Q)

    def sample(self, obs):
        """
        give action with exploitation
        :param obs:
        :return:
        """
        if np.random.rand() < self.greedy:
            act = np.random.randint(self.act_dim)
        else:
            act = self.predict(obs)

        self.greedy = max(0.01, self.greedy - self.greed_decrement)
        return act

    def learn(self, obs, act, reward, next_obs, done):
        """
        learn by input batch data
        :param obs: np.array
        :param act: int
        :param reward: int
        :param next_obs: np.array
        :param done: bool
        :return:
        """
        if self.global_step % self.update_target_model_steps == 0:
            self.algorithm.sync_target()

        self.global_step += 1

        action = np.expand_dims(act, -1)
        feed = {
            'obs': obs.astype('float32'),
            'action': action.astype('int32'),
            'reward': reward,
            'next_obs': next_obs.astype('float32'),
            'done': done
        }
        return self.fluid_executor.run(self.learn_program,
                                       feed=feed,
                                       fetch_list=[self.cost])[0]

