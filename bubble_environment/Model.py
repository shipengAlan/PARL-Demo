#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
module introduction
 
Authors: shipeng
Date:    2020/6/27 下午1:20
"""
import parl
from parl import layers


class Model(parl.Model):

    def __init__(self, act_num):
        """
        init
        :param act_num:
        """
        self.fc1 = layers.fc(size=64, act='relu')
        self.fc2 = layers.fc(size=64, act='relu')
        self.fc3 = layers.fc(size=act_num, act=None)

    def value(self, obs):
        """
        2 hidden layers with full connected
        :param obs:
        :return:
        """
        h1 = self.fc1(obs)
        h2 = self.fc2(h1)
        Q = self.fc3(h2)
        return Q
