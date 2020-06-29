#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
module introduction
 
Authors: shipeng
Date:    2020/6/27 下午4:37
"""
from bubble_environment import Paddle, ReplayMemory, Agent, Model
import numpy as np
from parl.algorithms import fluid
from parl.utils import logger

MINI_BATCH_SIZE = 200
BATCH_SIZE = 64
MEMORY_SIZE = 100000
LEARN_FREQ = 3


def run_episode(env: Paddle.Paddle, agent: Agent.Agent,
                replay_memory: ReplayMemory.ReplayMemory):
    """
    run one episode with environment
    :param env:
    :param agent:
    :param replay_memory:
    :return:
    """
    obs = env.reset()
    step = 0
    total_reward = 0
    while True:
        step += 1
        action = agent.sample(obs)

        reward, next_obs, done = env.step(action)
        replay_memory.append((obs, action, reward, next_obs, done))

        if len(replay_memory) >= MINI_BATCH_SIZE:
            obs_list, action_list, reward_list, next_obs_list, done_list = \
                replay_memory.sample(BATCH_SIZE)

            train_loss = agent.learn(obs_list, action_list, reward_list, next_obs_list, done_list)

        total_reward += reward
        obs = next_obs

        if done:
            break
    return total_reward


def evaluate(env: Paddle.Paddle, agent: Agent.Agent, render=False, episode: int=5):
    """
    test for the trained model
    :param env:
    :param agent:
    :param render:
    :param episode:
    :return:
    """
    reward_list = []
    for i in range(episode):
        obs = env.reset()
        total_reward = 0
        while True:
            action = agent.predict(obs)
            reward, next_obs, done = env.step(action)
            total_reward += reward
            reward_list.append(total_reward)
            obs = next_obs
            if done:
                break

    return np.mean(reward_list)


def train():
    gamma = 0.99
    learn_rate = 0.001
    greedy = 0.9
    max_episode = 200

    env = Paddle.Paddle()
    act_dim = 3
    obs_dim = 5

    model = Model.Model(act_dim)

    algorithm = fluid.DQN(model=model, act_dim=act_dim, gamma=gamma, lr=learn_rate)

    agent = Agent.Agent(algorithm, obs_dim, act_dim, greedy, greedy_decrement=0.00002)

    replay_memory = ReplayMemory.ReplayMemory(MEMORY_SIZE)

    while len(replay_memory) < MINI_BATCH_SIZE:
        run_episode(env, agent, replay_memory)

    logger.info("{}".format(len(replay_memory)))
    episode = 0

    while episode < max_episode:

        total_reward = run_episode(env, agent, replay_memory)
        episode += 1

        if episode % 50 == 0:
            print("evaluate" + str(episode))
            logger.info("reward: {}".format(evaluate(env, agent)))

    agent.save("./bubble_model.ckpt")


def test():
    gamma = 0.99
    learn_rate = 0.001
    greedy = 0.9
    max_episode = 200

    env = Paddle.Paddle()
    act_dim = 3
    obs_dim = 5

    model = Model.Model(act_dim)
    algorithm = fluid.DQN(model=model, act_dim=act_dim, gamma=gamma, lr=learn_rate)
    agent = Agent.Agent(algorithm, obs_dim, act_dim, greedy)
    agent.restore("./bubble_model.ckpt")
    for _ in range(max_episode):
        evaluate(env, agent)


if __name__ == "__main__":
    train()
    # test()
