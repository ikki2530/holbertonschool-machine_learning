#!/usr/bin/env python3
"""Monte Carlo Method"""

import gym
import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000,
               max_steps=100, alpha=0.1, gamma=0.99):
    """Function that performs the Monte Carlo algorithm
    Returns: V
    V: the updated value estimate"""
    state_d = env.observation_space.n
    Et = np.zeros(state_d)
    for _ in range(episodes):
        state = env.reset()
        for _ in range(max_steps):
            Et *= lambtha * gamma
            Et[state] += 1.0
            a = policy(state)
            new_s, reward, done, _ = env.step(a)
            if env.desc.reshape(state_d)[new_s] == b'G':
                reward = 1.0
            if env.desc.reshape(state_d)[new_s] == b'H':
                reward = -1.0
            delta_t = reward + gamma * V[new_s] - V[state]
            V[state] = V[state] + alpha * delta_t * Et[state]
            if done:
                break
            state = new_s
    return V
