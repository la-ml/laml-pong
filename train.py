from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import os
#
# from six.moves import xrange
import tensorflow as tf

from tensorflow.contrib.learn.python.learn import monitored_session as ms
#
# import meta
# import util

import gym

# env = gym.make('Pong-v0')
# env.reset()
# print(env.action_space.sample())
# for _ in range(1000):
#     env.render()
#     env.step(env.action_space.sample())

# env = gym.make('Pong-v0')
#
# for i_episode in range(20):
#     observation = env.reset()
#     for t in range(100):
#         env.render()
#         # print(observation)
#
#         # ML Goes Here
#
#         # DEEP Q Learning Pseudo-Code
#         #
#         '''
#             Initialize replay memory D to capacity N
#             Initialize action-value function Q with random weights
#             for episode = 1, M do
#                 Initialise sequence s1 = {x1} and preprocessed sequenced φ1 = φ(s1)
#                 for t = 1, T do
#                     With probability ε select a random action at
#                         otherwise select at = Maxa Q∗(φ(st), a; θ)
#                     Execute action at in emulator and observe reward rt and image xt+1
#                     Set st+1 = st, at, xt+1 and preprocess φt+1 = φ(st+1)
#                     Store transition (φt, at, rt, φt+1) in D
#                     Sample random minibatch of transitions (φj , aj , rj , φj +1 ) from D
#                     Set yj = rj                          -> for terminal φj+1
#                     Set yj = rj + γ maxa′ Q(φj+1, a′; θ) -> for non-terminal φj+1
#                     Perform a gradient descent step on (yj − Q(φj , aj ; θ))2 according to equation 3
#                 end for
#             end for
#         '''
#
#         # Initialize replay memory D to capacity N
#         # replay_memory
#
#         # Initialize action-value function Q with random weights
#         # TODO: Implement in TensorFlow
#
#         # Random Weights
#         # weights
#
#         # Define ε - Make a random number if that number is less than ε  then we generate a random action
#         epsilon = 0.1
#
#
#         def action_value_func_Q(weights):
#             # TODO: Make function
#
#         def pre_process_sequence(sequence):
#             # TODO: Make function
#
#
#         action = env.action_space.sample()
#         print(action)
#         observation, reward, done, info = env.step(action)
#         if done:
#             print("Episode finished after {} timesteps".format(t+1))
#             break

import gym
import numpy as np

env = gym.make('FrozenLake-v0')

#Initialize table with all zeros
Q = np.zeros([env.observation_space.n,env.action_space.n])
# Set learning parameters
lr = .8
y = .95
num_episodes = 2000
#create lists to contain total rewards and steps per episode
#jList = []
rList = []
for i in range(num_episodes):
    #Reset environment and get first new observation
    s = env.reset()
    env.render('human')
    rAll = 0
    d = False
    j = 0
    #The Q-Table learning algorithm
    while j < 99:
        env.render('human')
        j+=1
        #Choose an action by greedily (with noise) picking from Q table
        a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))
        #Get new state and reward from environment
        s1,r,d,_ = env.step(a)
        #Update Q-Table with new knowledge
        Q[s,a] = Q[s,a] + lr*(r + y*np.max(Q[s1,:]) - Q[s,a])
        rAll += r
        s = s1
        if d == True:
            break
    #jList.append(j)
    rList.append(rAll)

print("Score over time: " + str(sum(rList)/num_episodes))

print("Final Q-Table Values")
print(Q)

