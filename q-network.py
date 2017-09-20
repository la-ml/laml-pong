import gym
import numpy as np
import random
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the Environment
env = gym.make('FrozenLake-v0')

# Impliment the Network
tf.reset_default_graph()

# These lines establish the feed-forward part of the network used to choose actions
inputs1 = tf.placeholder(shape=[1, 16], dtype=tf.float32)
W = tf.Variable(tf.random_uniform([16, 4], 0, 0.01))
Qout = tf.matmul(inputs1, W)
predict = tf.argmax(Qout, 1)

# Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
nextQ = tf.placeholder(shape=[1, 4], dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
updateModel = trainer.minimize(loss)

## TRAINING THE NETWORK

init = tf.global_variables_initializer()

# Set learning parameters
y = .99  # Gamma
e = 0.1  # Epsilon

'''TODO: Episodes?  Number of times it tries to play the game to a designated point'''
num_episodes = 2000

# create lists to contain total rewards and steps per episode
stageList = [] # List of States that you go though
rewardsList = [] # Rewards List

with tf.Session() as sess:
    sess.run(init)
    for i in range(num_episodes):
        # Reset environment and get first new observation
        s = env.reset()
        rewardsAll = 0
        done = False # Done
        stage = 0
        # The Q-Network
        while stage < 99:
            stage += 1
            # Choose an action by greedily (with e chance of random action) from the Q-network
            action, allQ = sess.run([predict, Qout], feed_dict={inputs1: np.identity(16)[s:s + 1]})
            if np.random.rand(1) < e:
                action[0] = env.action_space.sample()

            # Get new state and reward from environment
            s1, reward, done, _ = env.step(action[0])

            # Obtain the Q' values by feeding the new state through our network
            Q1 = sess.run(Qout, feed_dict={inputs1: np.identity(16)[s1:s1 + 1]})

            # Obtain maxQ' and set our target value for chosen action.
            maxQ1 = np.max(Q1)
            targetQ = allQ
            targetQ[0, action[0]] = reward + y * maxQ1

            # Train our network using target and predicted Q values
            _, W1 = sess.run([updateModel, W], feed_dict={inputs1: np.identity(16)[s:s + 1], nextQ: targetQ})
            rewardsAll += reward
            s = s1

            if done == True:
                # Reduce chance of random action as we train the model.
                e = 1. / ((i / 50) + 10)
                break
        stageList.append(stage)
        rewardsList.append(rewardsAll)

print("Percent of successful episodes: " + str(sum(rewardsList) / num_episodes) + "%")

