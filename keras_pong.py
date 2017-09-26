import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import numpy as np

import gym
import os

# Stop complaining about my CPU's Power Level
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

model = Sequential()


# Q Network
env = gym.make('Pong-v0')
env.reset()
state, reward, done, meta = env.step(env.action_space.sample())
height, width, color = len(state), len(state[0]), len(state[0][0])

print('Screen Dimentions')
print('Height', height)
print('Width', width)
print('Color Depth', color)

env.reset()

steps = 50
step = 0

# input image dimensions
img_rows, img_cols = height, width


model = Sequential()
'''
[2] The first hidden layer convolves 16 8×8 filters with stride 4 with the 
    input image and applies a rectifier non-linearity.  
'''
model.add(Conv2D(16,
                 kernel_size=(8, 8),
                 activation='relu',
                 strides=4,
                 input_shape=(height, width, color)))
'''
[3] The second hidden layer convolves 32 4×4 filters with stride 2, 
    - [3.1] again followed by a rectifier nonlinearity. 
'''
model.add(Conv2D(32,
                 kernel_size=(4, 4),
                 activation='relu',
                 strides=2))
'''
[4] The final hidden layer is fully-connected and consists of 256 
    rectifier units.  
'''
model.add(Dense(256, activation='relu'))

'''
[5] The output layer is a fully-connected linear layer with a single output 
for each valid action. The number of valid actions varied between 4 and 18 on 
the games we considered. 
'''
model.add(Dense(3, activation=None))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])


# model.fit(state, reward, epochs=1, verbose=0)

episodes = 20

future_action = 0

for i in range(episodes):
    env.reset()
    while not done:
        random_action = env.action_space.sample()
        if future_action == 0:
            action = random_action
        else:
            action = future_action

        state, reward, done, _ = env.step(random_action)
        state = state.reshape(1, height, width, color)
        future_action = np.argmax(model.predict(state, batch_size=1))
        if done:
            break

# score = model.evaluate(x_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])

# while step < steps:
#     step += 1
#     random_action = env.action_space.sample()
#
#     state, reward, done, _ = env.step(random_action)
#
#     print('Reward:', reward)



'''
We now describe the exact architecture used for all seven Atari games.

[1] The input to the neural network consists is an 84×84×4 image produced by φ. 
    [1.1] LAML: We will use the original screen dimensions 
    
[2] The first hidden layer convolves 16 8×8 filters with stride 4 with the input 
image and applies a rectifier non-linearity.  

[3] The second hidden layer convolves 32 4×4 filters with stride 2, 
- [3.1] again followed by a rectifier nonlinearity. 

[4] The final hidden layer is fully-connected and consists of 256 rectifier units.  

[5] The output layer is a fully-connected linear layer with a single output 
for each valid action. The number of valid actions varied between 4 and 18 on 
the games we considered. 

We refer to convolutional networks trained with our approach as 
Deep Q-Networks (DQN).
'''

'''
Gets an 84x84x4 Tensor from Gym of the past 4 frames
'''
def get_frames(gym):
    return True


'''
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
'''
