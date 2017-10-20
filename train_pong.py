import sys

import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D

import numpy as np
import random
import math

import gym
import os

import uuid
import time

# Stop complaining about my CPU's Power Level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Q Network
env = gym.make('Pong-v0')
env.reset()
state, reward, done, meta = env.step(env.action_space.sample())
height, width, color = len(state), len(state[0]), len(state[0][0])

# Make Greyscale
color = 1

print('Screen Dimentions')
print('Height', height)
print('Width', width)
print('Color Depth', color)

# input image dimensions
img_rows, img_cols = height, width

possibleActions = [1, 2, 3]
numFrames = 4

model = Sequential()
'''
[2] The first hidden layer convolves 16 8×8 filters with stride 4 with the
    input image and applies a rectifier non-linearity.
'''
model.add(Conv2D(16,
                 kernel_size=(8, 8),
                 activation='relu',
                 strides=4,
                 input_shape=(height * numFrames, width, color)))
'''
[3] The second hidden layer convolves 32 4×4 filters with stride 2,
    - [3.1] again followed by a rectifier nonlinearity.
'''
model.add(Conv2D(32,
                 kernel_size=(4, 4),
                 activation='relu',
                 strides=2))

model.add(Flatten())
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
model.add(Dense(len(possibleActions), activation=None))

model.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

print(model.summary())

episodes = 5000
#epsilon = 0.05
epsilon = 1
gamma = 0.9999
replayMemory = []
firstAction = True
batchSize = 10
maxMemorySize = 10000

frames = 0

uuid = str(uuid.uuid4())
filename = 'Pong-v1_{0}'.format(uuid)
print("Saving {0}".format(filename))

for i in range(episodes):

    #sum = 0
    # for mem in replayMemory:
    #    sum += mem[0].nbytes + mem[4].nbytes

    if (i % 10 == 0):
        model.save(filename + "_snapshot")

    if (i % 250 == 0):
        temp_filename = '{0}-{1}'.format(filename, i)
        print("Saving file: ", temp_filename)
        model.save(temp_filename)

    epsilon = max(0.05, 1 - (0.001 * i))
    state = env.reset()
    greyscale_state = np.reshape(state[:, :, 1], (height, width, 1)) / 255
    frames += 1
    frameSequence = np.zeros((height * numFrames, width, color))
    frameSequence[:height, :, :] = greyscale_state
    framesFilled = 1
    done = False

    # Debugging
    #totalMem = 0

    # for mem in replayMemory:
    #    totalMem += mem[0].nbytes
    #    totalMem += mem[4].nbytes
    #print("Total Memory:", totalMem)

    start_time = time.time()
    #print("Episode: ", i, "Frames: ", frames, "Replay Memory: ", sum)
    print("Episode: ", i, "Frames: ", frames)

    while not done:

        #sum = 0

        # for mem in replayMemory:
        #    sum += mem[0].nbytes + mem[4].nbytes

        if frames % 60 == 0:
            end_time = time.time() - start_time
            #print("Frames: ", frames, "Frame Time: ", (end_time / 60), "Replay Memory: ", sum )
            print("Frames: ", frames, "Frame Time: ", (end_time / 60))
            start_time = time.time()

        # if it's the first action, do random because network hasn't been
        # trained yet
        if firstAction:
            action = math.floor(random.random() * len(possibleActions))
            firstAction = False
        else:
            # with probability epsilon do a random action, otherwise use the
            # neural network to choose best action
            randomNumber = random.random()
            if randomNumber < epsilon:
                action = math.floor(random.random() * len(possibleActions))
            else:
                reshapedSequence = frameSequence.reshape(
                    1, height * numFrames, width, color)
                model_prediction = model.predict(
                    reshapedSequence, batch_size=1)
                #print("Model Prediction: ", model_prediction)
                action = np.argmax(model_prediction)
                #print("Action: ", action)

        # save the current sequence, get the next frame, and then add it to the
        # frame sequence
        currentFrameSequence = frameSequence
        state, reward, done, _ = env.step(possibleActions[action])
        greyscale_state = np.reshape(state[:, :, 1], (height, width, 1)) / 255
        frames += 1

        if framesFilled < numFrames:
            frameSequence[height * framesFilled:height *
                          (framesFilled + 1), :, :] = greyscale_state
            framesFilled += 1
        else:
            # shift all the frames down, and then add the new frame
            frameSequence[:height * (numFrames - 1), :,
                          :] = frameSequence[height:, :, :]
            frameSequence[height * (numFrames - 1):, :, :] = greyscale_state
            #print("Frame Sequence:", frameSequence)

        # save to replayMemory
        if len(replayMemory) >= maxMemorySize:
            replayMemory.pop(0)

        replayMemory.append([currentFrameSequence, action,
                             reward, done, frameSequence])

        # if replay memory is less than the batch size,
        # use all of the replay memory, otherwise take a random sample
        if len(replayMemory) <= batchSize:
            actualBatchSize = len(replayMemory)
            memoryIndices = list(range(actualBatchSize))
        else:
            actualBatchSize = batchSize
            memoryIndices = random.sample(range(len(replayMemory)), batchSize)

        print(
            "Memory Indicies: ",
            memoryIndices,
            "Replay Memory Size: ",
            len(replayMemory))

        # create the batched states and y_true arrays
        batchedStates = np.zeros(
            (actualBatchSize, height * numFrames, width, color))
        y_true = np.zeros((actualBatchSize, len(possibleActions)))
        for j in range(len(memoryIndices)):
            theMemory = replayMemory[memoryIndices[j]]
            # add the current state from the replay memory to the batched
            # states
            batchedStates[j, :, :, :] = theMemory[0]
            # if done = true in the replay memory, just set the index of y_true corresponding
            # to the action taken to the reward received
            if theMemory[3]:
                y_true[j, theMemory[1]] = theMemory[2]
            else:
                # in addition to setting the index of y_true corresponding to the
                # action taken to the reward received, add the value of Q for the next state,
                # discounted by gamma
                reshapedSequence = theMemory[4].reshape(
                    1, height * numFrames, width, color)
                futureQ = model.predict(reshapedSequence, batch_size=1)
                y_true[j, :] = gamma * futureQ
                y_true[j, theMemory[1]] += theMemory[2]
                if(theMemory[2] != 0):
                    print("FutureQ:", futureQ, "y_true[j,:]: ", y_true[j, :])

        model.train_on_batch(batchedStates, y_true)

final_filename = filename + "_final"
print("Saving Filename: ", final_filename)
model.save(final_filename)

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
