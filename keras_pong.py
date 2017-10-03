import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import numpy as np
import random
import math 

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
                 input_shape=(height*numFrames, width, color)))
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
model.add(Dense(len(possibleActions), activation='softmax'))

model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

episodes = 20
epsilon = 0.05
gamma = 0.95
replayMemory = [None]
firstAction = true
batchSize = 10

for i in range(episodes):
    state = env.reset()
    frameSequence = np.zeros((height*numFrames, width, color))
    frameSequence[:height,:,:] = state
    framesFilled = 1
    while not done:
        
        #if it's the first action, do random because network hasn't been trained yet
        if firstAction
            action = math.floor(random.random()*len(possibleActions))
            firstAction = false
        else
            #with probability epsilon do a random action, otherwise use the neural network to choose best action
            randomNumber = random.random()
            if randomNumber < epsilon
                action = math.floor(random.random()*len(possibleActions))
            else
                reshapedSequence = frameSequence.reshape(1,height*numFrames,width,color)
                action = np.argmax(model.predict(reshapedSequence, batch_size=1))
                
        #save the current sequence, get the next frame, and then add it to the frame sequence
        currentFrameSequence = frameSequence
        state, reward, done, _ = env.step(possibleActions[action])
        
        if framesFilled < numFrames:
            frameSequence[height*framesFilled:height*(framesFilled+1),:,:] = state
            framesFilled += 1
        else:
            #shift all the frames down, and then add the new frame
            frameSequence[height:,:,:] = frameSequence[:height*(numFrames-1),:,:]
            frameSequence[:height,:,:] = state
            
        #save to replayMemory
        replayMemory.append([currentFrameSequence, action, reward, done, frameSequence])
        
        #if replay memory is less than the batch size, 
        #use all of the replay memory, otherwise take a random sample
        if len(replayMemory) <= batchSize
            actualBatchSize = len(replayMemory)
            memoryIndices = list(range(actualBatchSize))
        else
            actualBatchSize = batchSize
            memoryIndices = random.sample(range(len(replayMemory)), batchSize)
            
        #create the batched states and y_true arrays
        batchedStates = np.zeros((actualBatchSize,height*numFrames,width,color))
        y_true = np.zeros((actualBatchSize,len(possibleActions)))
        for j in range(memoryIndices):
            theMemory = replayMemory[memoryIndices[j]]
            #add the current state from the replay memory to the batched states
            batchedStates[j,:,:,:] = theMemory[0]
            #if done = true in the replay memory, just set the index of y_true corresponding 
            #to the action taken to the reward received
            if theMemory[3]
                y_true[j,theMemory[1]] = theMemory[2]
            else
                #in addition to setting the index of y_true corresponding to the 
                #action taken to the reward received, add the value of Q for the next state,
                #discounted by gamma
                reshapedSequence = theMemory[4].reshape(1,height*numFrames,width,color)
                futureQ = model.predict(reshapedSequence,batch_size=1)
                y_true[j,:] = gamma*futureQ
                y_true[j,theMemory[1]] += theMemory[2]
        model.train_on_batch(batchedStates,y_true)
        
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
