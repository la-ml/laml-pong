import keras

import numpy as np

import gym

import matplotlib.pyplot as plt

model = keras.models.load_model('Pong-v1_79360087-af11-4db5-9878-12158cfee1bd')

env = gym.make('Pong-v0')

state = env.reset()

done = False

height, width, color = len(state), len(state[0]), len(state[0][0])

color = 1

possibleActions = [1, 2, 3]
numFrames = 4
frameSequence = np.zeros((height*numFrames, width, color))
#frameSequence[:height,:,:] = state
#frameSequence = np.random.randint(0,255,(1,height*numFrames,width,color))
framesFilled = 0

epsilon = 0.1

pause = 25

pause_action = 0
do_up = 0

plt.ion()

greyscale_state = None

while not done:
    greyscale_state = np.reshape(state[:,:,1], (height, width, 1))/255

    if (framesFilled < numFrames):
        frameSequence[height*framesFilled:height*(framesFilled+1),:,:] = greyscale_state
        framesFilled += 1
    else:
        #shift all the frames down, and then add the new frame
        frameSequence[:height*(numFrames-1),:,:] = frameSequence[height:,:,:]         
        frameSequence[height*(numFrames-1):,:,:] = greyscale_state

#    frameSequence = np.random.randint(0,255,(height*numFrames,width,color))

    
    #plt.clf()
    #plt.imshow(frameSequence)
    #plt.draw()

    #input("Press the Any Key")


    reshapedSequence = frameSequence.reshape(1, height*numFrames, width, color)
    model_prediction = model.predict(reshapedSequence, batch_size=1)

    #if (np.random.random() < epsilon):
    #    action = int(np.random.random() * 3)
    #else:
    #    action = np.argmax(model_prediction)

    action = np.argmax(model_prediction)

    if (do_up < pause):
        action = pause_action
        do_up += 1

    print("Sequence Sum: ", reshapedSequence.sum())
    print("Frame Sequence Sum: ", frameSequence.sum())
    print("Model Predition: ", model_prediction)
    print("Action: ", action)
    #env.render('human')
    state, _, done, meta = env.step(possibleActions[action])
