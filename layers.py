import os
import keras
import numpy as np
import gym
import matplotlib.pyplot as plt
import argparse

# Stop complaining about my CPU's Power Level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='Pong-v1_1eddef71-6471-4334-9744-3bf2dd3c93d4')
args = parser.parse_args()

print("args.model: ", args.model)

model = keras.models.load_model(args.model)

#for index in range(len(model.layers)):
#	print("Layer["+str(index)+"]: ", model.layers[index].output)
#	print("Weights[0]: ", model.layers[index].get_weights()[0])

print("Layer[0]: ", model.layers[0].output)
print("Weights[0]: ", model.layers[0].get_weights()[0])

