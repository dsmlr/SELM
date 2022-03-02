# WKNEIL

# Import libraries
import os
import pandas as pd
import numpy as np
import random

# Path
selm_path = os.getcwd() + os.sep

# Load dataset - Labeled Faces in the Wild (pairsDevTest)
anchor_features = pd.read_csv(selm_path + 'anchor_features.csv', header=None, sep=',').to_numpy()
compared_features = pd.read_csv(selm_path + 'compared_features.csv', header=None, sep=',').to_numpy()
pairing_id = pd.read_csv(selm_path + 'pairing_id.csv', header=None, sep=',').squeeze().to_numpy()
labels = pd.read_csv(selm_path + 'labels.csv', header=None, sep=',').squeeze().to_numpy()

# Shuffle training and test data
shufflePosIdx = np.arange(0, 500, 1)
shuffleNegIdx = np.arange(500, 1000, 1)
random.Random(38).shuffle(shufflePosIdx)
random.Random(38).shuffle(shuffleNegIdx)

# Assign training and test data
trainingIdx = np.append(shufflePosIdx[0:400], shuffleNegIdx[0:400])
testIdx = np.append(shufflePosIdx[400:], shuffleNegIdx[400:])

# Initial SELM
from selm import SELM
selm_model = SELM()

# Train SELM
weights, trainingWeightDataID, beta, label_classes, run_time = selm_model.train(anchor_features[trainingIdx], compared_features[trainingIdx], labels[trainingIdx], trainingDataID=pairing_id[trainingIdx])

# Predict
predictedScores, predictedY, run_time = selm_model.predict(anchor_features[testIdx], compared_features[testIdx], weights, beta, label_classes, kernelFunc='euclidean', siameseCondition='sum')

# Accuracy
print('accuracy =', (np.sum(labels[testIdx] == predictedY))/predictedY.size)