# WKNEIL

# Import libraries
import random
import numpy as np
import time
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import pairwise_distances
from scipy import linalg

class SELM(object):

    def __init__(self):
        pass

    def train(self, trainingXA, trainingXB, trainingY, **kwargs):
        # Check parameters
        modelParams = {}
        modelParams['hiddenNode'] = 1.0
        modelParams['regC'] = 1
        modelParams['randomseed'] = 38
        modelParams['distanceFunc'] = 'euclidean'
        modelParams['trainingDataID'] = np.array(range(0, trainingXA.shape[0]))
        modelParams['siameseCondition'] = 'sum'
        for key, value in kwargs.items():
            if key in modelParams:
                modelParams[key] = value
            else:
                raise Exception('Error key ({}) exists in dict'.format(key))

        tic = time.perf_counter()
        
        # Calculate Siamese layer
        trainingX = self.siamese_layer(trainingXA, trainingXB, modelParams['siameseCondition'])
        
        # Random selected weights of the networks
        [weights, trainingWeightDataID] = self.initHidden(trainingX, modelParams['trainingDataID'], modelParams['hiddenNode'], modelParams['randomseed'])
        
        # Train model
        [beta, label_classes] = self.trainModel(trainingX, trainingY, weights, modelParams['regC'], modelParams['distanceFunc'])
        
        # Timer
        toc = time.perf_counter()
        run_time = toc-tic

        return weights, trainingWeightDataID, beta, label_classes, run_time
    

    def trainModel(self, trainingDataX, trainingDataY, weights, regC, distanceFunc):
        # Calculate kernel
        kernel_mat = self.calculate_kernel(trainingDataX, weights, distanceFunc)
        del trainingDataX, weights

        # integer encode
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(trainingDataY)

        # binary encode
        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        trainingDataY_onehot = onehot_encoder.fit_transform(integer_encoded)

        # Balance classes
        label_classes = label_encoder.classes_
        class_freq = trainingDataY_onehot.sum(axis=0)
        max_freq = max(class_freq)
        penalized_value = np.sqrt(max_freq / class_freq)
        penalized_array = penalized_value[integer_encoded]
        
        # Optimize output weights Beta
        H = np.matrix(np.multiply(penalized_array, kernel_mat))
        regC_mat = np.matrix((1/regC) * np.identity(H.shape[1]))
        Two_H = H.T * H
        inv_data = Two_H + regC_mat
        del Two_H, regC_mat
        inv_data = np.matrix(linalg.inv(inv_data))
        penalized_trainingDataY_onehot = trainingDataY_onehot * penalized_value
        del trainingDataY_onehot, penalized_value
        penalized_trainingDataY_onehot = H.T * penalized_trainingDataY_onehot
        del H
        beta = inv_data * penalized_trainingDataY_onehot

        return beta, label_classes

    def siamese_layer(self, XA, XB, condition):
        # Calculate Siamese condition
        if condition == 'sum':
            return XA + XB
        elif condition == 'dist':
            return np.absolute(XA - XB)
        elif condition == 'multiply':
            return np.multiply(XA, XB)
        elif condition == 'mean':
            return (XA + XB)/2
    
    def initHidden(self, trainingDataX, trainingDataID, hiddenNode, randomseed):
        # Calculate number of hidden nodes
        if hiddenNode <= 0:
            raise Exception('The range of hidden node is wrong')
        else:
            # Quantity number of hidden nodes
            if isinstance(hiddenNode, int):
                hiddenNodeNum = hiddenNode
            else:
                # Convert hidden node in percent to quantity
                hiddenNodeNum = round(hiddenNode * trainingDataX.shape[0])
        
        # Limit the node equal to number of samples
        hiddenNodeNum = min(hiddenNodeNum, trainingDataX.shape[0])
        if not isinstance(hiddenNodeNum, int):
            hiddenNodeNum = hiddenNodeNum.astype(int)
        
        # random.seed( randomseed )
        weightIdx = random.Random(38).sample(range(trainingDataX.shape[0]), hiddenNodeNum)
        weights = trainingDataX[weightIdx, :]
        trainingWeightDataID = trainingDataID[weightIdx]
        return weights, trainingWeightDataID

    def predict(self, testDataXA, testDataXB, weights, beta, label_classes, kernelFunc='euclidean', siameseCondition='sum'):
        # Predictt
        tic = time.perf_counter()
        testDataX = self.siamese_layer(testDataXA, testDataXB, siameseCondition)
        kernel_mat = self.calculate_kernel(testDataX, weights, kernelFunc)
        predictedScores = kernel_mat * beta
        toc = time.perf_counter()
        predictedY = np.argmax(predictedScores, axis=1)
        predictedY = label_classes[predictedY]
        
        # Timer
        run_time = toc-tic

        return predictedScores, predictedY.squeeze(), run_time

    def calculate_kernel(self, m1, m2, kernelFunc):
        if kernelFunc == 'euclidean':
            kernal_matrix = pairwise_distances(m1, m2, metric=kernelFunc)
        return kernal_matrix
