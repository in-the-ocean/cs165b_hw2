#!/usr/bin/env python3

import math
import numpy as np
from time import time
from collections import Counter

SETOSA = "Iris-setosa"

class Data:
    def __init__(self):
        self.features = []	# list of lists (size: number_of_examples x number_of_features)
        self.labels = []	# list of strings (lenght: number_of_examples)

    def addSample(self, sample):
        sample = sample.strip()
        feature = sample.split(",")
        for i in range(4):
            feature[i] = float(feature[i])
        label = feature[-1]

        if label == SETOSA:
            l = 1
        else:
            l = -1
        self.features.append(feature[0:4] + [1])
        self.labels.append(l)


def read_data(path):
    data = Data()
    # TODO: function that will read the input file and store it in the data structure
    # use the Data class defined above to store the information
    with open(path) as f:
        for line in f:
            data.addSample(line)
    return data

def dot_kf(u, v):
    """
    The basic dot product kernel returns u*v.

    Args:
        u: list of numbers
        v: list of numbers

    Returns:
        u*v
    """
    # TODO: implement the kernel function
    product = 0
    for i in range(len(u)):
        product += u[i] * v[i]
    return product

def poly_kernel(d):
    """
    The polynomial kernel.

    Args:
        d: a number

    Returns:
        A function that takes two vectors u and v,
        and returns (u*v+1)^d.
    """
    def kf(u, v):
        # TODO: implement the kernel function
        return pow(dot_kf(u,v) + 1, d)
    return kf

def exp_kernel(s):
    """
    The exponential kernel.

    Args:
        s: a number

    Returns:
        A function that takes two vectors u and v,
        and returns exp(-||u-v||/(2*s^2))
    """
    def kf(u, v):
        # TODO: implement the kernel function
        dist = 0
        for i in range(len(u)):
            dist += (u[i] - v[i]) ** 2
        return math.exp(-(dist/2*s*s))
    return kf

class Perceptron:
    def __init__(self, kf, lr):
        """
        Args:
            kf - a kernel function that takes in two vectors and returns
            a single number.
        """
        self.MissedPoints = []
        self.MissedLabels = []
        self.w = []
        self.kf = kf
        self.lr = lr

    def train(self, data):
        # TODO: Main function - train the perceptron with data
        converged = False
        while not converged:
            converged = True
            for p in range(len(data.features)):
                if self.update(data.features[p], data.labels[p]):
                    self.MissedLabels.append(data.labels[p])
                    self.MissedPoints.append(data.features[p])
                    converged = False
        self.w = [0 for i in range(5)]
        for i in range(len(self.MissedLabels)):
            for j in range(5):
                self.w[j] += self.MissedPoints[i][j] * self.MissedLabels[i]


    def update(self, point, label):
        """
        Updates the parameters of the perceptron, given a point and a label.

        Args:
            point: a list of numbers
            label: either 1 or -1

        Returns:
            True if there is an update (prediction is wrong),
            False otherwise (prediction is accurate).
        """
        # TODO
        score = 0
        for i in range(len(self.MissedPoints)):
            score += self.kf(self.MissedPoints[i], point) * self.MissedLabels[i]
        score *= label

        return score <= 0 

    def predict(self, point):
        """
        Given a point, predicts the label of that point (1 or -1).
        """
        # TODO
        score = 0
        for i in range(len(self.MissedPoints)):
            score += self.kf(self.MissedPoints[i], point) * self.MissedLabels[i]
        print(score)
        if score >= 0:
            return 1
        else:
            return -1

    def test(self, data):
        predictions = []
        # TODO: given data and a perceptron - return a list of integers (+1 or -1).
        # +1 means it is Iris Setosa
        # -1 means it is not Iris Setosa
        for d in data.features:
            predictions.append(self.predict(d))
        return predictions


# Feel free to add any helper functions as needed.
if __name__ == '__main__':
    train = read_data("hw2_train.txt")
    test = read_data("hw2_test.txt")
    perceptron = Perceptron(exp_kernel(1), 0.5)
    perceptron.train(train)
    predict = perceptron.test(test)

    correct = 0
    for i in range(len(predict)):
        if predict[i] == test.labels[i]:
            correct += 1
    print(correct/len(predict))
