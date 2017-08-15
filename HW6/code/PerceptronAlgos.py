from cs475_types import Predictor
import numpy as np

class Perceptron(Predictor):
    def __init__(self, instances, learning_rate, iterations):
        super(Perceptron, self).__init__()
        self._learning_rate = learning_rate
        self._iterations = iterations
        self._w = {1:0}
        self.train(instances)
        pass

    def train(self, instances):
        for k in range(0, self._iterations):
            for i in instances:
                # initial prediction
                pred = 0
                for key, value in i._feature_vector.vector.iteritems():
                    if not self._w.has_key(key):
                        self._w[key] = 0
                    pred = pred + self._w[key]*value
                if pred >= 0:
                    pred = 1
                    yi = -1 # for update step
                else:
                    pred = 0
                    yi = 1 # for update step
                
                if not (pred == i._label.label):
                    for key, value in i._feature_vector.vector.iteritems():
                        self._w[key] = self._w[key] + self._learning_rate*yi*value
                
        pass

    def predict(self, instance):
        pred = 0
        for key, value in instance._feature_vector.vector.iteritems():
            if not self._w.has_key(key):
                self._w[key] = 0
            pred = pred + self._w[key]*value
        if pred >= 0:
            pred = 1
        else:
            pred = 0
        return pred

class AvgPerceptron(Predictor):
    def __init__(self, instances, learning_rate, iterations):
        super(AvgPerceptron, self).__init__()
        self._learning_rate = learning_rate
        self._iterations = iterations
        self._w = {1:0}
        self._finalw = {1:0}
        self.train(instances)
        pass

    def train(self, instances):
        for k in range(0, self._iterations):
            for i in instances:
                # initial prediction
                pred = 0
                for key, value in i._feature_vector.vector.iteritems():
                    if not self._w.has_key(key):
                        self._w[key] = 0
                        self._finalw[key] = 0
                    pred = pred + self._w[key]*value
                if pred >= 0:
                    pred = 1
                    yi = -1 # for update step
                else:
                    pred = 0
                    yi = 1 # for update step
                
                if not (pred == i._label.label):
                    for key, value in i._feature_vector.vector.iteritems():
                        self._w[key] = self._w[key] + self._learning_rate*yi*value
                # update _finalw
                for key, value in self._finalw.iteritems():
                    self._finalw[key] = self._w[key] + value
        pass

    def predict(self, instance):
        pred = 0
        for key, value in instance._feature_vector.vector.iteritems():
            if not self._finalw.has_key(key):
                self._w[key] = 0
                self._finalw[key] = 0
            pred = pred + self._finalw[key]*value
        if pred >= 0:
            pred = 1
        else:
            pred = 0
        return pred

class MCPerceptron(Predictor):
    def __init__(self, instances, iterations):
        super(MCPerceptron, self).__init__()
        self._iterations = iterations

        # get highest feature index for future ease
        self._M = 0
        self._K = 0
        for i in instances:
            for key, value in i._feature_vector.vector.iteritems():
                if key > self._M:
                    self._M = key

            if i._label.label > self._K:
                self._K = i._label.label

        self._w = np.zeros((self._M + 1, self._K + 1))
        self.train(instances)
        pass

    def train(self, instances):
        for iters in range(0, self._iterations):
            for i in instances:
                # initial prediction
                max_k = 1
                max_sum = 0
                for k in range(1, self._K + 1):
                    pred = 0
                    for key, value in i._feature_vector.vector.iteritems():
                        pred = pred + self._w[key][k]*value
                    if pred > max_sum:
                        max_sum = pred
                        max_k = k
                
                if not (max_k == i._label.label):
                    f1 = np.zeros((self._M + 1, self._K + 1))
                    f2 = np.zeros((self._M + 1, self._K + 1))
                    for key, value in i._feature_vector.vector.iteritems():
                        f1[key][i._label.label] = value
                        f2[key][max_k] = value

                    self._w = self._w + f1 - f2
            
        pass

    def predict(self, instance):
        max_k = 1
        max_sum = 0
        for k in range(1, self._K + 1):
            pred = 0
            for key, value in instance._feature_vector.vector.iteritems():
                pred = pred + self._w[key][k]*value
            if pred > max_sum:
                max_sum = pred
                max_k = k
        return max_k
