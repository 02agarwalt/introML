from cs475_types import Predictor
import math
import numpy as np
import sys

class LambdaMeans(Predictor):
    def __init__(self, instances, t, lam):
        super(LambdaMeans, self).__init__()
        self._num_iterations = t
        self._instances = instances
        self._N = len(instances) # number of instances
        
        # get highest feature index for future ease
        self._max_feature_index = 0
        for i in instances:
            for key, value in i._feature_vector.vector.iteritems():
                if key > self._max_feature_index:
                    self._max_feature_index = key

        # initialize mean vector of data
        self._data_mean = {}
        for index in range(1, self._max_feature_index + 1):
            sum = 0
            for i in instances:
                val = 0
                if i._feature_vector.vector.has_key(index):
                    val = i._feature_vector.vector[index]
                sum = sum + val
            mean = float(sum) / float(self._N)
            if not (mean == 0):
                self._data_mean[index] = mean

        # initialize lambda
        self._lambda = lam
        if (lam == 0):
            sum = 0
            for i in instances:
                dist = self.eucDistSquared(self._data_mean, i._feature_vector.vector)
                sum = sum + dist
            self._lambda = float(sum) / float(self._N)

        # initialize clusters
        self._K = 1
        self._means = {1:self._data_mean}
        
        self.train(instances)
        pass
    
    def train(self, instances):
        for iter in range(0, self._num_iterations):
            # E-step
            r = np.zeros((self._K, self._N))
            n = 0
            for i in instances:
                min_k = sys.maxint
                min_dist = float('inf')
                for k in range(1, self._K + 1):
                    cluster_mean = self._means[k]
                    dist = self.eucDistSquared(i._feature_vector.vector, cluster_mean)
                    if (dist < min_dist):
                        min_k = k
                        min_dist = dist
                if (min_dist <= self._lambda):
                    r[min_k - 1, n] = 1
                else:
                    newRow = np.zeros(self._N)
                    newRow[n] = 1
                    r = np.vstack([r, newRow])
                    self._K = self._K + 1
                    self._means[self._K] = i._feature_vector.vector

                n = n + 1
            # M-step
            for k in range(1, self._K + 1):
                new_mean = {}
                denominator = 0
                for n in range(0, self._N):
                    denominator = denominator + r[k - 1, n]
            
                for index in range(1, self._max_feature_index + 1):
                    numerator = 0
                    n = 0
                    for i in instances:
                        if i._feature_vector.vector.has_key(index):
                            numerator = numerator + (r[k - 1, n] * i._feature_vector.vector[index])
                        n = n + 1
                    if (not (numerator == 0)) and (not (denominator == 0)):
                        new_mean[index] = float(numerator) / float(denominator)

                self._means[k] = new_mean
        pass

    def eucDistSquared(self, x, y):
        dist = 0.0
        for i in range(1, self._max_feature_index + 1):
            x_val = 0.0
            y_val = 0.0
            if x.has_key(i):
                x_val = x[i]
            if y.has_key(i):
                y_val = y[i]

            dist = dist + pow((y_val - x_val), 2)
        return dist

    def predict(self, instance):
        min_k = sys.maxint
        min_dist = float('inf')
        for k in range(1, self._K + 1):
            cluster_mean = self._means[k]
            dist = self.eucDistSquared(instance._feature_vector.vector, cluster_mean)
            if (dist < min_dist):
                min_k = k
                min_dist = dist
        return min_k
