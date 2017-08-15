from cs475_types import Predictor
import math
import numpy as np
import sys
from scipy.stats import multivariate_normal as mvn
import scipy.stats

class NaiveBayesClustering(Predictor):
    def __init__(self, instances, t, num_clusters):
        super(NaiveBayesClustering, self).__init__()
        self._num_iterations = t
        self._instances = instances
        self._N = len(instances) # number of instances
        self._K = num_clusters
        
        # get highest feature index for future ease
        self._max_feature_index = 0
        for i in instances:
            for key, value in i._feature_vector.vector.iteritems():
                if key > self._max_feature_index:
                    self._max_feature_index = key
        
        # initialize clusters
        self._means = {}
        self._variances = {}
        self._phi = {}
        clusters = {k:[] for k in range(0, self._K)}
        data_mean = self.mean(instances)
        self._S = self.Svariance(instances, data_mean)
        n = 0
        for i in instances:
            k = n % self._K
            (clusters[k]).append(i)
            n = n + 1
        for k in range(0, self._K):
            self._means[k] = self.mean(clusters[k])
            self._phi[k] = float(len(clusters[k]) + 1) / float(self._N + self._K)
            if len(clusters[k]) <= 1:
                self._variances[k] = self._S
            else:
                self._variances[k] = self.variance(clusters[k], self._means[k])

        self.train(instances)
        pass
    
    def train(self, instances):
        for iter in range(0, self._num_iterations):
            # E-step
            r = np.zeros((self._K, self._N))
            n = 0
            for i in instances:
                max_k = -1
                max_ll = float('-inf')
                for k in range(0, self._K):
                    logphi = math.log(self._phi[k])
                    sum = 0
                    for j in range(1, self._max_feature_index + 1):
                        x_ij = 0
                        u = 0
                        sig = self._S[j]
                        if i._feature_vector.vector.has_key(j):
                            x_ij = i._feature_vector.vector[j]
                        if self._means[k].has_key(j):
                            u = self._means[k][j]
                        if self._variances[k].has_key(j):
                            sig = self._variances[k][j]
                        #val = scipy.stats.norm(u, sig).logpdf(x_ij)
                        val = self.lognormalpdf(x_ij, u, sig)
                        sum = sum + val
                    curr_ll = sum + logphi

                    if (curr_ll > max_ll):
                        max_k = k
                        max_ll = curr_ll
                r[max_k, n] = 1
                n = n + 1
            # M-step
            self._phi = {}
            self._means = {}
            self._variances = {}
            for k in range(0, self._K): # update phi and means
                count = 0
                for n in range(0, self._N):
                    count = count + r[k, n]
                self._phi[k] = float(count + 1) / float(self._N + self._K)
                
                self._means[k] = {}
                for j in range(1, self._max_feature_index + 1):
                    sum = 0
                    for n in range(0, self._N):
                        val = 0
                        if (r[k, n] == 1) and (instances[n]._feature_vector.vector.has_key(j)):
                            val = instances[n]._feature_vector.vector[j]
                        sum = sum + val
                    val = float(sum) / float(count)
                    if not (val == 0):
                        self._means[k][j] = val
            
            for k in range(0, self._K): # update variances
                self._variances[k] = {}
                count = 0
                for n in range(0, self._N):
                    count = count + r[k, n]
                
                if (count <= 1):
                    self._variances[k] = self._S
                else:
                    for j in range(1, self._max_feature_index + 1):
                        mean_val = 0
                        if self._means[k].has_key(j):
                            mean_val = self._means[k][j]
                        sum = 0
                        for n in range(0, self._N):
                            if (r[k, n] == 1):
                                x_ij = 0
                                if instances[n]._feature_vector.vector.has_key(j):
                                    x_ij = instances[n]._feature_vector.vector[j]
                                val = (x_ij - mean_val)**2
                                sum = sum + val
                        val = float(sum) / float(count - 1)
                        if (val < self._S[j]):
                            self._variances[k][j] = self._S[j]
                        else:
                            self._variances[k][j] = val

        pass

    def predict(self, instance):
        i = instance
        max_k = -1
        max_ll = float('-inf')
        for k in range(0, self._K):
            logphi = math.log(self._phi[k])
            sum = 0
            for j in range(1, self._max_feature_index + 1):
                x_ij = 0
                u = 0
                sig = self._S[j]
                if i._feature_vector.vector.has_key(j):
                    x_ij = i._feature_vector.vector[j]
                if self._means[k].has_key(j):
                    u = self._means[k][j]
                if self._variances[k].has_key(j):
                    sig = self._variances[k][j]
                #val = scipy.stats.norm(u, sig).logpdf(x_ij)
                val = self.lognormalpdf(x_ij, u, sig)
                #val = math.log(val)
                sum = sum + val
            curr_ll = sum + logphi
            if (curr_ll > max_ll):
                max_k = k
                max_ll = curr_ll
        return max_k

    def mean(self, instances):
        mean = {}
        N = len(instances)
        for index in range(1, self._max_feature_index + 1):
            sum = 0
            for i in instances:
                val = 0
                if i._feature_vector.vector.has_key(index):
                    val = i._feature_vector.vector[index]
                sum = sum + val
            temp = float(sum) / float(N)
            if not (temp == 0):
                mean[index] = temp
        return mean

    def variance(self, instances, mean):
        variance = {}
        N = len(instances)
        for index in range(1, self._max_feature_index + 1):
            mean_val = 0
            if mean.has_key(index):
                mean_val = mean[index]
            sum = 0
            for i in instances:
                x_ij = 0
                if i._feature_vector.vector.has_key(index):
                    x_ij = i._feature_vector.vector[index]
                val = (x_ij - mean_val)**2
                sum = sum + val
            temp = float(sum) / float(N - 1)
            if (temp < self._S[index]):
                variance[index] = self._S[index]
            else:
                variance[index] = temp
        return variance

    def Svariance(self, instances, mean):
        variance = {}
        N = len(instances)
        for index in range(1, self._max_feature_index + 1):
            mean_val = 0
            if mean.has_key(index):
                mean_val = mean[index]
            sum = 0
            for i in instances:
                x_ij = 0
                if i._feature_vector.vector.has_key(index):
                    x_ij = i._feature_vector.vector[index]
                val = (x_ij - mean_val)**2
                sum = sum + val
            temp = float(sum) / float(N - 1)
            temp = 0.01 * temp
            variance[index] = temp
            if (temp == 0):
                variance[index] = 1
        return variance

    def lognormalpdf(self, x, u, var):
        temp = float(x - u)
        #temp2 = (float(1) / float(math.sqrt(2 * math.pi * abs(sig))) * math.exp(-temp * temp / float(2 * sig))
        #return math.log(temp2)

        return (float(-temp * temp) / float(2 * var)) - math.log(math.sqrt(2 * math.pi * var))
