from cs475_types import Predictor
import math
import operator

class StandardKNN(Predictor):
    def __init__(self, instances, k):
        super(StandardKNN, self).__init__()
        self._k = k
        self._instances = instances
        self._max_feature_index = 0
        self.train(instances)
        pass
    
    def train(self, instances):
        for i in instances:
            for key, value in i._feature_vector.vector.iteritems():
                if key > self._max_feature_index:
                    self._max_feature_index = key
        pass

    def eucDistance(self, x, y):
        dist = 0.0
        for i in range(1, self._max_feature_index + 1):
            x_val = 0.0
            y_val = 0.0
            if x.has_key(i):
                x_val = x[i]
            if y.has_key(i):
                y_val = y[i]

            dist = dist + pow((y_val - x_val), 2)
        return math.sqrt(dist)

    def predict(self, instance):
        neighbors = []
        for i in self._instances:
            dist = self.eucDistance(i._feature_vector.vector, instance._feature_vector.vector)
            neighbors.append((i, dist))
        neighbors = sorted(neighbors, key = operator.itemgetter(1))
        #neighbors.sort(key = operator.itemgetter(1)) # all instances sorted by distance
        
        freqs = {}
        for x in range(0, self._k):
            key = neighbors[x][0]._label.label
            if freqs.has_key(key):
                freqs[key] = freqs[key] + 1 # increment frequency of label
            else:
                freqs[key] = 1
            
        sorted_labels = sorted(freqs)
        
        # get max voted label
        max_voted_label = 0;
        max_voted_label_freq = 0;
        for label in sorted_labels:
            freq = freqs[label]
            if freq > max_voted_label_freq:
                max_voted_label = label
                max_voted_label_freq = freq
        
        return max_voted_label


class DistanceKNN(Predictor):
    def __init__(self, instances, k):
        super(DistanceKNN, self).__init__()
        self._k = k
        self._instances = instances
        self._max_feature_index = 0
        self.train(instances)
        pass
    
    def train(self, instances):
        for i in instances:
            for key, value in i._feature_vector.vector.iteritems():
                if key > self._max_feature_index:
                    self._max_feature_index = key
        pass

    def eucDistance(self, x, y):
        dist = 0.0
        for i in range(1, self._max_feature_index + 1):
            x_val = 0.0
            y_val = 0.0
            if x.has_key(i):
                x_val = x[i]
            if y.has_key(i):
                y_val = y[i]

            dist = dist + pow((y_val - x_val), 2)
        return math.sqrt(dist)

    def predict(self, instance):
        neighbors = []
        for i in self._instances:
            dist = self.eucDistance(i._feature_vector.vector, instance._feature_vector.vector)
            neighbors.append((i, dist))
        neighbors = sorted(neighbors, key = operator.itemgetter(1))
        #neighbors.sort(key = operator.itemgetter(1)) # all instances sorted by distance
        
        freqs = {}
        for x in range(0, self._k):
            key = neighbors[x][0]._label.label
            if freqs.has_key(key):
                freqs[key] = freqs[key] + (1.0/(1.0 + pow(neighbors[x][1], 2))) # update vote count
            else:
                freqs[key] = 1.0/(1.0 + pow(neighbors[x][1], 2))
            
        sorted_labels = sorted(freqs)
        
        # get max voted label
        max_voted_label = 0;
        max_voted_label_freq = 0;
        for label in sorted_labels:
            freq = freqs[label]
            if freq > max_voted_label_freq:
                max_voted_label = label
                max_voted_label_freq = freq
        
        return max_voted_label
