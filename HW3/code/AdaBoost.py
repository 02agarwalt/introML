from cs475_types import Predictor
import math

class AdaBoost(Predictor):
    def __init__(self, instances, num_boosting_iterations):
        super(AdaBoost, self).__init__()
        self._instances = instances
        self._num_boosting_iterations = num_boosting_iterations
        self._max_feature_index = 0
        self._D = {}
        self._H = {}
        #self._A = np.empty(num_boosting_iterations, dtype=float)
        self._A = {}
        self._num_instances = len(instances)
        self._num_realized_iterations = num_boosting_iterations

        for i in instances:
            self._D[i] = 1.0/self._num_instances
            for key, value in i._feature_vector.vector.iteritems():
                if key > self._max_feature_index:
                    self._max_feature_index = key

        self._newCs = {}

        for j in range(1, self._max_feature_index + 1):
            Cs = {}
            for i in instances:
                if i._feature_vector.vector.has_key(j):
                    value = i._feature_vector.vector[j]
                    Cs[value] = value
                else:
                    Cs[0] = 0
            Cs = sorted(Cs)
            newCs = []
            for x in range(len(Cs)-1):
                avg = (Cs[x] + Cs[x+1])/2.0
                newCs.append(avg)
            if (len(newCs) == 0):
                newCs.append(Cs[0])
            self._newCs[j] = newCs

        self.train(instances)
        pass

    def err(self, j, c, direction1, direction2):
        error = 0
        for i in self._instances:
            xij = 0
            if i._feature_vector.vector.has_key(j):
                xij = i._feature_vector.vector[j]
            pred = direction2
            if (xij > c):
                pred = direction1
            
            curr_error = 0
            if not (pred == (2*i._label.label - 1)):
                curr_error = self._D[i]
            error = error + curr_error
        return error

    def train(self, instances):
        for t in range(1, self._num_boosting_iterations + 1):
            # pick some h_t
            h_j = 0.0
            h_c = 0.0
            h_direction1 = 0.0
            h_direction2 = 0.0
            min_error = float("inf")
            for j in range(1, self._max_feature_index + 1):
                for c in self._newCs[j]:
                    for direc1 in [1.0, -1.0]:
                        for direc2 in [1.0, -1.0]:
                            error = self.err(j, c, direc1, direc2)
                            if error < min_error:
                                min_error = error
                                h_j = j
                                h_c = c
                                h_direction1 = direc1
                                h_direction2 = direc2
            
            # store h
            self._H[t] = (h_j, h_c, h_direction1, h_direction2)


            # stopping critera
            if min_error < 0.000001:
                self._num_realized_iterations = t - 1
                if (t == 1):
                    self._num_realized_iterations = 1
                    self._A[1] = 1
                return
                
            # calculate alpha
            self._A[t] = 0.5*math.log((1-min_error)/min_error)
            # calculate Z
            Z = 0
            for i in instances:
                # calculate h(xi)
                xij = 0
                if i._feature_vector.vector.has_key(h_j):
                    xij = i._feature_vector.vector[h_j]
                h = h_direction2
                if (xij > h_c):
                    h = h_direction1
                
                label = (2 * i._label.label) - 1
                Z = Z + self._D[i] * math.exp(-1*self._A[t]*label*h)
                
            
            # calculate new D
            for i in instances:
                label = (2 * i._label.label) - 1
                # calculate h(xi)
                xij = 0
                if i._feature_vector.vector.has_key(h_j):
                    xij = i._feature_vector.vector[h_j]
                h = h_direction2
                if (xij > h_c):
                    h = h_direction1
                
                # update D
                self._D[i] = self._D[i] * math.exp(-1*self._A[t]*label*h)
                self._D[i] = self._D[i] / Z

            if (t == t):
            #    print "D = ", self._D 
            #    print "maxfi = ", self._max_feature_index
            #    print "newCs = ", self._newCs[h_j]
                print "iteration = ", t
                print "h_j = ", h_j
                print "h_c = ", h_c
                print "direc1 = ", h_direction1
                print "direc2 = ", h_direction2
                print "error = ", min_error
                print "A = ", self._A
            #    print "H = ", self._H
                print "Z = ", Z
        pass

    def predict(self, instance):
        pred = 0
        for t in range(1, self._num_realized_iterations + 1):
            h_j = self._H[t][0]
            h_c = self._H[t][1]
            h_direction1 = self._H[t][2]
            h_direction2 = self._H[t][3]
            xij = 0
            if instance._feature_vector.vector.has_key(h_j):
                xij = instance._feature_vector.vector[h_j]
            h = h_direction2
            if (xij > h_c):
                h = h_direction1
                
            pred = pred + (self._A[t] * h)
        
        if (pred < 0):
            return 0
        else:
            return 1
