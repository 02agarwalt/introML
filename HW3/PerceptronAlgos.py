from cs475_types import Predictor

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

class MargPerceptron(Predictor):
    def __init__(self, instances, learning_rate, iterations):
        super(MargPerceptron, self).__init__()
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
                
                yi = 0
                if (i._label.label == 1): yi = 1
                if (i._label.label == 0): yi = -1

                if (pred * yi < 1):
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
