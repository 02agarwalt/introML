from cs475_types import Predictor

class Pegasos(Predictor):
    def __init__(self, instances, learning_rate, iterations, pegasos_lambda):
        super(Pegasos, self).__init__()
        self._learning_rate = learning_rate
        self._iterations = iterations
        self._w = {1:0}
        self._lambda = pegasos_lambda
        self.train(instances)
        pass

    def train(self, instances):
        timestamp = 1.0
        for k in range(0, self._iterations):
            for i in instances:
                # dot product < w , x > calculation
                dot_prod = 0
                for key, value in i._feature_vector.vector.iteritems():
                    if not self._w.has_key(key):
                        self._w[key] = 0.0
                    dot_prod = dot_prod + self._w[key]*value

                label = 0.0
                if (i._label.label == 1): label = 1.0
                if (i._label.label == 0): label = -1.0
                
                # update w
                #for key, value in i._feature_vector.vector.iteritems():
                #    self._w[key] = self._w[key] * (1.0 - (1.0/timestamp))
                    
                #    if (float(dot_prod) * label < 1):
                #        self._w[key] = self._w[key] + (label*float(value))*(1/(timestamp*self._lambda))

                # update w
                for key, value in self._w.iteritems():
                    self._w[key] = value * (1.0 - (1.0/timestamp))
                    
                    if dot_prod * label < 1 and i._feature_vector.vector.has_key(key):
                        x = i._feature_vector.vector[key]
                        self._w[key] = self._w[key] + (label*x)*(1/(timestamp*self._lambda))


                timestamp = timestamp + 1

        pass

    def predict(self, instance):
        # calculate dot product
        pred = 0
        for key, value in instance._feature_vector.vector.iteritems():
            if not self._w.has_key(key):
                self._w[key] = 0
            pred = pred + self._w[key]*value
        
        if pred < 0: 
            return 0
        else:
            return 1
