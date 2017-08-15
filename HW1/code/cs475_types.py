from abc import ABCMeta, abstractmethod

# abstract base class for defining labels
class Label:
    __metaclass__ = ABCMeta

    @abstractmethod
    def __str__(self): pass

       
class ClassificationLabel(Label):
    def __init__(self, label):
        self.label = label
        pass

    def get(self):
        return self.label
        
    def __str__(self):
        return str(self.label)
        

class FeatureVector:
    def __init__(self):
        self.vector = {}
        pass
        
    def add(self, index, value):
        self.vector[index] = value
        pass
        
    def get(self, index):
        return self.vector.get(index, default=0.0)
    
    def __iter__(self):
        return iter(self.vector.iteritems())
        

class Instance:
    def __init__(self, feature_vector, label):
        self._feature_vector = feature_vector
        self._label = label

# abstract base class for defining predictors
class Predictor:
    __metaclass__ = ABCMeta

    @abstractmethod
    def train(self, instances): pass

    @abstractmethod
    def predict(self, instance): pass

       
