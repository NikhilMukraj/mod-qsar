from tensorflow.keras import models
import numpy as np


def load_model(filename):
    return models.load_model(filename)

class Model:
    def __init__(self, model=None):
        self.model = load_model(model)

    def compile(self):
        self.model.compile()

class ClassifierModel(Model):
    def __init__(self, model=None):
        super().__init__(model)

    def predict(self, sequence):
        return self.model.predict(sequence, verbose=0)

class RegressionModel(Model):
    def __init__(self, model=None, maximum_value=500, minimum_value=0):
        super().__init__(model)
        self.maximum_value = maximum_value
        self.minimum_value = minimum_value

    def predict(self, sequence):
        value = self.model.predict(sequence, verbose=0)

        value = np.clip(value, self.minimum_value, self.maximum_value)
        value = (value - self.minimum_value) / (self.maximum_value - self.minimum_value)
        
        return np.power(2, value) - 1
