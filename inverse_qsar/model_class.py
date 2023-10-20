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
    def __init__(self, model=None):
        super().__init__(model)

    def predict(self, sequence):
        value = self.model.predict(sequence, verbose=0)

        value = np.clip(value, 0, 500)
        value = (value - 0) / (500 - 0)
        
        return np.power(2, value) - 1
