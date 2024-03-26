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
    def __init__(self, model=None, minimum=0, maximum=500):
        super().__init__(model)
        self.minimum = minimum
        self.maximum = maximum
        self.max_scaled = 1
        self.min_scaled = 0
        self.scale_factor = (self.max_scaled - self.min_scaled) / (self.maximum - self.minimum)

    def predict(self, sequence):
        value = self.model.predict(sequence, verbose=0)
        
        value = np.clip(value, self.minimum, self.maximum)

        return ((value - self.minimum) * self.scale_factor) + self.min_scaled
