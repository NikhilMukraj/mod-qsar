from tensorflow.keras import models
import numpy as np


def load_model(filename):
    return models.load_model(filename)

class Model:
    def __init__(self, model):
        self.model = load_model(model)

    def compile(self):
        self.model.compile()

class ClassifierModel(Model):
    def __init__(self, model):
        super().__init__(model)

    def predict(self, sequence):
        return self.model.predict(sequence, verbose=0)

    def get_output_shape(self):
        return self.model.layers[-1].output_shape[1]

    def get_input_shape(self):
        return self.model.layers[0].output_shape[1]

class RegressionModel(Model):
    def __init__(self, model, minimum=0, maximum=500):
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

    def get_output_shape(self):
        return self.model.layers[-1].output_shape[1]
    
    def get_input_shape(self):
        return self.model.layers[0].output_shape[1]

class MultipleClassifierModels():
    def __init__(self, models, combination_function, output_shape):
        if len(models) == 0:
            raise ValueError('Models array must have at least one model')

        self.models = [load_model(i) for i in models]

        if len(set(i.layers[0].output_shape[1] for i in self.models)) != 1:
            raise ValueError('All models must have same input shape')

        self.combine = combination_function
        self.output_shape = output_shape
    
    def compile(self):
        [i.compile() for i in self.models]

    def predict(self, sequence):
        preds = np.array([i.predict(sequence, verbose=0) for i in self.models])
        preds = self.combine(preds)

        return preds

    def get_output_shape(self):
        return self.output_shape

    def get_input_shape(self):
        return self.models[0].layers[0].output_shape[1]

class MultipleRegressionModels():
    def __init__(self, models, combination_function, output_shape, minimum=0, maximum=500):
        self.minimum = minimum
        self.maximum = maximum
        self.max_scaled = 1
        self.min_scaled = 0

        if len(models) == 0:
            raise ValueError('Models array must have at least one model')

        self.models = [load_model(i) for i in models]

        if len(set(i.layers[0].output_shape[1] for i in self.models)) != 1:
            raise ValueError('All models must have same input shape')

        self.combine = combination_function
        self.output_shape = output_shape
    
    def compile(self):
        [i.compile() for i in self.models]

    def predict(self, sequence):
        preds = np.array([i.predict(sequence, verbose=0) for i in self.models])
        preds = self.combine(preds)

        preds = np.clip(preds, self.minimum, self.maximum)

        return ((preds - self.minimum) * self.scale_factor) + self.min_scaled

    def get_output_shape(self):
        return self.output_shape

    def get_input_shape(self):
        return self.models[0].layers[0].output_shape[1]
        