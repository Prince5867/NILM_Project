
import tf_keras as keras
from keras.layers import Layer, Conv1D, Multiply, Activation, Input, Dense, MaxPooling1D, Dropout, LSTM, Concatenate
from keras.models import Model
import tensorflow as tf

import keras
from keras.layers import Layer, Add, Conv1D, Multiply

# Register the custom layer for saving/loading without needing custom_objects
@keras.saving.register_keras_serializable(package="MyLayers")
class Conv1DGLU(Layer):
    def __init__(self, filters, kernel_size, **kwargs):
        super(Conv1DGLU, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.conv_linear = Conv1D(filters, kernel_size, padding='same')
        self.conv_gate = Conv1D(filters, kernel_size, padding='same')

    def call(self, inputs):
        linear_out = self.conv_linear(inputs)
        gate_out = keras.activations.relu(self.conv_gate(inputs), max_value=6) / 6
        return Multiply()([linear_out, gate_out])

    def get_config(self):
        print("it works")
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size
        })
        return config

    @classmethod
    def from_config(cls, config):
        # config["filters"] = keras.saving.deserialize_keras_object(config["filters"])
        # config["kernel_size"] = keras.saving.deserialize_keras_object(config["kernel_size"])
        return cls(**config)

