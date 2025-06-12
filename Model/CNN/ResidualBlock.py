from keras.layers import Layer, Add, Conv1D
import keras

@keras.saving.register_keras_serializable(package="MyLayers")
class ResidualBlock(Layer):
    def __init__(self, sublayer, filters, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.sublayer = sublayer
        self.filters = filters
        self.projection = Conv1D(filters, kernel_size=1, padding='same')

    def call(self, inputs):
        x = self.sublayer(inputs)
        if inputs.shape[-1] != x.shape[-1]:
            inputs = self.projection(inputs)
        return Add()([x, inputs])

    def get_config(self):
        config = super().get_config()
        config.update({
            # "sublayer": keras.saving.serialize_keras_object(self.sublayer),
            "sublayer": keras.saving.serialize_keras_object(self.sublayer),
            "filters": self.filters,
        })
        return config

    @classmethod
    def from_config(cls, config):
        config["sublayer"] = keras.saving.deserialize_keras_object(config["sublayer"])
        # config["filters"] = keras.saving.deserialize_keras_object(config["filters"])
        # sublayer = keras.saving.deserialize_keras_object(config.pop("sublayer"))
        return cls(**config)
