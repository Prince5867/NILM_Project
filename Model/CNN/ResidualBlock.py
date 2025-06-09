from keras.layers import Layer, Add, Conv1D

class ResidualBlock(Layer):
    def __init__(self, sublayer, filters, **kwargs):
        """
        Residual block that applies the given sublayer and adds a skip connection.
        
        Args:
            sublayer (Layer): A Keras layer instance (e.g., Conv1DGLU).
            filters (int): Number of output filters of the sublayer.
        """
        super(ResidualBlock, self).__init__(**kwargs)
        self.sublayer = sublayer
        self.projection = Conv1D(filters, kernel_size=1, padding='same')

    def call(self, inputs):
        x = self.sublayer(inputs)

        # Match the input to the output if necessary
        if inputs.shape[-1] != x.shape[-1]:
            inputs = self.projection(inputs)

        return Add()([x, inputs])
