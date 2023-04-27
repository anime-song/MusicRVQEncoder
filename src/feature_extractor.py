import tensorflow as tf
from tensorflow.keras import layers as L


class FeatureExtractorLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        filter_sizes,
        kernel_sizes,
        strides,
        is_gelu_approx=False,
        layer_id=0,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.filter_sizes = filter_sizes
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.is_gelu_approx = is_gelu_approx
        self.layer_id = layer_id
        
        conv_dim = filter_sizes[layer_id]
        kernel_size = kernel_sizes[layer_id]
        stride = strides[layer_id]

        self.conv_layer = L.Conv1D(
            conv_dim,
            kernel_size,
            strides=stride,
            use_bias=False,
            padding="same",
            kernel_initializer="he_normal"
        )

    def call(self, inputs, training=False):
        inputs = self.conv_layer(inputs)
        inputs = tf.keras.activations.gelu(inputs, approximate=self.is_gelu_approx)
        return inputs
    
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filter_sizes": self.filter_sizes,
                "kernel_sizes": self.kernel_sizes,
                "strides": self.strides,
                "is_gelu_approx": self.is_gelu_approx,
                "layer_id": self.layer_id,
            }
        )
        return config
