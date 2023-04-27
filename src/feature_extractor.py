import tensorflow as tf
from tensorflow.keras import layers as L
from vector_quantization import ResidualVQ


class FeatureExtractorLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        filter_sizes,
        kernel_sizes,
        strides,
        codebook_sizes,
        commitment_costs,
        num_quantizers_list,
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

        self.codebook_sizes = codebook_sizes
        self.commitment_costs = commitment_costs
        self.num_quantizers_list = num_quantizers_list
        
        conv_dim = filter_sizes[layer_id]
        kernel_size = kernel_sizes[layer_id]
        stride = strides[layer_id]

        self.residual_vq = ResidualVQ(
            codebook_size=codebook_sizes[layer_id],
            embedding_dim=conv_dim,
            commitment_cost=commitment_costs[layer_id],
            num_quantizers=num_quantizers_list[layer_id])

        self.conv_layer = L.Conv1D(
            conv_dim,
            kernel_size,
            strides=stride,
            use_bias=False,
            padding="same",
            kernel_initializer="he_normal"
        )

    def call(self, inputs, training=False):
        vq_output = self.residual_vq(inputs, training=training)
        inputs = vq_output['quantized_out']

        inputs = self.conv_layer(inputs)
        inputs = tf.keras.activations.gelu(inputs, approximate=self.is_gelu_approx)
        return inputs, vq_output
    
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filter_sizes": self.filter_sizes,
                "kernel_sizes": self.kernel_sizes,
                "strides": self.strides,
                "is_gelu_approx": self.is_gelu_approx,
                "layer_id": self.layer_id,
                "codebook_sizes": self.codebook_sizes,
                "commitment_costs": self.commitment_costs,
                "num_quantizers_list": self.num_quantizers_list
            }
        )
        return config
