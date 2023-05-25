import tensorflow as tf


class WeightNormDense(tf.keras.layers.Dense):
    def build(self, input_shape):
        super().build(input_shape)
        self.g = self.add_weight(
            name='g',
            shape=[self.units, ],
            initializer='one',
            dtype=self.dtype,
            trainable=True)

    def call(self, inputs):
        kernel = self.kernel * self.g / \
            tf.keras.backend.sqrt(tf.keras.backend.sum(tf.keras.backend.square(self.kernel), axis=0))
        output = tf.keras.backend.dot(inputs, kernel)

        if self.use_bias:
            output = tf.keras.backend.bias_add(output, self.bias)

        if self.activation is not None:
            output = self.activation(output)

        return output

    def get_config(self):
        base_config = super(WeightNormDense, self).get_config()
        return dict(list(base_config.items()))
    

class ReGLU(tf.keras.layers.Layer):
    def __init__(self, bias=True, dim=-1, **kwargs):
        super(ReGLU, self).__init__(**kwargs)
        self.bias = bias
        self.dim = dim

    def call(self, x):
        out, gate = tf.split(x, num_or_size_splits=2, axis=self.dim)
        gate = tf.nn.relu(gate)
        x = tf.multiply(out, gate)
        return x
    
    def get_config(self):
        config = {
            'bias': self.bias,
            'dim': self.dim
            }
        base_config = super(ReGLU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))