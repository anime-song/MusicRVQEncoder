import tensorflow as tf
from tensorflow.keras import layers as L
from tensorflow.keras import backend as K

from vector_quantization import ResidualVQ


class WeightNormDense(L.Dense):
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
            K.sqrt(K.sum(K.square(self.kernel), axis=0))
        output = K.dot(inputs, kernel)

        if self.use_bias:
            output = K.bias_add(output, self.bias)

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
        self.dense = tf.keras.layers.Dense(2, use_bias=bias)

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
    

class PositionalEncoding(L.Layer):
    def __init__(self, feature_size, batch_size, max_shift_size=10, patch_length=8192, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.max_shift_size = max_shift_size
        self.feature_size = feature_size
        self.patch_length = patch_length
        self.batch_size = batch_size

    def call(self, inputs, training=False):
        x = inputs
        x *= tf.math.sqrt(tf.cast(tf.shape(inputs)[-1], tf.float32))

        def get_angles(pos, i, d_model):
            angle_rates = 1 / tf.math.pow(tf.cast(10000, dtype=tf.float64), (2 * (i//2)) / d_model)
            return tf.cast(angle_rates, dtype=tf.float32) * tf.cast(pos, dtype=tf.float32)

        def positional_encoding(position, d_model):
            angle_rads = get_angles(tf.range(position)[:, tf.newaxis],
                                    tf.range(d_model)[tf.newaxis, :],
                                    d_model)

            indices = tf.range(d_model)
            indices = indices[:, tf.newaxis]

            angle_rads = tf.transpose(angle_rads, (1, 0))

            # 配列中の偶数インデックスにはsinを適用; 2i
            angle_rads = tf.tensor_scatter_nd_update(angle_rads, indices[0::2, :], tf.math.sin(angle_rads[0::2, :]))

            # 配列中の奇数インデックスにはcosを適用; 2i+1
            angle_rads = tf.tensor_scatter_nd_update(angle_rads, indices[1::2, :], tf.math.cos(angle_rads[1::2, :]))

            angle_rads = tf.transpose(angle_rads, (1, 0))

            pos_encoding = angle_rads[tf.newaxis, ...]

            return tf.cast(pos_encoding, dtype=tf.float32)

        if training is False or not self.trainable or self.max_shift_size == 0:
            x += positional_encoding(tf.shape(x)[-2], self.feature_size)
            return x

        position_offset = tf.random.uniform(minval=0, maxval=self.max_shift_size, shape=(self.batch_size,), dtype=tf.int64)
        pos_enc = positional_encoding(self.patch_length + self.max_shift_size, self.feature_size)[0]

        def body_(i, body_x):
            pos_enc_slice = tf.slice(pos_enc, [position_offset[i], 0], [self.patch_length, -1])
            y = x[i] + pos_enc_slice
            return (i + 1, body_x.write(i, y))

        ind = tf.constant(0)
        v1 = tf.TensorArray(dtype=inputs.dtype, size=self.batch_size, dynamic_size=False)
        _, loop = tf.while_loop(
            cond=lambda i, x: i < self.batch_size,
            body=body_,
            loop_vars=[ind, v1]
        )

        x = loop.stack()
        return x
    
    def get_config(self):
        config = {'feature_size': self.feature_size, 'max_shift_size': self.max_shift_size}
        base_config = super(PositionalEncoding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    

class TransformerLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        hidden_size,
        num_heads,
        intermediate_size,
        batch_size,
        layer_norm_eps=1e-5,
        is_gelu_approx=False,
        dropout=0.1,
        attention_norm_type="postnorm",
        **kwargs
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.batch_size = batch_size
        self.layer_norm_eps = layer_norm_eps
        self.is_gelu_approx = is_gelu_approx
        self.dropout = dropout
        self.attention_norm_type = attention_norm_type

        self.attention_layer = L.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=hidden_size
        )

        self.dropout = L.Dropout(dropout)

        self.layer_norm = L.LayerNormalization(
            epsilon=layer_norm_eps, name="layer_norm"
        )

        self.intermediate = WeightNormDense(
            intermediate_size, name="feed_forward1", kernel_initializer="he_normal"
        )
        self.attention_output = WeightNormDense(
            hidden_size, name="feed_forward2", kernel_initializer="he_normal"
        )

        self.final_layer_norm = L.LayerNormalization(
            epsilon=layer_norm_eps,
            name="final_layer_norm"
        )

        self.conv_m_pointwise_conv = L.Conv1D(hidden_size, kernel_size=1, padding="same")
        self.conv_m_batch_norm = L.BatchNormalization()
        self.conv_m_depthwise_conv = L.DepthwiseConv1D(kernel_size=3, padding="same")
        self.conv_m_dropout = L.Dropout(dropout)
        self.conv_m_layer_norm = L.LayerNormalization(epsilon=layer_norm_eps)

    def call(self, inputs, attention_mask=None, training=False):
        # Attention
        residual = inputs
        if self.attention_norm_type == "prenorm":
            inputs = self.layer_norm(inputs)
        inputs, scores = self.attention_layer(inputs, inputs, return_attention_scores=True, attention_mask=attention_mask, training=training)
        inputs = self.dropout(inputs, training=training)
        inputs = inputs + residual
        if self.attention_norm_type == "postnorm":
            inputs = self.layer_norm(inputs)

        # conv module
        residual = inputs
        inputs = self.conv_m_pointwise_conv(inputs)
        inputs = tf.keras.activations.gelu(inputs, approximate=self.is_gelu_approx)
        inputs = self.conv_m_depthwise_conv(inputs)
        inputs = self.conv_m_batch_norm(inputs)
        inputs = self.conv_m_dropout(inputs)
        inputs = inputs + residual
        inputs = self.conv_m_layer_norm(inputs)

        # FFN
        residual = inputs
        if self.attention_norm_type == "prenorm":
            inputs = self.final_layer_norm(inputs)
        inputs = ReGLU()(self.intermediate(inputs))
        inputs = self.dropout(self.attention_output(inputs))
        inputs = inputs + residual
        if self.attention_norm_type == "postnorm":
            inputs = self.final_layer_norm(inputs)

        return inputs, scores
    
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_heads": self.num_heads,
                "intermediate_size": self.intermediate_size,
                "layer_norm_eps": self.layer_norm_eps,
                "is_gelu_approx": self.is_gelu_approx,
                "dropout": self.dropout,
                "attention_norm_type": self.attention_norm_type
            }
        )

        return config


class MaskedEncoder(tf.keras.layers.Layer):
    def __init__(
        self,
        hidden_size,
        num_heads,
        num_layers,
        intermediate_size,
        batch_size,
        patch_length,
        codebook_size,
        embedding_dim,
        num_quantizers,
        ema_decay,
        commitment_cost,
        threshold_ema_dead_code,
        temperature,
        dropout=0.1,
        layer_norm_eps=1e-5,
        is_gelu_approx=False,
        attention_norm_type="postnorm",
        **kwargs
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.intermediatee_size = intermediate_size
        self.dropout = dropout
        self.layer_norm_eps = layer_norm_eps
        self.is_gelu_approx = is_gelu_approx
        self.attention_norm_type = attention_norm_type
        self.temperature = temperature

        self.pos_embed = PositionalEncoding(
            hidden_size,
            batch_size,
            max_shift_size=1024,
            patch_length=patch_length
        )

        self.layers = [
            TransformerLayer(
                hidden_size,
                num_heads,
                intermediate_size,
                batch_size,
                layer_norm_eps,
                is_gelu_approx,
                dropout,
                attention_norm_type
            )
            for i in range(num_layers)
        ]

        self.residual_vq = ResidualVQ(
            codebook_size=codebook_size,
            embedding_dim=embedding_dim,
            num_quantizers=num_quantizers,
            batch_size=batch_size,
            ema_decay=ema_decay,
            threshold_ema_dead_code=threshold_ema_dead_code,
            commitment_cost=commitment_cost
        )

    def call(self, inputs, attention_mask=None, training=False, add_loss=True):
        if add_loss:
            quantized = self.residual_vq(inputs, training=training)
            mask_index = tf.random.uniform(shape=(), minval=0, maxval=self.patch_length, dtype=tf.int32)
            mask = tf.expand_dims(tf.expand_dims(tf.one_hot(mask_index, self.patch_length), axis=0), axis=-1)
            inputs = inputs * (1 - mask)
        
        inputs = self.pos_embed(inputs, training=training)
        attention_scores = []
        for i, layer in enumerate(self.layers):
            inputs, scores = layer(inputs, attention_mask=attention_mask, training=training)
            attention_scores.append(scores)

        if add_loss:
            loss = self.contrastive_loss(mask_index, inputs, quantized)
            self.add_loss(loss)
            self.add_metric(loss, name="context")

        return inputs, attention_scores
    
    def contrastive_loss(self, mask_index, context, quantized):
        c_t = tf.expand_dims(context[:, mask_index, :], axis=1)
        q_t = tf.expand_dims(quantized[:, mask_index, :], axis=1)

        numerator = tf.exp(cosine_similarity(c_t, q_t) / self.temperature)
        mask = tf.expand_dims(tf.one_hot(mask_index, self.patch_length), axis=0)
        denominator = tf.reduce_sum(tf.exp(cosine_similarity(c_t, quantized) / self.temperature) * (1 - mask), axis=1)

        loss = -tf.math.log(numerator / denominator)
        
        return tf.reduce_mean(loss)
    
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_heads": self.num_heads,
                "num_layers": self.num_layers,
                "intermediate_size": self.intermediate_size,
                "dropout": self.dropout,
                "layer_norm_eps": self.layer_norm_eps,
                "is_gelu_approx": self.is_gelu_approx,
                "attention_norm_type": self.attention_norm_type,
                "temperature": self.temperature
            }
        )
        return config
    

def cosine_similarity(a, b):
    a_normalized = tf.nn.l2_normalize(a, axis=-1)
    b_normalized = tf.nn.l2_normalize(b, axis=-1)
    cosine_sim = a_normalized * b_normalized
    return tf.reduce_sum(cosine_sim, axis=-1)
