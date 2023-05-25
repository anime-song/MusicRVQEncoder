import tensorflow as tf

from vector_quantization import ResidualVQ
from model_utils import ReGLU
    

class PositionalEncoding(tf.keras.layers.Layer):
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
        self.layer_norm_eps = layer_norm_eps
        self.is_gelu_approx = is_gelu_approx
        self.dropout = dropout
        self.attention_norm_type = attention_norm_type

        self.attention_layer = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=hidden_size
        )

        self.conv_m_pointwise_conv = tf.keras.layers.Conv1D(hidden_size, kernel_size=1, padding="same")
        self.conv_m_layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=layer_norm_eps)
        self.conv_m_depthwise_conv = tf.keras.layers.DepthwiseConv1D(kernel_size=3, padding="same")
        self.conv_m_dropout = tf.keras.layers.Dropout(dropout)
        self.conv_m_layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=layer_norm_eps)

        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)

        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=layer_norm_eps)

        self.intermediate = tf.keras.layers.Dense(intermediate_size)
        self.attention_output = tf.keras.layers.Dense(hidden_size)

        self.final_layer_norm = tf.keras.layers.LayerNormalization(epsilon=layer_norm_eps)

    def call(self, inputs, attention_mask=None, training=False):
        # Attention
        residual = inputs
        b2_connection = inputs
        if self.attention_norm_type == "prenorm":
            inputs = self.layer_norm(inputs)
        inputs, scores = self.attention_layer(inputs, inputs, return_attention_scores=True, attention_mask=attention_mask, training=training)
        inputs = self.dropout1(inputs, training=training)
        inputs = inputs + residual
        if self.attention_norm_type == "postnorm":
            inputs = self.layer_norm(inputs)

        # conv module
        residual = inputs
        inputs = self.conv_m_pointwise_conv(inputs)
        inputs = tf.keras.activations.gelu(inputs, approximate=self.is_gelu_approx)
        inputs = self.conv_m_depthwise_conv(inputs)
        inputs = self.conv_m_layer_norm1(inputs)
        inputs = self.conv_m_dropout(inputs)
        inputs = inputs + residual
        inputs = self.conv_m_layer_norm2(inputs)

        # FFN
        residual = inputs
        if self.attention_norm_type == "prenorm":
            inputs = self.final_layer_norm(inputs)
        inputs = ReGLU()(self.intermediate(inputs))
        inputs = self.dropout2(self.attention_output(inputs), training=training)
        inputs = inputs + residual + b2_connection
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
        sample_codebook_temperature,
        dropout=0.1,
        layer_norm_eps=1e-5,
        is_gelu_approx=False,
        attention_norm_type="postnorm",
        temperature=1.0,
        use_quantizer=False,
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
        self.patch_length = patch_length
        self.batch_size = batch_size
        self.temperature = temperature
        self.use_quantizer = use_quantizer

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
                layer_norm_eps,
                is_gelu_approx,
                dropout,
                attention_norm_type
            )
            for i in range(num_layers)
        ]

        if self.use_quantizer:
            self.residual_vq = ResidualVQ(
                codebook_size=codebook_size,
                embedding_dim=embedding_dim,
                num_quantizers=num_quantizers,
                batch_size=batch_size,
                ema_decay=ema_decay,
                threshold_ema_dead_code=threshold_ema_dead_code,
                commitment_cost=commitment_cost,
                sample_codebook_temperature=sample_codebook_temperature
            )
        self.mask_samples = 5
        self.num_negative_samples = 100

    def apply_mask(self, inputs):
        mask_indices = tf.random.uniform((self.batch_size, self.mask_samples,), minval=0, maxval=self.patch_length, dtype=tf.int32)
        mask = tf.one_hot(mask_indices, self.patch_length)
        mask = tf.reduce_sum(mask, axis=1)
        mask = tf.clip_by_value(mask, 0, 1)
        mask = tf.cast(mask, dtype=tf.bool)
        mask_vector = tf.random.uniform(tf.shape(inputs), minval=0, maxval=1, dtype=tf.float32)
        inputs = tf.where(tf.expand_dims(mask, axis=-1), mask_vector, inputs)
        return inputs, mask

    def call(self, inputs, attention_mask=None, training=False, add_loss=True):
        if self.use_quantizer:
            quantized = self.residual_vq(inputs, training=training)
        else:
            quantized = inputs
            
        if add_loss:
            inputs, mask = self.apply_mask(inputs)

        # encoder
        attention_scores = []
        inputs = self.pos_embed(inputs, training=training)
        for i, layer in enumerate(self.layers):
            inputs, scores = layer(inputs, training=training)
            attention_scores.append(scores)

        if add_loss:
            loss = self.contrastive_loss(mask, inputs, quantized)
            self.add_loss(loss)
            self.add_metric(loss, name="context")
        return inputs, attention_scores
    
    def contrastive_loss(self, mask, context, quantized):
        loss = []
        for batch_index in range(self.batch_size):
            mask_indices = tf.where(mask[batch_index])
            c_t = tf.gather_nd(context[batch_index], mask_indices)
            q_t = tf.gather_nd(quantized[batch_index], mask_indices)
            pos_similarity = cosine_similarity(c_t, q_t)

            # negative_samples = tf.boolean_mask(quantized, tf.range(self.batch_size) != batch_index, axis=0)
            # flat_negative_samples = tf.reshape(negative_samples, [-1, tf.shape(negative_samples)[-1]])
            # target_indices = tf.random.shuffle(tf.range(tf.shape(flat_negative_samples)[0]))[:self.num_negative_samples]
            # targets = tf.gather(flat_negative_samples, target_indices)
            targets = tf.gather_nd(quantized[batch_index], tf.where(tf.logical_not(mask[batch_index])))

            neg_similarity = cosine_similarity_matmul(c_t, targets)

            numerator = tf.exp(pos_similarity / self.temperature)
            denominator = tf.reduce_sum(tf.exp(neg_similarity / self.temperature), axis=-1)
            loss_m = -tf.math.log(numerator / denominator)
            loss.append(tf.reduce_mean(loss_m))
        
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


def cosine_similarity_matmul(a, b):
    a_normalized = tf.nn.l2_normalize(a, axis=-1)
    b_normalized = tf.nn.l2_normalize(b, axis=-1)

    cosine_similarities = tf.matmul(a_normalized, b_normalized, transpose_b=True)
    return cosine_similarities
