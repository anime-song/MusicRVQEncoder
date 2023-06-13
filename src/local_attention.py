import tensorflow as tf

from model_utils import ReGLU, WeightNormDense


def get_relative_positions_matrix(length, max_relative_position):
    """
    Generate relative positions matrix.
    :param length: The length of the sequence.
    :param max_relative_position: Maximum relative position considered.
    :return: A relative positions matrix.
    """
    positions = tf.range(length, dtype=tf.int32)
    positions_matrix = positions[None, :] - positions[:, None]
    positions_matrix = tf.clip_by_value(positions_matrix, -max_relative_position, max_relative_position)
    return positions_matrix + max_relative_position


def get_relative_positions_embeddings(length, max_relative_position, d_k):
    """
    Generate relative positions embeddings.
    :param length: The length of the sequence.
    :param max_relative_position: Maximum relative position considered.
    :param d_k: Dimension of query, key, and value.
    :return: A relative positions embeddings matrix.
    """
    vocab_size = max_relative_position * 2 + 1
    positions_matrix = get_relative_positions_matrix(length, max_relative_position)
    positions_embeddings = tf.keras.layers.Embedding(vocab_size, d_k, trainable=True)(positions_matrix)
    return positions_embeddings


@tf.function
def sliding_chunks_no_overlap_matmul_qk(q, k, batch_size, window_size):
    _, _, num_heads, head_dim = list(q.shape)
    # chunk seqlen into non-overlapping chunks of size w
    chunk_q = tf.reshape(q, [batch_size, -1, window_size, num_heads, head_dim])
    chunk_k = tf.reshape(k, [batch_size, -1, window_size, num_heads, head_dim])
    chunk_k_expanded = tf.stack([
        tf.pad(chunk_k[:, :-1], ((0, 0), (1, 0), (0, 0), (0, 0), (0, 0)), constant_values=0.0),
        chunk_k,
        tf.pad(chunk_k[:, 1:], ((0, 0), (0, 1), (0, 0), (0, 0), (0, 0)), constant_values=0.0)
    ], axis=-1)
    diagonal_attn = tf.einsum('bcxhd,bcyhde->bcxhey', chunk_q, chunk_k_expanded)  # multiply
    return tf.reshape(diagonal_attn, [batch_size, -1, num_heads, 3 * window_size])


@tf.function
def sliding_chunks_no_overlap_matmul_pv(prob, v, batch_size, window_size):
    _, _, num_heads, head_dim = list(v.shape)
    chunk_prob = tf.reshape(prob, [batch_size, -1, window_size, num_heads, 3, window_size])
    chunk_v = tf.reshape(v, [batch_size, -1, window_size, num_heads, head_dim])
    chunk_v_extended = tf.stack([
        tf.pad(chunk_v[:, :-1], ((0, 0), (1, 0), (0, 0), (0, 0), (0, 0)), constant_values=0.0),
        chunk_v,
        tf.pad(chunk_v[:, 1:], ((0, 0), (0, 1), (0, 0), (0, 0), (0, 0)), constant_values=0.0)
    ], axis=-1)
    context = tf.einsum('bcwhpd,bcdhep->bcwhe', chunk_prob, chunk_v_extended)
    return tf.reshape(context, [batch_size, -1, num_heads, head_dim])


class SlidingWindowAttentionNoOverlap(tf.keras.layers.Layer):
    def __init__(self, window_size, batch_size, droprate=0.0, **kwargs):
        super(SlidingWindowAttentionNoOverlap, self).__init__(**kwargs)
        self.window_size = window_size
        self.batch_size = batch_size
        self.dropout = tf.keras.layers.Dropout(droprate)

    def call(self, query, key, value, mask=None, training=False):
        batch_size = self.batch_size
        if not training:
            batch_size = tf.shape(query)[0]
        
        attention = sliding_chunks_no_overlap_matmul_qk(query, key, batch_size=batch_size, window_size=self.window_size)
        attention = self.dropout(tf.keras.activations.softmax(attention, axis=-1), training=training)
        
        out = sliding_chunks_no_overlap_matmul_pv(attention, value, batch_size=batch_size, window_size=self.window_size)

        return out, attention


class MultiHeadWindowAttention(tf.keras.layers.Layer):
    def __init__(self, window_size, num_heads, d_model, key_dim, batch_size, droprate=0.1, **kwargs):
        super(MultiHeadWindowAttention, self).__init__(**kwargs)
        self.window_size = window_size
        self.num_heads = num_heads
        self.d_model = d_model
        self.droprate = droprate
        self.batch_size = batch_size
        self.key_dim = key_dim

        self.depth = d_model // self.num_heads

        self.wq = WeightNormDense(d_model, kernel_initializer="he_normal")
        self.wk = WeightNormDense(d_model, kernel_initializer="he_normal")
        self.wv = WeightNormDense(d_model, kernel_initializer="he_normal")

        self.dense = WeightNormDense(self.key_dim, kernel_initializer="he_normal")
        self.dropout = tf.keras.layers.Dropout(self.droprate)
        self.attention = SlidingWindowAttentionNoOverlap(self.window_size, batch_size=batch_size)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return x
    
    def call(self, query, key, value=None, mask=None, training=False):
        if value is None:
            value = key

        batch_size = tf.shape(query)[0]

        query = self.wq(query)
        key = self.wk(key)
        value = self.wv(value)
        query = query / tf.math.sqrt(tf.cast(self.depth, tf.float32))
        
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        query, attention_weights = self.attention(query, key, value, mask=mask, training=training)
        query = tf.reshape(query, (batch_size, -1, self.d_model))
        attention_weights = tf.transpose(attention_weights, perm=[0, 2, 1, 3])

        query = self.dropout(self.dense(query), training=training)
        return query, attention_weights
    
    def get_config(self):
        config = {
            'window_size': self.window_size,
            'num_heads': self.num_heads,
            'd_model': self.d_model,
            'droprate': self.droprate,
            'key_dim': self.key_dim
        }
        base_config = super(MultiHeadWindowAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    

class LocalAttentionTransformer(tf.keras.layers.Layer):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 ff_dim,
                 window_size,
                 batch_size,
                 attention_axes=None,
                 layer_norm_eps=1e-5,
                 is_gelu_approx=False,
                 rate=0.1,
                 attention_norm_type="postnorm",
                 **kwargs):
        super(LocalAttentionTransformer, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.is_gelu_approx = is_gelu_approx
        self.layer_norm_eps = layer_norm_eps
        self.attention_norm_type = attention_norm_type

        self.attention_layer = MultiHeadWindowAttention(
            window_size=window_size,
            num_heads=num_heads,
            d_model=ff_dim,
            key_dim=embed_dim,
            batch_size=batch_size,
            droprate=rate)
        self.ffn = WeightNormDense(ff_dim, kernel_initializer="he_normal")
        self.ffn_out = WeightNormDense(embed_dim, kernel_initializer="he_normal")
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=layer_norm_eps)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=layer_norm_eps)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, mask=None, training=False, return_attention_scores=False):
        residual = inputs
        b2_connection = inputs
        if self.attention_norm_type == "prenorm":
            inputs = self.layernorm1(inputs)
        inputs, attention_scores = self.attention_layer(inputs, inputs, training=training)
        inputs = self.dropout1(inputs, training=training)
        inputs = inputs + residual
        if self.attention_norm_type == "postnorm":
            inputs = self.layernorm1(inputs)

        # FFN
        residual = inputs
        if self.attention_norm_type == "prenorm":
            inputs = self.layernorm2(inputs)
        inputs = self.ffn(inputs)
        inputs = ReGLU()(inputs)
        inputs = self.ffn_out(inputs)
        inputs = self.dropout2(inputs, training=training)
        inputs = inputs + residual + b2_connection
        if self.attention_norm_type == "postnorm":
            inputs = self.layernorm2(inputs)

        if return_attention_scores:
            return inputs, attention_scores

        return inputs
    
    def get_config(self):
        config = {
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'layer_norm_eps': self.layer_norm_eps,
            'is_gelu_approx': self.is_gelu_approx,
            'attention_norm_type': self.attention_norm_type}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
    

class LocalPositonalEncoding(tf.keras.layers.Layer):
    def __init__(self, batch_dim, feature_size, window_size, **kwargs):
        super(LocalPositonalEncoding, self).__init__(**kwargs)
        self.window_size = window_size
        self.batch_dim = batch_dim
        self.feature_size = feature_size

    def call(self, inputs, training=False):
        x = inputs
        x *= tf.math.sqrt(tf.cast(self.feature_size, tf.float32))

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

        x = tf.reshape(x, (-1, self.window_size, self.feature_size))
        x += positional_encoding(self.window_size, self.feature_size)

        if not training:
            batch_size = tf.shape(inputs)[0]
        else:
            batch_size = self.batch_dim
        x = tf.reshape(x, (batch_size, -1, self.feature_size))
        return x
    
    def get_config(self):
        config = {'feature_size': self.feature_size, 'window_size': self.window_size, 'batch_dim': self.batch_dim}
        base_config = super(LocalPositonalEncoding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
