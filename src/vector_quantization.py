import tensorflow as tf


class VectorQuantizer(tf.keras.layers.Layer):
    """
    Args:
        embedding_dim: 埋め込み次元
        num_embeddings: コードブックのサイズ
    """

    def __init__(
            self,
            embedding_dim,
            codebook_size,
            commitment_cost=0.25,
            ema_decay=0.8,
            epsilon=1e-6,
            threshold_ema_dead_code=2,
            **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.codebook_size = codebook_size
        self.commitment_cost = commitment_cost
        self.ema_decay = ema_decay
        self.epsilon = epsilon
        self.threshold_ema_dead_code = threshold_ema_dead_code

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            name="embeddings",
            shape=(self.embedding_dim, self.codebook_size),
            initializer="random_normal",
            trainable=True,
            dtype="float32")

        self.ema_count = self.add_weight(
            name="ema_count",
            shape=(self.codebook_size,),
            dtype=tf.float32,
            initializer='zeros',
            aggregation=tf.VariableAggregation.MEAN,
            trainable=False,
            use_resource=True)
        self.ema_sum = self.add_weight(
            name="ema_sum",
            shape=(self.embedding_dim, self.codebook_size),
            dtype=tf.float32,
            initializer='zeros',
            aggregation=tf.VariableAggregation.MEAN,
            trainable=False,
            use_resource=True)

    def call(self, inputs, training=False):
        flat_inputs = tf.reshape(inputs, [-1, self.embedding_dim])

        encoding_indices = self.get_code_indices(flat_inputs)
        encodings = tf.one_hot(encoding_indices, self.codebook_size)
        encoding_indices = tf.reshape(encoding_indices, tf.shape(inputs)[:-1])
        quantized = tf.nn.embedding_lookup(tf.transpose(self.embeddings, [1, 0]), encoding_indices)

        e_latent_loss = tf.reduce_mean(tf.square(tf.stop_gradient(quantized) - inputs))
        q_latent_loss = tf.reduce_mean(tf.square(quantized - tf.stop_gradient(inputs)))
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = inputs + tf.stop_gradient(quantized - inputs)

        if training:
            updated_count = tf.reduce_sum(encodings, axis=0)
            updated_sum = tf.matmul(flat_inputs, encodings, transpose_a=True)
            self.ema_count.assign((self.ema_count - updated_count) * self.ema_decay)
            self.ema_sum.assign((self.ema_sum - updated_sum) * self.ema_decay)

            n = tf.reduce_sum(self.ema_count)
            cluster_size = (self.ema_count + self.epsilon) / (n + self.codebook_size * self.epsilon) * n
            updated_embeddings = self.ema_sum / tf.reshape(cluster_size, [1, self.codebook_size])

            # カウントが閾値以下のコードを現在のバッチからランダムに選ばれたベクトルで置き換える
            dead_code_indices = tf.where(self.ema_count < self.threshold_ema_dead_code)
            dead_code_indices = tf.reshape(dead_code_indices, (-1, 1))

            dead_code_mask = tf.math.less(self.ema_count, self.threshold_ema_dead_code)
            remaining_indices_mask = tf.math.logical_not(dead_code_mask)

            remaining_indices = tf.boolean_mask(tf.range(0, self.codebook_size), remaining_indices_mask)
            if tf.shape(remaining_indices)[0] > 0:
                random_index = tf.random.uniform(
                    (tf.shape(dead_code_indices)[0],),
                    minval=0,
                    maxval=tf.shape(remaining_indices)[0],
                    dtype=tf.int32)
                sampled_indices = tf.gather(remaining_indices, random_index)
                random_vectors = tf.gather(flat_inputs, sampled_indices)
                random_vectors = tf.reshape(random_vectors, (-1, self.embedding_dim))

                updated_embeddings = tf.transpose(updated_embeddings)
                updated_embeddings = tf.tensor_scatter_nd_update(updated_embeddings, dead_code_indices, random_vectors)
                updated_embeddings = tf.transpose(updated_embeddings)

            self.embeddings.assign(updated_embeddings)

        self.add_loss(loss)
        # self.add_metric(loss, name="vq_loss_" + self.name)
        return {
            "quantized": quantized,
            "encodings": encodings,
            "encoding_indices": encoding_indices
        }

    def get_code_indices(self, flat_inputs):
        similarity = tf.matmul(flat_inputs, self.embeddings)
        distances = (
            tf.reduce_sum(flat_inputs ** 2, axis=1, keepdims=True) + tf.reduce_sum(self.embeddings ** 2, axis=0) - 2 * similarity
        )

        encoding_indices = tf.argmin(distances, axis=1)
        return encoding_indices

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embedding_dim": self.embedding_dim,
                "codebook_size": self.codebook_size,
                "commitment_cost": self.commitment_cost,
                "ema_decay": self.ema_decay,
                "epsilon": self.epsilon,
                "threshold_ema_dead_code": self.threshold_ema_dead_code
            }
        )
        return config


class ResidualVQ(tf.keras.layers.Layer):
    def __init__(
            self,
            codebook_size,
            embedding_dim,
            commitment_cost,
            num_quantizers,
            **kwargs):
        super().__init__(**kwargs)
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.num_quantizers = num_quantizers
        self.vq_layers = [
            VectorQuantizer(
                embedding_dim=embedding_dim,
                codebook_size=codebook_size,
                commitment_cost=commitment_cost)
            for i in range(num_quantizers)]

    def call(self, inputs, training=False):
        residual = inputs
        quantized_out = 0.

        quantized_list = []
        for layer in self.vq_layers:
            vq_output = layer(residual, training=training)

            residual = residual - tf.stop_gradient(vq_output['quantized'])
            quantized_out = quantized_out + vq_output['quantized']
            quantized_list.append(vq_output['quantized'])

        return {
            "quantized_out": quantized_out,
            "quantized_list": quantized_list
        }

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "codebook_size": self.codebook_size,
                "embedding_dim": self.embedding_dim,
                "commitment_cost": self.commitment_cost,
                "num_quantizers": self.num_quantizers
            }
        )
