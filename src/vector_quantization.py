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
            batch_size,
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
        self.batch_size = batch_size

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            name="embeddings",
            shape=(self.embedding_dim, self.codebook_size),
            initializer=tf.initializers.random_normal(),
            trainable=True)

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
        
    def expire_codes(self, batch_samples):
        dead_codes = self.ema_count < self.threshold_ema_dead_code

        seq_len = tf.reduce_sum(tf.ones_like(batch_samples)[:, :, 0], axis=1)[0]
        seq_len = tf.cast(seq_len, tf.int32)

        for i in range(self.batch_size):
            samples = batch_samples[i]
            sampled_indices = tf.random.uniform((self.codebook_size,), minval=0, maxval=seq_len, dtype=tf.int32)
            sampled_vectors = tf.gather(samples, sampled_indices)

            indices_to_update = tf.where(dead_codes)
            vectors_to_update = tf.gather(sampled_vectors, tf.range(tf.shape(indices_to_update)[0]))

            # 更新
            updated_embeddings = tf.transpose(self.embeddings)
            updated_embeddings = tf.tensor_scatter_nd_update(updated_embeddings, indices_to_update, vectors_to_update)
            updated_embeddings = tf.transpose(updated_embeddings)
            self.embeddings.assign(updated_embeddings)

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
            self.ema_count.assign((1 - self.ema_decay) * self.ema_count + self.ema_decay * updated_count)
            self.ema_sum.assign((1 - self.ema_decay) * self.ema_sum + self.ema_decay * updated_sum)

            n = tf.reduce_sum(self.ema_count)
            cluster_size = (self.ema_count + self.epsilon) / (n + self.codebook_size * self.epsilon) * n
            updated_embeddings = self.ema_sum / tf.reshape(cluster_size, [1, self.codebook_size])

            self.embeddings.assign(updated_embeddings)

            self.expire_codes(inputs)

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
            tf.reduce_sum(flat_inputs ** 2, axis=1, keepdims=True) - 2 * similarity + tf.reduce_sum(self.embeddings ** 2, axis=0, keepdims=True)
        )

        encoding_indices = tf.argmax(-distances, axis=1)
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
            batch_size,
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
                commitment_cost=commitment_cost,
                batch_size=batch_size)
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

        codebook_list = [
            vq_layer.embeddings for vq_layer in self.vq_layers
        ]

        return {
            "quantized_out": quantized_out,
            "quantized_list": quantized_list,
            "codebook_list": codebook_list
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
