import tensorflow as tf
from tensorflow.keras import backend as K


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
            ema_decay,
            epsilon=1e-12,
            commitment_cost=1.0,
            threshold_ema_dead_code=2,
            **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.codebook_size = codebook_size
        self.batch_size = batch_size
        self.ema_decay = ema_decay
        self.epsilon = epsilon
        self.commitment_cost = commitment_cost
        self.threshold_ema_dead_code = threshold_ema_dead_code

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            name="embeddings",
            shape=(self.embedding_dim, self.codebook_size),
            dtype=tf.float32,
            initializer=tf.initializers.random_normal(),
            trainable=False)

        self.ema_cluster_size = self.add_weight(
            name="ema_cluster_size",
            shape=(self.codebook_size,),
            dtype=tf.float32,
            initializer='zeros',
            trainable=False)
        self.ema_w = self.add_weight(
            name="ema_w",
            shape=(self.embedding_dim, self.codebook_size),
            dtype=tf.float32,
            initializer=tf.initializers.Constant(self.embeddings.numpy()),
            trainable=False)
        
    def expire_codes(self, batch_samples):
        if self.threshold_ema_dead_code <= 0.0:
            return
        
        dead_codes = self.ema_cluster_size < self.threshold_ema_dead_code
        indices_to_update = tf.where(dead_codes)

        reshaped_samples = tf.reshape(batch_samples, (self.batch_size, -1, tf.shape(batch_samples)[-1]))
        seq_len = tf.reduce_sum(tf.ones_like(reshaped_samples)[:, :, 0], axis=1)[0]
        seq_len = tf.cast(seq_len, tf.int32)

        updated_embeddings = tf.transpose(self.embeddings)
        for i in range(self.batch_size):
            samples = reshaped_samples[i]
            if seq_len >= self.codebook_size:
                sampled_indices = tf.random.shuffle(tf.range(seq_len))[:self.codebook_size]
            else:
                sampled_indices = tf.random.uniform((self.codebook_size,), minval=0, maxval=seq_len, dtype=tf.int32)
            sampled_vectors = tf.gather(samples, sampled_indices)

            vectors_to_update = tf.gather(sampled_vectors, tf.range(tf.minimum(tf.shape(indices_to_update)[0], tf.shape(sampled_vectors)[0])))
            updated_embeddings = tf.tensor_scatter_nd_update(updated_embeddings, indices_to_update, vectors_to_update)

        updated_embeddings = tf.transpose(updated_embeddings)
        self.embeddings.assign(updated_embeddings)

        updated_ema_cluster_size = tf.where(dead_codes, tf.ones_like(self.ema_cluster_size) * self.threshold_ema_dead_code, self.ema_cluster_size)
        self.ema_cluster_size.assign(updated_ema_cluster_size)
        self.ema_w.assign(updated_embeddings * self.threshold_ema_dead_code)

    def call(self, inputs, training=False):
        flat_inputs = tf.reshape(inputs, [-1, self.embedding_dim])

        encoding_indices = self.get_code_indices(flat_inputs)
        encodings = tf.one_hot(encoding_indices, self.codebook_size)
        encoding_indices = tf.reshape(encoding_indices, tf.shape(inputs)[:-1])
        quantized = tf.nn.embedding_lookup(tf.transpose(self.embeddings, [1, 0]), encoding_indices)

        if training:
            updated_ema_cluster_size = K.moving_average_update(self.ema_cluster_size, tf.reduce_sum(encodings, 0), self.ema_decay)

            dw = tf.matmul(flat_inputs, encodings, transpose_a=True)
            updated_ema_w = K.moving_average_update(self.ema_w, dw, self.ema_decay)

            n = tf.reduce_sum(updated_ema_cluster_size)
            updated_ema_cluster_size = ((updated_ema_cluster_size + self.epsilon) / (n + self.codebook_size * self.epsilon) * n)
            normalized_updated_ema_w = updated_ema_w / tf.reshape(updated_ema_cluster_size, [1, -1])
            self.embeddings.assign(normalized_updated_ema_w)
            
            self.expire_codes(inputs)

        e_latent_loss = tf.reduce_mean((tf.stop_gradient(quantized) - inputs) ** 2)
        loss = e_latent_loss * self.commitment_cost
        quantized = inputs + tf.stop_gradient(quantized - inputs)

        self.add_loss(loss)
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
            num_quantizers,
            batch_size,
            ema_decay,
            threshold_ema_dead_code,
            commitment_cost,
            **kwargs):
        super().__init__(**kwargs)
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.num_quantizers = num_quantizers
        self.batch_size = batch_size
        self.vq_layers = [
            VectorQuantizer(
                embedding_dim=embedding_dim,
                codebook_size=codebook_size,
                batch_size=batch_size,
                ema_decay=ema_decay,
                threshold_ema_dead_code=threshold_ema_dead_code,
                commitment_cost=commitment_cost)
            for i in range(num_quantizers)]

    def call(self, inputs, training=False):
        residual = inputs
        quantized_out = 0.

        losses = []
        for layer in self.vq_layers:
            vq_output = layer(residual, training=training)

            residual = residual - tf.stop_gradient(vq_output['quantized'])
            quantized_out = quantized_out + vq_output['quantized']

            losses.extend(layer.losses)

        self.add_metric(tf.math.reduce_sum(losses), name="residual_vq_commitment")
        return quantized_out

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "codebook_size": self.codebook_size,
                "embedding_dim": self.embedding_dim,
                "num_quantizers": self.num_quantizers
            }
        )
