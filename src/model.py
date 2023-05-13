import tensorflow as tf
import numpy as np
from tensorflow.keras import layers as L
from tensorflow.keras import backend as K

from config import MusicRVQAEConfig, MusicRVQLMConfig
from encoder import MaskedEncoder, WeightNormDense
from decoder import DecoderLayer
from feature_extractor import FeatureExtractorLayer


class MixStripes(L.Layer):
    def __init__(
            self,
            dim,
            mix_width,
            stripes_num,
            batch_size,
            patch_length,
            **kwargs):
        super(MixStripes, self).__init__(**kwargs)

        self.batch_size = batch_size
        self.dim = dim
        self.mix_width = mix_width
        self.stripes_num = stripes_num
        self.time_steps = patch_length
        self.supports_masking = True

    def call(self, inputs, mask=None, training=None):
        # inputs: (batch_size, time_steps, freqs)

        if training is False:
            return inputs

        if self.dim == 1:
            total_width = self.time_steps
        elif self.dim == 2:
            total_width = K.int_shape(inputs)[-1]

        ind = tf.constant(0)
        v1 = tf.TensorArray(dtype=inputs.dtype, size=0, dynamic_size=True)
        _, loop = tf.while_loop(
            cond=lambda i, x: i < self.batch_size,
            body=lambda i, x: (
                i + 1,
                x.write(i, self.transform_slice(inputs[i], total_width, self.time_steps))
            ),
            loop_vars=[ind, v1]
        )

        inputs = loop.stack()

        if mask is not None:
            inputs * tf.expand_dims(tf.cast(mask, dtype=tf.float32), -1)

        return inputs

    def transform_slice(self, e, total_width, time_steps):
        # e: (time_steps, freqs)
        # r: (time_steps, freqs)

        x_range = tf.range(time_steps)[:, tf.newaxis]
        y_range = tf.range(K.int_shape(e)[-1])[tf.newaxis, :]

        mask = tf.zeros([time_steps, K.int_shape(e)[-1]], dtype=tf.bool)

        for _ in range(self.stripes_num):
            distance = self.mix_width
            bgn = np.random.randint(0, total_width - distance)

            if self.dim == 1:
                mask |= (
                    (bgn <= x_range) & (
                        x_range < bgn +
                        distance)) | (
                    0 > y_range)

            elif self.dim == 2:
                mask |= (
                    (bgn <= y_range) & (
                        y_range < bgn +
                        distance)) | (
                    0 > x_range)

        e = tf.where(mask, 0.0, e)

        return e

    def get_config(self):
        config = super(MixStripes, self).get_config()
        return dict(list(config.items()))


class SpecMixAugmentation(L.Layer):
    def __init__(self,
                 time_mix_width,
                 time_stripes_num,
                 freq_mix_width,
                 freq_stripes_num,
                 batch_size,
                 patch_length, **kwargs):
        super(SpecMixAugmentation, self).__init__(**kwargs)

        self.batch_size = batch_size
        self.time_mixer = MixStripes(
            dim=1,
            mix_width=time_mix_width,
            stripes_num=time_stripes_num,
            batch_size=batch_size,
            patch_length=patch_length)
        self.freq_mixer = MixStripes(
            dim=2,
            mix_width=freq_mix_width,
            stripes_num=freq_stripes_num,
            batch_size=batch_size,
            patch_length=patch_length)
        self.supports_masking = True

    def call(self, inputs, mask=None, training=None):
        # inputs: (batch_size, time_steps, freqs)

        x = self.time_mixer(inputs, mask=mask, training=training)
        x = self.freq_mixer(x, mask=mask, training=training)
        return x

    def get_config(self):
        config = super(SpecMixAugmentation, self).get_config()
        return dict(list(config.items()))
    

class MusicRVQAE(tf.keras.Model):
    def __init__(self, config: MusicRVQAEConfig, batch_size, seq_len, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.batch_size = batch_size
        self.seq_len = seq_len

        self.feature_extract_layers = [
            FeatureExtractorLayer(
                filter_sizes=config.filter_sizes,
                kernel_sizes=config.kernel_sizes,
                strides=config.strides,
                batch_size=batch_size,
                num_heads=config.num_heads,
                intermediate_size=config.intermediate_size,
                layer_norm_eps=config.layer_norm_eps,
                dropout=config.dropout,
                attention_norm_type=config.attention_norm_type,
                is_gelu_approx=config.is_gelu_approx,
                layer_id=i
            )
            for i in range(len(config.filter_sizes))
        ]

        self.decoder_layers = [
            DecoderLayer(
                filter_sizes=config.filter_sizes,
                kernel_sizes=config.kernel_sizes,
                strides=config.strides,
                is_gelu_approx=config.is_gelu_approx,
                layer_id=i
            )
            for i in range(len(config.filter_sizes))
        ]

        self.encoded_seq_len = self.seq_len // np.prod(config.strides)
        self.spec_augment = SpecMixAugmentation(3, 3, 0, 0, batch_size, self.encoded_seq_len)
        self.output_layer = WeightNormDense(252 * 2, kernel_initializer="he_normal")

    def call(self, inputs, attention_mask=None, training=False):
        inputs = tf.transpose(inputs, (0, 1, 3, 2))
        inputs = tf.reshape(inputs, (self.batch_size, -1, 252 * 2))

        for feature_extractor_layer in self.feature_extract_layers:
            inputs = feature_extractor_layer(inputs, training=training)

        inputs = self.spec_augment(inputs, training=training)

        for decoder_layer in self.decoder_layers:
            inputs = decoder_layer(inputs, training=training)

        outputs = self.output_layer(inputs)
        outputs = tf.reshape(outputs, (self.batch_size, -1, 2, 252))
        outputs = tf.transpose(outputs, (0, 1, 3, 2))
        outputs = tf.keras.activations.relu(outputs)
        return outputs
    
    def freeze_feature_extractor(self):
        for i in range(len(self.feature_extract_layers)):
            self.feature_extract_layers[i].trainable = False


class MusicRVQLM(tf.keras.Model):
    def __init__(self, rvq_ae: MusicRVQAE, config: MusicRVQLMConfig, batch_size, **kwargs):
        super().__init__(**kwargs)
        self.rvq_ae = rvq_ae
        self.config = config
        self.batch_size = batch_size

        self.masked_encoder = MaskedEncoder(
            config.hidden_size,
            config.num_heads,
            config.num_layers,
            config.intermediate_size,
            batch_size,
            patch_length=rvq_ae.encoded_seq_len,
            dropout=config.dropout,
            layer_norm_eps=config.layer_norm_eps,
            attention_norm_type=config.attention_norm_type,
            codebook_size=config.codebook_size,
            embedding_dim=config.embedding_dim,
            num_quantizers=config.num_quantizers,
            ema_decay=config.ema_decay,
            commitment_cost=config.commitment_cost,
            threshold_ema_dead_code=config.threshold_ema_dead_code
        )

    def call(self, inputs, training=False, return_scores=False, add_loss=True):
        inputs = tf.transpose(inputs, (0, 1, 3, 2))
        inputs = tf.reshape(inputs, (self.batch_size, -1, 252 * 2))

        for feature_extractor_layer in self.rvq_ae.feature_extract_layers:
            inputs = feature_extractor_layer(inputs, training=training)

        inputs, attention_scores = self.masked_encoder(inputs, training=training, add_loss=add_loss)

        if return_scores:
            return inputs, attention_scores

        return inputs
