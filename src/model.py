import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import models
from omegaconf import OmegaConf
import numpy as np


def to_model(cfg, n_output_units):
    cfg.config.layers[-1].config.units = n_output_units
    return models.model_from_yaml(OmegaConf.to_yaml(cfg, resolve=True))


class Model(object):
    def __init__(self, model_conf):
        self.batch_size = model_conf.batch_size
        self.learning_rate = model_conf.learning_rate
        self.n_sources = model_conf.n_sources
        self.dim_sources = model_conf.dim_sources
        self.dim_shared = model_conf.dim_shared
        self.dim_correlate = model_conf.correlate_dilation_factor * (self.dim_shared + self.dim_sources)
        self.dim_latent = model_conf.dim_latent
        self.decoding_mode = model_conf.decoding_mode
        if self.decoding_mode == 'sources':
            self.dim_reconstruction = (self.dim_sources + self.dim_shared) * self.n_sources
        elif self.decoding_mode == 'sources_no_repetition':
            self.dim_reconstruction = self.dim_sources * self.n_sources + self.dim_shared
        elif self.decoding_mode == 'correlates':
            self.dim_reconstruction = self.dim_correlate * self.n_sources
        else:
            raise VallueError("unrecognized option in conf: {}".format(self.decoding_mode))
        self.default_layer_size = model_conf.default_layer_size
        self.noise = model_conf.noise
        ### models ###
        self.correlator_models = [to_model(model_conf.correlator_model_arch, self.dim_correlate) for i in range(self.n_sources)]
        self.encoder_model = to_model(model_conf.encoder_model_arch, self.dim_latent)
        self.decoder_model = to_model(model_conf.decoder_model_arch, self.dim_reconstruction)
        self.shared_readout_model = to_model(model_conf.shared_readout_model_arch, self.dim_shared)
        self.source_readout_models = [to_model(model_conf.source_readout_model_arch, self.dim_sources) for i in range(self.n_sources)]
        self.correlates_means = [tf.Variable(np.zeros(shape=self.dim_correlate), dtype=np.float32) for _ in range(self.n_sources)]
        self.correlates_stds = [tf.Variable(np.ones(shape=self.dim_correlate), dtype=np.float32) for _ in range(self.n_sources)]
        ### optimizer ###
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

    @tf.function
    def _get_noise(self, shape):
        if self.noise.type == "gaussian":
            return tf.random.normal(
                shape=shape,
                mean=self.noise.mean,
                stddev=self.noise.std,
            )
        elif self.noise.type == "choice":
            values = tf.constant(list(self.noise.val))
            dist = tfp.distributions.Multinomial(1, probs=list(self.noise.probs))
            return tf.reduce_sum(dist.sample(sample_shape=shape) * values, axis=-1)
        elif self.noise.type == "exponential":
            return tfp.distributions.Exponential(
                rate=1 / self.noise.mean,
            ).sample(sample_shape=shape)

    @tf.function
    def get_sources(self):
        return [self._get_noise((self.batch_size, self.dim_sources)) for i in range(self.n_sources)]

    @tf.function
    def get_shared(self):
        return self._get_noise((self.batch_size, self.dim_shared))

    @tf.function
    def get_inputs(self, sources, shared):
        return [tf.concat([source, shared], axis=-1) for source in sources]

    @tf.function
    def get_correlates(self, sources, shared, format='list'):
        inputs = self.get_inputs(sources, shared)
        ret = [
            (correlator_model(inp) - correlate_mean) / correlate_std
            for correlator_model, inp, correlate_mean, correlate_std in
            zip(self.correlator_models, inputs, self.correlates_means, self.correlates_stds)
        ]
        if format == 'list':
            return ret
        elif format == 'tensor':
            return tf.concat(ret, axis=-1)
        else:
            raise ValueError("unrecognized option ({})".format(format))

    @tf.function
    def get_latent(self, correlates):
        return self.encoder_model(correlates)

    @tf.function
    def get_reconstructions(self, latent):
        return self.decoder_model(latent)

    @tf.function
    def get_sources_readouts(self, latent):
        return [source_readout_model(latent) for source_readout_model in self.source_readout_models]

    @tf.function
    def get_shared_readout(self, latent):
        return self.shared_readout_model(latent)

    @tf.function
    def get_readouts_recerrs(self):
        sources = self.get_sources()
        shared = self.get_shared()
        correlates = self.get_correlates(sources, shared, format='tensor')
        latent = self.get_latent(correlates)
        sources_readouts = self.get_sources_readouts(latent)
        shared_readout = self.get_shared_readout(latent)
        sources_recerrs = [
            tf.reduce_mean((source - source_readout) ** 2, axis=-1)
            for source, source_readout in zip(sources, sources_readouts)
        ]
        all_sources_recerrs = tf.stack(sources_recerrs, axis=-1)
        sources_recerrs = tf.reduce_sum(all_sources_recerrs, axis=-1)
        shared_recerrs = tf.reduce_mean((shared - shared_readout) ** 2, axis=-1)
        return sources_recerrs, shared_recerrs

    def z_score(self):
        sources = self.get_sources()
        shared = self.get_shared()
        correlates = self.get_correlates(sources, shared, format='list')
        means = [tf.reduce_mean(correlate, axis=0) for correlate in correlates]
        stds = [tf.math.reduce_std(correlate, axis=0) for correlate in correlates]
        for correlate_mean, mean in zip(self.correlates_means, means):
            correlate_mean.assign_add(mean)
        for correlate_std, std in zip(self.correlates_stds, stds):
            correlate_std.assign(correlate_std * std)
        return self.correlates_means, self.correlates_stds

    @tf.function
    def train_encoder(self):
        with tf.GradientTape() as tape:
            sources = self.get_sources()
            shared = self.get_shared()
            correlates = self.get_correlates(sources, shared, format='tensor')
            latent = self.get_latent(correlates)
            if self.decoding_mode == 'sources':
                inputs = self.get_inputs(sources, shared)
                inputs = tf.concat(inputs, axis=-1)
            elif self.decoding_mode == 'sources_no_repetition':
                inputs = tf.concat(sources + [shared], axis=-1)
            elif self.decoding_mode == 'correlates':
                inputs = correlates
            else:
                raise VallueError("unrecognized option in conf: {}".format(self.decoding_mode))
            reconstructions = self.get_reconstructions(latent)
            loss = tf.reduce_sum(tf.reduce_mean((reconstructions - inputs) ** 2, axis=-1))
            variables = self.encoder_model.variables + self.decoder_model.variables
            grads = tape.gradient(loss, variables)
            self.optimizer.apply_gradients(zip(grads, variables))
        return loss / self.batch_size

    @tf.function
    def train_readout(self):
        with tf.GradientTape() as tape:
            sources_recerrs, shared_recerrs = self.get_readouts_recerrs()
            sources_loss = tf.reduce_sum(sources_recerrs)
            shared_loss = tf.reduce_sum(shared_recerrs)
            loss = sources_loss + shared_loss
            variables = sum([model.variables for model in self.source_readout_models], self.shared_readout_model.variables)
            grads = tape.gradient(loss, variables)
            self.optimizer.apply_gradients(zip(grads, variables))
        return sources_loss / self.batch_size, shared_loss / self.batch_size

    def dump_test_data(self, filepath, n_batch):
        data_sources = []
        data_shared = []
        for i in range(n_batch):
            sources_recerrs, shared_recerrs = self.get_readouts_recerrs()
            data_sources.append(sources_recerrs.numpy())
            data_shared.append(shared_recerrs.numpy())
        data_sources = np.concatenate(data_sources, axis=0) / self.n_sources
        data_shared = np.concatenate(data_shared, axis=0)
        np.savez(filepath, sources=data_sources, shared=data_shared)
