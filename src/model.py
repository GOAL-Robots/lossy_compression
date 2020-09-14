import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import models
from omegaconf import OmegaConf


def to_model(cfg):
    return models.model_from_yaml(OmegaConf.to_yaml(cfg, resolve=True))


class Model(object):
    def __init__(self, model_conf):
        self.batch_size = model_conf.batch_size
        self.learning_rate = model_conf.learning_rate
        self.n_sources = model_conf.n_sources
        self.dim_sources = model_conf.dim_sources
        self.dim_shared = model_conf.dim_shared
        self.dim_correlate = model_conf.dim_correlate
        self.dim_latent = model_conf.dim_latent
        self.decoding_mode = model_conf.decoding_mode
        self.default_layer_size = model_conf.default_layer_size
        self.networks = model_conf.networks
        self.noise = model_conf.noise
        ### models ###
        self.correlator_models = [to_model(model_conf.networks.correlator_model_arch) for i in range(self.n_sources)]
        self.encoder_model = to_model(model_conf.networks.encoder_model_arch)
        self.correlate_decoder_model = to_model(model_conf.networks.correlate_decoder_model_arch)
        self.source_decoder_model = to_model(model_conf.networks.source_decoder_model_arch)
        self.shared_readout_model = to_model(model_conf.networks.shared_readout_model_arch)
        self.source_readout_models = [to_model(model_conf.networks.source_readout_model_arch) for i in range(self.n_sources)]
        ### optimizer ###
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

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
            return dist.sample(sample_shape=self.batch_size) * values
        elif self.noise.type == "exponential":
            return tfp.distributions.Exponential(
                rate=1 / self.noise.mean,
            ).sample(sample_shape=shape)

    def get_sources(self):
        return [self._get_noise((self.batch_size, self.dim_sources)) for i in range(self.n_sources)]

    def get_shared(self):
        return self._get_noise((self.batch_size, self.dim_shared))

    def get_inputs(self, sources, shared):
        return [tf.concat([source, shared], axis=-1) for source in sources]

    def get_correlates(self, sources, shared, format='list'):
        inputs = self.get_inputs(sources, shared)
        ret = [correlator_model(inp) for correlator_model, inp in zip(self.correlator_models, inputs)]
        if format == 'list':
            return ret
        elif format == 'tensor':
            return tf.concat(ret, axis=-1)
        else:
            raise ValueError("unrecognized option ({})".format(format))

    def get_latent(self, correlates):
        return self.encoder_model(correlates)

    def get_correlates_reconstructions(self, latent):
        return self.correlate_decoder_model(latent)

    def get_sources_reconstructions(self, latent):
        return self.source_decoder_model(latent)

    def get_sources_readouts(self, latent):
        return [source_readout_model(latent) for source_readout_model in self.source_readout_models]

    def get_shared_readout(self, latent):
        return self.shared_readout_model(latent)

    def pretrain(self):
        with tf.GradientTape() as tape:
            sources = self.get_sources()
            shared = self.get_shared()
            correlates = self.get_correlates(sources, shared, format='tensor')
            std = tf.math.reduce_std(correlates)
            loss = (1 - std) ** 2
            variables = sum([model.variables for model in self.correlator_models], [])
            grads = tape.gradient(loss, variables)
            self.optimizer.apply_gradients(zip(grads, variables))
        return loss / self.batch_size

    def train_encoder(self):
        with tf.GradientTape() as tape:
            sources = self.get_sources()
            shared = self.get_shared()
            correlates = self.get_correlates(sources, shared, format='tensor')
            latent = self.get_latent(correlates)
            if self.decoding_mode == 'sources':
                inputs = self.get_inputs(sources, shared)
                inputs = tf.concat(inputs, axis=-1)
                reconstructions = self.get_sources_reconstructions(latent)
                loss = tf.reduce_sum(tf.reduce_mean((reconstructions - inputs) ** 2, axis=-1))
                variables = self.encoder_model.variables + self.source_decoder_model.variables
            elif self.decoding_mode == 'correlates':
                reconstructions = self.get_correlates_reconstructions(latent)
                loss = tf.reduce_sum(tf.reduce_mean((reconstructions - correlates) ** 2, axis=-1))
                variables = self.encoder_model.variables + self.correlate_decoder_model.variables
            else:
                raise VallueError("unrecognized option in conf: {}".format(self.decoding_mode))
            grads = tape.gradient(loss, variables)
            self.optimizer.apply_gradients(zip(grads, variables))
        return loss / self.batch_size

    def train_readout(self):
        with tf.GradientTape() as tape:
            sources = self.get_sources()
            shared = self.get_shared()
            correlates = self.get_correlates(sources, shared, format='tensor')
            latent = self.get_latent(correlates)
            sources_readouts = self.get_sources_readouts(latent)
            shared_readout = self.get_shared_readout(latent)
            sources_losses = [tf.reduce_sum(tf.reduce_mean(
                (source - source_readout) ** 2,
                axis=-1))
                for source, source_readout in zip(sources, sources_readouts)
            ]
            all_sources_losses = tf.stack(sources_losses, axis=0)
            sources_loss = tf.reduce_sum(all_sources_losses)
            shared_loss = tf.reduce_sum(tf.reduce_mean((shared - shared_readout) ** 2, axis=-1))
            loss = sources_loss + shared_loss
            variables = sum([model.variables for model in self.source_readout_models], self.shared_readout_model.variables)
            grads = tape.gradient(loss, variables)
            self.optimizer.apply_gradients(zip(grads, variables))
        return sources_loss / self.n_sources / self.batch_size, shared_loss / self.batch_size
