from model import Model
import hydra
import custom_interpolations
import tensorflow as tf
import numpy as np
from time import gmtime, strftime
import os


class Experiment(object):
    def __init__(self, experiment_conf, model_conf, repetition=0):
        self.model = Model(model_conf)
        self.experiment_conf = experiment_conf
        self.model_conf = model_conf
        self.n_train_encoder = experiment_conf.n_train_encoder
        self.n_train_readout = experiment_conf.n_train_readout
        self.n_test_batch = experiment_conf.n_test_batch
        ### summary ###
        self.summary_writer = tf.summary.create_file_writer("logs_{}".format(repetition))
        ### data storage ###
        self.data_collection_path = experiment_conf.data_collection_path
        self.data_readout_sources = np.zeros(self.n_train_readout)
        self.data_readout_shared = np.zeros(self.n_train_readout)

    def train_encoder(self):
        with self.summary_writer.as_default():
            for i in range(self.n_train_encoder):
                print("train_encoder: {: 4d}/{: 4d}".format(i + 1, self.n_train_encoder), end='\r')
                tf.summary.scalar("train_encoder_loss", self.model.train_encoder(), step=i)
        print('')

    def train_readout(self):
        with self.summary_writer.as_default():
            for i in range(self.n_train_readout):
                print("train_readout: {: 4d}/{: 4d}".format(i + 1, self.n_train_readout), end='\r')
                sources_loss, shared_loss = self.model.train_readout()
                tf.summary.scalar("train_readout_sources", sources_loss, step=i)
                tf.summary.scalar("train_readout_shared", shared_loss, step=i)
                self.data_readout_sources[i] = sources_loss
                self.data_readout_shared[i] = shared_loss
        print('')

    def dump_data(self):
        filename = "{}_{:04d}_{:04d}_{:04d}_{:04d}_{:04d}".format(
            strftime("%Y_%m_%d_%H_%M_%S", gmtime()),
            self.model_conf.n_sources,
            self.model_conf.dim_sources,
            self.model_conf.dim_shared,
            self.model_conf.dim_correlate,
            self.model_conf.dim_latent,
        )
        path = self.data_collection_path + "/" + filename
        os.makedirs(self.data_collection_path, exist_ok=True)
        print("saving", path)
        self.model.dump_test_data(path, n_batch=self.n_test_batch)

    def __call__(self):
        self.model.z_score()
        self.train_encoder()
        self.train_readout()
        self.dump_data()


@hydra.main(config_path='../config/', config_name='config.yaml')
def main(cfg):
    experiment(cfg)


def experiment(cfg):
    for repetition in range(cfg.experiment_conf.n_repetitions):
        print("repetition nb {}".format(repetition + 1))
        exp = Experiment(cfg.experiment_conf, cfg.model_conf, repetition=repetition)
        exp()


if __name__ == '__main__':
    main()
