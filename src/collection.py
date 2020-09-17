import numpy as np
import os
from collections import defaultdict, namedtuple
import re


class Collection(object):
    def __init__(self, path):
        self.path = path
        self.data = defaultdict(list)
        self.RunDescription = namedtuple('RunDescription', ['n_sources', 'dim_sources', 'dim_shared', 'dim_correlate', 'dim_latent'])
        self.RunData = namedtuple('RunData', ['sources', 'shared'])
        for filename in os.listdir(self.path):
            self.add_data(filename)

    def add_data(self, filename):
        match = re.match("[0-9]+_[0-9]+_[0-9]+_[0-9]+_[0-9]+_[0-9]+_([0-9]+)_([0-9]+)_([0-9]+)_([0-9]+)_([0-9]+).npz", filename)
        if match:
            new_data = np.load(self.path + '/' + filename)
            run_description = self.RunDescription(*(int(x) for x in match.groups()))
            run_data = self.RunData(new_data["sources"], new_data["shared"])
            self.data[run_description].append(run_data)

    def get_final_reconstruction_errors_means_stds(self, n=100):
        sources_data = {
            run_description: np.array([run_data.sources[-n:] for run_data in run_data_list])
            for run_description, run_data_list in self.data.items()
        }
        shared_data = {
            run_description: np.array([run_data.shared[-n:] for run_data in run_data_list])
            for run_description, run_data_list in self.data.items()
        }
        sources_means = {
            run_description: np.mean(data)
            for run_description, data in sources_data.items()
        }
        sources_stds = {
            run_description: np.std(data)
            for run_description, data in sources_data.items()
        }
        shared_means = {
            run_description: np.mean(data)
            for run_description, data in shared_data.items()
        }
        shared_stds = {
            run_description: np.std(data)
            for run_description, data in shared_data.items()
        }
        return sources_means, sources_stds, shared_means, shared_stds


if __name__ == '__main__':
    c = Collection('../data/trash/')

    c.compute_fits()
    fits = c.compute_fits_fixed_u0()
    print(fits)
