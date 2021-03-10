import numpy as np
import os
from collections import defaultdict, namedtuple
import re
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


class Collection(object):
    def __init__(self, path):
        self.path = path
        self.name = path.strip("/").split("/")[-1]
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

    def get_data(self):
        sources_data = {
            run_description: np.array([run_data.sources for run_data in run_data_list])
            for run_description, run_data_list in self.data.items()
        }
        shared_data = {
            run_description: np.array([run_data.shared for run_data in run_data_list])
            for run_description, run_data_list in self.data.items()
        }
        return sources_data, shared_data

    def get_final_reconstruction_errors_means_stds(self):
        sources_data, shared_data = self.get_data()
        sources_means = {
            run_description: np.mean(data)
            for run_description, data in sources_data.items()
        }
        sources_stds = {
            run_description: np.std(np.mean(data, axis=-1))
            for run_description, data in sources_data.items()
        }
        shared_means = {
            run_description: np.mean(data)
            for run_description, data in shared_data.items()
        }
        shared_stds = {
            run_description: np.std(np.mean(data, axis=-1))
            for run_description, data in shared_data.items()
        }
        return sources_means, sources_stds, shared_means, shared_stds

    def plot_wrt_latent_dim(self, ax, legend=True, lasts=3, inset=False, xlabel=True, ylabel=False, title='exclusive', vline=True):
        sources_data, shared_data = self.get_data()
        sources_means, sources_stds, shared_means, shared_stds = self.get_final_reconstruction_errors_means_stds()
        keys = list(sources_means.keys())
        keys.sort(key=lambda x: x.dim_latent)
        x = np.array([key.dim_latent for key in keys])
        sources_data = np.array([np.mean(sources_data[key], axis=-1) for key in keys])
        shared_data = np.array([np.mean(shared_data[key], axis=-1) for key in keys])
        sources_means = np.array([sources_means[key] for key in keys])
        sources_stds = np.array([sources_stds[key] for key in keys])
        shared_means = np.array([shared_means[key] for key in keys])
        shared_stds = np.array([shared_stds[key] for key in keys])
        # ax.plot([0], [1], color='grey', marker='o')
        # ax.plot([0, x[0]], [1, sources_means[0]], color='grey', linestyle='--')
        # ax.plot([0, x[0]], [1, shared_means[0]], color='grey', linestyle='--')
        x = np.concatenate([[0], x], axis=0)
        sources_means = np.concatenate([[1], sources_means], axis=0)
        sources_stds = np.concatenate([[0], sources_stds], axis=0)
        shared_means = np.concatenate([[1], shared_means], axis=0)
        shared_stds = np.concatenate([[0], shared_stds], axis=0)
        ax.plot(x, sources_means, color='b', linestyle='--', marker='o', label="exclusive", markersize=2)
        ax.plot(x, shared_means, color='r', linestyle='--', marker='o', label="shared", markersize=2)
        ax.fill_between(x, sources_means - sources_stds, sources_means + sources_stds, color='b', alpha=0.2)
        ax.fill_between(x, shared_means - shared_stds, shared_means + shared_stds, color='r', alpha=0.2)
        if vline:
            ax.axvline(keys[0].dim_shared + (keys[0].n_sources * keys[0].dim_sources), color='grey', linestyle=':')
        # for repetition in shared_data.T:
        #     repetition = np.concatenate([[1], repetition], axis=0)
        #     ax.plot(x, repetition, marker='x', alpha=0.5, color='r', linestyle='')
        if xlabel:
            ax.set_xlabel("latent\ndimension $d_z$")
        if ylabel:
            ax.set_ylabel("mean reconstruction\nerrors $r_m$ and $r_e$")
        else:
            ax.set_yticks([])
        if title == 'exclusive':
            title = r"$d_{e} = " + "{}$".format(keys[0].dim_sources)
        elif title == 'n_sources':
            title = r"$n = {}$".format(keys[0].n_sources)
        ax.set_title(title)
        if legend:
            ax.legend(loc='center right')
        ax.set_ylim([-0.05, 1.05])

        if inset:
            inset = inset_axes(ax, width="15%", height="30%", loc=1)
            inset.plot(x[-lasts:], sources_means[-lasts:], color='b', linestyle='--', marker='o', label="exclusive")
            inset.plot(x[-lasts:], shared_means[-lasts:], color='r', linestyle='--', marker='o', label="shared")
            inset.fill_between(x[-lasts:], sources_means[-lasts:] - sources_stds[-lasts:], sources_means[-lasts:] + sources_stds[-lasts:], color='b', alpha=0.5)
            inset.fill_between(x[-lasts:], shared_means[-lasts:] - shared_stds[-lasts:], shared_means[-lasts:] + shared_stds[-lasts:], color='r', alpha=0.5)
            inset.set_ylim([0, None])



if __name__ == '__main__':
    c = Collection('../data/trash/')

    c.compute_fits()
    fits = c.compute_fits_fixed_u0()
    print(fits)
