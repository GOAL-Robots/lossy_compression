import numpy as np
import matplotlib.pyplot as plt
from collection import Collection


class FigureManager:
    def __init__(self, filepath, save=True):
        self._fig = plt.figure(figsize=(8.53, 4.8), dpi=200)
        self._filepath = filepath
        self._save = save

    def __enter__(self):
        return self._fig

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self._save:
            print("saving plot {}  ...  ".format(self._filepath), end="")
            self._fig.savefig(self._filepath)
            print("done")
            plt.close(self._fig)
        else:
            plt.show()


def plot_convergence_value_wrt_1d(ax, sources_means, sources_stds, shared_means, shared_stds,
        wrt, xlabel=None, ylabel=None, title=None):
    keys = list(sources_means.keys())
    keys.sort(key=lambda x: getattr(x, wrt))
    x = np.array([getattr(key, wrt) for key in keys])
    sources_means = np.array([sources_means[key] for key in keys])
    sources_stds = np.array([sources_stds[key] for key in keys])
    shared_means = np.array([shared_means[key] for key in keys])
    shared_stds = np.array([shared_stds[key] for key in keys])
    ax.plot(x, sources_means, color='b', linestyle='--', marker='o', label="sources")
    ax.plot(x, shared_means, color='r', linestyle='--', marker='o', label="shared")
    ax.fill_between(x, sources_means - sources_stds, sources_means + sources_stds, color='b', alpha=0.5)
    ax.fill_between(x, shared_means - shared_stds, shared_means + shared_stds, color='r', alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()


if __name__ == '__main__':

    for suffix in ["", "_mode_correlates", "_mode_sources_no_repetition", "_binary", "_binary_0.1", "_mode_correlates_5_sources", "_mode_correlates_fancy_loss", "_mode_correlates_dim_correlate_10", "_mode_correlates_dim_correlate_15", "_mode_correlates_dim_correlate_20", "_mode_correlates_dim_correlate_25", "_mode_correlates_dim_correlate_50", "_mode_correlates_dim_correlate_100", "_mode_correlates_dim_correlate_200", "_mode_correlates_dim_correlate_500", "_dim_correlate_1000"]:
        col = Collection('../data/wrt_dim_latent{}/'.format(suffix))
        sources_means, sources_stds, shared_means, shared_stds = col.get_final_reconstruction_errors_means_stds(n=50)

        with FigureManager("/tmp/new_final_recerr{}.png".format(suffix)) as fig:
            ax = fig.add_subplot(111)
            plot_convergence_value_wrt_1d(ax, sources_means, sources_stds, shared_means, shared_stds,
                wrt="dim_latent",
                xlabel="latent dimension",
                ylabel="mean reconstruction error",
                title="Reconstruction error of the readouts (n_shared=10, n_non_shared=5x2)"
            )

        # with FigureManager("/tmp/speed{}.png".format(suffix)) as fig:
        #     ax = fig.add_subplot(111)
        #     plot_speed_value_wrt_1d(ax, fits, wrt="dim_latent", xlabel="latent dimension", ylabel="batch reconstruction error", title="Reconstruction error of the readouts")
