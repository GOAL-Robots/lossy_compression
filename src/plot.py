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


def plot_convergence_value_wrt_1d(ax, fits, wrt, xlabel=None, ylabel=None, title=None):
    keys = list(fits.keys())
    keys.sort(key=lambda x: getattr(x, wrt))
    n_runs = len(fits[keys[0]])
    x = np.array([getattr(key, wrt) for key in keys])
    y_sources_mean = [np.mean([fit.sources.r for fit in fits[key]]) for key in keys]
    y_shared_mean = [np.mean([fit.shared.r for fit in fits[key]]) for key in keys]
    width = 0.4
    if n_runs > 1:
        y_sources_std = [np.std([fit.sources.r for fit in fits[key]]) for key in keys]
        y_shared_std = [np.std([fit.shared.r for fit in fits[key]]) for key in keys]
        sources_rects = ax.bar(x - width / 2, y_sources_mean, width, yerr=y_sources_std, label='sources')
        shared_rects = ax.bar(x + width / 2, y_shared_mean, width, yerr=y_shared_std, label='shared')
    else:
        sources_rects = ax.bar(x - width / 2, y_sources_mean, width, label='sources')
        shared_rects = ax.bar(x + width / 2, y_shared_mean, width, label='shared')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()


def plot_speed_value_wrt_1d(ax, fits, wrt, xlabel=None, ylabel=None, title=None):
    keys = list(fits.keys())
    keys.sort(key=lambda x: getattr(x, wrt))
    n_runs = len(fits[keys[0]])
    x = np.array([getattr(key, wrt) for key in keys])
    y_sources_mean = [np.mean([fit.sources.a for fit in fits[key]]) for key in keys]
    y_shared_mean = [np.mean([fit.shared.a for fit in fits[key]]) for key in keys]
    width = 0.4
    if n_runs > 1:
        y_sources_std = [np.std([fit.sources.a for fit in fits[key]]) for key in keys]
        y_shared_std = [np.std([fit.shared.a for fit in fits[key]]) for key in keys]
        sources_rects = ax.bar(x - width / 2, y_sources_mean, width, yerr=y_sources_std, label='sources')
        shared_rects = ax.bar(x + width / 2, y_shared_mean, width, yerr=y_shared_std, label='shared')
    else:
        sources_rects = ax.bar(x - width / 2, y_sources_mean, width, label='sources')
        shared_rects = ax.bar(x + width / 2, y_shared_mean, width, label='shared')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()

if __name__ == '__main__':

    for suffix in ["", "_mode_correlates", "_mode_sources_no_repetition", "_binary", "_binary_0.1", "_mode_correlates_5_sources", "_mode_correlates_fancy_loss", "_mode_correlates_dim_correlate_10", "_mode_correlates_dim_correlate_15", "_mode_correlates_dim_correlate_20", "_mode_correlates_dim_correlate_25", "_mode_correlates_dim_correlate_50", "_mode_correlates_dim_correlate_100", "_mode_correlates_dim_correlate_200", "_mode_correlates_dim_correlate_500", "_dim_correlate_1000"]:
        col = Collection('../data/wrt_dim_latent{}/'.format(suffix))
        fits = col.compute_fits()

        with FigureManager("/tmp/final_recerr{}.png".format(suffix)) as fig:
            ax = fig.add_subplot(111)
            plot_convergence_value_wrt_1d(ax, fits, wrt="dim_latent", xlabel="latent dimension", ylabel="mean reconstruction error", title="Reconstruction error of the readouts (n_shared=10, n_non_shared=5x2)")

        # with FigureManager("/tmp/speed{}.png".format(suffix)) as fig:
        #     ax = fig.add_subplot(111)
        #     plot_speed_value_wrt_1d(ax, fits, wrt="dim_latent", xlabel="latent dimension", ylabel="batch reconstruction error", title="Reconstruction error of the readouts")
