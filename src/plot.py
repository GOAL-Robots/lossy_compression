import numpy as np
import matplotlib.pyplot as plt
from collection import Collection
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


class FigureManager:
    def __init__(self, filepath, save=True, figsize=(8.53, 4.8), dpi=200):
        self._fig = plt.figure(figsize=figsize, dpi=dpi)
        self._filepath = filepath
        self._save = save

    def subplots_adjust(self, left=None, bottom=None, right=None, top=None, wspace=None, hspace=None):
        self._fig.subplots_adjust(
            left=left,
            bottom=bottom,
            right=right,
            top=top,
            wspace=wspace,
            hspace=hspace
        )

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
        wrt, xlabel=None, ylabel=None, title=None, legend=True):
    keys = list(sources_means.keys())
    keys.sort(key=lambda x: getattr(x, wrt))
    x = np.array([getattr(key, wrt) for key in keys])
    sources_means = np.array([sources_means[key] for key in keys])
    sources_stds = np.array([sources_stds[key] for key in keys])
    shared_means = np.array([shared_means[key] for key in keys])
    shared_stds = np.array([shared_stds[key] for key in keys])
    ax.plot(x, sources_means, color='b', linestyle='--', marker='o', label="exclusive")
    ax.plot(x, shared_means, color='r', linestyle='--', marker='o', label="shared")
    ax.fill_between(x, sources_means - sources_stds, sources_means + sources_stds, color='b', alpha=0.5)
    ax.fill_between(x, shared_means - shared_stds, shared_means + shared_stds, color='r', alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if legend:
        ax.legend(loc='center right')


if __name__ == '__main__':

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica"]})

    # with FigureManager("../tmp/je_vary_exclusive_dim.png") as fig:
    #     ax = fig.add_subplot(141)
    #     col = Collection('../data/je_mutual_4_exclusive_4_n_2')
    #     col.plot_wrt_latent_dim(ax, legend=False, ylabel=True, title='exclusive')
    #
    #     ax = fig.add_subplot(142)
    #     col = Collection('../data/je_mutual_4_exclusive_10_n_2')
    #     col.plot_wrt_latent_dim(ax, legend=False, title='exclusive')
    #
    #     ax = fig.add_subplot(143)
    #     col = Collection('../data/je_mutual_4_exclusive_16_n_2')
    #     col.plot_wrt_latent_dim(ax, legend=False, title='exclusive')
    #
    #     ax = fig.add_subplot(144)
    #     col = Collection('../data/je_mutual_4_exclusive_22_n_2')
    #     col.plot_wrt_latent_dim(ax, legend=False, title='exclusive')
    #
    # with FigureManager("../tmp/je_vary_n_sources.png") as fig:
    #     ax = fig.add_subplot(141)
    #     col = Collection('../data/je_mutual_4_exclusive_4_n_2_exp')
    #     col.plot_wrt_latent_dim(ax, legend=False, ylabel=True, title='n_sources')
    #
    #     ax = fig.add_subplot(142)
    #     col = Collection('../data/je_mutual_4_exclusive_4_n_3_exp')
    #     col.plot_wrt_latent_dim(ax, legend=False, title='n_sources')
    #
    #     ax = fig.add_subplot(143)
    #     col = Collection('../data/je_mutual_4_exclusive_4_n_4_exp')
    #     col.plot_wrt_latent_dim(ax, legend=False, title='n_sources')
    #
    #     ax = fig.add_subplot(144)
    #     col = Collection('../data/je_mutual_4_exclusive_4_n_5_exp')
    #     col.plot_wrt_latent_dim(ax, legend=False, title='n_sources')
    #
    #
    #
    # with FigureManager("../tmp/jes_vary_exclusive_dim.png") as fig:
    #     ax = fig.add_subplot(141)
    #     col = Collection('../data/jes_mutual_4_exclusive_4_n_2')
    #     col.plot_wrt_latent_dim(ax, legend=False, ylabel=True, title='exclusive')
    #
    #     ax = fig.add_subplot(142)
    #     col = Collection('../data/jes_mutual_4_exclusive_10_n_2')
    #     col.plot_wrt_latent_dim(ax, legend=False, title='exclusive')
    #
    #     ax = fig.add_subplot(143)
    #     col = Collection('../data/jes_mutual_4_exclusive_16_n_2')
    #     col.plot_wrt_latent_dim(ax, legend=False, title='exclusive')
    #
    #     ax = fig.add_subplot(144)
    #     col = Collection('../data/jes_mutual_4_exclusive_22_n_2')
    #     col.plot_wrt_latent_dim(ax, legend=False, title='exclusive')
    #
    # with FigureManager("../tmp/jes_vary_n_sources.png") as fig:
    #     ax = fig.add_subplot(141)
    #     col = Collection('../data/jes_mutual_4_exclusive_4_n_2_exp')
    #     col.plot_wrt_latent_dim(ax, legend=False, ylabel=True, title='n_sources')
    #
    #     ax = fig.add_subplot(142)
    #     col = Collection('../data/jes_mutual_4_exclusive_4_n_3_exp')
    #     col.plot_wrt_latent_dim(ax, legend=False, title='n_sources')
    #
    #     ax = fig.add_subplot(143)
    #     col = Collection('../data/jes_mutual_4_exclusive_4_n_4_exp')
    #     col.plot_wrt_latent_dim(ax, legend=False, title='n_sources')
    #
    #     ax = fig.add_subplot(144)
    #     col = Collection('../data/jes_mutual_4_exclusive_4_n_5_exp')
    #     col.plot_wrt_latent_dim(ax, legend=False, title='n_sources')
    #
    #
    #
    # with FigureManager("../tmp/cm_vary_exclusive_dim.png") as fig:
    #     ax = fig.add_subplot(141)
    #     col = Collection("../data/cm_mutual_4_exclusive_4_n_2_wrt_de_after_debug")
    #     col.plot_wrt_latent_dim(ax, legend=False, ylabel=True, title='exclusive', vline=False)
    #
    #     ax = fig.add_subplot(142)
    #     col = Collection("../data/cm_mutual_4_exclusive_10_n_2_wrt_de_after_debug")
    #     col.plot_wrt_latent_dim(ax, legend=False, title='exclusive', vline=False)
    #
    #     ax = fig.add_subplot(143)
    #     col = Collection("../data/cm_mutual_4_exclusive_16_n_2_wrt_de_after_debug")
    #     col.plot_wrt_latent_dim(ax, legend=False, title='exclusive', vline=False)
    #
    #     ax = fig.add_subplot(144)
    #     col = Collection("../data/cm_mutual_4_exclusive_22_n_2_wrt_de_after_debug")
    #     col.plot_wrt_latent_dim(ax, legend=False, title='exclusive', vline=False)
    #
    # with FigureManager("../tmp/cm_vary_n_sources.png") as fig:
    #     ax = fig.add_subplot(141)
    #     col = Collection('../data/cm_mutual_4_exclusive_4_n_2_wrt_n_after_debug')
    #     col.plot_wrt_latent_dim(ax, legend=False, ylabel=True, title='n_sources', vline=False)
    #
    #     ax = fig.add_subplot(142)
    #     col = Collection('../data/cm_mutual_4_exclusive_4_n_3_wrt_n_after_debug')
    #     col.plot_wrt_latent_dim(ax, legend=False, title='n_sources', vline=False)
    #
    #     ax = fig.add_subplot(143)
    #     col = Collection('../data/cm_mutual_4_exclusive_4_n_4_wrt_n_after_debug')
    #     col.plot_wrt_latent_dim(ax, legend=False, title='n_sources', vline=False)
    #
    #     ax = fig.add_subplot(144)
    #     col = Collection('../data/cm_mutual_4_exclusive_4_n_5_wrt_n_after_debug')
    #     col.plot_wrt_latent_dim(ax, legend=False, title='n_sources', vline=False)


    ############################################################################
    ############################################################################
    ############################################################################
    ############################################################################

    offset = 0.016

    with FigureManager("../tmp/vary_exclusive_dim.png", figsize=(5, 6)) as fig:
        ax = fig.add_subplot(3, 4, 1)
        col = Collection('../data/je_mutual_4_exclusive_4_n_2')
        col.plot_wrt_latent_dim(ax, legend=False, ylabel=True, title='exclusive', xlabel=False)

        ax = fig.add_subplot(3, 4, 2)
        col = Collection('../data/je_mutual_4_exclusive_10_n_2')
        col.plot_wrt_latent_dim(ax, legend=False, title='exclusive', xlabel=False)

        ax = fig.add_subplot(3, 4, 3)
        col = Collection('../data/je_mutual_4_exclusive_16_n_2')
        col.plot_wrt_latent_dim(ax, legend=False, title='exclusive', xlabel=False)

        ax = fig.add_subplot(3, 4, 4)
        col = Collection('../data/je_mutual_4_exclusive_22_n_2')
        col.plot_wrt_latent_dim(ax, legend=False, title='exclusive', xlabel=False)

        #

        ax = fig.add_subplot(3, 4, 5)
        col = Collection('../data/jes_mutual_4_exclusive_4_n_2')
        col.plot_wrt_latent_dim(ax, legend=False, ylabel=True, title='exclusive', xlabel=False)

        ax = fig.add_subplot(3, 4, 6)
        col = Collection('../data/jes_mutual_4_exclusive_10_n_2')
        col.plot_wrt_latent_dim(ax, legend=False, title='exclusive', xlabel=False)

        ax = fig.add_subplot(3, 4, 7)
        col = Collection('../data/jes_mutual_4_exclusive_16_n_2')
        col.plot_wrt_latent_dim(ax, legend=False, title='exclusive', xlabel=False)

        ax = fig.add_subplot(3, 4, 8)
        col = Collection('../data/jes_mutual_4_exclusive_22_n_2')
        col.plot_wrt_latent_dim(ax, legend=False, title='exclusive', xlabel=False)

        #

        ax = fig.add_subplot(3, 4, 9)
        col = Collection("../data/cm_mutual_4_exclusive_4_n_2_wrt_de_after_debug")
        col.plot_wrt_latent_dim(ax, legend=False, ylabel=True, title='exclusive', vline=False)

        ax = fig.add_subplot(3, 4, 10)
        col = Collection("../data/cm_mutual_4_exclusive_10_n_2_wrt_de_after_debug")
        col.plot_wrt_latent_dim(ax, legend=False, title='exclusive', vline=False)

        ax = fig.add_subplot(3, 4, 11)
        col = Collection("../data/cm_mutual_4_exclusive_16_n_2_wrt_de_after_debug")
        col.plot_wrt_latent_dim(ax, legend=False, title='exclusive', vline=False)

        ax = fig.add_subplot(3, 4, 12)
        col = Collection("../data/cm_mutual_4_exclusive_22_n_2_wrt_de_after_debug")
        col.plot_wrt_latent_dim(ax, legend=False, title='exclusive', vline=False)

        fig.subplots_adjust(hspace=0.9, wspace=0.1, top=0.9, bottom=0.1, left=0.2, right=0.95)
        fig.text(0.02, 1.0000 + - 0.05 + 1 * offset, "A", fontsize='xx-large')
        fig.text(0.02, 0.6666 + - 0.05 + 2 * offset, "B", fontsize='xx-large')
        fig.text(0.02, 0.3333 + - 0.05 + 3 * offset, "C", fontsize='xx-large')
        fig.text(0.07, 1.0000 + - 0.05 + 1 * offset, "joint encoding of the $y_i$~s", fontsize='large')
        fig.text(0.07, 0.6666 + - 0.05 + 2 * offset, "joint encoding of the $y_i$~s, reconstruction of the $x_i$~s", fontsize='large')
        fig.text(0.07, 0.3333 + - 0.05 + 3 * offset, "cross-modality prediction based encoding", fontsize='large')


    with FigureManager("../tmp/vary_n_sources.png", figsize=(5, 6)) as fig:
        ax = fig.add_subplot(3, 4, 1)

        col = Collection('../data/je_mutual_4_exclusive_4_n_2_exp')
        col.plot_wrt_latent_dim(ax, legend=False, ylabel=True, title='n_sources', xlabel=False)

        ax = fig.add_subplot(3, 4, 2)
        col = Collection('../data/je_mutual_4_exclusive_4_n_3_exp')
        col.plot_wrt_latent_dim(ax, legend=False, title='n_sources', xlabel=False)

        ax = fig.add_subplot(3, 4, 3)
        col = Collection('../data/je_mutual_4_exclusive_4_n_4_exp')
        col.plot_wrt_latent_dim(ax, legend=False, title='n_sources', xlabel=False)

        ax = fig.add_subplot(3, 4, 4)
        col = Collection('../data/je_mutual_4_exclusive_4_n_5_exp')
        col.plot_wrt_latent_dim(ax, legend=False, title='n_sources', xlabel=False)

        #

        ax = fig.add_subplot(3, 4, 5)
        col = Collection('../data/jes_mutual_4_exclusive_4_n_2_exp')
        col.plot_wrt_latent_dim(ax, legend=False, ylabel=True, title='n_sources', xlabel=False)

        ax = fig.add_subplot(3, 4, 6)
        col = Collection('../data/jes_mutual_4_exclusive_4_n_3_exp')
        col.plot_wrt_latent_dim(ax, legend=False, title='n_sources', xlabel=False)

        ax = fig.add_subplot(3, 4, 7)
        col = Collection('../data/jes_mutual_4_exclusive_4_n_4_exp')
        col.plot_wrt_latent_dim(ax, legend=False, title='n_sources', xlabel=False)

        ax = fig.add_subplot(3, 4, 8)
        col = Collection('../data/jes_mutual_4_exclusive_4_n_5_exp')
        col.plot_wrt_latent_dim(ax, legend=False, title='n_sources', xlabel=False)

        #

        ax = fig.add_subplot(3, 4, 9)
        col = Collection('../data/cm_mutual_4_exclusive_4_n_2_wrt_n_after_debug')
        col.plot_wrt_latent_dim(ax, legend=False, ylabel=True, title='n_sources', vline=False)

        ax = fig.add_subplot(3, 4, 10)
        col = Collection('../data/cm_mutual_4_exclusive_4_n_3_wrt_n_after_debug')
        col.plot_wrt_latent_dim(ax, legend=False, title='n_sources', vline=False)

        ax = fig.add_subplot(3, 4, 11)
        col = Collection('../data/cm_mutual_4_exclusive_4_n_4_wrt_n_after_debug')
        col.plot_wrt_latent_dim(ax, legend=False, title='n_sources', vline=False)

        ax = fig.add_subplot(3, 4, 12)
        col = Collection('../data/cm_mutual_4_exclusive_4_n_5_wrt_n_after_debug')
        col.plot_wrt_latent_dim(ax, legend=False, title='n_sources', vline=False)

        fig.subplots_adjust(hspace=0.9, wspace=0.1, top=0.9, bottom=0.1, left=0.2, right=0.95)
        fig.text(0.02, 1.0000 + - 0.05 + 1 * offset, "A", fontsize='xx-large')
        fig.text(0.02, 0.6666 + - 0.05 + 2 * offset, "B", fontsize='xx-large')
        fig.text(0.02, 0.3333 + - 0.05 + 3 * offset, "C", fontsize='xx-large')
        fig.text(0.07, 1.0000 + - 0.05 + 1 * offset, "joint encoding of the $y_i$~s", fontsize='large')
        fig.text(0.07, 0.6666 + - 0.05 + 2 * offset, "joint encoding of the $y_i$~s, reconstruction of the $x_i$~s", fontsize='large')
        fig.text(0.07, 0.3333 + - 0.05 + 3 * offset, "cross-modality prediction based encoding", fontsize='large')


    with FigureManager("../tmp/biig_batch.png", figsize=(5, 5)) as fig:
        ax = fig.add_subplot(111)

        col = Collection('../data/jes_mutual_4_exclusive_4_n_2_bs_1024')
        col.plot_wrt_latent_dim(ax, legend=False, ylabel=True, title='n_sources')
