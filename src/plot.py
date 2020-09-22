import numpy as np
import matplotlib.pyplot as plt
from collection import Collection
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


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

    # for suffix in ["", "_mode_correlates", "_mode_sources_no_repetition", "_binary", "_binary_0.1", "_mode_correlates_5_sources", "_mode_correlates_fancy_loss", "_mode_correlates_dim_correlate_10", "_mode_correlates_dim_correlate_15", "_mode_correlates_dim_correlate_20", "_mode_correlates_dim_correlate_25", "_mode_correlates_dim_correlate_50", "_mode_correlates_dim_correlate_100", "_mode_correlates_dim_correlate_200", "_mode_correlates_dim_correlate_500", "_dim_correlate_1000"]:
    #     col = Collection('../data/wrt_dim_latent{}/'.format(suffix))
    #     sources_means, sources_stds, shared_means, shared_stds = col.get_final_reconstruction_errors_means_stds(n=50)
    #
    #     with FigureManager("/tmp/new_final_recerr{}.png".format(suffix)) as fig:
    #         ax = fig.add_subplot(111)
    #         plot_convergence_value_wrt_1d(ax, sources_means, sources_stds, shared_means, shared_stds,
    #             wrt="dim_latent",
    #             xlabel="latent dimension",
    #             ylabel="mean reconstruction error",
    #             title="Reconstruction error of the readouts (n_shared=10, n_non_shared=5x2)"
    #         )



    # for collection_name in ["mutual_4_exclusive_10_n_2", "mutual_4_exclusive_16_n_2", "mutual_4_exclusive_22_n_2", "mutual_4_exclusive_4_n_2", "mutual_4_exclusive_4_n_1_exp", "mutual_4_exclusive_4_n_2_exp", "mutual_4_exclusive_4_n_3_exp", "mutual_4_exclusive_4_n_4_exp", "mutual_4_exclusive_4_n_5_exp", "mutual_4_exclusive_4_n_6_exp"]:
    #     col = Collection('../data/{}/'.format(collection_name))
    #     with FigureManager("../tmp/{}.png".format(collection_name)) as fig:
    #         ax = fig.add_subplot(111)
    #         col.plot_wrt_latent_dim(ax)
        # sources_means, sources_stds, shared_means, shared_stds = col.get_final_reconstruction_errors_means_stds()
        #
        # with FigureManager("../tmp/{}.png".format(collection_name)) as fig:
        #     ax = fig.add_subplot(111)
        #     plot_convergence_value_wrt_1d(ax, sources_means, sources_stds, shared_means, shared_stds,
        #         wrt="dim_latent",
        #         xlabel="latent dimension",
        #         ylabel="mean reconstruction error",
        #         title="Reconstruction error of the readouts",
        #     )
        #
        #     keys = [key for key in sources_means.keys()]
        #     keys.sort(key=lambda x: x.dim_latent)
        #     keys = keys[-3:]
        #     sources_means = {key: sources_means[key] for key in keys}
        #     sources_stds = {key: sources_stds[key] for key in keys}
        #     shared_means = {key: shared_means[key] for key in keys}
        #     shared_stds = {key: shared_stds[key] for key in keys}
        #     inset = inset_axes(ax, width="30%", height="30%", loc=1)
        #     plot_convergence_value_wrt_1d(inset, sources_means, sources_stds, shared_means, shared_stds,
        #         wrt="dim_latent",
        #         legend=False,
        #     )
        #     inset.set_ylim([0, None])

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica"]})

    with FigureManager("../tmp/vary_exclusive_dim.png") as fig:
        ax = fig.add_subplot(141)
        col = Collection('../data/mutual_4_exclusive_4_n_2')
        col.plot_wrt_latent_dim(ax, legend=False, ylabel=True, title='exclusive')

        ax = fig.add_subplot(142)
        col = Collection('../data/mutual_4_exclusive_10_n_2')
        col.plot_wrt_latent_dim(ax, legend=False, title='exclusive')

        ax = fig.add_subplot(143)
        col = Collection('../data/mutual_4_exclusive_16_n_2')
        col.plot_wrt_latent_dim(ax, legend=False, title='exclusive')

        ax = fig.add_subplot(144)
        col = Collection('../data/mutual_4_exclusive_22_n_2')
        col.plot_wrt_latent_dim(ax, legend=False, title='exclusive')

    with FigureManager("../tmp/vary_n_sources.png") as fig:
        # ax = fig.add_subplot(141)
        # col = Collection('../data/mutual_4_exclusive_4_n_1_exp')
        # col.plot_wrt_latent_dim(ax, legend=False, ylabel=True, title='n_sources')

        ax = fig.add_subplot(141)
        col = Collection('../data/mutual_4_exclusive_4_n_2_exp')
        col.plot_wrt_latent_dim(ax, legend=False, ylabel=True, title='n_sources')

        ax = fig.add_subplot(142)
        col = Collection('../data/mutual_4_exclusive_4_n_3_exp')
        col.plot_wrt_latent_dim(ax, legend=False, title='n_sources')

        ax = fig.add_subplot(143)
        col = Collection('../data/mutual_4_exclusive_4_n_4_exp')
        col.plot_wrt_latent_dim(ax, legend=False, title='n_sources')

        ax = fig.add_subplot(144)
        col = Collection('../data/mutual_4_exclusive_4_n_5_exp')
        col.plot_wrt_latent_dim(ax, legend=False, title='n_sources')

        # ax = fig.add_subplot(146)
        # col = Collection('../data/mutual_4_exclusive_4_n_6_exp')
        # col.plot_wrt_latent_dim(ax, legend=False, title='n_sources')
