import numpy as np
import os
from collections import defaultdict, namedtuple
import re
from scipy.optimize import curve_fit


def fit_function(x_data, u0, a, r):
    return (a ** x_data) *(u0 - r) + r


def fit_function_factory(u0):
    def func(x_data, a, r):
        return (a ** x_data) *(u0 - r) + r
    return func


class Collection(object):
    def __init__(self, path):
        self.path = path
        self.data = defaultdict(list)
        self.Fit = namedtuple('Fit', ['a', 'r'])
        self.U0Fit = namedtuple('U0Fit', ['u0', 'a', 'r'])
        self.RunFit = namedtuple('RunFit', ['sources', 'shared'])
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

    def compute_fits_fixed_u0(self):
        fits = self.compute_fits()
        count = 0
        u0 = 0
        for run_description, fits_list in fits.items():
            u0 += np.sum(fit.sources.u0 for fit in fits_list)
            u0 += np.sum(fit.shared.u0 for fit in fits_list)
            count += 2 * len(fits_list)
        u0 /= count
        return self.compute_fits(sources_u0=u0, shared_u0=u0)

    def compute_fits(self, sources_u0=None, shared_u0=None):
        fits = defaultdict(list)
        for run_description, run_data_list in self.data.items():
            fits[run_description] = [self.compute_fit(run_data, sources_u0=sources_u0, shared_u0=shared_u0) for run_data in run_data_list]
        return fits

    def compute_fit(self, run_data, sources_u0=None, shared_u0=None):
        sources_upper = np.max(run_data.sources)
        shared_upper = np.max(run_data.shared)
        sources_lower = np.min(run_data.sources)
        shared_lower = np.min(run_data.shared)
        if sources_u0 is not None:
            try:
                bounds = ([0.0, sources_lower], [1.0, sources_upper])
                popt_sources, pcov_sources = curve_fit(
                    fit_function_factory(sources_u0),
                    np.arange(len(run_data.sources)),
                    run_data.sources,
                    bounds=bounds,
                )
            except RuntimeError as e:
                mean = np.mean(run_data.sources)
                popt_sources = [0.0, mean]
                print("Could not fit run, using {}".format(popt_sources))
            sources_fit = self.Fit(*popt_sources)
        else:
            try:
                bounds = ([sources_lower, 0.0, sources_lower], [sources_upper, 1.0, sources_upper])
                popt_sources, pcov_sources = curve_fit(
                    fit_function,
                    np.arange(len(run_data.sources)),
                    run_data.sources,
                    bounds=bounds,
                )
            except RuntimeError as e:
                mean = np.mean(run_data.sources)
                popt_sources = [mean, 0.0, mean]
                print("Could not fit run, using {}".format(popt_sources))
            sources_fit = self.U0Fit(*popt_sources)
        if shared_u0 is not None:
            try:
                bounds = ([0.0, shared_lower], [1.0, shared_upper])
                popt_shared, pcov_shared = curve_fit(
                    fit_function_factory(shared_u0),
                    np.arange(len(run_data.shared)),
                    run_data.shared,
                    bounds=bounds,
                )
            except RuntimeError as e:
                mean = np.mean(run_data.shared)
                popt_shared = [0.0, mean]
                print("Could not fit run, using {}".format(popt_shared))
            shared_fit = self.Fit(*popt_shared)
        else:
            try:
                bounds = ([shared_lower, 0.0, shared_lower], [shared_upper, 1.0, shared_upper])
                popt_shared, pcov_shared = curve_fit(
                    fit_function,
                    np.arange(len(run_data.shared)),
                    run_data.shared,
                    bounds=bounds,
                )
            except RuntimeError as e:
                mean = np.mean(run_data.shared)
                popt_shared = [mean, 0.0, mean]
                print("Could not fit run, using {}".format(popt_shared))
            shared_fit = self.U0Fit(*popt_shared)
        return self.RunFit(sources_fit, shared_fit)


if __name__ == '__main__':
    c = Collection('../data/trash/')

    c.compute_fits()
    fits = c.compute_fits_fixed_u0()
    print(fits)
