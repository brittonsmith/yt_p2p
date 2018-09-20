from matplotlib import pyplot
import numpy as np
import os
from yt.visualization.color_maps import \
    yt_colormaps

from yt.extensions.p2p.plots import \
    make_phase_plot

from grid_figure import GridFigure

def plot_density_H2_fraction(data_dir, halo_id):
    filename = os.path.join(data_dir, "H2_fraction_%06d.h5" % halo_id)

    fontsize = 14
    my_fig = GridFigure(
        1, 1, figsize=(6, 4.5),
        top_buffer = 0.03, bottom_buffer = 0.13,
        left_buffer = 0.14, right_buffer = 0.19)

    my_axes = my_fig[0]
    xscale = 'log'
    yscale = 'log'

    field = 'cell_mass'
    units = 'Msun'
    cmap = yt_colormaps['dusk']
    clabel = "M [M$_{\\odot}$]"

    xlim = (1e-3, 1e13)
    xmajor = np.logspace(-3, 12, 6)
    xminor = np.logspace(-3, 13, 17)
    xlabel = "n [cm$^{-3}$]"

    ylim = (1e-8, 1)
    ymajor = np.logspace(-8, 0, 9)
    yminor = None
    ylabel = "f$_{\\rm H_{2}}$"

    output_filename = "figures/density_H2_fraction.pdf"

    make_phase_plot(
        my_fig, my_axes, filename,
        field, units, cmap, clabel,
        xlim, xmajor, xminor, xscale, xlabel,
        ylim, ymajor, yminor, yscale, ylabel,
        output_filename)

def plot_density_temperature(data_dir, halo_id):
    filename = os.path.join(data_dir, "temperature_%06d.h5" % halo_id)

    my_fig = GridFigure(
        1, 1, figsize=(6, 4.5),
        top_buffer = 0.02, bottom_buffer = 0.13,
        left_buffer = 0.12, right_buffer = 0.19)

    my_axes = my_fig[0]
    xscale = 'log'
    yscale = 'log'

    field = 'cell_mass'
    units = 'Msun'
    cmap = yt_colormaps['dusk']
    clabel = "M [M$_{\\odot}$]"

    xlim = (1e-3, 1e13)
    xmajor = np.logspace(-3, 12, 6)
    xminor = np.logspace(-3, 13, 17)
    xlabel = "n [cm$^{-3}$]"

    ylim = (10, 2e4)
    ymajor = np.logspace(1, 4, 4)
    yminor = None
    ylabel = "T [K]"

    output_filename = "figures/density_temperature.pdf"

    make_phase_plot(
        my_fig, my_axes, filename,
        field, units, cmap, clabel,
        xlim, xmajor, xminor, xscale, xlabel,
        ylim, ymajor, yminor, yscale, ylabel,
        output_filename)

if __name__ == "__main__":
    data_dir = "../halo_catalogs/profile_catalogs/DD0560/density_profiles"
    halo_id = 41732
    plot_density_H2_fraction(data_dir, halo_id)
    pyplot.clf()
    plot_density_temperature(data_dir, halo_id)
