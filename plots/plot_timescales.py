from matplotlib import pyplot, ticker
import numpy as np
import os
import yt
from yt.visualization.color_maps import \
    yt_colormaps
from yt.units.yt_array import \
    YTQuantity, YTArray

import matplotlib as mpl
mpl.rcParams['axes.unicode_minus'] = False

from yt.extensions.p2p.plots import \
    plot_profile_distribution, \
    plot_profile_distribution_legend, \
    draw_major_grid, \
    twin_unit_axes, \
    mirror_yticks

from grid_figure import GridFigure

if __name__ == "__main__":
    my_fig = GridFigure(
        [3, 1], 1, figsize=(6, 4.5),
        top_buffer = 0.13, bottom_buffer = 0.12,
        left_buffer = 0.14, right_buffer = 0.02,
        horizontal_buffer = 0.05, vertical_buffer = 0)

    fontsize = 12

    colors = ["red", "green", "blue"]

    data_dir = "../halo_catalogs/profile_catalogs/DD0560/timescale_profiles"
    halo_id = 41732

    for i, my_axes in enumerate(my_fig):
        my_axes.set_xscale('log')
        my_axes.set_yscale('log')

        xlim = (2e-6, 3e2)
        tx = twin_unit_axes(
            my_axes, xlim, "r",
            "pc", top_units="AU")

        if i == 0:

            fields = ["dynamical_time", "cooling_time", "vortical_time"]
            units = ["%f*yr" % np.sqrt(2), "yr", "%f*yr" % (1/(2*np.pi))]
            for i, field in enumerate(fields):
                filename = os.path.join(data_dir, "%s_%06d.h5" % (field, halo_id))
                plot_profile_distribution(
                    my_axes, filename, 'cell_mass',
                    x_units="pc", y_units=units[i], alpha_scale=0.7,
                    pkwargs=dict(color=colors[i], linewidth=1))

            my_axes.xaxis.set_visible(False)

            ylim = (10, 1e10)
            ymajor = np.logspace(2, 10, 5)
            yminor = np.logspace(1, 9, 5)
            draw_major_grid(my_axes, 'y', ymajor,
                        color='black', linestyle='-',
                        linewidth=1, alpha=0.2)
            ty = mirror_yticks(my_axes, ylim, ymajor, yminor=yminor)
            my_axes.yaxis.set_label_text("t [yr]")
            # my_axes.yaxis.labelpad = 4

            labels = ("free-fall", "cooling", "mixing")
            dist = [True]*3
            plot_items = list(zip(colors, labels, dist))
            plot_profile_distribution_legend(
                my_axes, plot_items, alpha_scale=0.7,
                lheight=0.27, label_rotation=315)

        if i == 1:

            fields = ["cooling_dynamical_ratio", "vortical_dynamical_ratio"]
            units = [str(1/np.sqrt(2)), str(1/(2*np.pi*np.sqrt(2)))]
            for i, field in enumerate(fields):
                filename = os.path.join(data_dir, "%s_%06d.h5" % (field, halo_id))
                plot_profile_distribution(
                    my_axes, filename, 'cell_mass',
                    x_units="pc", y_units=units[i], alpha_scale=0.7,
                    pkwargs=dict(color=colors[i+1], linewidth=1))

            ds = yt.load(filename)
            x_data = ds.profile.x.to("pc")
            z_data = ds.profile['cell_mass'].transpose()
            z_sum = z_data.sum(axis=0)
            rfil = z_sum > 0
            gmin = np.where(~rfil)[0].max() + 1
            my_axes.plot(x_data[gmin:], np.ones(x_data[gmin:].size),
                         alpha=0.9, color=colors[0], linewidth=1, zorder=1)

            tx.xaxis.set_visible(False)

            ylim = (0.01, 100)
            ymajor = np.logspace(-1, 1, 2)
            yminor = np.logspace(-2, 2, 5)
            draw_major_grid(my_axes, 'y', ymajor,
                            color='black', linestyle='-',
                            linewidth=1, alpha=0.2)
            ty = mirror_yticks(my_axes, ylim, ymajor, yminor=yminor)
            my_axes.yaxis.set_label_text("t / t$_{\\rm ff}$")
            my_axes.yaxis.labelpad = 0

    pyplot.savefig("timescales.pdf")
