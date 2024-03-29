from matplotlib import pyplot, ticker
import numpy as np
import os
import yt
from yt.visualization.color_maps import \
    yt_colormaps
from yt.utilities.physical_constants import G

import matplotlib as mpl
mpl.rcParams['axes.unicode_minus'] = False

from yt.extensions.p2p.plots import \
    plot_profile_distribution, \
    plot_profile_distribution_legend, \
    draw_major_grid, \
    twin_unit_axes, \
    mirror_yticks

from grid_figure import GridFigure

def plot_velocity_profiles(data_dir, file_prefix):
    my_fig = GridFigure(
        1, 1, figsize=(6, 4.5),
        top_buffer = 0.14, bottom_buffer = 0.12,
        left_buffer = 0.09, right_buffer = 0.02)

    # http://www.colourlovers.com/palette/1329926/graphic_artist.
    # colors = ["#B00029", "#90B004", "#19849C", "#851370", "#544D4D", "black"]
    colors = ["red", "green", "blue", "purple", "#544D4D", "black"]

    my_axes = my_fig[0]
    my_axes.set_xscale('log')

    fields = ["velocity_magnitude",
              "tangential_velocity_magnitude",
              "velocity_spherical_radius",
              "sound_speed"]

    for i, field in enumerate(fields):
        filename = os.path.join(data_dir, f"{file_prefix}_2D_profile_radius_{field}_None.h5")
        plot_profile_distribution(
            my_axes, filename, 'cell_mass',
            x_units="pc", y_units='km/s', alpha_scale=0.7,
            pkwargs=dict(color=colors[i], linewidth=1))

    fn = os.path.join(data_dir, f"{file_prefix}_1D_profile_radius_cell_mass.h5")
    pds = yt.load(fn)
    pradius = pds.profile.x.to("pc")
    vsigma = pds.profile.standard_deviation['data', 'velocity_magnitude'].to("km/s")
    my_axes.plot(pradius[vsigma > 0], vsigma[vsigma > 0], alpha=0.9,
                 linewidth=1, color=colors[4], zorder=998)

    field = "matter_mass"
    fn = os.path.join(data_dir, f"{file_prefix}_1D_profile_radius_None.h5")
    mds = yt.load(fn)
    radius = mds.profile.x.to("pc")
    mass = mds.profile[field]
    dfil = mass > 0
    v_sp = np.sqrt(G * mass[dfil].cumsum() / radius[dfil]).to("km/s")
    my_axes.plot(radius[dfil], v_sp, alpha=0.9, linewidth=1,
                 color=colors[5], zorder=997)

    ylim = (-5, 13)
    ymajor = np.arange(-5, 16, 5.)
    yminor = np.arange(-5, 15, 1.)
    my_axes.yaxis.set_label_text("v [km / s]")
    my_axes.yaxis.labelpad = -3
    draw_major_grid(my_axes, 'y', ymajor,
                    color='black', linestyle='-',
                    linewidth=1, alpha=0.2)
    ty = mirror_yticks(my_axes, ylim, ymajor, yminor=yminor)

    xlim = (1e-1, 2e2)
    tx = twin_unit_axes(
        my_axes, xlim, "r",
        "pc", top_units="AU")

    labels = ["|v|", "v$_{\\rm tan}$", "v$_{\\rm rad}$",
              "c$_{\\rm s}$", "$\\sigma$", "v$_{\\rm c}$"]
    dist = [True]*4 + [False]*2
    plot_items = list(zip(colors, labels, dist))
    plot_profile_distribution_legend(
        my_axes, plot_items, alpha_scale=0.7)

    pyplot.savefig("figures/velocity_profiles.pdf")

if __name__ == "__main__":
    data_dir = "minihalo_analysis/node_7254372/profiles"
    plot_velocity_profiles(data_dir, "DD0295")
