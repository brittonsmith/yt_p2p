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

if __name__ == "__main__":
    fontsize = 14
    my_fig = GridFigure(
        1, 1, figsize=(6, 4.5),
        top_buffer = 0.14, bottom_buffer = 0.12,
        left_buffer = 0.09, right_buffer = 0.02)

    # http://www.colourlovers.com/palette/1329926/graphic_artist.
    # colors = ["#B00029", "#90B004", "#19849C", "#851370", "#544D4D", "black"]
    colors = ["red", "green", "blue", "purple", "#544D4D", "black"]

    my_axes = my_fig[0]
    my_axes.set_xscale('log')
    data_dir = "../halo_catalogs/profile_catalogs/DD0560/velocity_profiles"
    halo_id = 41732

    fields = ["velocity_magnitude",
              "tangential_velocity_magnitude",
              "velocity_spherical_radius",
              "sound_speed"]

    for i, field in enumerate(fields):
        filename = os.path.join(data_dir, "%s_%06d.h5" % (field, halo_id))
        plot_profile_distribution(
            my_axes, filename, 'cell_mass',
            x_units="pc", y_units="km/s", alpha_scale=0.7,
            pkwargs=dict(color=colors[i], linewidth=1))

    pds = yt.load(
        "../halo_catalogs/profile_catalogs/DD0560/profiles/profiles_%06d.h5" % halo_id)
    pradius = pds.profile.x.to("pc")
    vsigma = pds.profile.standard_deviation['data', 'velocity_magnitude'].to("km/s")
    my_axes.plot(pradius[vsigma > 0], vsigma[vsigma > 0], alpha=0.9,
                 linewidth=1, color=colors[4], zorder=998)

    field = "cell_mass"
    mds = yt.load(os.path.join(data_dir, "%s_%06d.h5" % (field, halo_id)))
    radius = mds.profile.x.to("pc")
    mass = mds.profile[field]
    dfil = mass > 0
    v_sp = np.sqrt(G * mass[dfil].cumsum() / radius[dfil]).to("km/s")
    my_axes.plot(radius[dfil], v_sp, alpha=0.9, linewidth=1,
                 color=colors[5], zorder=999)

    ylim = (-5, 13)
    ymajor = np.arange(-5, 16, 5.)
    yminor = np.arange(-5, 15, 1.)
    my_axes.yaxis.set_label_text("v [km / s]")
    my_axes.yaxis.labelpad = -3
    draw_major_grid(my_axes, 'y', ymajor,
                    color='black', linestyle='-',
                    linewidth=1, alpha=0.2)
    ty = mirror_yticks(my_axes, ylim, ymajor, yminor=yminor)

    xlim = (2e-6, 2e2)
    tx = twin_unit_axes(
        my_axes, xlim, "r",
        "pc", top_units="AU")

    labels = ["|v|", "v$_{\\rm tan}$", "v$_{\\rm rad}$",
              "c$_{\\rm s}$", "$\\sigma$", "v$_{\\rm c}$"]
    dist = [True]*4 + [False]*2
    plot_items = list(zip(colors, labels, dist))
    plot_profile_distribution_legend(
        my_axes, plot_items, alpha_scale=0.7)

    pyplot.savefig("velocities.pdf")
