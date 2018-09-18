from matplotlib import \
    pyplot
import numpy as np
import os
from yt.visualization.color_maps import \
    yt_colormaps

import matplotlib as mpl
mpl.rcParams['axes.unicode_minus'] = False

from yt.extensions.p2p.plots import \
    plot_phase, \
    twin_unit_axes, \
    draw_major_grid, \
    mirror_xticks, \
    mirror_yticks

from grid_figure import GridFigure

def plot_radius_metallicity(data_dir, halo_id):
    filename = os.path.join(data_dir, "metallicity3_min7_%06d.h5" % halo_id)

    fontsize = 14
    my_fig = GridFigure(
        1, 1, figsize=(6, 4.5),
        top_buffer = 0.14, bottom_buffer = 0.13,
        left_buffer = 0.14, right_buffer = 0.19)

    my_axes = my_fig[0]
    my_axes.set_xscale('log')
    my_axes.set_yscale('log')

    my_cax = my_fig.add_cax(my_axes, "right", buffer=0.02,
                            length=0.95, width=0.04)
    plot_phase(filename, 'cell_mass', 'Msun',
               my_axes, my_cax=my_cax,
               cmap=yt_colormaps['dusk'])

    my_cax.yaxis.set_label_text("M [M$_{\\odot}$]")

    xlim = (2e-6, 2e2)
    tx = twin_unit_axes(
        my_axes, xlim, "r",
        "pc", top_units="AU")

    ylim = (4.5e-8, 0.1)
    ymajor = np.logspace(-7, -1, 7)
    mirror_yticks(my_axes, ylim, ymajor)
    draw_major_grid(my_axes, 'y', ymajor)
    my_axes.yaxis.set_label_text("Z [Z$_{\\odot}$]")

    pyplot.savefig("figures/radius_metallicity.pdf")

if __name__ == "__main__":
    data_dir = "../halo_catalogs/profile_catalogs/DD0560/radial_profiles"
    halo_id = 41732
    plot_radius_metallicity(data_dir, halo_id)
