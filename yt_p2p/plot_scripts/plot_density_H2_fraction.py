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
    draw_major_grid, \
    mirror_xticks, \
    mirror_yticks

from grid_figure import GridFigure

def plot_density_H2_fraction(data_dir, halo_id):
    filename = os.path.join(data_dir, "H2_fraction_%06d.h5" % halo_id)

    fontsize = 14
    my_fig = GridFigure(
        1, 1, figsize=(6, 4.5),
        top_buffer = 0.03, bottom_buffer = 0.13,
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

    xlim = (1e-3, 1e13)
    xmajor = np.logspace(-3, 12, 6)
    xminor = np.logspace(-3, 13, 17)
    mirror_xticks(my_axes, xlim, xmajor, xminor=xminor)
    draw_major_grid(my_axes, 'x', xmajor)
    my_axes.xaxis.set_label_text("n [cm$^{-3}$]")

    ylim = (1e-8, 1)
    ymajor = np.logspace(-8, 0, 9)
    mirror_yticks(my_axes, ylim, ymajor)
    draw_major_grid(my_axes, 'y', ymajor)
    my_axes.yaxis.set_label_text("f$_{\\rm H_{2}}$")

    pyplot.savefig("figures/density_H2_fraction.pdf")

if __name__ == "__main__":
    data_dir = "../halo_catalogs/profile_catalogs/DD0560/density_profiles"
    halo_id = 41732
    plot_density_H2_fraction(data_dir, halo_id)
