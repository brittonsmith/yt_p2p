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
    draw_major_grid, \
    twin_unit_axes, \
    mirror_yticks

from grid_figure import GridFigure

def plot_timescale_profiles(data_dir, file_prefix):
    my_fig = GridFigure(
        1, 1, figsize=(6, 4.5),
        top_buffer = 0.13, bottom_buffer = 0.12,
        left_buffer = 0.14, right_buffer = 0.02,
        horizontal_buffer = 0.05, vertical_buffer = 0)

    my_axes = my_fig[0]
    my_axes.set_xscale('log')
    my_axes.set_yscale('log')

    xlim = (5e-3, 3e2)
    tx = twin_unit_axes(
        my_axes, xlim, "r",
        "pc", top_units="AU")

    fields = [
        "sound_crossing_time",
        "total_dynamical_time",
        "cooling_time",
        "vortical_time"
    ]
    units = [
        "yr",
        f"{np.sqrt(2)}*yr",
        "yr",
        f"{1/(2*np.pi)}*yr"
    ]
    labels = [
        "sound-crossing",
        "free-fall",
        "cooling",
        "mixing"
    ]
    colors = ["orange", "red", "green", "blue"]


    filename = os.path.join(data_dir, "DD0295_1D_profile_radius_cell_mass.h5")
    ds = yt.load(filename)

    x_data = ds.profile.x.to("pc")
    used = ds.profile.used
    for field, unit, label, color in zip(fields, units, labels, colors):
        if field == "sound_crossing_time":
            cs = ds.profile["data", "sound_speed"]
            vt = ds.profile.standard_deviation["data", "velocity_magnitude"]
            v = np.sqrt(cs**2 + vt**2)
            y_data = (2 * x_data / v).to(unit)
        else:
            y_data = ds.profile["data", field].to(unit)

        my_axes.plot(x_data[used], y_data[used], color=color,
                     alpha=0.7, linewidth=1.5,
                     label=label)

    ylim = (1e4, 1e8)
    ymajor = np.logspace(2, 10, 5)
    yminor = np.logspace(1, 9, 5)
    draw_major_grid(my_axes, 'y', ymajor,
                color='black', linestyle='-',
                linewidth=1, alpha=0.2)
    ty = mirror_yticks(my_axes, ylim, ymajor, yminor=yminor)
    my_axes.yaxis.set_label_text("t [yr]")
    my_axes.legend()
    # my_axes.yaxis.labelpad = 4

    pyplot.savefig("figures/timescale_profiles.pdf")

if __name__ == "__main__":
    data_dir = "minihalo_analysis/node_7254372/profiles"
    plot_timescale_profiles(data_dir, "DD0295")
