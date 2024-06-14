import matplotlib as mpl
from matplotlib import pyplot
from matplotlib.ticker import FuncFormatter
import numpy as np
import os
import yt
import ytree

from matplotlib.collections import LineCollection

pyplot.rcParams['font.size'] = 16

from grid_figure import GridFigure
from yt.extensions.p2p.stars import get_star_data
from yt.utilities.physical_constants import G
from yt.visualization.color_maps import yt_colormaps
from ytree.data_structures.tree_container import TreeContainer
from yt.extensions.p2p.tree_analysis_operations import get_progenitor_line

def _z_from_t(t, pos):
    global co
    return "%d" % np.round(co.z_from_t(co.quan(t, "Myr")))

def _int_fmt(t, pos):
    return f"{t:d}"

def _flt_fmt(t, pos):
    return np.format_float_positional(t, trim="-")

if __name__ == "__main__":
    data_dir = "minihalo_models/onezone_control_runs"
    variant_dir = "minihalo_models/onezone_variants"
    star_id = 334267102

    filekey = os.path.join(data_dir, f"star_{star_id}")
    model_ds = yt.load(f"{filekey}.h5")
    ed_ds = yt.load(f"{filekey}_external_data.h5")
    co = ed_ds.cosmology

    star_data = get_star_data("star_hosts.yaml")
    my_star = star_data[star_id]
    creation_time = my_star["creation_time"]

    a = ytree.load(my_star["arbor"])
    my_root = a[my_star["_arbor_index"]]
    my_tree = my_root.get_node("forest", my_star["tree_id"])
    prog = TreeContainer(a, get_progenitor_line(my_tree))

    my_fig = GridFigure([0.5, 1, 0.5, 0.5, 0.5], 1, figsize=(8.5, 11),
                    left_buffer=0.13, right_buffer=0.12,
                    bottom_buffer=0.06, top_buffer=0.06,
                    vertical_buffer=0, horizontal_buffer=0.12)

    vstyles = [":", "--", (5, (10, 3)), "-."]

    mass_row = 0
    density_row = 1
    temperature_row = 2
    fh2_row = 3
    mbe_row = 4

    xlim = (170, 350)
    for i, my_axes in enumerate(my_fig):
        my_axes.set_xscale("linear")
        my_axes.set_yscale("log")
        my_axes.tick_params(axis="y", direction="inout", which="both",
                            left=True, right=True)
        my_axes.grid(visible=True, axis="both", zorder=0, linestyle=":",
                     color="black", alpha=0.6)
        my_axes.set_xlim(*xlim)
        # my_axes.xaxis.set_ticks(np.linspace(90, 140, 11), minor=True, labels="")

    for my_axes in list(my_fig.middle_axes) + list(my_fig.top_axes):
        my_axes.tick_params(axis="x", labelbottom=False)

    for my_axes in my_fig.bottom_axes:
        my_axes.set_xlabel("t [Myr]", labelpad=3)

    ecolor = "black"
    mcolor = "green"

    model_data = model_ds.data
    model_time = model_data["data", "time"].to("Myr") + ed_ds.parameters["start_time"]

    ed_data = ed_ds.data    
    ed_time = ed_data["data", "absolute_time"].to("Myr")
    tfilter = ed_time < creation_time
    ed_time = ed_time[tfilter]

    my_axes = my_fig[density_row]
    my_axes.plot(ed_time, ed_data["data", "density"], color=ecolor,
                 label="data")
    # my_axes.plot(ed_time, ed_data["data", "dark_matter_density"][:ed_time.size],
    #              color=ecolor, linestyle=":")
    my_axes.plot(model_time, model_data["data", "density"], color=mcolor,
                 label="model")

    t_ratio = model_data["data", "freefall_time"] / \
      model_data["data", "sound_crossing_time"]
    my_ylim = my_axes.get_ylim()
    my_y = my_ylim[0] * np.ones(model_time.size)
    points = np.array([model_time, my_y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    # my_norm = pyplot.Normalize(t_ratio.min(), t_ratio.max())
    my_norm = pyplot.Normalize(0.8, 1.2)
    # my_cmap = "PiYG"
    my_cmap = "seismic"
    lc = LineCollection(segments, cmap=my_cmap, norm=my_norm)
    lc.set_array(t_ratio)
    lc.set_linewidth(20)
    my_line = my_axes.add_collection(lc)
    my_axes.set_ylim(my_ylim)
    my_cax = my_fig.add_cax(my_axes, "right", buffer=0.01, width=0.02)
    my_cbar = mpl.colorbar.ColorbarBase(
        my_cax, cmap=my_cmap, norm=my_norm,
        orientation='vertical')
    my_cbar.set_label("$t_{ff}/t_{sc}$")

    my_axes = my_fig[mass_row]
    my_axes.plot(prog["time"], prog["mass"], color=ecolor)

    my_axes = my_fig[temperature_row]
    my_axes.plot(model_time, model_data["data", "temperature"], color=mcolor)
    my_axes.plot(ed_time, ed_data["data", "temperature"], color=ecolor)
    # my_z = co.z_from_t(ed_ds.arr(xlim, "Myr"))
    # my_axes.plot(xlim, 2.73 * (1 + my_z), color="red", label="T$_{\\rm CMB}$")

    my_axes = my_fig[fh2_row]
    model_fH2 = model_data["data", "H2I"] / model_data["data", "density"]
    my_axes.plot(model_time, model_fH2, color=mcolor)
    ed_fH2 = ed_data["data", "H2_p0_density"] / ed_data["data", "density"]
    my_axes.plot(ed_time, ed_fH2, color=ecolor)

    my_axes = my_fig[mbe_row]
    a = 1.67
    b = (225 / (32 * np.sqrt(5 * np.pi))) * a**-1.5
    model_cs = np.sqrt(model_data["data", "gamma"] * model_data["data", "pressure"] /
                       model_data["data", "density"])
    model_m_BE = b * (model_cs**4 / G**1.5) * model_data["data", "pressure"]**-0.5
    model_BE_ratio = model_data["data", "gas_mass"] / model_m_BE
    my_axes.plot(model_time, model_BE_ratio, color=mcolor)
    my_axes.plot(ed_time, ed_data["data", "bonnor_ebert_ratio"], color=ecolor)

    my_axes = my_fig[density_row]
    for iv in range(4):
        filekey = os.path.join(variant_dir, f"star_{star_id}_v{iv}")
        model_ds = yt.load(f"{filekey}.h5")
        model_data = model_ds.data
        model_time = model_data["data", "time"].to("Myr") + ed_ds.parameters["start_time"]
        my_axes.plot(model_time, model_data["data", "density"], color=mcolor,
                     linestyle=vstyles[iv], linewidth=1.0)

    my_fig[density_row].yaxis.set_label_text("$\\rho_{\\rm gas}$ [g/cm$^{3}$]")
    my_fig[density_row].yaxis.set_ticks(np.logspace(-24, -20, 3))
    my_fig[density_row].yaxis.set_ticks(np.logspace(-23, -19, 3), minor=True, labels="")
    my_fig[density_row].legend()

    my_fig[mass_row].yaxis.set_label_text("M$_{\\rm halo}$ [M$_{\odot}$]")
    my_fig[mass_row].set_ylim(1e4, 1e6)

    my_fig[temperature_row].yaxis.set_label_text("T [K]")
    my_fig[temperature_row].yaxis.set_ticks([100, 1000])
    my_fig[temperature_row].yaxis.set_major_formatter(FuncFormatter(_int_fmt))
    # my_fig[temperature_row].set_ylim(200, 1500)
    # my_fig[temperature_row].yaxis.set_ticks([100, 1000])
    # my_fig[temperature_row].legend(loc="upper left")
    my_fig[fh2_row].yaxis.set_label_text("f$_{\\rm H_{2}}}$")
    my_fig[fh2_row].yaxis.set_ticks([1e-5, 1e-4])
    # my_fig[temperature_row].yaxis.set_major_formatter(FuncFormatter(_flt_fmt))
    my_fig[mbe_row].yaxis.set_label_text("M$_{\\rm gas, enc}$ / M$_{\\rm BE}$")
    my_fig[mbe_row].yaxis.set_ticks(np.logspace(-3, 0, 4))

    my_axes = my_fig[0]
    tx = my_axes.twiny()
    tx.xaxis.tick_top()

    z_ticks_in_t = co.t_from_z(
        np.array([21, 20, 19, 18, 17, 16, 15, 14, 13])).in_units("Myr")
    tx.xaxis.set_ticks(z_ticks_in_t.d)
    tx.xaxis.set_major_formatter(FuncFormatter(_z_from_t))
    tx.set_xlim(*xlim)
    tx.xaxis.set_label_text("z")

    pyplot.savefig(f"onezone_control_model_8.pdf")
