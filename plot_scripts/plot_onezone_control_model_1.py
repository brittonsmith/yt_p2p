from matplotlib import pyplot
from matplotlib.ticker import FuncFormatter
import numpy as np
import os
import yt

pyplot.rcParams['font.size'] = 14

from grid_figure import GridFigure
from yt.extensions.p2p.stars import get_star_data
from yt.utilities.physical_constants import G
from yt.visualization.color_maps import yt_colormaps

def _z_from_t(t, pos):
    global co
    return "%d" % np.round(co.z_from_t(co.quan(t, "Myr")))

def _int_fmt(t, pos):
    return f"{t:d}"

def _flt_fmt(t, pos):
    return np.format_float_positional(t, trim="-")

if __name__ == "__main__":
    data_dir = "minihalo_models/onezone_control_runs"
    star_id = 334267081

    filekey = os.path.join(data_dir, f"star_{star_id}")
    model_ds = yt.load(f"{filekey}.h5")
    ed_ds = yt.load(f"{filekey}_external_data.h5")
    co = ed_ds.cosmology
    
    star_data = get_star_data("star_hosts.yaml")
    my_star = star_data[star_id]
    creation_time = my_star["creation_time"]

    my_fig = GridFigure(4, 1, figsize=(8, 8),
                    left_buffer=0.15, right_buffer=0.02,
                    bottom_buffer=0.07, top_buffer=0.07,
                    vertical_buffer=0, horizontal_buffer=0.12)

    xlim = (90, 145)
    for my_axes in my_fig:
        my_axes.set_xscale("linear")
        my_axes.set_yscale("log")
        # my_axes.tick_params(axis="x", direction="inout", which="both",
        #                     top=True, bottom=True)
        my_axes.grid(visible=True, axis="both", zorder=0, linestyle=":",
                     color="black", alpha=0.6)
        my_axes.set_xlim(*xlim)
        my_axes.xaxis.set_ticks(np.linspace(90, 140, 11), minor=True, labels="")

    for my_axes in list(my_fig.middle_axes) + list(my_fig.top_axes):
        my_axes.tick_params(axis="x", labelbottom=False)

    for my_axes in my_fig.bottom_axes:
        my_axes.set_xlabel("t [Myr]", labelpad=3)

    ecolor = "black"
    mcolor = "blue"

    model_data = model_ds.data
    model_time = model_data["data", "time"].to("Myr") + ed_ds.parameters["start_time"]

    ed_data = ed_ds.data    
    ed_time = ed_data["data", "absolute_time"].to("Myr")
    tfilter = ed_time < creation_time
    ed_time = ed_time[tfilter]

    my_axes = my_fig[0]
    my_axes.plot(ed_time, ed_data["data", "density"], color=ecolor,
                 label="data")
    my_axes.plot(model_time, model_data["data", "density"], color=mcolor,
                 label="model")

    my_axes = my_fig[1]
    my_axes.plot(model_time, model_data["data", "temperature"], color=mcolor)
    my_axes.plot(ed_time, ed_data["data", "temperature"], color=ecolor)
    # my_z = co.z_from_t(ed_ds.arr(xlim, "Myr"))
    # my_axes.plot(xlim, 2.73 * (1 + my_z), color="red", label="T$_{\\rm CMB}$")

    my_axes = my_fig[2]
    model_fH2 = model_data["data", "H2I"] / model_data["data", "density"]
    my_axes.plot(model_time, model_fH2, color=mcolor)
    ed_fH2 = ed_data["data", "H2_p0_density"] / ed_data["data", "density"]
    my_axes.plot(ed_time, ed_fH2, color=ecolor)

    my_axes = my_fig[3]
    a = 1.67
    b = (225 / (32 * np.sqrt(5 * np.pi))) * a**-1.5
    model_cs = np.sqrt(model_data["data", "gamma"] * model_data["data", "pressure"] /
                       model_data["data", "density"])
    model_m_BE = b * (model_cs**4 / G**1.5) * model_data["data", "pressure"]**-0.5
    model_BE_ratio = model_data["data", "gas_mass"] / model_m_BE
    my_axes.plot(model_time, model_BE_ratio, color=mcolor)
    my_axes.plot(ed_time, ed_data["data", "bonnor_ebert_ratio"], color=ecolor)

    my_fig[0].yaxis.set_label_text("$\\rho_{\\rm gas}$ [g/cm$^{3}$]")
    my_fig[0].yaxis.set_ticks(np.logspace(-23, -19, 3))
    my_fig[0].yaxis.set_ticks(np.logspace(-22, -20, 2), minor=True, labels="")
    my_fig[0].legend()
    my_fig[1].yaxis.set_label_text("T [K]")
    # my_fig[1].yaxis.set_ticks([50, 100, 200, 400])
    my_fig[1].yaxis.set_ticks([100, 1000])
    my_fig[1].yaxis.set_major_formatter(FuncFormatter(_int_fmt))
    # my_fig[1].legend(loc="center left")
    my_fig[2].yaxis.set_label_text("f$_{\\rm H_{2}}}$")
    my_fig[2].yaxis.set_ticks([5e-4, 1e-3])
    my_fig[2].yaxis.set_major_formatter(FuncFormatter(_flt_fmt))
    my_fig[2].yaxis.set_ticks(np.linspace(4e-4, 1e-3, 7), minor=True, labels="")
    my_fig[3].yaxis.set_label_text("M$_{\\rm gas, enc}$ / M$_{\\rm BE}$")
    my_fig[3].yaxis.set_ticks(np.logspace(-4, 0, 3))
    my_fig[3].yaxis.set_ticks(np.logspace(-3, -1, 2), minor=True, labels="")

    my_axes = my_fig[0]
    tx = my_axes.twiny()
    tx.xaxis.tick_top()

    z_ticks_in_t = co.t_from_z(
        np.array([32, 31, 30, 29, 28, 27, 26, 25, 24])).in_units("Myr")
    tx.xaxis.set_ticks(z_ticks_in_t.d)
    tx.xaxis.set_major_formatter(FuncFormatter(_z_from_t))
    tx.set_xlim(*xlim)
    tx.xaxis.set_label_text("z")

    pyplot.savefig(f"onezone_control_model_1.pdf")
