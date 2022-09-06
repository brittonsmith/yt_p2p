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

def _int_fmt(t, pos):
    return f"{t:d}"

def _flt_fmt(t, pos):
    return np.format_float_positional(t, trim="-")

if __name__ == "__main__":
    star_id = 334267081
    star_data = get_star_data("star_hosts.yaml")
    my_star = star_data[star_id]
    creation_time = my_star["creation_time"]

    filename = os.path.join("star_cubes", f"star_{star_id}_radius.h5")
    pds = yt.load(filename)
    profile_data = pds.data

    times = profile_data["data", "time"].to("Myr")
    ibefore = (times < creation_time).sum()
    iafter = times.size - ibefore

    xlabel = "M$_{\\rm gas, enc}$ [M$_{\\odot}$]"

    my_fig = GridFigure(2, 2, figsize=(11, 6),
                    left_buffer=0.09, right_buffer=0.08,
                    bottom_buffer=0.1, top_buffer=0.1,
                    vertical_buffer=0, horizontal_buffer=0.12)

    for my_axes in my_fig:
        my_axes.set_xscale("log")
        my_axes.set_yscale("log")
        my_axes.tick_params(axis="x", direction="inout", which="both",
                            top=True, bottom=True)
        my_axes.grid(visible=True, axis="both", zorder=0, linestyle=":",
                     color="black", alpha=0.6)
        my_axes.xaxis.set_ticks(np.logspace(-3, 5, 9), labels="", minor=True)

    for my_axes in my_fig.left_axes:
        my_axes.tick_params(axis="y", left=True, direction="inout", which="both")
        my_axes.tick_params(axis="y", right=True, direction="in", which="both")
    for my_axes in my_fig.right_axes:
        my_axes.tick_params(axis="y", right=True, direction="inout", which="both",
                            labelright=True)
        my_axes.tick_params(axis="y", left=True, direction="in", which="both",
                            labelleft=False)
        my_axes.yaxis.set_label_position("right")
    for my_axes in my_fig.bottom_axes:
        my_axes.set_xlabel(xlabel, labelpad=3)
    for my_axes in my_fig.top_axes:
        my_axes.tick_params(axis="x", top=True, labeltop=True)
        my_axes.xaxis.set_label_position("top")
        my_axes.set_xlabel(xlabel, labelpad=8)

    for i, current_time in enumerate(times):
        before = current_time < creation_time
        if not before:
            break

        used = profile_data["data", "used"][i].d.astype(bool)
        m_gas_enc = profile_data["data", "gas_mass_enclosed"][i, used].to("Msun")

        radius_bins = profile_data["data", "radius"]
        lrb = np.log10(radius_bins)
        radius = np.power(10, (lrb[:-1] + lrb[1:]) / 2)[used] * radius_bins.units

        rho = profile_data["data", "density"][i, used]
        T = profile_data["data", "temperature"][i, used]

        p = profile_data["data", "pressure"][i, used]
        p_hyd = profile_data["data", "hydrostatic_pressure"][i, used]

        a = 1.67
        b = (225 / (32 * np.sqrt(5 * np.pi))) * a**-1.5
        cs = profile_data["data", "sound_speed"][i, used]
        m_BE = (b * (cs**4 / G**1.5) * p**-0.5).to("Msun")

        color = pyplot.cm.turbo(float(i/(ibefore-1)))
        alpha = 0.85
        lt = (current_time - creation_time).to("Myr")
        # if np.abs(lt) < pds.quan(1, "Myr"):
        #     lt.convert_to_units("kyr")
        label = f"{lt.d:.1f}"

        my_axes = my_fig[0]
        my_axes.plot(m_gas_enc, rho, color=color, alpha=alpha, label=label)

        my_axes = my_fig[1]
        my_axes.plot(m_gas_enc, T, color=color, alpha=alpha)

        my_axes = my_fig[2]
        my_axes.plot(m_gas_enc, p/p_hyd, color=color, alpha=alpha)
    
        my_axes = my_fig[3]
        my_axes.plot(m_gas_enc, m_gas_enc / m_BE, color=color, alpha=alpha)

    my_fig[0].yaxis.set_label_text("$\\rho_{\\rm gas}$ [g/cm$^{3}$]")
    my_fig[0].yaxis.set_ticks(np.logspace(-24, -18, 4))
    my_fig[0].yaxis.set_ticks(np.logspace(-23, -19, 3), minor=True, labels="")
    my_fig[0].legend(bbox_to_anchor=(1, 1.05), framealpha=0, title="t - t$_{*}$ [Myr]")
    my_fig[1].yaxis.set_label_text("T [K]")
    my_fig[1].yaxis.set_ticks([100, 200, 400])
    my_fig[1].yaxis.set_major_formatter(FuncFormatter(_int_fmt))
    my_fig[2].yaxis.set_label_text("p / p$_{\\rm HSE}$")
    my_fig[2].yaxis.set_ticks([0.5, 1, 2, 4])
    my_fig[2].yaxis.set_major_formatter(FuncFormatter(_flt_fmt))
    my_fig[3].yaxis.set_label_text("M$_{\\rm gas, enc}$ / M$_{\\rm BE}$")
    my_fig[3].yaxis.set_ticks(np.logspace(-8, 0, 5))
    my_fig[3].yaxis.set_ticks(np.logspace(-7, -1, 4), minor=True, labels="")

    pyplot.savefig("model_profiles.pdf")
