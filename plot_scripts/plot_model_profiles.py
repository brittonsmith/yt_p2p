from matplotlib import pyplot
from matplotlib.ticker import FuncFormatter
import numpy as np
import os
import yt

pyplot.rcParams['font.size'] = 16

from grid_figure import GridFigure
from yt.extensions.p2p.stars import get_star_data
from yt.utilities.physical_constants import G
from yt.visualization.color_maps import yt_colormaps
from unyt import uvstack

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

    my_fig = GridFigure(3, 2, figsize=(11, 9),
                    left_buffer=0.1, right_buffer=0.09,
                    bottom_buffer=0.08, top_buffer=0.08,
                    vertical_buffer=0, horizontal_buffer=0.14)

    for my_axes in my_fig:
        my_axes.set_xscale("log")
        my_axes.set_yscale("log")
        my_axes.tick_params(axis="x", direction="inout", which="both",
                            top=True, bottom=True)
        my_axes.grid(visible=True, axis="both", zorder=0, linestyle=":",
                     color="black", alpha=0.6)
        my_axes.xaxis.set_ticks(np.logspace(-3, 5, 9), labels="", minor=True)
        my_axes.set_xlim(1e-3, 1e5)

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
        my_axes.tick_params(axis="x", top=True, bottom=True,
                            labeltop=True, labelbottom=False)
        my_axes.xaxis.set_label_position("top")
        my_axes.set_xlabel(xlabel, labelpad=8)
    for my_axes in my_fig.middle_axes:
        my_axes.tick_params(axis="x", top=True, bottom=True,
                            labeltop=False, labelbottom=False)

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

        t_cs = profile_data["data", "sound_crossing_time"][i, used]
        t_ff = profile_data["data", "total_dynamical_time"][i, used] / np.sqrt(2)
        t_cool = profile_data["data", "cooling_time"][i, used]

        a = 1.67
        b = (225 / (32 * np.sqrt(5 * np.pi))) * a**-1.5
        cs = profile_data["data", "sound_speed"][i, used]
        p_max = uvstack([p, p_hyd]).max(axis=0)
        m_BE = (b * (cs**4 / G**1.5) * p_max**-0.5).to("Msun")

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

        my_axes = my_fig[4]
        my_axes.plot(m_gas_enc, t_cs / t_ff, color=color, alpha=alpha)

        my_axes = my_fig[5]
        yval = t_cool / t_ff
        if (yval[m_gas_enc < 1000] > 100).any():
            gind = np.where((yval < 5) & (m_gas_enc < 1000))[0]
            segments = np.split(gind, np.where(gind[1:] != gind[:-1]+1)[0] + 1)
            for iseg, segment in enumerate(segments):
                if segment.size < 2:
                    continue
                my_axes.plot(m_gas_enc[segment], yval[segment],
                             color=color, alpha=alpha)

                asegs = []
                # if iseg == 0 and segment[0] > 0:
                #     asegs.append(slice(0, segment[0]+1))
                if iseg < len(segments) - 1:
                    asegs.append(slice(segment[-1], segments[iseg+1][0]+1))
                if iseg == len(segments) - 1 and segment[-1] < gind[-1]:
                    asegs.append(slice(segment[-1], gind[-1]+1))

                for aseg in asegs:
                    if yval[aseg].size < 2:
                        continue
                    my_axes.plot(m_gas_enc[aseg], yval[aseg],
                                 color=color, alpha=0.3*alpha)
            my_axes.plot(m_gas_enc[m_gas_enc >= 1000], yval[m_gas_enc >= 1000], color=color, alpha=alpha)
        else:
            my_axes.plot(m_gas_enc, yval, color=color, alpha=alpha)

        m_crit = m_gas_enc[(m_gas_enc / m_BE).argmax()]

    for my_axes in my_fig:
        my_axes.axvline(x=m_crit, color="red", linestyle="--")

    my_fig[0].yaxis.set_label_text("$\\rho_{\\rm gas}$ [g/cm$^{3}$]")
    my_fig[0].yaxis.set_ticks(np.logspace(-24, -18, 4))
    my_fig[0].yaxis.set_ticks(np.logspace(-23, -19, 3), minor=True, labels="")
    my_fig[0].legend(bbox_to_anchor=(1, 1.05), framealpha=0, title="t - t$_{*}$ [Myr]")
    my_fig[1].yaxis.set_label_text("T [K]")
    my_fig[1].yaxis.set_ticks([100, 200, 400])
    my_fig[1].yaxis.set_ticks([], minor=True)
    my_fig[1].yaxis.set_major_formatter(FuncFormatter(_int_fmt))
    my_fig[2].yaxis.set_label_text("p / p$_{\\rm hyd}$")
    my_fig[2].yaxis.set_ticks([0.5, 1, 2, 4])
    my_fig[2].yaxis.set_ticks([], minor=True)
    my_fig[2].yaxis.set_major_formatter(FuncFormatter(_flt_fmt))
    my_fig[3].set_ylim(1e-6, 1)
    my_fig[3].yaxis.set_label_text("M$_{\\rm gas, enc}$ / M$_{\\rm BE}$")
    my_fig[3].yaxis.set_ticks(np.logspace(-5, -1, 3))
    my_fig[3].yaxis.set_ticks(np.logspace(-6, 0, 4), minor=True, labels="")
    my_fig[4].set_ylim(0.1, 2)
    my_fig[4].yaxis.set_label_text("t$_{\\rm sc}$ / t$_{\\rm ff}$")
    my_fig[5].set_ylim(0.5, 100)
    my_fig[5].yaxis.set_label_text("t$_{\\rm cool}$ / t$_{\\rm ff}$")

    pyplot.savefig("model_profiles.pdf")
