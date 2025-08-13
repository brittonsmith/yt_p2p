import cmyt
from matplotlib import pyplot
import numpy as np
import os
import yt

pyplot.rcParams['font.size'] = 16

from grid_figure import GridFigure
from yt.extensions.p2p.stars import get_star_data
from yt.utilities.physical_constants import G
from yt.visualization.color_maps import yt_colormaps

if __name__ == "__main__":
    star_ids = [
        334267081,
        334267082,
        334267086,
        334267090,
        334267093,
        334267099,
        334267102,
    ]

    slabels = [1, 2, 4, 5, 6, 7, 8]

    star_data = get_star_data("star_hosts.yaml")

    xlabel = "M$_{\\rm gas, enc}$ [M$_{\\odot}$]"
    my_fig = GridFigure(1, 1, figsize=(8, 5),
                    left_buffer=0.12, right_buffer=0.03,
                    bottom_buffer=0.14, top_buffer=0.02)

    my_axes = my_fig[0]
    my_axes.set_xscale("log")
    my_axes.set_yscale("log")
    my_axes.grid(visible=True, axis="both", zorder=0, linestyle=":",
                 color="black", alpha=0.6)


    for i, star_id in enumerate(star_ids):
        my_star = star_data[star_id]
        filename = os.path.join("star_cubes", f"star_{star_id}_radius.h5")
        pds = yt.load(filename)
        profile_data = pds.data
        
        creation_time = my_star["creation_time"]
        times = profile_data["data", "time"].to("Myr")
        ilast = np.where(times < creation_time)[0][-1]

        used = profile_data["data", "used"][ilast].d.astype(bool)
        m_gas_enc = profile_data["data", "gas_mass_enclosed"][ilast, used].to("Msun")
        m_tot_enc = profile_data["data", "total_mass_enclosed"][ilast, used].to("Msun")

        p = profile_data["data", "pressure"][ilast, used]

        a = 1.67
        b = (225 / (32 * np.sqrt(5 * np.pi))) * a**-1.5
        cs = profile_data["data", "sound_speed"][ilast, used]
        m_BE = (b * (cs**4 / G**1.5) * p**-0.5).to("Msun")

        # cmap = pyplot.cm.turbo
        # cmap = cmyt.pastel
        cmap = cmyt.algae
        color = cmap(float(i/(len(star_ids)-1)))
        alpha = 0.85

        my_axes.plot(m_gas_enc, m_tot_enc / m_BE, color=color, alpha=alpha,
                     linewidth=2, label=str(slabels[i]))

    my_axes.xaxis.set_label_text(xlabel)
    my_axes.yaxis.set_label_text("M$_{\\rm tot, enc}$ / M$_{\\rm BE}$")
    my_axes.xaxis.set_ticks(np.logspace(-3, 5, 9), minor=True, labels="")
    my_axes.legend(title="halo")
    my_axes.set_ylim(1e-4, 3)
    my_axes.set_xlim(1e-3, 1e5)

    # my_fig[0].legend(bbox_to_anchor=(1, 1.05), framealpha=0, title="t - t$_{*}$ [Myr]")
    # my_fig[1].yaxis.set_label_text("T [K]")
    # my_fig[2].yaxis.set_label_text("p / p$_{\\rm HSE}$")
    # my_fig[3].yaxis.set_ticks(np.logspace(-8, 0, 5))
    # my_fig[3].yaxis.set_ticks(np.logspace(-7, -1, 4), minor=True, labels="")

    pyplot.savefig("model_BE_Mtot_profiles.pdf")
