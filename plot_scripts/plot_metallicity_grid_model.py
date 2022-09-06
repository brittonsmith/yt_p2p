import glob
from matplotlib import pyplot, colors
from matplotlib.ticker import FuncFormatter
import matplotlib as mpl
import numpy as np
import os
import re
from scipy.interpolate import interp1d
import sys
import yaml
import yt
import ytree

from ytree.data_structures.tree_container import TreeContainer
from yt.extensions.p2p.model_profiles import get_star_data
from yt.extensions.p2p.tree_analysis_operations import get_progenitor_line

from grid_figure import GridFigure

yt.mylog.setLevel(40)

def load_all_files(star_id):
    data_dir = os.path.join(base_data_dir, f"star_{star_id}")
    fns = glob.glob(os.path.join(data_dir, "*.h5"))
    my_files = []
    zreg = re.compile("lZ\_(.+)\.h5$")
    for fn in fns:
        zmatch = zreg.search(fn)
        if zmatch is None:
            continue
        metal = float(zmatch.groups()[0])
        my_files.append({"fn": fn, "Z": metal})
    my_files.sort(key=lambda x: x["Z"])
    all_Z = np.array([x["Z"] for x in my_files])
    
    return my_files, all_Z

if __name__ == "__main__":
    with open("models.yaml", 'r') as f:
        models = yaml.load(f, Loader=yaml.FullLoader)
    star_data = get_star_data("star_hosts.yaml")
    star_ids = list(star_data.keys())

    if len(sys.argv) > 1:
        model_id = int(sys.argv[1])
    else:
        model_id = 1
    star_id = star_ids[model_id-1]

    base_data_dir = "minihalo_models/metallicity_grids"
    my_star = star_data[star_id]

    Z_min = -5
    Z_max = -3

    tolerance = 1e-3
    my_tol = f"{tolerance:g}"

    my_files, all_Z = load_all_files(star_id)

    my_model = models.get(star_id, {})
    my_solutions = my_model.get("solutions", {})

    if my_tol in my_solutions:
        solution_fn = my_solutions[my_tol]["filename"]
        solution_value = my_solutions[my_tol]["value"]
        print (f"Solution ({my_tol}): {solution_value} - {solution_fn}")
    else:
        solution_fn = ""
        solution_value = None

    my_fig = GridFigure(1, 1, figsize=(8, 5),
                    left_buffer=0.12, right_buffer=0.2,
                    bottom_buffer=0.12, top_buffer=0.11)

    my_axes = my_fig[0]
    my_axes.set_xscale("linear")
    my_axes.set_yscale("log")
    my_axes.tick_params(axis="x", direction="in", which="minor",
                    labelbottom=True, pad=-12,
                    top=False, bottom=True)
    # my_axes.grid(visible=True, axis="both", zorder=0, linestyle=":",
    #              color="black", alpha=0.6)

    min_t = None
    max_t = None

    pbar = yt.get_pbar("Plotting", len(my_files))
    for i, my_file in enumerate(my_files):
        pbar.update(i+1)

        my_Z = my_file["Z"]
        if my_Z < Z_min or my_Z > Z_max:
            continue

        fn = my_file["fn"]

        ds = yt.load(fn)
        t = ds.data["data", "time"].to("Myr") + ds.parameters["absolute_start_time"]
        min_t = t.min() if min_t is None else min(min_t, t.min())
        max_t = t.max() if max_t is None else max(max_t, t.max())

        gas_mass = ds.data["data", "gas_mass"]
        be_mass = ds.data["data", "bonnor_ebert_mass"]
        ratio = gas_mass / be_mass
        my_y = ratio.max(axis=1)

        my_model = models.get(star_id, {})
        my_solutions = my_model.get("solutions", {})

        if fn == solution_fn:
            linewidth = 2
            alpha = 1.0
            color = "black"
        else:
            linewidth = 1.5
            alpha = 0.75
            color = pyplot.cm.turbo((my_Z - Z_min) / (Z_max - Z_min))

        my_axes.plot(t, my_y, color=color, linewidth=linewidth, alpha=alpha)

    pbar.finish()

    xlim = (np.floor(min_t/10)*10, np.ceil(max_t/10)*10)
    my_axes.set_xlim(*xlim)
    my_axes.xaxis.set_label_text("t [Myr]")
    my_axes.xaxis.set_ticks([my_star["creation_time"].d], labels="*", minor=True)
    my_axes.yaxis.set_label_text("M$_{\\rm gas, enc}$ / M$_{\\rm BE}$")
    norm = colors.Normalize(vmin=Z_min, vmax=Z_max)
    cmap = "turbo"
    # Z_vals = all_Z[(all_Z >= Z_min) & (all_Z <= Z_max)]
    Z_vals = np.linspace(Z_min, Z_max, 41)

    my_cax = my_fig.add_cax(my_axes, "right")
    cbar = mpl.colorbar.ColorbarBase(
        my_cax, cmap=cmap, boundaries=Z_vals,
        norm=norm, orientation='vertical')

    cbar.set_label("log$_{\\rm 10}$ (Z [Z$_{\\odot}$])")
    my_ticks = list(np.linspace(-5, -3, 9))
    cbar.set_ticks(my_ticks)
    cbar.solids.set_edgecolor("face")
    # if solution_value is not None:
    #     cbar.set_ticks([solution_value], minor=True)
    my_cax.axhline(solution_value, color="black")

    ### arbor stuff
    a = ytree.load(my_star["arbor"])
    my_root = a[my_star["_arbor_index"]]
    my_tree = my_root.get_node("forest", my_star["tree_id"])
    ct = my_star["creation_time"]
    prog = TreeContainer(a, get_progenitor_line(my_tree))
    good = (prog["time"] <= ct) | ((prog["time"] > ct) & (prog["mass"] > 1e4))


    my_x1 = prog["time"][good].to("Myr").d
    my_y1 = np.log10(prog["mass"][good].to("Msun").d)
    f1 = interp1d(my_x1, my_y1)

    i1 = np.where(good)[0][-10]
    i2 = np.where(good)[0][-1]
    my_x2 = np.array([prog["time"][i1], prog["time"][i2]])
    my_y2 = np.log10([prog["mass"][i1], prog["mass"][i2]])
    ikwargs = {"kind": "linear", "fill_value": "extrapolate"}
    f2 = interp1d(my_x2, my_y2, **ikwargs)

    def _m_from_t(t, pos):
        if t <= my_x1[-1]:
            my_f = f1
        else:
            my_f = f2
        val = my_f(t)
        if np.abs(np.round(val) - val) < 1e-2:
            val = np.round(val)
        dec = 10**(val - int(val))
        if dec > 2:
            return "%dx10$^{%d}$" % (int(np.round(dec)), my_f(t))
        if dec > 1.1:
            return ""
        return "10$^{%d}$" % val

    f1r = interp1d(my_y1, my_x1)
    f2r = interp1d(my_y2, my_x2, **ikwargs)

    m_ticks = np.arange(5, int(my_y1.max()) + 1)
    m_ticks_in_t = f1r(m_ticks)
    if my_y1[-1] > m_ticks.max():
        m_ticks_in_t = np.append(m_ticks_in_t, my_x1[-1])
    # m_ticks_in_t = np.append(m_ticks_in_t, f2r(7))

    my_axes = my_fig[0]
    tx = my_axes.twiny()
    tx.xaxis.tick_top()

    tx.xaxis.set_ticks(m_ticks_in_t)
    tx.xaxis.set_major_formatter(FuncFormatter(_m_from_t))
    tx.set_xlim(*xlim)
    tx.xaxis.set_label_text("M$_{\\rm halo}$ [M$_{\\odot}$]")

    pyplot.savefig(f"metallicity_grids/metallicity_grid_model_{model_id}.pdf")
