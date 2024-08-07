import matplotlib as mpl
from matplotlib import pyplot, colors
from matplotlib.ticker import FuncFormatter
import numpy as np
import os
from scipy.interpolate import interp1d
import sys
import yaml
import yt
import ytree
from ytree.data_structures.tree_container import TreeContainer
from yt.extensions.p2p.tree_analysis_operations import get_progenitor_line

pyplot.rcParams['font.size'] = 18
pyplot.rcParams["axes.axisbelow"] = False

from grid_figure import GridFigure
from yt.extensions.p2p.stars import get_star_data
from yt.utilities.cosmology import Cosmology
from yt.utilities.physical_constants import G
from yt.visualization.color_maps import yt_colormaps

def _z_from_t(t, pos):
    global co
    return "%d" % np.round(co.z_from_t(co.quan(t, "Myr")))

def _int_fmt(t, pos):
    return f"{t:d}"

def _flt_fmt(t, pos):
    return np.format_float_positional(t, trim="-")

def _log_to_exp(val, decimals=1):
    exp = np.floor(val)
    dec = 10**(val - exp)
    if dec >= 2:
        return "%d$\\times$10$^{%d}$" % (int(np.round(dec)), exp)
    else:
        return "10$^{%d}$" % exp

if __name__ == "__main__":
    with open("models.yaml", 'r') as f:
        models = yaml.load(f, Loader=yaml.FullLoader)

    data_dir = "minihalo_models/metallicity_grids"
    star_data = get_star_data("star_hosts.yaml")
    star_ids = list(star_data.keys())

    if len(sys.argv) > 1:
        model_id = int(sys.argv[1])
    else:
        model_id = 1
    star_id = star_ids[model_id-1]

    my_model = models.get(star_id, {})
    my_solutions = my_model.get("solutions", {})

    tolerance = 1e-3
    my_tol = f"{tolerance:g}"
    my_evaluators = ["evaluate_model_collapsed", "evaluate_model_turnaround"]

    solution_fns = []
    solution_values = []

    if my_tol in my_solutions:
        for my_evaluator in my_evaluators:
            if my_evaluator in my_solutions[my_tol]:
                solution_fns.append(my_solutions[my_tol][my_evaluator]["filename"])
                solution_values.append(my_solutions[my_tol][my_evaluator]["value"])
                print (f"Solution ({my_tol}): {solution_values[-1]} - {solution_fns[-1]}")

    star_dir = os.path.join(data_dir, f"star_{star_id}")
    co = Cosmology(
        omega_lambda=0.734,
        omega_matter=0.266,
        hubble_constant=0.71)

    star_data = get_star_data("star_hosts.yaml")
    my_star = star_data[star_id]
    creation_time = my_star["creation_time"]

    my_fig = GridFigure(3, 1, figsize=(8, 6),
                    left_buffer=0.16, right_buffer=0.21,
                    bottom_buffer=0.11, top_buffer=0.12,
                    vertical_buffer=0, horizontal_buffer=0.12)

    for my_axes in my_fig:
        my_axes.set_xscale("linear")
        my_axes.set_yscale("log")
        # my_axes.tick_params(axis="x", direction="inout", which="both",
        #                     top=True, bottom=True)
        my_axes.grid(visible=True, axis="both", zorder=0, linestyle=":",
                     color="black", alpha=0.6)
        # my_axes.set_xlim(*xlim)
        # my_axes.xaxis.set_ticks(np.linspace(90, 140, 11), minor=True, labels="")

    for my_axes in list(my_fig.middle_axes) + list(my_fig.top_axes):
        my_axes.tick_params(axis="x", labelbottom=False)

    for my_axes in my_fig.bottom_axes:
        my_axes.set_xlabel("t [Myr]", labelpad=3)
        my_axes.tick_params(axis="x", direction="in", which="minor",
                            labelbottom=True, pad=-12,
                            top=False, bottom=True)

    mcolor = "blue"

    # log_metallicities = [-np.inf, -4, -3.5, -3]
    # log_metallicities = np.concatenate([[-np.inf], np.linspace(-5, -3, 21)])
    log_metallicities = np.concatenate([np.linspace(-5, -3, 21), solution_values])

    dss = {}
    min_t = None
    max_t = None
    isol = 0

    for iZ, lZ in enumerate(log_metallicities):
        if iZ >= log_metallicities.size - len(solution_fns):
            filekey = solution_fns[isol]
            isol += 1
        else:
            filekey = os.path.join(star_dir, f"model_lZ_{lZ:.2f}.h5")
        dss[lZ] = yt.load(filekey)
        ds = dss[lZ]
        t = ds.data["data", "time"].to("Myr") + \
          ds.parameters["absolute_start_time"]
        min_t = t.min() if min_t is None else min(min_t, t.min())
        max_t = t.max() if max_t is None else max(max_t, t.max())

    ds = dss[solution_values[0]]
    be_ratio = ds.data["data", "gas_mass"] / ds.data["data", "bonnor_ebert_mass"][-1]
    icoord = be_ratio.argmax()

    xlim = (np.floor(min_t/10)*10, np.ceil(max_t/10)*10)
    for i, lZ in enumerate(log_metallicities):
        model_ds = dss[lZ]
        if i > len(log_metallicities)- 1 - len(solution_fns):
            color = "black"
            linewidth = 2
        else:
            color = pyplot.cm.turbo(i / (len(log_metallicities)- 1 - len(solution_fns)))
            linewidth = 1.5
        # color = pyplot.cm.turbo(i / 6)

        model_data = model_ds.data
        model_time = model_data["data", "time"].to("Myr") + \
          model_ds.parameters["absolute_start_time"]

        if lZ == -np.inf:
            label = "0"
        else:
            label = f"{_log_to_exp(lZ)}" + " Z$_{\\odot}$"

        my_axes = my_fig[0]
        my_axes.plot(model_time, model_data["data", "density"][:, icoord], color=color,
                     label=label, linewidth=linewidth)

        my_axes = my_fig[1]
        my_axes.plot(model_time, model_data["data", "temperature"][:, icoord], color=color,
                     linewidth=linewidth)
        # my_z = co.z_from_t(model_ds.arr(xlim, "Myr"))
        # my_axes.plot(xlim, 2.73 * (1 + my_z), color="red", label="T$_{\\rm CMB}$")

        my_axes = my_fig[2]
        a = 1.67
        b = (225 / (32 * np.sqrt(5 * np.pi))) * a**-1.5
        model_cs = np.sqrt(model_data["data", "gamma"] * model_data["data", "pressure"] /
                           model_data["data", "density"])
        model_m_BE = b * (model_cs**4 / G**1.5) * model_data["data", "pressure"]**-0.5
        model_BE_ratio = model_data["data", "gas_mass"] / model_m_BE
        my_axes.plot(model_time, model_BE_ratio[:, icoord], color=color, linewidth=linewidth)

    my_fig[0].yaxis.set_label_text("$\\rho_{\\rm gas}$ [g/cm$^{3}$]")
    my_fig[0].yaxis.set_ticks(np.logspace(-23, -19, 3))
    my_fig[0].yaxis.set_ticks(np.logspace(-22, -20, 2), minor=True, labels="")
    # my_fig[0].legend(markerfirst=False, fontsize=14)
    my_fig[1].yaxis.set_label_text("T [K]")
    # my_fig[1].yaxis.set_ticks([50, 100, 200, 400])
    my_fig[1].yaxis.set_ticks([100, 1000])
    my_fig[1].yaxis.set_major_formatter(FuncFormatter(_int_fmt))
    # my_fig[1].legend(loc="center left")
    my_fig[2].yaxis.set_label_text("M$_{\\rm gas, enc}$ / M$_{\\rm BE}$")
    my_fig[2].yaxis.set_ticks(np.logspace(-4, 0, 3))
    my_fig[2].yaxis.set_ticks(np.logspace(-3, -1, 2), minor=True, labels="")
    my_fig[2].xaxis.set_ticks([my_star["creation_time"].d], labels="*", minor=True, color="blue", fontsize=20)

    Z_min = log_metallicities[0]
    Z_max = log_metallicities[-len(solution_values)-1]
    # breakpoint()
    norm = colors.Normalize(vmin=Z_min, vmax=Z_max)
    # Z_vals = np.linspace(Z_min, Z_max, 41)
    cmap = "turbo"
    my_cax = my_fig.add_cax(my_fig[1], "right", length=2.95)
    cbar = mpl.colorbar.ColorbarBase(
        my_cax, cmap=cmap, boundaries=log_metallicities[:-len(solution_values)],
        norm=norm, orientation='vertical')

    cbar.set_label("log$_{\\rm 10}$ (Z [Z$_{\\odot}$])")
    my_ticks = list(np.linspace(-5, -3, 9))
    # my_ticks = list(np.linspace(-5, -3, 5))
    cbar.set_ticks(my_ticks)
    cbar.solids.set_edgecolor("face")
    if solution_values:
        cbar.set_ticks(solution_values, minor=True)
    for solution_value in solution_values:
        my_cax.axhline(solution_value, color="black", linewidth=2)

    for my_axes in my_fig:
        my_axes.set_xlim(*xlim)

    # my_axes = my_fig[0]
    # tx = my_axes.twiny()
    # tx.xaxis.tick_top()
    # z_ticks_in_t = co.t_from_z(
    #     np.array([25, 20, 15, 12, 10, 9])).in_units("Myr")
    # tx.xaxis.set_ticks(z_ticks_in_t.d)
    # z_ticks_in_t_m = co.t_from_z(
    #     np.arange(9, 33)).in_units("Myr")
    # tx.xaxis.set_ticks(z_ticks_in_t_m.d, minor=True, labels="")
    # tx.xaxis.set_major_formatter(FuncFormatter(_z_from_t))
    # tx.set_xlim(*xlim)
    # tx.xaxis.set_label_text("z")

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
    # if my_y1[-1] > m_ticks.max():
    #     m_ticks_in_t = np.append(m_ticks_in_t, my_x1[-1])
    # m_ticks_in_t = np.append(m_ticks_in_t, f2r(7))

    my_axes = my_fig[0]
    tx = my_axes.twiny()
    tx.xaxis.tick_top()

    tx.xaxis.set_ticks(m_ticks_in_t)
    tx.xaxis.set_major_formatter(FuncFormatter(_m_from_t))
    tx.set_xlim(*xlim)
    tx.xaxis.set_label_text("M$_{\\rm halo}$ [M$_{\\odot}$]")
    tx.xaxis.labelpad = 8

    pyplot.savefig(f"metallicity_grids/metallicity_model_{model_id}.pdf")
