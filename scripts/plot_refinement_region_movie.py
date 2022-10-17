import copy
import gc
import glob
import h5py
from matplotlib.patches import Circle
import numpy as np
import os
import sys
import yt
from yt.funcs import ensure_dir
from yt.units.yt_array import YTArray, YTQuantity
from yt.visualization.color_maps import *

import matplotlib as mpl
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['font.family'] = "monospace"

from yt_p2p.timeline import *

from yt_p2p.projection_image import *

def my_timeline(t_current):
    my_fig = pyplot.figure(figsize=(8, 8))
    my_axes = my_fig.add_axes((left_timeline, bottom_timeline,
                               timeline_width, timeline_height),
                               facecolor="black")
    create_timeline(my_axes, co, t_initial, t_final,
                    t_units="Myr",
                    t_major=co.arr(np.arange(0, 501, 50), "Myr"),
                    t_minor=co.arr(np.arange(0, 501, 10), "Myr"),
                    t_current=t_current,
                    redshifts=np.array([100, 50, 40, 30, 25, 22,
                                        20, 18, 16, 15, 14, 14,
                                        13, 12, 11, 10]),
                    text_color="white")
    return my_fig

def plot_box(axes, x, y, **kwargs):
    axes.plot([x[0], x[1]], [y[0], y[0]], **kwargs)
    axes.plot([x[0], x[1]], [y[1], y[1]], **kwargs)
    axes.plot([x[0], x[0]], [y[0], y[1]], **kwargs)
    axes.plot([x[1], x[1]], [y[0], y[1]], **kwargs)
    return x[0], y[0]

smallfont = 8

from yt.utilities.cosmology import Cosmology
co = Cosmology(omega_matter=0.266, omega_lambda=0.734, hubble_constant=0.71)

t_final = co.quan(501, "Myr")
z_final = co.z_from_t(t_final)
t_initial = YTQuantity(-0.5, "Myr")
top_timeline = 0.82
bottom_timeline = 0.16
left_timeline = 0.03
right_timeline = 0.03
timeline_width = 1.0 - left_timeline - right_timeline
timeline_height = 1.0 - top_timeline - bottom_timeline

panels = {}
panels["dark_matter"] = {"filename": "temp_proj.h5", "quantity": ("data", "particle_mass"),
                     "range": [0.3, 'max'], "cmap": "algae", "scale_to_rhom": True,
                     "label": "$\\rho_{dm} / \\bar{\\rho_{m}}$", "ceiling": 225,
                     "cbar_tick_formatter": intcommaformat2}
panels["Density"] = {"filename": "temp_proj.h5", "quantity": ("data", "density"),
                     "range": None, "cmap": "algae", "ceiling": 5e-23,
                     "label": "$\\rho_{b}$ [g / cm$^{3}$]",
                     "cbar_tick_formatter": powformat}
panels["Temperature"] = {"filename": "temp_proj.h5", "quantity": ("data", "temperature"),
                         "range": None, "cmap": "gist_heat", "ceiling": 5e4,
                         "label": "T [K]",
                         "cbar_tick_formatter": intcommaformat}
panels["Metallicity3"] = {"filename": "temp_proj.h5", "quantity": ("data", "metallicity3"),
                          "range": [1e-8, "max"], "cmap": "kamae", "ceiling": 1e-2,
                          "label": "Z [Z$_{\\odot}$]",
                          "cbar_tick_formatter": powformat}

axis = "x"
output_dir = f"frames_{axis}"
ensure_dir(output_dir)

if __name__ == "__main__":
    es = yt.load("simulation.h5")
    times = es.data["data", "time"]
    redshifts = es.data["data", "redshift"]
    fns = es.data["data", "filename"].astype(str)

    length_bar = {"length_bar": True, "length_bar_units": "kpc",
                  "length_bar_left": 50, "length_bar_scale": 50,
                  "length_bar_color": "white"}
    
    # initial evolution
    fields = ["dark_matter", "Density", "Temperature", "Metallicity3"]
    data_dir = "projections"

    image_max = []

    for iframe, fn in enumerate(fns):
        my_time = times[iframe]
        my_redshift = redshifts[iframe]

        ofn = os.path.join(output_dir, "frame_%04d.png" % iframe)
        if os.path.exists(ofn):
            continue

        filename = os.path.join(data_dir, f"{os.path.basename(fn)}_{axis}.h5")
        if not os.path.exists(filename):
            continue
        print (f"Creating frame {iframe} ({filename}), "
               f"z = {my_redshift}, t = {my_time}.")

        my_panels = [copy.deepcopy(panels[field]) for field in fields]
        for my_panel in my_panels:
            my_panel["filename"] = filename
        my_panels[3].update(length_bar)
        if len(image_max) > 0:
            for ip, panel in enumerate(my_panels):
                if ip == 1: continue
                if image_max[ip] >= panel["ceiling"]:
                    if panel["range"] is None:
                        panel["range"] = ["min", panel["ceiling"]]
                    else:
                        panel["range"][1] = panel["ceiling"]

        my_fig = my_timeline(my_time)
        my_fig = multi_image(my_panels, ofn, figsize=(8, 8), fontsize=12, dpi=200,
                          n_columns=2, bg_color="black", text_color="white",
                          fig=my_fig,
                          bottom_buffer=0.1, top_buffer=0.0,
                          left_buffer=0.12, right_buffer=0.12)
        image_max = [panel["image_max"] for panel in my_panels]
        print ()
        pyplot.clf()
        pyplot.close("all")
        del my_fig

        val = gc.collect()
        print (f"Collected {val} garbages!")
