import copy
import gc
from matplotlib import pyplot
import numpy as np
import os
import sys
import time
import yt
from yt.funcs import ensure_dir
from yt.visualization.color_maps import *

import matplotlib as mpl
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['font.family'] = "monospace"

from yt_p2p.misc import reunit
from yt_p2p.projection_image import \
    multi_image, \
    intcommaformat2, \
    intcommaformat, \
    powformat

class FakeDS:
    def __init__(self, attrs):
        for attr, val in attrs.items():
            setattr(self, attr, val)

def interpolate_proj(t, ds1, ds2, field, modifier=None):
    t1 = ds1.current_time.to(t.units)
    t2 = ds2.current_time.to(t.units)
    if modifier is None:
        my_d1 = ds1.data[field]
        my_d2 = ds2.data[field]
    else:
        my_d1 = modifier(ds1, field)
        my_d2 = modifier(ds2, field)
    d1 = np.log10(my_d1)
    d2 = np.log10(my_d2)
    m = (d2 - d1) / (t2 - t1)
    di = np.power(10, m * (t - t1) + d1)
    return ds1.arr(di, ds1.data[field].units)

def fix_pmass(ds, field):
    d = ds.data[field]
    if field != ("data", "particle_mass"):
        return d

    my_w = ds.parameters["width"]
    my_dx2 = my_w**2 / np.prod(d.shape)
    rhom = ds.cosmology.omega_matter * \
      ds.cosmology.critical_density(0) * (1 + ds.current_redshift)**3
    d /= my_dx2 * my_w * rhom
    return d

def my_sigmoid(x, x0, ymin, ymax, speed):
    c = speed * (x - x0)
    return (ymax - ymin) * (1 - np.exp(c) / (1 + np.exp(c))) + ymin

panels = {}
panels["dark_matter"] = {"quantity": ("data", "particle_mass"),
                     "range": [0.3, 'max'], "cmap": "algae", "scale_to_rhom": True,
                     "label": "$\\rho_{dm} / \\bar{\\rho_{m}}$", "ceiling": 225,
                     "cbar_tick_formatter": intcommaformat2}
panels["Density"] = {"quantity": ("data", "density"),
                     "range": None, "cmap": "algae", "ceiling": 5e-23,
                     "label": "$\\rho_{b}$ [g / cm$^{3}$]",
                     "cbar_tick_formatter": powformat}
panels["Temperature"] = {"quantity": ("data", "temperature"),
                         "range": None, "cmap": "gist_heat", "ceiling": 5e4,
                         "label": "T [K]",
                         "cbar_tick_formatter": intcommaformat}
panels["Metallicity3"] = {"quantity": ("data", "metallicity3"),
                          "range": [1e-8, "max"], "cmap": "kamae", "ceiling": 2e-2,
                          "label": "Z [Z$_{\\odot}$]",
                          "cbar_tick_formatter": powformat}

if __name__ == "__main__":
    es = yt.load("simulation.h5")
    times = es.data["data", "time"]
    redshifts = es.data["data", "redshift"]
    fns = es.data["data", "filename"].astype(str)

    length_bar = {"length_bar": True,
                  "length_bar_left": 50,
                  "length_bar_scale": es.cosmology.quan(50, "kpc"),
                  "length_bar_color": "white"}
    
    # initial evolution
    fields = ["dark_matter", "Density", "Temperature", "Metallicity3"]
    data_dir = "projections"

    t_final = es.quan(501, "Myr")
    z_final = es.cosmology.z_from_t(t_final)
    t_initial = es.quan(-0.5, "Myr")

    axis = "x"
    output_dir = f"frames_{axis}"
    ensure_dir(output_dir)
    oformat = os.path.join(output_dir, "frame_%04d.png")
    image_max = None

    my_time = times[1]
    iframe = 0
    t1 = time.time()

    while my_time <= times[-1]:
        i = np.digitize([my_time], times) - 1
        i = np.clip(i, 0, times.size - 2)[0]

        my_redshift = es.cosmology.z_from_t(my_time)

        dt = my_sigmoid(my_time.d, 145, 0.25, 5, 0.1)
        dt = es.quan(dt, "Myr")

        ofn = oformat % iframe
        # if os.path.exists(ofn):
        #     my_time += dt
        #     iframe += 1
        #     continue

        fn1 = os.path.join(data_dir, f"{os.path.basename(fns[i])}_{axis}.h5")
        fn2 = os.path.join(data_dir, f"{os.path.basename(fns[i+1])}_{axis}.h5")

        print (f"Creating frame {iframe}: z = {my_redshift}, t = {my_time}, ",
               f"{fn1} - {fn2}.")

        ds1 = yt.load(fn1)
        ds2 = yt.load(fn2)
        idata = {}
        for field in fields:
            fieldname = panels[field]["quantity"]
            idata[fieldname] = interpolate_proj(
                my_time, ds1, ds2, fieldname, modifier=fix_pmass)

        my_width = reunit(es.cosmology, ds1.parameters["width"], "kpccm")
        my_width.convert_to_units("kpc")
        parameters = {"width": my_width}
        ds_attrs = {
            "data": idata,
            "current_time": my_time,
            "current_redshift": my_redshift,
            "cosmology": es.cosmology,
            "parameters": parameters,
        }
        dsi = FakeDS(ds_attrs)
        dsi.quan = es.cosmology.quan
        dsi.arr = es.cosmology.arr

        my_panels = [copy.deepcopy(panels[field]) for field in fields]
        for my_panel in my_panels:
            my_panel["ds"] = dsi
        my_panels[3].update(length_bar)

        # make image max increase monotonically to avoid flashing
        if image_max is not None:
            for ip in [0, 1, 3]:
                my_panels[ip]["floor"] = image_max[ip]

        timeline = {"current_time": my_time,
                    "initial_time": t_initial,
                    "final_time": t_final,
                    "cosmology": es.cosmology,
                    "height": 0.03}

        my_fig = multi_image(my_panels, ofn, figsize=(8, 7), fontsize=12, dpi=200,
                             n_columns=2, bg_color="black", text_color="white",
                             bottom_buffer=0.18, top_buffer=0.01,
                             left_buffer=0.15, right_buffer=0.15,
                             timeline=timeline)
        # print (f"Figsize: {my_fig.figsize}.")

        my_image_max = [panel["image_max"] for panel in my_panels]
        if image_max is None:
            image_max = my_image_max
        else:
            for ip in range(len(image_max)):
                image_max[ip] = max(image_max[ip], my_image_max[ip])
        print ()
        pyplot.clf()
        pyplot.close("all")
        del my_fig

        if time.time() - t1 > 30:
            val = gc.collect()
            print (f"Collected {val} garbages!")
            t1 = time.time()

        my_time += dt
        iframe += 1

    plot_cmd = f"ffmpeg -i {oformat} -vcodec libx264 -vf scale=1280:-2,format=yuv420p movie.mp4"
    print (f"Now run:\n\t {plot_cmd}")
