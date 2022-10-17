from matplotlib import pyplot
from matplotlib.ticker import FuncFormatter
import numpy as np

from yt.units.yt_array import YTQuantity, YTArray
from yt.utilities.cosmology import Cosmology

pyplot.rcParams['axes.unicode_minus'] = False

def create_timeline(my_axes, cosmology, t_initial, t_final, t_units="Myr",
                    t_major=None, t_minor=None, t_current=None, redshifts=None,
                    fontsize=12, text_color="white"):

    def _z_from_t(t, pos):
        return "%d" % np.round(cosmology.z_from_t(YTQuantity(t, t_units)))

    my_axes.set_xlim(t_initial.in_units(t_units),
                     t_final.in_units(t_units))

    if t_major is None: t_major = []
    my_axes.xaxis.set_ticks(t_major.d)
    if t_minor is None: t_minor = []
    my_axes.xaxis.set_ticks(t_minor.d, minor=True)
    my_axes.xaxis.set_label_text("t [%s]" % t_units, 
                                 fontsize=fontsize,
                                 color=text_color)
    my_axes.xaxis.labelpad = 2

    tx = my_axes.twiny()
    tx.xaxis.tick_top()
    tx.set_xlim(t_initial.in_units(t_units),
                t_final.in_units(t_units))
    
    if redshifts is not None:
        time_from_z = cosmology.t_from_z(redshifts).in_units(t_units)
        tx.xaxis.set_ticks(time_from_z.d)
        tx.xaxis.set_major_formatter(FuncFormatter(_z_from_t))

    tx.xaxis.set_label_text("z",
                            fontsize=fontsize,
                            color=text_color)

    ticklabels = my_axes.xaxis.get_ticklabels() + \
      tx.xaxis.get_ticklabels()
    for ticklabel in ticklabels:
        ticklabel.set_color(text_color)
        ticklabel.set_size(fontsize)

    ticklines = my_axes.xaxis.get_ticklines() + \
      my_axes.xaxis.get_ticklines(minor=True) + \
      tx.xaxis.get_ticklines()
    for tickline in ticklines:
        tickline.set_visible(True)
        tickline.set_color(text_color)

    my_axes.yaxis.set_visible(False)
    my_axes.axhline(y=0., color=text_color, alpha=1.0, linewidth=2)
    my_axes.axhline(y=0.99, color=text_color, alpha=1.0, linewidth=2)

    if t_current is not None:
        my_t = t_current.in_units(t_units)
        my_axes.axvline(x=my_t, color=text_color, linewidth=2)
