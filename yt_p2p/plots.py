"""
Pop2Prime plotting functions.



"""

#-----------------------------------------------------------------------------
# Copyright (c) Britton Smith <brittonsmith@gmail.com>.  All rights reserved.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

from matplotlib import \
    pyplot, \
    colors, \
    ticker
import numpy as np
import yt

from yt.units.yt_array import \
    YTArray
from yt.visualization.color_maps import \
    yt_colormaps

def plot_profile_distribution(
        my_axes, filename, field,
        x_units=None, y_units=None,
        pkwargs=None, show_dist=True, step=0.01,
        alpha_scale=1.0):

    if pkwargs is None:
        pkwargs = {}

    linewidth = pkwargs.pop("linewidth", 1)

    ds = yt.load(filename)
    x_data = ds.profile.x
    if x_units is not None:
        x_data.convert_to_units(x_units)
    y_data = ds.profile.y
    if y_units is not None:
        y_data.convert_to_units(y_units)
    z_data = ds.profile[field].transpose()
    z_sum = z_data.sum(axis=0)
    rfil = z_sum > 0
    gmin = np.where(~rfil)[0].max() + 1

    z_data = z_data[:, gmin:]
    x_data = x_data[gmin:]

    z_sort = z_data.cumsum(axis=0) / z_data.sum(axis=0)
    y_med = y_data[np.abs(z_sort - 0.5).argmin(axis=0)]
    pfilter = y_med == y_med
    my_axes.plot(x_data[pfilter], y_med[pfilter],
                 alpha=0.9, linewidth=linewidth, **pkwargs)

    my_alpha = alpha_scale * 2 * step
    if show_dist:
        for offset in np.arange(step, 0.5+step, step):
            y1 = y_data[np.abs(z_sort - (0.5 - offset)).argmin(axis=0)]
            y2 = y_data[np.abs(z_sort - (0.5 + offset)).argmin(axis=0)]
            pfilter &= (y1 != y_data.min())
            my_axes.fill_between(
                x_data[pfilter], y1[pfilter], y2[pfilter],
                linewidth=0, alpha=my_alpha, **pkwargs)

def plot_profile_distribution_legend(
        my_axes, items, step=1, label_rotation=0,
        fontsize=12, lfontsize=8, alpha_scale=1.0,
        lwidth=0.1, lheight=0.2):

    my_pos = np.array(my_axes.get_position())
    panel_width = my_pos[1, 0] - my_pos[0, 0]
    panel_height = my_pos[1, 1] - my_pos[0, 1]

    my_lwidth = lwidth * len(items) * panel_width
    my_lheight = lheight * panel_height
    lax = my_axes.figure.add_axes(
        (my_pos[0, 0] + 0.1,
         my_pos[1, 1] - my_lheight - 0.05,
         my_lwidth, my_lheight))

    ihalf = 50
    my_alpha = alpha_scale * step / ihalf
    ones = np.ones(2)
    for i, item in enumerate(items):
        color, label, dist = item
        lax.plot([i-0.4, i+0.4], [50, 50], linewidth=1,
                 color=color, alpha=0.9)
        if dist:
            for offset in range(step, ihalf+1, step):
                lax.fill_between([i-0.4, i+0.4],
                                 y1=ihalf-offset,
                                 y2=ihalf+offset,
                                 alpha=my_alpha, color=color,
                                 linewidth=0)

    lax.xaxis.set_tick_params(direction="in")
    lax.yaxis.set_tick_params(direction="inout", length=4)
    lax.set_xlim(-0.5, len(items)-0.5)
    lax.set_ylim(0, 100)
    lax.yaxis.tick_left()
    lax.yaxis.set_ticks(np.arange(0, 101, 25))
    lax.yaxis.set_label_text("%", fontsize=lfontsize)
    lax.yaxis.set_label_position("left")
    lax.yaxis.labelpad = 0

    lty = lax.twinx()
    lty.yaxis.tick_right()
    lty.set_ylim(0, 100)
    lty.yaxis.set_ticks(np.arange(0, 101, 25))
    lty.yaxis.set_tick_params(direction="in", length=2)
    for tl in lty.yaxis.get_majorticklabels():
        tl.set_visible(False)

    lax.xaxis.tick_bottom()
    lax.xaxis.set_ticks(np.arange(len(items)))
    lax.xaxis.set_ticklabels([item[1] for item in items],
                             rotation=label_rotation)
    for tick in lax.xaxis.get_ticklines():
        tick.set_visible(False)
    for tl in lax.xaxis.get_majorticklabels():
        tl.set_visible(True)
        tl.set_horizontalalignment("center")
        tl.set_verticalalignment("top")
        tl.set_fontsize(fontsize)
    for tl in lax.yaxis.get_majorticklabels():
        tl.set_fontsize(lfontsize)

def plot_phase(filename, field, units,
               my_axes, cmap=None, my_cax=None):

    ds = yt.load(filename)
    x_data = ds.profile.x
    y_data = ds.profile.y
    z_data = ds.profile[field].to(units)

    if cmap is None:
        cmap = yt_colormaps['dusk']

    nz = z_data > 0
    my_min = z_data[nz].min()
    my_max = z_data[nz].max()

    c_max = np.floor(np.log10(my_max))
    c_min = np.ceil(np.log10(my_min))
    c_step = np.ceil((c_max - c_min + 1) / 6)
    c_ticks = 10**np.arange(c_min, c_max+1, c_step)
    my_norm = colors.LogNorm(my_min, my_max)

    my_image = my_axes.pcolormesh(x_data, y_data, z_data.T,
                                  norm=my_norm, cmap=cmap,
                                  zorder=9999)
    cbar = pyplot.colorbar(my_image, orientation="vertical",
                           cax=my_cax, ticks=c_ticks)

def get10s(lim):
    bds = np.ceil(np.log10(lim))
    return np.logspace(bds[0], bds[1], bds[1]-bds[0]+1)

def draw_major_grid(my_axes, axis, ticks, **pkwargs):
    if not pkwargs:
        pkwargs = dict(color='black', linestyle='-',
                       linewidth=1, alpha=0.2)

    my_axis = getattr(my_axes, "%saxis" % axis)
    for tick in ticks:
        if axis == 'x':
            my_axes.axvline(x=tick, zorder=1, **pkwargs)
        elif axis == 'y':
            my_axes.axhline(y=tick, zorder=1, **pkwargs)
        else:
            raise RuntimeError("Axis must be x or y.")

def twin_unit_axes(
        my_axes, xlim, xlabel,
        bottom_units, top_units=None,
        bottom_grid=True, top_grid=True):

    my_axes.xaxis.set_ticks(get10s(xlim))
    my_axes.set_xlim(xlim)
    my_axes.xaxis.labelpad = 2
    my_axes.xaxis.set_label_text(
        "%s [%s]" % (xlabel, bottom_units))
    if bottom_grid:
        draw_major_grid(my_axes, 'x', get10s(xlim),
                        color='black', linestyle='-',
                        linewidth=1, alpha=0.2)

    if top_units is not None:
        tx = my_axes.twiny()
        tx.set_xscale('log')
        tx.xaxis.tick_top()
        txlim = YTArray(xlim, bottom_units).to(top_units)
        tx.xaxis.set_ticks(get10s(txlim))
        tx.set_xlim(tuple(txlim))
        tx.xaxis.set_label_text(
            "%s [%s]" % (xlabel, top_units))
        tx.xaxis.labelpad = 8
        if top_grid:
            draw_major_grid(tx, 'x', get10s(txlim),
                            color='black', linestyle=':',
                            linewidth=1, alpha=0.2)
        return tx

def mirror_xticks(my_axes, xlim, xmajor, xminor=None):
    my_axes.xaxis.set_ticks(xmajor)
    if xminor is not None:
        my_axes.xaxis.set_ticks(xminor, minor=True)
        my_axes.xaxis.set_minor_formatter(ticker.NullFormatter())
    my_axes.set_xlim(xlim)

    tx = my_axes.twiny()
    tx.set_xscale(my_axes.get_xscale())
    tx.xaxis.tick_top()
    tx.xaxis.set_tick_params(direction='in', which='both')
    tx.xaxis.set_ticks(xmajor)
    if xminor is not None:
        tx.xaxis.set_ticks(xminor, minor=True)
        tx.xaxis.set_minor_formatter(ticker.NullFormatter())
    tx.set_xlim(xlim)
    for ticklabel in tx.xaxis.get_majorticklabels():
        ticklabel.set_visible(False)
    return tx

def mirror_yticks(my_axes, ylim, ymajor, yminor=None):
    my_axes.yaxis.set_ticks(ymajor)
    if yminor is not None:
        my_axes.yaxis.set_ticks(yminor, minor=True)
        my_axes.yaxis.set_minor_formatter(ticker.NullFormatter())
    my_axes.set_ylim(ylim)

    ty = my_axes.twinx()
    ty.set_yscale(my_axes.get_yscale())
    ty.yaxis.tick_right()
    ty.yaxis.set_tick_params(direction='in', which='both')
    ty.yaxis.set_ticks(ymajor)
    if yminor is not None:
        ty.yaxis.set_ticks(yminor, minor=True)
        ty.yaxis.set_minor_formatter(ticker.NullFormatter())
    ty.set_ylim(ylim)
    for ticklabel in ty.yaxis.get_majorticklabels():
        ticklabel.set_visible(False)
    return ty
