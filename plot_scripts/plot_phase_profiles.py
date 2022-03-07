from matplotlib import pyplot
import numpy as np
import os
from yt.visualization.color_maps import \
    yt_colormaps

from yt.extensions.p2p.plots import \
    plot_phase, \
    twin_unit_axes, \
    draw_major_grid, \
    mirror_yticks, \
    make_phase_plot

from grid_figure import GridFigure

def plot_density_HD_fraction(data_dir, halo_id):
    xfield = 'density'
    yfield = 'HD_fraction'
    filename = os.path.join(data_dir, "%s_%06d.h5" %
                            (yfield, halo_id))

    my_fig = GridFigure(
        1, 1, figsize=(6, 4.5),
        top_buffer = 0.03, bottom_buffer = 0.13,
        left_buffer = 0.16, right_buffer = 0.19)

    my_axes = my_fig[0]
    xscale = 'log'
    yscale = 'log'

    field = 'cell_mass'
    units = 'Msun'
    cmap = yt_colormaps['dusk']
    clabel = "M [M$_{\\odot}$]"

    xlim = (1e-3, 1e13)
    xmajor = np.logspace(-3, 12, 6)
    xminor = np.logspace(-3, 13, 17)
    xlabel = "n [cm$^{-3}$]"

    ylim = (1e-10, 1e-4)
    ymajor = np.logspace(-10, -4, 7)
    yminor = None
    ylabel = "f$_{\\rm HD}$"

    output_filename = "figures/%s_%s.pdf" % \
      (xfield, yfield)

    make_phase_plot(
        my_fig, my_axes, filename,
        field, units, cmap, clabel,
        xlim, xmajor, xminor, xscale, xlabel,
        ylim, ymajor, yminor, yscale, ylabel,
        output_filename)

def plot_density_H2_fraction(data_dir, halo_id):
    xfield = 'density'
    yfield = 'H2_fraction'
    filename = os.path.join(data_dir, "%s_%06d.h5" %
                            (yfield, halo_id))

    fontsize = 14
    my_fig = GridFigure(
        1, 1, figsize=(6, 4.5),
        top_buffer = 0.03, bottom_buffer = 0.13,
        left_buffer = 0.14, right_buffer = 0.19)

    my_axes = my_fig[0]
    xscale = 'log'
    yscale = 'log'

    field = 'cell_mass'
    units = 'Msun'
    cmap = yt_colormaps['dusk']
    clabel = "M [M$_{\\odot}$]"

    xlim = (1e-3, 1e13)
    xmajor = np.logspace(-3, 12, 6)
    xminor = np.logspace(-3, 13, 17)
    xlabel = "n [cm$^{-3}$]"

    ylim = (1e-8, 1)
    ymajor = np.logspace(-8, 0, 9)
    yminor = None
    ylabel = "f$_{\\rm H_{2}}$"

    output_filename = "figures/%s_%s.pdf" % \
      (xfield, yfield)

    make_phase_plot(
        my_fig, my_axes, filename,
        field, units, cmap, clabel,
        xlim, xmajor, xminor, xscale, xlabel,
        ylim, ymajor, yminor, yscale, ylabel,
        output_filename)

def plot_density_HD_H2_ratio(data_dir, halo_id):
    xfield = 'density'
    yfield = 'HD_H2_ratio'
    filename = os.path.join(data_dir, "%s_%06d.h5" %
                            (yfield, halo_id))

    my_fig = GridFigure(
        1, 1, figsize=(6, 4.5),
        top_buffer = 0.03, bottom_buffer = 0.13,
        left_buffer = 0.14, right_buffer = 0.19)

    my_axes = my_fig[0]
    xscale = 'log'
    yscale = 'log'

    field = 'cell_mass'
    units = 'Msun'
    cmap = yt_colormaps['dusk']
    clabel = "M [M$_{\\odot}$]"

    xlim = (1e-3, 1e13)
    xmajor = np.logspace(-3, 12, 6)
    xminor = np.logspace(-3, 13, 17)
    xlabel = "n [cm$^{-3}$]"

    ylim = (2e-5, 200)
    ymajor = np.logspace(-5, 2, 8)
    yminor = None
    ylabel = "f$_{\\rm HD}$ / f$_{\\rm H_{2}}$"

    output_filename = "figures/%s_%s.pdf" % \
      (xfield, yfield)

    make_phase_plot(
        my_fig, my_axes, filename,
        field, units, cmap, clabel,
        xlim, xmajor, xminor, xscale, xlabel,
        ylim, ymajor, yminor, yscale, ylabel,
        output_filename)

def plot_density_temperature(data_dir, halo_id):
    xfield = 'density'
    yfield = 'temperature'
    filename = os.path.join(data_dir, "%s_%06d.h5" %
                            (yfield, halo_id))

    my_fig = GridFigure(
        1, 1, figsize=(6, 4.5),
        top_buffer = 0.02, bottom_buffer = 0.13,
        left_buffer = 0.12, right_buffer = 0.19)

    my_axes = my_fig[0]
    xscale = 'log'
    yscale = 'log'

    field = 'cell_mass'
    units = 'Msun'
    cmap = yt_colormaps['dusk']
    clabel = "M [M$_{\\odot}$]"

    xlim = (1e-3, 1e13)
    xmajor = np.logspace(-3, 12, 6)
    xminor = np.logspace(-3, 13, 17)
    xlabel = "n [cm$^{-3}$]"

    ylim = (10, 2e4)
    ymajor = np.logspace(1, 4, 4)
    yminor = None
    ylabel = "T [K]"

    output_filename = "figures/%s_%s.pdf" % \
      (xfield, yfield)

    make_phase_plot(
        my_fig, my_axes, filename,
        field, units, cmap, clabel,
        xlim, xmajor, xminor, xscale, xlabel,
        ylim, ymajor, yminor, yscale, ylabel,
        output_filename)

def plot_radius_density(data_dir, halo_id):
    xfield = 'radius'
    yfield = 'density'
    filename = os.path.join(data_dir, "%s_%06d.h5" %
                            (yfield, halo_id))

    fontsize = 14
    my_fig = GridFigure(
        1, 1, figsize=(6, 4.5),
        top_buffer = 0.14, bottom_buffer = 0.13,
        left_buffer = 0.16, right_buffer = 0.19)

    my_axes = my_fig[0]
    my_axes.set_xscale('log')
    my_axes.set_yscale('log')

    my_cax = my_fig.add_cax(my_axes, "right", buffer=0.02,
                            length=0.95, width=0.04)
    plot_phase(filename, 'cell_mass', 'Msun',
               my_axes, my_cax=my_cax,
               cmap=yt_colormaps['dusk'])

    my_cax.yaxis.set_label_text("M [M$_{\\odot}$]")

    xlim = (2e-6, 2e2)
    tx = twin_unit_axes(
        my_axes, xlim, "r",
        "pc", top_units="AU")

    ylim = (1e-25, 1e-10)
    ymajor = np.logspace(-25, -10, 6)
    yminor = np.logspace(-25, -10, 16)
    ylabel = "$\\rho$ [g/cm$^{-3}$]"
    mirror_yticks(my_axes, ylim, ymajor, yminor=yminor)
    draw_major_grid(my_axes, 'y', ymajor)
    my_axes.yaxis.set_label_text(ylabel)

    output_filename = "figures/%s_%s.pdf" % \
      (xfield, yfield)
    pyplot.savefig(output_filename)

def plot_radius_H2_fraction(data_dir, halo_id):
    xfield = 'radius'
    yfield = 'H2_fraction'
    filename = os.path.join(data_dir, "%s_%06d.h5" %
                            (yfield, halo_id))

    fontsize = 14
    my_fig = GridFigure(
        1, 1, figsize=(6, 4.5),
        top_buffer = 0.14, bottom_buffer = 0.13,
        left_buffer = 0.16, right_buffer = 0.19)

    my_axes = my_fig[0]
    my_axes.set_xscale('log')
    my_axes.set_yscale('log')

    my_cax = my_fig.add_cax(my_axes, "right", buffer=0.02,
                            length=0.95, width=0.04)
    plot_phase(filename, 'cell_mass', 'Msun',
               my_axes, my_cax=my_cax,
               cmap=yt_colormaps['dusk'])

    my_cax.yaxis.set_label_text("M [M$_{\\odot}$]")

    xlim = (2e-6, 2e2)
    tx = twin_unit_axes(
        my_axes, xlim, "r",
        "pc", top_units="AU")

    ylim = (1e-8, 1)
    ymajor = np.logspace(-8, 0, 9)
    ylabel = "f$_{\\rm H_{2}}$"
    draw_major_grid(my_axes, 'y', ymajor)
    mirror_yticks(my_axes, ylim, ymajor)
    my_axes.yaxis.set_label_text(ylabel)

    output_filename = "figures/%s_%s.pdf" % \
      (xfield, yfield)
    pyplot.savefig(output_filename)

def plot_radius_metallicity(data_dir, halo_id):
    xfield = 'radius'
    yfield = 'metallicity3_min7'
    filename = os.path.join(data_dir, "%s_%06d.h5" %
                            (yfield, halo_id))

    fontsize = 14
    my_fig = GridFigure(
        1, 1, figsize=(6, 4.5),
        top_buffer = 0.14, bottom_buffer = 0.13,
        left_buffer = 0.14, right_buffer = 0.19)

    my_axes = my_fig[0]
    my_axes.set_xscale('log')
    my_axes.set_yscale('log')

    my_cax = my_fig.add_cax(my_axes, "right", buffer=0.02,
                            length=0.95, width=0.04)
    plot_phase(filename, 'cell_mass', 'Msun',
               my_axes, my_cax=my_cax,
               cmap=yt_colormaps['dusk'])

    my_cax.yaxis.set_label_text("M [M$_{\\odot}$]")

    xlim = (2e-6, 2e2)
    tx = twin_unit_axes(
        my_axes, xlim, "r",
        "pc", top_units="AU")

    ylim = (4.5e-8, 0.1)
    ymajor = np.logspace(-7, -1, 7)
    ylabel = "Z [Z$_{\\odot}$]"
    mirror_yticks(my_axes, ylim, ymajor)
    draw_major_grid(my_axes, 'y', ymajor)
    my_axes.yaxis.set_label_text(ylabel)

    output_filename = "figures/%s_%s.pdf" % \
      (xfield, yfield)
    pyplot.savefig(output_filename)

def plot_radius_temperature(data_dir, halo_id):
    xfield = 'radius'
    yfield = 'temperature'
    filename = os.path.join(data_dir, "%s_%06d.h5" %
                            (yfield, halo_id))

    fontsize = 14
    my_fig = GridFigure(
        1, 1, figsize=(6, 4.5),
        top_buffer = 0.14, bottom_buffer = 0.13,
        left_buffer = 0.14, right_buffer = 0.19)

    my_axes = my_fig[0]
    my_axes.set_xscale('log')
    my_axes.set_yscale('log')

    my_cax = my_fig.add_cax(my_axes, "right", buffer=0.02,
                            length=0.95, width=0.04)
    plot_phase(filename, 'cell_mass', 'Msun',
               my_axes, my_cax=my_cax,
               cmap=yt_colormaps['dusk'])

    my_cax.yaxis.set_label_text("M [M$_{\\odot}$]")

    xlim = (2e-6, 2e2)
    tx = twin_unit_axes(
        my_axes, xlim, "r",
        "pc", top_units="AU")

    ylim = (10, 2e4)
    ymajor = np.logspace(1, 4, 4)
    ylabel = "T [K]"
    mirror_yticks(my_axes, ylim, ymajor)
    draw_major_grid(my_axes, 'y', ymajor)
    my_axes.yaxis.set_label_text(ylabel)

    output_filename = "figures/%s_%s.pdf" % \
      (xfield, yfield)
    pyplot.savefig(output_filename)

if __name__ == "__main__":
    halo_id = 41732

    data_dir = "../halo_catalogs/profile_catalogs/DD0560/density_profiles"
    plot_density_HD_fraction(data_dir, halo_id)
    pyplot.clf()
    plot_density_H2_fraction(data_dir, halo_id)
    pyplot.clf()
    plot_density_HD_H2_ratio(data_dir, halo_id)
    pyplot.clf()
    plot_density_temperature(data_dir, halo_id)

    pyplot.clf()
    data_dir = "../halo_catalogs/profile_catalogs/DD0560/radial_profiles"
    plot_radius_density(data_dir, halo_id)
    pyplot.clf()
    plot_radius_H2_fraction(data_dir, halo_id)
    pyplot.clf()
    plot_radius_metallicity(data_dir, halo_id)
    pyplot.clf()
    plot_radius_temperature(data_dir, halo_id)
