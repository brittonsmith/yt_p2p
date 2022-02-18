"""
Merger tree analysis operations.
"""

import gc
import os
import time
import yaml

from yt import \
    ProjectionPlot, \
    ParticleProjectionPlot, \
    load as yt_load
from yt.utilities.logger import ytLogger as mylog

from yt.extensions.p2p import add_p2p_fields
from yt.extensions.p2p.misc import \
    sphere_icom, \
    reunit

_dataset_dicts = {}
def get_dataset_dict(data_dir):
    data_dir = os.path.abspath(data_dir)
    if data_dir in _dataset_dicts:
        return _dataset_dicts[data_dir]

    with open(os.path.join(data_dir, 'simulation.yaml'), 'r') as f:
        sim_data = yaml.load(f, Loader=yaml.FullLoader)
    _dataset_dicts[data_dir] = \
        dict((snap['Snap_idx'], snap['filename']) for snap in sim_data)

    return _dataset_dicts[data_dir]

_es_dict = {}
def get_dataset_filename(node, data_dir):
    if "Snap_idx" in node.arbor.field_list:
        ddict = get_dataset_dict(data_dir)
        dsfn = ddict.get(int(node['Snap_idx']), None)
        if dsfn is not None:
            return os.path.join(data_dir, dsfn)

    if os.path.exists(os.path.join(data_dir, "simulation.h5")):
        if data_dir not in _es_dict:
            _es_dict[data_dir] = yt_load(os.path.join(data_dir, "simulation.h5"))
        es = _es_dict[data_dir]

        efns = es.data["filename"].astype(str)
        etimes = reunit(node.arbor, es.data["time"].to("Myr"), "Myr")
        ifn = np.abs(etimes - node["time"]).argmin()
        return os.path.join(data_dir, efns[ifn])

    raise RuntimeError(f"Could not associate {node} with a dataset.")

def get_yt_dataset(node, data_dir):
    dsfn = get_dataset_filename(node, data_dir)
    return yt_load(dsfn)

def yt_dataset(node, data_dir, add_fields=True):
    if hasattr(node, "ds"):
        dsfn = get_dataset_filename(node, data_dir)
        if os.path.basename(dsfn) == node.ds.basename:
            return
    node.ds = get_yt_dataset(node, data_dir)
    if add_fields:
        add_p2p_fields(node.ds)

def get_node_sphere(node, ds=None,
                    position_field="position",
                    radius=None,
                    radius_field="virial_radius",
                    radius_factor=1.0):
    if ds is None:
        ds = node.ds
    center = reunit(ds, node[position_field], "unitary")
    if radius is None:
        radius = radius_factor * reunit(ds, node[radius_field], "unitary")
    return ds.sphere(center, radius)

def node_sphere(node,
                position_field="position",
                radius=None,
                radius_field="virial_radius",
                radius_factor=1.0):
    node.sphere = get_node_sphere(
        node,
        position_field=position_field,
        radius=radius,
        radius_field=radius_field,
        radius_factor=radius_factor)

def node_icom(node):
    sphere = get_node_sphere(node)
    center = sphere_icom(sphere, 4*sphere["gas", "dx"].min(),
                         com_kwargs=dict(use_particles=False, use_gas=True))
    sphere.clear_data()
    del sphere
    center.convert_to_units("unitary")
    for iax, ax in enumerate("xyz"):
        node[f"icom_gas_position_{ax}"] = center[iax]

    sphere = get_node_sphere(node)
    center = sphere_icom(sphere, 4*sphere["gas", "dx"].min(),
                         com_kwargs=dict(use_particles=True, use_gas=True))
    sphere.clear_data()
    del sphere
    center.convert_to_units("unitary")
    for iax, ax in enumerate("xyz"):
        node[f"icom_all_position_{ax}"] = center[iax]

def decorate_plot(node, p):
    p.set_axes_unit("pc")
    title = f"t = {node['time'].to('Myr'):.2f}, z = {node['redshift']:.2f}, M = {node['mass'].to('Msun'):.2g}"
    p.annotate_title(title)

def get_projection_filename(
        node, axis, field=None, weight_field=("gas", "density"),
        particle_projections=False,
        output_format="ds", output_dir="."):

    output_key = get_output_key(node, node.ds, output_format)
    if field is not None:
        return os.path.join(output_dir, f"{output_key}_Projection_{axis}_{field[1]}_{weight_field[1]}.png")
    if particle_projections:
        return os.path.join(output_dir, f"{output_key}_Particle_{axis}_particle_mass.png")

def get_region_projection_filenames(
        node, axes="xyz", fields=None, weight_field=("gas", "density"),
        particle_projections=False,
        output_format="ds", output_dir="."):

    fns = []
    if fields is not None:
        my_fns = [
            get_projection_filename(
                node, ax, field=field, weight_field=weight_field,
                output_format=output_format, output_dir=output_dir)
            for field in fields
            for ax in axes]
        fns.extend(my_fns)

    if particle_projections:
        my_fns = [
            get_projection_filename(
                node, ax, particle_projections=True,
                output_format=output_format, output_dir=output_dir)
            for ax in axes]
        fns.extend(my_fns)

    return fns

def region_projections_not_done(
        node, fields, weight_field=("gas", "density"),
        axes="xyz", particle_projections=True,
        output_format="ds", output_dir="."):

    my_fns = [fn for fn in
              get_region_projection_filenames(
                  node, axes=axes, fields=fields, weight_field=weight_field,
                  particle_projections=particle_projections,
                  output_format=output_format, output_dir=output_dir)
              if not os.path.exists(fn)]
    return len(my_fns)

def get_output_key(node, ds, output_format):
    if output_format == "ds":
        output_key = str(ds)
    elif output_format == "node":
        output_key = f"node_{node.uid}"
    else:
        raise ValueError(f"Bad {output_format=}.")
    return output_key

def region_projections(node, fields, weight_field=("gas", "density"),
                       axes="xyz", particle_projections=True,
                       output_format="ds", output_dir="."):

    ds = node.ds

    sphere = node.sphere
    left  = sphere.center - 1.05 * sphere.radius
    right = sphere.center + 1.05 * sphere.radius
    region = ds.box(left, right)

    output_key = get_output_key(node, ds, output_format)

    for ax in axes:
        do_fields = \
            [field for field in fields
             if not os.path.exists(
                     get_projection_filename(
                         node, ax, field=field, weight_field=weight_field,
                         output_format=output_format, output_dir=output_dir))]

        if do_fields:
            add_p2p_fields(ds)
            p = ProjectionPlot(
                ds, ax, do_fields, weight_field=weight_field,
                center=sphere.center, width=2*sphere.radius,
                data_source=region)
            for field, cmap in fields.items():
                p.set_cmap(field, cmap)
            decorate_plot(node, p)
            p.save(output_dir + "/" + output_key)

    if not particle_projections:
        return

    do_axes = \
        [ax for ax in axes
         if not os.path.exists(
                 get_projection_filename(
                     node, ax, particle_projections=True,
                     output_format=output_format, output_dir=output_dir))]

    for ax in do_axes:
        p = ParticleProjectionPlot(
            ds, ax, ("all", "particle_mass"),
            center=sphere.center, width=2*sphere.radius,
            data_source=region)
        p.set_unit(("all", "particle_mass"), "Msun")
        p.set_cmap(("all", "particle_mass"), "turbo")
        decorate_plot(node, p)
        p.save(output_dir + "/" + output_key)

def delattrs(node, attrs):
    for attr in attrs:
        if hasattr(node, attr):
            delattr(node, attr)

time_last_gc = time.time()
def garbage_collect(node, time_between_gc):
    global time_last_gc
    ctime = time.time()
    if ctime - time_last_gc > time_between_gc:
        val = gc.collect()
        mylog.info(f"Collected {val} garbages!")
        time_last_gc = ctime

def fields_not_assigned(node, fields):
    for field in fields:
        default = fields[field]["default"]
        if node[field] == default:
            return True
    return False

def add_analysis_fields(a, fields):
    for field, finfo in fields.items():
        if field in a.field_list:
            continue
        a.add_analysis_field(
            field, units=finfo.get('units', None),
            default=finfo.get('default', -1),
            dtype=finfo.get('dtype', None))
