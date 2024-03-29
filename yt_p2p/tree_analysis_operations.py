"""
Merger tree analysis operations.
"""

import gc
import numpy as np
import os
import time
import yaml

from ytree.data_structures.tree_node import TreeNode

from yt import \
    ProjectionPlot, \
    ParticleProjectionPlot, \
    load as yt_load
from yt.utilities.logger import ytLogger as mylog

from yt.extensions.p2p.fields import add_p2p_fields
from yt.extensions.p2p.misc import \
    sphere_icom, \
    reunit
from yt.extensions.p2p.profiles import my_profile

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

def _get_node_time_index(node, etimes, offset=0):
    """
    Get the time index associated with this node.

    Check the node and its two last ancestors against the time array.
    To be valid, all three indices must be unique, meaning neither
    the node in question has a duplicate index with its ancestor nor
    the ancestor has a duplicate index with its ancestor.
    """
    ifn = np.abs(etimes - node["time"]).argmin()
    ancs = list(node.ancestors)
    if not ancs:
        return ifn + offset

    iancs = []
    for i, anc in enumerate(node["prog"]):
        if i == 0:
            continue
        if i == 1:
            my_anc = anc
        if i > 2:
            break
        iancs.append(np.abs(etimes - anc["time"]).argmin())

    if ifn != iancs[0] and iancs[0] != iancs[1]:
        return ifn + offset
    return  _get_node_time_index(my_anc, etimes, offset+1)

_es_dict = {}
def _get_dataset_filename_h5(node, data_dir):
    if not os.path.exists(os.path.join(data_dir, "simulation.h5")):
        return None

    if data_dir not in _es_dict:
        _es_dict[data_dir] = yt_load(os.path.join(data_dir, "simulation.h5"))
    es = _es_dict[data_dir]

    efns = es.data["filename"].astype(str)
    etimes = reunit(node.arbor, es.data["time"].to("Myr"), "Myr")
    ifn = _get_node_time_index(node, etimes)
    return efns[ifn]

def get_dataset_filename(node, data_dir):
    if "Snap_idx" in node.arbor.field_list:
        ddict = get_dataset_dict(data_dir)
        dsfn = ddict.get(int(node['Snap_idx']), None)
        if dsfn is not None:
            return os.path.join(data_dir, dsfn)

    dsfn = _get_dataset_filename_h5(node, data_dir)
    if dsfn is not None:
        return os.path.join(data_dir, dsfn)

    raise RuntimeError(f"Could not associate {node} with a dataset.")

def get_yt_dataset(node, data_dir):
    dsfn = get_dataset_filename(node, data_dir)
    return yt_load(dsfn)

def delattrs(node, attrs):
    for attr in attrs:
        try:
            delattr(node, attr)
        except AttributeError:
            pass

def _setds(node, value):
    node._ds = value

_ds_attrs = ("_ds", "_ds_filename")
def _delds(node):
    delattrs(node, _ds_attrs)

def _getds(node):
    if node._ds is None:
        node._ds = yt_load(node._ds_filename)
    return node._ds

def _yt_dataset_pre():
    TreeNode.ds = property(fget=_getds, fset=_setds, fdel=_delds)

def yt_dataset(node, data_dir, add_fields=True):
    node._ds_filename = get_dataset_filename(node, data_dir)

    # If we already have one, check it's the right one.
    ds = getattr(node, "_ds", None)
    if ds is None or os.path.basename(node._ds_filename) != ds.basename:
        node._ds = None
    del ds

    if add_fields:
        add_p2p_fields(node.ds)

yt_dataset.preprocess = _yt_dataset_pre

def get_node_sphere(node, ds=None,
                    center_field="position",
                    radius=None,
                    radius_field="virial_radius",
                    radius_factor=1.0):
    if ds is None:
        ds = node.ds
    center = reunit(ds, node[center_field], "unitary")
    if radius is None:
        radius = radius_factor * reunit(ds, node[radius_field], "unitary")
    return ds.sphere(center, radius)

def _setsp(node, value):
    node._sphere = value

_sp_attrs = (
    "_sphere",
    "_sphere_center_field",
    "_sphere_radius",
)

def _delsp(node):
    delattrs(node, _sp_attrs)

def _getsp(node):
    if node._sphere is not None:
        return node._sphere

    ds = node.ds
    center = reunit(ds, node[node._sphere_center_field], "unitary")
    radius = ds.arr(*node._sphere_radius)
    node._sphere = ds.sphere(center, radius)
    del center, radius, ds
    return node._sphere

def _node_sphere_pre():
    TreeNode.sphere = property(fget=_getsp, fset=_setsp, fdel=_delsp)

def node_sphere(node,
                center_field="position",
                radius=None,
                radius_field="virial_radius",
                radius_factor=1.0):

    if radius is None:
        radius = radius_factor * node[radius_field]
        radius.convert_to_units("unitary")
    node._sphere_radius = (radius.d, str(radius.units))
    del radius
    node._sphere = None
    node._sphere_center_field = center_field

node_sphere.preprocess = _node_sphere_pre

def do_node_icom(node, itype):
    if itype not in ("all", "gas"):
        raise ValueError(f"itype ({itype}) must be either \"all\" or \"gas\".")

    default = node.arbor.quan(-1, "unitary")
    if all([node[f"icom_{itype}_position_{ax}"] > default for ax in "xyz"]):
        return

    if itype == "all":
        com_kwargs = {"use_gas": True, "use_particles": True}
    elif itype == "gas":
        com_kwargs = {"use_gas": True, "use_particles": False}
    else:
        raise RuntimeError()

    sphere = get_node_sphere(node)
    center = sphere_icom(sphere, 4*sphere["gas", "dx"].min(),
                         com_kwargs=com_kwargs)
    sphere.clear_data()
    del sphere

    center.convert_to_units("unitary")
    for iax, ax in enumerate("xyz"):
        node[f"icom_{itype}_position_{ax}"] = center[iax]

def node_icom(node):
    for itype in ("all", "gas"):
        do_node_icom(node, itype)

def node_profile(node, bin_fields, profile_fields, weight_field,
                 data_object="sphere", profile_kwargs=None,
                 output_format="ds", output_dir=".",):

    nd = len(bin_fields)
    for field in bin_fields + profile_fields + [weight_field]:
        if field is not None and not isinstance(field, tuple):
            raise ValueError(f"Field {field} must be a tuple.")

    if weight_field is None:
        wname = "None"
    else:
        wname = weight_field[1]

    output_key = get_output_key(node, output_format)
    fpre = f"{output_key}_{nd}D_profile"
    fkey = "_".join([field[1] for field in bin_fields]) + f"_{wname}"
    fn = f"{fpre}_{fkey}.h5"
    ofn = os.path.join(output_dir, fn)
    if os.path.exists(ofn):
        return

    data_source = getattr(node, data_object, None)
    if data_source is None:
        raise ValueError("Cannot find node attribute {data_object}.")

    if profile_kwargs is None:
        profile_kwargs = {}
    profile_kwargs["weight_field"] = weight_field

    profile = my_profile(
        data_source,
        bin_fields,
        profile_fields,
        **profile_kwargs,
    )

    profile.save_as_dataset(filename=ofn)
    del profile

def decorate_plot(node, p):
    p.set_axes_unit("pc")
    title = f"t = {node['time'].to('Myr'):.2f}, z = {node['redshift']:.2f}, M = {node['mass'].to('Msun'):.2g}"
    p.annotate_title(title)

def get_projection_filename(
        node, axis, field=None, weight_field=("gas", "density"),
        particle_projections=False,
        output_format="ds", output_dir="."):

    output_key = get_output_key(node, output_format)
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

def get_output_key(node, output_format):
    if output_format == "ds":
        dsfn = getattr(node, "_ds_filename", None)
        if dsfn is None:
            output_key = str(node.ds)
        else:
            output_key = os.path.basename(dsfn)
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

    output_key = get_output_key(node, output_format)

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
            del p

    sphere.clear_data()
    region.clear_data()
    if not particle_projections:
        del sphere, region, ds
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
        del p

    sphere.clear_data()
    region.clear_data()
    del sphere, region, ds

def sphere_radial_profiles(node, fields, weight_field=None, profile_kwargs=None,
                           output_format="ds", output_dir="."):

    if weight_field is None:
        weight_name = "None"
    else:
        if not isinstance(weight_field, tuple) and len(weight_field) != 2:
            raise ValueError("weight_field must be a tuple of length 2.")
        weight_name = weight_field[1]

    output_key = get_output_key(node, output_format)
    fn = os.path.join(output_dir, f"{output_key}_profile_weight_field_{weight_name}.h5")
    if os.path.exists(fn):
        return

    add_p2p_fields(node.ds)

    pkwargs = {"accumulation": False, "bin_density": 20}
    if profile_kwargs is not None:
        pkwargs.update(profile_kwargs)

    data_source = node.sphere

    if ("gas", "velocity_spherical_radius") in fields:
        bulk_velocity = data_source.field_parameters["bulk_velocity"]
        if bulk_velocity.sum().to("cm/s") < 1e-3:
            max_vals = data_source.quantities.sample_at_max_field_values(
                ("gas", "density"), ["velocity_%s" % ax for ax in "xyz"])
            bulk_velocity = node.ds.arr(max_vals[1:]).to("km/s")
            mylog.info(f"Setting bulk velocity to {bulk_velocity}.")
            data_source.set_field_parameter("bulk_velocity", bulk_velocity)

    profile = my_profile(
        data_source,
        ("index", "radius"), fields,
        units={("index", "radius"): "pc"},
        weight_field=weight_field,
        **pkwargs)
    profile.save_as_dataset(filename=fn)
    data_source.clear_data()
    del profile, data_source

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

def get_progenitor_line(node):
    """
    Get line of nodes that pass through this node to the root.

    For nodes from earlier in time, take the main progenitor line.
    After this node, that the descendents.
    """

    node = list(node.get_leaf_nodes(selector="prog"))[0]
    while node is not None:
        yield node
        node = node.descendent

def add_analysis_fields(a, fields):
    for field, finfo in fields.items():
        if field in a.field_list:
            continue
        a.add_analysis_field(
            field, units=finfo.get('units', None),
            default=finfo.get('default', -1),
            dtype=finfo.get('dtype', None))
