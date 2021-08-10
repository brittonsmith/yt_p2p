"""
Merger tree analysis operations.
"""

import gc
import os
import yaml

from yt.loaders import load as yt_load
from yt.utilities.logger import ytLogger as mylog

from yt.extensions.p2p.misc import \
    sphere_icom, \
    reunit

from ytree.analysis import \
    add_filter, \
    add_operation

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

def get_yt_dataset(node, data_dir):
    ddict = get_dataset_dict(data_dir)
    dsfn = os.path.join(data_dir, ddict[int(node['Snap_idx'])])
    return yt_load(dsfn)

def yt_dataset(node, data_dir):
    node.ds = get_yt_dataset(node, data_dir)

add_operation("yt_dataset", yt_dataset)

def get_node_sphere(node, ds=None,
                    position_field="position",
                    radius_field="virial_radius",
                    radius_factor=1.0):
    if ds is None:
        ds = node.ds
    center = reunit(ds, node[position_field], "unitary")
    radius = radius_factor * reunit(ds, node[radius_field], "unitary")
    return ds.sphere(center, radius)

def node_sphere(node,
                position_field="position",
                radius_field="virial_radius",
                radius_factor=1.0):
    node.sphere = get_node_sphere(
        position_field=position_field,
        radius_field=radius_field,
        radius_factor=radius_factor)

add_operation("node_sphere", node_sphere)

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

add_operation("node_icom", node_icom)

def delattrs(node, attrs):
    for attr in attrs:
        delattr(node, attr)
    val = gc.collect()
    mylog.info(f"Collected {val} garbages!")

add_operation("delattrs", delattrs)

def fields_not_assigned(node, fields):
    for field in fields:
        default = fields[field]["default"]
        if node[field] == default:
            return True
    return False

add_filter("fields_not_assigned", fields_not_assigned)

def add_analysis_fields(a, fields):
    for field, finfo in fields.items():
        if field in a.field_list:
            continue
        a.add_analysis_field(
            field, units=finfo.get('units', None),
            default=finfo.get('default', -1),
            dtype=finfo.get('dtype', None))
