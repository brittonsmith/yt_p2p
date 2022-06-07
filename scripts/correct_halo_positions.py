"""
Used to correct the halo positions in the pisn_solo simulation.

The iterative center-of-mass positions (icom_gas_position) becomes
corrupted by a halo approaching from outside the refinement region.

The icom_gas2_position correction uses linear interpolation from
the "good" positions, determined manually. This seems to have
worked.

The icom_gas3_position correction starts a new iterative center of
mass calculation from that position, but using only the gas. I do
not think it worked all that well.
"""

import copy
import numpy as np
import os
from scipy.interpolate import interp1d
import time
import yaml
import yt
yt.enable_parallelism()
import ytree

from yt.frontends.enzo.data_structures import EnzoDataset

from ytree.analysis import \
    AnalysisPipeline

from yt.extensions.p2p.misc import \
    sphere_icom, \
    reunit
from yt.extensions.p2p.profiles import my_profile
from yt.extensions.p2p.tree_analysis_operations import yt_dataset, delattrs
from yt.extensions.p2p import add_p2p_fields

def icom_gas3_position(node, sphere_radius):
    arbor = node.arbor
    ds = node.ds

    npos = node["icom_gas_position"].to("unitary")
    apos = node["icom_all_position"].to("unitary")
    dis = np.sqrt(((apos-npos)**2).sum())
    yt.mylog.info(f"Displacement from all_icom_position: {dis.to('kpc')}.")
    if dis > arbor.quan(100, "pc"):
        center = reunit(ds, apos.to("unitary"), "unitary")
        sphere = ds.sphere(center, sphere_radius)
        center = sphere_icom(sphere, 4*sphere["gas", "dx"].min(),
                             com_kwargs=dict(use_particles=False, use_gas=True))
        new_center = reunit(arbor, center.to("unitary"), "unitary")
    else:
        new_center = npos

    dis = np.sqrt(((apos-new_center)**2).sum())
    yt.mylog.info(f"New displacement from all_icom_position: {dis.to('kpc')}.")
    for i, ax in enumerate("xyz"):
        node[f"icom_gas3_position_{ax}"] = new_center[i]

def icom_gas2_position(node, interp, sphere_radius):
    arbor = node.arbor
    ds = node.ds

    nt = node["time"].to("Myr")
    npos = node["icom_gas_position"].to("unitary")
    ipos = arbor.arr(interp(nt), "unitary")
    dis = np.sqrt(((ipos-npos)**2).sum())
    yt.mylog.info(f"Displacement from interpolation: {dis.to('kpc')}.")
    if dis > arbor.quan(500, "pc"):
        center = reunit(ds, ipos.to("unitary"), "unitary")
        sphere = ds.sphere(center, sphere_radius)
        center = sphere_icom(sphere, 4*sphere["gas", "dx"].min(),
                             com_kwargs=dict(use_particles=False, use_gas=True))
        new_center = reunit(arbor, center.to("unitary"), "unitary")
    else:
        new_center = npos

    dis = np.sqrt(((ipos-new_center)**2).sum())
    yt.mylog.info(f"New displacement from interpolation: {dis.to('kpc')}.")
    for i, ax in enumerate("xyz"):
        node[f"icom_gas2_position_{ax}"] = new_center[i]

def icom_sphere(node):
    ds = node.ds

    radius = reunit(ds, node["virial_radius"], "unitary")
    if "icom_gas_position_x" in node.arbor.field_list:
        center = reunit(ds, node["icom_gas_position"], "unitary")
    else:
        center = reunit(ds, node["position"], "unitary")
        sphere = ds.sphere(center, radius)
        center = sphere_icom(sphere, 4*sphere["gas", "dx"].min(),
                             com_kwargs=dict(use_particles=False, use_gas=True))

    sphere = ds.sphere(center, radius)
    node.sphere = sphere

def my_yt_dataset(node, data_dir, es):
    if "Snap_idx" in node.arbor.field_list:
        yt_dataset(node, data_dir)
    else:
        efns = es.data["filename"].astype(str)
        etimes = reunit(node.arbor, es.data["time"].to("Myr"), "Myr")
        ifn = np.abs(etimes - node["time"]).argmin()
        dsfn = os.path.join(data_dir, efns[ifn])
        node.ds = yt.load(dsfn)

if __name__ == "__main__":
    output_dir = "minihalo_analysis"

    es = yt.load("simulation.h5")

    data_dir = "/disk12/brs/pop2-prime/firstpop2_L2-Seed3_large/pisn_solo"

    a = ytree.load("merger_trees/target_halos/target_halos.h5")

    for field in ["icom_gas2", "icom_gas3"]:
        for ax in "xyz":
            a.add_analysis_field(f"{field}_position_{ax}", units="unitary", default=-1)
        a.add_vector_field(field)

    # assemble data for interpolation
    tree = a[0]
    tt = np.flip(tree["prog", "time"].to("Myr"))
    tpos = np.flip(tree["prog", "icom_gas_position"].to("unitary"), axis=0)
    good = np.where((np.diff(tt, prepend=0) > 0))[0]
    tt = tt[good]
    tpos = tpos[good]

    # find jumps
    dis = np.sqrt((np.diff(tpos, axis=0)**2).sum(axis=1)).to('kpc')
    dis = np.insert(dis, 0, 0)
    breaks = np.where(dis > 1)[0]
    breaks = np.concatenate([[0], breaks, [dis.size]])
    groups = [slice(breaks[i], breaks[i+1]) for i in range(breaks.size-1)]
    # the good groups are 1 and 3
    good_groups = [1, 3]
    t_int = a.arr(np.concatenate([tt[groups[i]] for i in good_groups]), tt.units)
    pos_int = a.arr(np.concatenate([tpos[groups[i]] for i in good_groups]), tpos.units)
    interp = interp1d(t_int, pos_int, kind="linear", axis=0, fill_value="extrapolate")

    ap = AnalysisPipeline(output_dir=output_dir)
    ap.add_operation(my_yt_dataset, data_dir, es)
    ap.add_operation(icom_gas2_position, interp, (100, "pc"))
    ap.add_operation(icom_gas3_position, (100, "pc"))
    ap.add_operation(delattrs, ["ds"])

    for node in ytree.parallel_tree_nodes(tree, group="prog"):
        ap.process_target(node)

    if yt.is_root():
        a.save_arbor(trees=[tree])
