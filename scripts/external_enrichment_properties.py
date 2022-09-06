"""
Calculate external enrichment properties for halos in merger tree.
"""

import gc
import numpy as np
import os
import yaml
import yt
yt.enable_parallelism()
import ytree

from ytree.analysis import AnalysisPipeline
from yt.extensions.p2p.fields import add_p2p_fields
from yt.extensions.p2p.misc import reunit
from yt.extensions.p2p.tree_analysis_operations import \
    add_analysis_fields, \
    get_yt_dataset, \
    yt_dataset, \
    delattrs, \
    garbage_collect, \
    node_sphere, \
    node_icom, \
    fields_not_assigned
from yt.utilities.parallel_tools.parallel_analysis_interface import _get_comm

def pop3_dataset(node, data_dir):
    basename = os.path.basename(node._ds_filename)
    p3fn = os.path.join(data_dir, "pop3", f"{basename}.h5")

    p3ds = getattr(node, "pop3_ds", None)
    if p3ds is None or p3ds.parameter_filename != p3fn:
        node.pop3_ds = yt.load(p3fn)
    del p3ds

def n_stars(node):
    pos = reunit(node.arbor, node.pop3_ds.data["pop3", "particle_position"], "unitary")
    d = np.sqrt(((node["position"] - pos)**2).sum(axis=1))
    node["n_stars"] = (d < node["virial_radius"]).sum()
    yt.mylog.info(f"{node} has {node['n_stars']} stars.")

def halo_properties(node):
    add_p2p_fields(node.ds)
    sp = node.sphere

    rho_max, Z_dense = sp.quantities.sample_at_max_field_values(
        ("gas", "density"), [("gas", "metallicity3")])
    node["max_density"] = reunit(node.arbor, rho_max, "g/cm**3")
    node["densest_metallicity"] = reunit(node.arbor, Z_dense, "")
    Z_ex = sp.quantities.extrema([("gas", "metallicity3")])
    Z_ex = reunit(node.arbor, Z_ex, "")
    node["max_metallicity"] = Z_ex[1]
    del rho_max, Z_dense, Z_ex

    yt.mylog.info(f"{node} analysis fields:")
    for field in ("max_density", "max_metallicity", "densest_metallicity"):
        yt.mylog.info(f"{field}: {node[field]}.")

    sp.clear_data()
    del sp

def get_analysis_nodes(a, isnap):
    if yt.is_root():
        yt.mylog.info(f"Searching for nodes with snap index: {isnap}.")
    selection = a.get_yt_selection(equal=[("Snap_idx", isnap, "")], above=[("mass", 1e4, "Msun")])

    do = False
    for field, config in afields.items():
        if yt.is_root():
            yt.mylog.info(f"Checking selection values for {field}.")
        my_def = a.quan(config["default"], config["units"])

        field_storage = {}
        for my_store, my_chunk in yt.parallel_objects(selection.chunks([], "io"),
                                                      storage=field_storage):
            my_store.result = (my_chunk["halos", field] == my_def).any()
        do |= any(list(field_storage.values()))
        if do:
            break
        # just check one field
        break

    if do:
        if yt.is_root():
            yt.mylog.info(f"Generating nodes...")
        my_nodes = list(a.get_nodes_from_selection(selection, deconstructed=True))
    else:
        if yt.is_root():
            yt.mylog.info(f"Skipping snap index: {isnap}.")
        my_nodes = []

    del selection
    val = gc.collect()
    yt.mylog.info(f"Collected {val} garbages!")
    return my_nodes

if __name__ == "__main__":
    output_dir = "."
    #data_dir = "/cephfs/brs/pop2-prime/firstpop2_L2-Seed3_large/cc_512_no_dust_continue/"
    data_dir = "/disk12/brs/pop2-prime/firstpop2_L2-Seed3_large/cc_512_no_dust_continue/"

    comm = _get_comm(())
    dynamic = comm.comm is not None and comm.comm.size > 1

    a = ytree.load("merger_trees/p2p_nd_enrichment/p2p_nd.h5")    
    my_trees = list(a[:])
   
    afields = {
        "n_stars": {"units": "", "dtype": int, "default": -1},
        "max_density": {"units": "g/cm**3", "default": -1},
        "max_metallicity": {"units": "", "default": -1},
        "densest_metallicity": {"units": "", "default": -1},
    }
    add_analysis_fields(a, afields)

    ap = AnalysisPipeline(output_dir=output_dir)
    ap.add_operation(fields_not_assigned, afields)
    ap.add_operation(yt_dataset, data_dir=data_dir, add_fields=False)
    ap.add_operation(pop3_dataset, data_dir)
    ap.add_operation(n_stars)
    ap.add_operation(node_icom)
    ap.add_operation(node_sphere, center_field="position")
    ap.add_operation(halo_properties)
    ap.add_operation(delattrs, ["sphere"], always_do=True)
    ap.add_operation(garbage_collect, 60, always_do=True)

    done_file = "ee_done"
    if os.path.exists(done_file):
        with open(done_file, mode="r") as f:
            istart = int(f.readline().strip()) - 1
    else:
        istart = a[0]["Snap_idx"]
    if yt.is_root():
        yt.mylog.info(f"Starting from snapshot {istart}.")

    t = a[0]
    ds = get_yt_dataset(t, data_dir)
    a.unit_registry.modify("Zsun", ds.parameters["SolarMetalFractionByMass"])
    del t, ds

    handoff_attrs=["ds", "pop3_ds"]
    for isnap in range(istart, 0, -1):
        my_nodes = get_analysis_nodes(a, isnap)

        if yt.is_root():
            yt.mylog.info(f"Analyzing {len(my_nodes)} nodes with snap index: {isnap}.")
        for node in ytree.parallel_trees(
                my_nodes, base_trees=my_trees, dynamic=dynamic, save_every=None,
                save_nodes_only=True, save_in_place = True):

            ap.process_target(node, handoff_attrs=handoff_attrs)

        ap._handoff_store.clear()
        if len(my_nodes) > 0:
            a._ytds = None
        del my_nodes, my_trees

        val = gc.collect()
        yt.mylog.info(f"Collected {val} garbages!")

        na = a.reload_arbor()
        del a
        a = na
        my_trees = list(a[:])

        if yt.is_root():
            with open(done_file, mode="w") as f:
                f.write(f"{isnap}\n")
