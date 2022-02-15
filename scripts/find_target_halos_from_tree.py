"""
Use this to locate your target halo, i.e., the halo with extremely dense gas.

Usage: python location_trees.py <merger tree path>
"""

import numpy as np
import os
import sys
import yt
yt.enable_parallelism()
import ytree

from ytree.analysis import AnalysisPipeline

from yt.extensions.p2p.tree_analysis_operations import \
    region_projections, \
    yt_dataset, \
    node_sphere, \
    delattrs

pfields = {
    ("gas", "density"): "algae",
    ("gas", "temperature"): "gist_heat",
    ("gas", "metallicity3"): "kamae",
    ("gas", "H2_p0_fraction"): "cool"
}

def above_some_density(node, density):
    density = node.ds.quan(*density)
    max_density = node.sphere["gas", "density"].max()
    yt.mylog.info(f"{node} ({node['mass']}) has max density {max_density}.")
    node["has_dense_gas"] = max_density >= density
    return node["has_dense_gas"]

if __name__ == "__main__":
    a = ytree.load(sys.argv[1])
    a.add_analysis_field("has_dense_gas", "", dtype=bool, default=0)

    if len(sys.argv) > 2:
        data_dir = sys.argv[2]
    else:
        data_dir = "."

    ap = AnalysisPipeline(output_dir="target_location")
    ap.add_operation(yt_dataset, data_dir)
    ap.add_operation(node_sphere)
    ap.add_operation(above_some_density, (1e-21, "g/cm**3"))
    ap.add_operation(region_projections, pfields, axes='xyz', particle_projections=True,
                     output_format="node", output_dir="projections")
    ap.add_operation(delattrs, ["sphere"], always_do=True, time_between_gc=60)

    trees = list(a[:])
    ds = None
    for tree in ytree.parallel_trees(trees, dynamic=False, save_every=False):
        if ds is not None:
            tree.ds = ds

        ap.process_target(tree)

        if ds is None:
            ds = tree.ds
        del tree.ds

    if yt.is_root():
        my_trees = [tree for tree in trees if tree["has_dense_gas"]]
        fn = "merger_trees/target_halos"
        a.save_arbor(filename=fn, trees=my_trees)
