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
from yt.extensions.p2p.tree_analysis_operations import \
    yt_dataset, \
    delattrs, \
    garbage_collect, \
    region_projections, \
    region_projections_not_done, \
    node_sphere

pfields = {
    ("gas", "density"): "algae",
    ("gas", "temperature"): "gist_heat",
    ("gas", "metallicity3"): "kamae",
    ("gas", "H2_p0_fraction"): "cool"
}

def all_projections(ap, a, field):
    pos_field = f"{field}_position"
    if f"{pos_field}_x" not in a.field_list:
        return

    ap.add_operation(region_projections_not_done,
                     pfields, output_dir=f"{field}_projections")
    ap.add_operation(node_sphere, center_field=pos_field)
    ap.add_operation(region_projections,
                     pfields, output_dir=f"{field}_projections")
    ap.add_operation(delattrs, ["sphere"])

if __name__ == "__main__":
    output_dir = "minihalo_analysis"
    data_dir = "."

    a = ytree.load("merger_trees/target_halos/target_halos.h5")
    cfields = ["icom_gas", "icom_all"]

    for tree in a:
        ap = AnalysisPipeline(output_dir=os.path.join(output_dir, f"node_{tree.uid}"))
        ap.add_operation(yt_dataset, data_dir, add_fields=False)

        for field in cfields:
            ap.add_recipe(all_projections, a, field)
        ap.add_operation(delattrs, ["ds"], always_do=True)
        ap.add_operation(garbage_collect, 60, always_do=True)

        for node in ytree.parallel_tree_nodes(tree, group="prog", dynamic=True):
            ap.process_target(node)
