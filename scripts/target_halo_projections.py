import numpy as np
import os
import time
import yt
yt.enable_parallelism()
import ytree

from ytree.analysis import AnalysisPipeline
from yt.extensions.p2p.stars import get_star_data
from yt.extensions.p2p.tree_analysis_operations import \
    get_progenitor_line, \
    yt_dataset, \
    delattrs, \
    garbage_collect, \
    node_sphere, \
    region_projections, \
    region_projections_not_done

pfields = {
    ("gas", "density"): "algae",
    ("gas", "temperature"): "gist_heat",
    ("gas", "metallicity3"): "kamae",
    ("gas", "H2_p0_fraction"): "cool"
}

if __name__ == "__main__":
    output_data_dir = "target_halos"
    # data_dir = "/disk12/brs/pop2-prime/firstpop2_L2-Seed3_large/cc_512_no_dust_continue"
    data_dir = "/cephfs/brs/pop2-prime/firstpop2_L2-Seed3_large/cc_512_no_dust_continue"

    center_fields = [
        "position",
        "icom_all_position",
        "icom_gas_position"
    ]

    a = ytree.load("merger_trees/target_halos/target_halos.h5")

    for tree in a:

        output_dir = os.path.join(output_data_dir, f"tree_{tree.uid}")
        done_file = os.path.join(output_dir, "proj_done")
        if os.path.exists(done_file):
            continue

        ap = AnalysisPipeline(output_dir=output_dir)
        ap.add_operation(yt_dataset, data_dir, add_fields=False)

        for field in center_fields:
            my_output_dir = f"projections/{field}"
            ap.add_operation(region_projections_not_done,
                             pfields, output_dir=my_output_dir)
            ap.add_operation(node_sphere, center_field=field)
            ap.add_operation(region_projections, pfields, output_dir=my_output_dir)
            ap.add_operation(delattrs, ["sphere"], always_do=True)

        ap.add_operation(delattrs, ["ds"], always_do=True)
        ap.add_operation(garbage_collect, 60, always_do=True)

        for node in ytree.parallel_tree_nodes(tree, group="prog", dynamic=True):
            ap.process_target(node)

        if yt.is_root():
            with open(done_file, mode='w') as f:
                f.write(f"{time.time()}\n")
