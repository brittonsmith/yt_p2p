"""
Check merger trees for bad sections (i.e., where halo mass/position
jumps abruptly).
"""

import numpy as np
import os
import time
import yaml
import yt
import ytree

from ytree.data_structures.tree_container import TreeContainer
from ytree.analysis import AnalysisPipeline
from yt.extensions.p2p.stars import get_star_data
from yt.extensions.p2p.tree_analysis_operations import \
    get_progenitor_line, \
    delattrs, \
    garbage_collect, \
    yt_dataset

if __name__ == "__main__":
    output_data_dir = "star_minihalos"

    # data_dir = "/disk12/brs/pop2-prime/firstpop2_L2-Seed3_large/cc_512_no_dust_continue"
    data_dir = "/cephfs/brs/pop2-prime/firstpop2_L2-Seed3_large/cc_512_no_dust_continue"

    star_data = get_star_data("star_hosts.yaml")

    for star_id, star_info in star_data.items():
        print (f"Checking {star_id}.")
        output_dir = os.path.join(output_data_dir, f"star_{star_id}")

        a = ytree.load(star_info["arbor"])

        ap = AnalysisPipeline(output_dir=output_dir)
        ap.add_operation(delattrs, ["ds"], always_do=True)
        ap.add_operation(garbage_collect, 60, always_do=True)

        my_tree = a[star_info["_arbor_index"]]

        # find the halo where the star formed
        form_node = my_tree.get_node("forest", star_info["tree_id"])
        nodes = list(get_progenitor_line(form_node))

        # get all nodes going back at least 50 Myr or until progenitor is 1e4 Msun
        m_min = a.quan(1e4, "Msun")
        tprior = a.quan(50, "Myr")

        last_good_mass = None
        min_rat = None
        max_rat = None
        for node in nodes:
            # if form_node["time"] < node["time"] and node["mass"] < m_min:
            #     continue
            if form_node["time"] - node["time"] > tprior and node["mass"] < m_min:
                continue

            if last_good_mass is not None:
                ratio = node["mass"] / last_good_mass
                if ratio < 0.1 or 1 / ratio < 0.1:
                    print (f"BAD {node} has mass {node['mass']} but last good is {last_good_mass} ({ratio}).")
                    yt_dataset(node, data_dir, add_fields=False)
                    print (node._ds_filename)
                else:
                    # print (f"GOOD {node} has mass {node['mass']} and last good is {last_good_mass} ({ratio}).")
                    last_good_mass = node["mass"]
                min_rat = ratio if min_rat is None else min(min_rat, ratio)
                max_rat = ratio if max_rat is None else max(max_rat, ratio)
            else:
                last_good_mass = node["mass"]

            # ap.process_target(node)

        print (f"Min: {min_rat}, max: {max_rat}.")
