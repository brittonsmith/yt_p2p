import sys
import yt
yt.enable_parallelism()
import ytree

from unyt import unyt_quantity

from ytree.analysis import AnalysisPipeline

from yt.extensions.p2p.stars import get_star_data
from yt.extensions.p2p.tree_analysis_operations import \
    add_analysis_fields, \
    get_progenitor_line, \
    fields_not_assigned, \
    yt_dataset, \
    node_icom, \
    delattrs, \
    garbage_collect

def minimum_mass(node, val):
    return node["mass"] >= val

if __name__ == "__main__":
    data_dir = '/disk12/brs/pop2-prime/firstpop2_L2-Seed3_large/cc_512_no_dust_continue'

    afields = {'icom_gas_position_x': {'units': 'unitary', 'default': -1},
               'icom_gas_position_y': {'units': 'unitary', 'default': -1},
               'icom_gas_position_z': {'units': 'unitary', 'default': -1},
               'icom_all_position_x': {'units': 'unitary', 'default': -1},
               'icom_all_position_y': {'units': 'unitary', 'default': -1},
               'icom_all_position_z': {'units': 'unitary', 'default': -1}}
    m_min = unyt_quantity(1e3, "Msun")

    star_data = get_star_data("star_hosts.yaml")

    ap = AnalysisPipeline()
    # ap.add_operation(minimum_mass, m_min)
    ap.add_operation(fields_not_assigned, afields)
    ap.add_operation(yt_dataset, data_dir)
    ap.add_operation(node_icom)
    ap.add_operation(garbage_collect, 60, always_do=True)

    for star_id, star_info in star_data.items():
        a = ytree.load(star_info["arbor"])
        add_analysis_fields(a, afields)

        trees = list(a[:])
        my_tree = trees[star_info["_arbor_index"]]
        form_node = my_tree.get_node("forest", star_info["tree_id"])
        nodes = list(get_progenitor_line(form_node))

        for node in ytree.parallel_tree_nodes(my_tree, nodes=nodes,
                                              dynamic=True):
            ap.process_target(node)

        if yt.is_root():
            a.save_arbor(trees=trees)
