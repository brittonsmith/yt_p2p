import sys
import yt
yt.enable_parallelism()
import ytree

from ytree.analysis import \
    AnalysisPipeline, \
    add_filter
from ytree.utilities.parallel import \
    parallel_nodes

from yt.extensions.p2p.misc import reunit
from yt.extensions.p2p.tree_analysis_operations import add_analysis_fields

def minimum_mass(node, val):
    return node["mass"] >= val

add_filter("minimum_mass", minimum_mass)

def in_region(node):
    ds = node.ds
    refine_left = ds.arr(ds.parameters['RefineRegionLeftEdge'], 'unitary')
    refine_right = ds.arr(ds.parameters['RefineRegionRightEdge'], 'unitary')
    center = reunit(ds, node['position'], 'unitary')
    radius = reunit(ds, node['virial_radius'], 'unitary')

    node["in_region"] = int((center - radius >= refine_left).all() and
                            (center + radius <= refine_right).all())
    return node["in_region"]

add_filter("in_region", in_region)

if __name__ == "__main__":
    afn = sys.argv[1]
    a = ytree.load(afn)

    afields = {'icom_gas_position_x': {'units': 'unitary', 'default': -1},
               'icom_gas_position_y': {'units': 'unitary', 'default': -1},
               'icom_gas_position_z': {'units': 'unitary', 'default': -1},
               'icom_all_position_x': {'units': 'unitary', 'default': -1},
               'icom_all_position_y': {'units': 'unitary', 'default': -1},
               'icom_all_position_z': {'units': 'unitary', 'default': -1}}
    afields['in_region'] = {'units': '', 'default': -1, 'dtype': np.int32}
    add_analysis_fields(a, afields)

    m_min = a.quan(1e3, 'Msun')
    trees = list(a[:])
    group = "forest"

    data_dir = '.'
    ap = AnalysisPipeline()
    ap.add_filter("minimum_mass", m_min)
    ap.add_filter("in_region")
    ap.add_filter("fields_not_assigned", afields)
    ap.add_operation("yt_dataset", data_dir)
    ap.add_operation("node_icom")
    ap.add_operation("delattrs", ["ds"])

    for node in parallel_nodes(trees, group=group, save_every=1,
                               njobs=(1, 0), dynamic=(False, True)):

        result = ap.process_target(node)
