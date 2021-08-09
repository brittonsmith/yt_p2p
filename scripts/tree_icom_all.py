import numpy as np
import os
import sys
import yaml
import yt
yt.enable_parallelism()
import ytree

from ytree.analysis import AnalysisPipeline

from yt.extensions.p2p.tree_analysis_operations import add_analysis_fields

if __name__ == "__main__":
    afn = sys.argv[1]
    a = ytree.load(afn)

    afields = {'icom_gas_position_x': {'units': 'unitary', 'default': -1},
               'icom_gas_position_y': {'units': 'unitary', 'default': -1},
               'icom_gas_position_z': {'units': 'unitary', 'default': -1},
               'icom_all_position_x': {'units': 'unitary', 'default': -1},
               'icom_all_position_y': {'units': 'unitary', 'default': -1},
               'icom_all_position_z': {'units': 'unitary', 'default': -1}}
    add_analysis_fields(a, afields)

    m_min = a.quan(1e3, 'Msun')
    trees = list(a[:])
    group = "forest"

    data_dir = '.'
    ap = AnalysisPipeline()
    ap.add_filter("fields_not_assigned", afields)
    ap.add_operation("yt_dataset", data_dir)
    ap.add_operation("node_icom")
    ap.add_operation("delattrs", ["ds"])

    for it in range(len(trees)):
        tree = trees[it]
        storage = {}

        n_nodes = (tree[group, 'mass'] >= m_min).sum()
        if n_nodes == 0:
            continue
        if yt.is_root():
            yt.mylog.info(f'Analyzing {n_nodes} nodes for tree {it+1}/{len(trees)}.')

        nodes = [node for node in tree[group]
                 if node['mass'] >= m_min]

        for my_storage, inode in yt.parallel_objects(range(len(nodes)),
                                                     dynamic=True, storage=storage):
            node = nodes[inode]
            result = ap.process_target(node)

            my_storage.result_id = inode
            my_storage.result = {}
            for field in afields:
                my_storage.result[field] = node[field].d

            yt.mylog.info(f'Finished node {inode+1}/{len(nodes)} of tree {it+1}/{len(trees)}.')

        if not nodes:
            continue

        if yt.is_root():
            yt.mylog.info(f'Assembling results for tree {it+1}/{len(trees)} ({tree}).')
            for inode, results in sorted(storage.items()):
                if results is None:
                    continue

                node = nodes[inode]
                for field in results:
                    node[field] = results[field]

            fn = a.save_arbor(trees=trees)
            a = ytree.load(fn)
            trees = list(a[:])
