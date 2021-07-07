import numpy as np
import os
import sys
import yaml
import yt
yt.enable_parallelism()

import ytree

from yt.funcs import \
    ensure_dir

from yt.extensions.p2p.misc import \
    sphere_icom, \
    reunit
from yt.extensions.p2p.profiles import \
    my_profile
from yt.extensions.p2p import \
    add_p2p_particle_filters, \
    add_p2p_fields

def dirname(path, up=0):
    return "/".join(path.split('/')[:-up-1])

if __name__ == "__main__":
    # afn = "/disk12/brs/pop2-prime/firstpop2_L2-Seed3_large/cc_512_no_dust_continue/merger_trees/p2p_nd/p2p_nd.h5"
    afn = "merger_trees/p2p_nd/p2p_nd.h5"
    a = ytree.load(afn)

    output_data_dir = 'tree_profiles'
    ensure_dir(output_data_dir)

    # data_dir = dirname(afn, 2)
    data_dir = "/disk12/brs/pop2-prime/firstpop2_L2-Seed3_large/cc_512_no_dust_continue"
    with open(os.path.join(data_dir, 'simulation.yaml'), 'r') as f:
        sim_data = yaml.load(f, Loader=yaml.FullLoader)

    dfile = dict((snap['filename'], snap['Snap_idx'])
                 for snap in sim_data)
    dsnap = dict((val, key) for key, val in dfile.items())

    afields = {'in_region': {'units': '', 'default': -1, 'dtype': np.int32},
               'icom_gas_position_x': {'units': 'unitary', 'default': -1},
               'icom_gas_position_y': {'units': 'unitary', 'default': -1},
               'icom_gas_position_z': {'units': 'unitary', 'default': -1},
               'icom_all_position_x': {'units': 'unitary', 'default': -1},
               'icom_all_position_y': {'units': 'unitary', 'default': -1},
               'icom_all_position_z': {'units': 'unitary', 'default': -1}}

    for field, finfo in afields.items():
        if field in a.field_list:
            continue
        a.add_analysis_field(
            field, units=finfo.get('units'),
            default=finfo.get('default', 0),
            dtype=finfo.get('dtype', None))

    m_min = a.quan(1e3, 'Msun')
    trees = a[:]

    for it in range(trees.size):
        tree = trees[it]
        storage = {}

        n_nodes = ((tree['forest', 'mass'] >= m_min) &
                   (tree['forest', 'in_region'] == -1)).sum()
        if n_nodes == 0:
            continue
        if yt.is_root():
            yt.mylog.info(f'Assembling {n_nodes} for tree {it}/{trees.size}.')

        nodes = [node for node in tree['forest']
                 if node['mass'] >= m_min and node['in_region'] == -1]

        for my_storage, inode in yt.parallel_objects(range(len(nodes)), dynamic=True, storage=storage):
            node = nodes[inode]
            my_storage.result_id = inode
            my_storage.result = {}

            dsfn = os.path.join(data_dir, dsnap[int(node['Snap_idx'])])
            ds = yt.load(dsfn)

            refine_left = ds.arr(ds.parameters['RefineRegionLeftEdge'], 'unitary')
            refine_right = ds.arr(ds.parameters['RefineRegionRightEdge'], 'unitary')
            center = reunit(ds, node['position'], 'unitary')
            radius = reunit(ds, node['virial_radius'], 'unitary')

            in_region = int((center - radius >= refine_left).all() and
                         (center + radius <= refine_right).all())
            my_storage.result['in_region'] = in_region

            if not in_region:
                yt.mylog.info(f'Node {inode}/{len(nodes)} of tree {it}/{trees.size} is outside refinement region.')
                continue

            sphere = ds.sphere(center, radius)
            center = sphere_icom(sphere, 4*sphere['dx'].min(),
                                 com_kwargs=dict(use_particles=False, use_gas=True))
            center.convert_to_units('unitary')
            for iax, ax in enumerate('xyz'):
                my_storage.result[f'icom_gas_position_{ax}'] = center.d[iax]

            # reset center
            center = reunit(ds, node['position'], 'unitary')
            sphere = ds.sphere(center, radius)
            center = sphere_icom(sphere, 4*sphere['dx'].min(),
                                 com_kwargs=dict(use_particles=True, use_gas=True))
            center.convert_to_units('unitary')
            for iax, ax in enumerate('xyz'):
                my_storage.result[f'icom_all_position_{ax}'] = center.d[iax]

            yt.mylog.info(f'Finished node {inode}/{len(nodes)} of tree {it}/{trees.size}.')

        if not nodes:
            continue

        if yt.is_root():
            yt.mylog.info(f'Assembling results for tree {it}/{trees.size} ({tree}).')
            for inode, results in sorted(storage.items()):
                if results is None:
                    continue

                node = nodes[inode]
                for field in results:
                    node[field] = results[field]

            fn = a.save_arbor(trees=trees)
            a = ytree.load(fn)
            trees = a[:]
