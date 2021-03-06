import numpy as np
import os
import yaml
import yt
import ytree

from yt.extensions.p2p.misc import \
    reunit

if __name__ == "__main__":
    ds = yt.load('pop3/DD0560.h5')
    es = yt.load('simulation.h5')
    efns = es.data['filename'].astype(str)
    etimes = es.data['time'].to('Myr')

    afn = "merger_trees/p2p_nd/p2p_nd.h5"
    a = ytree.load(afn)
    m_min = a.quan(1e5, 'Msun')
    my_trees = a[a['mass'] >= m_min]

    data_dir = "."
    with open(os.path.join(data_dir, 'simulation.yaml'), 'r') as f:
        sim_data = yaml.load(f, Loader=yaml.FullLoader)

    dfile = dict((snap['filename'], snap['Snap_idx'])
                 for snap in sim_data)
    dsnap = dict((val, key) for key, val in dfile.items())

    cts = ds.data['pop3', 'creation_time'].to('Myr')
    ctsort = cts.argsort()
    star_ids = ds.data['pop3', 'particle_index'].d.astype(int)

    star_hosts = {}

    for i in ctsort:
        my_fn = efns[etimes > cts[i]][0]
        my_snap_idx = dfile[my_fn]

        my_pid = int(star_ids[i])

        yt.mylog.info(f'Finding host halo for star {my_pid}.')

        sfn = os.path.join('pop3', f"{my_fn.split('/')[-1]}.h5")
        sds = yt.load(sfn)
        istar = np.where(sds.data['pop3', 'particle_index'].d.astype(int) == my_pid)[0][0]
        my_star_pos = reunit(a, sds.data['pop3', 'particle_position'][istar], 'unitary')

        star_hosts[my_pid] = {'creation_time': str(cts[i])}

        for my_tree in my_trees:
            in_ds = my_tree['forest', 'Snap_idx'] == my_snap_idx
            if not in_ds.any():
                continue

            d_star = np.sqrt(((my_tree['forest', 'position'][in_ds] - my_star_pos)**2).sum(axis=1))
            r_halo = my_tree['forest', 'virial_radius'][in_ds]

            in_halo = d_star <= r_halo
            if not in_halo.any():
                continue

            ihalos = np.where(in_ds)[0][in_halo]
            m_halo = my_tree['forest', 'mass'][ihalos]
            ihalo = ihalos[m_halo.argmax()]
            my_node = my_tree.get_node('forest', ihalo)

            star_hosts[my_pid]['tree_id'] = my_node.tree_id
            star_hosts[my_pid]['_arbor_index'] = int(my_node.root._arbor_index)
            star_hosts[my_pid]['arbor'] = a.filename
            break

        if 'tree_id' in star_hosts[my_pid] is None:
            yt.mylog.info(f'Host halo for star {my_pid} not found.')
        else:
            yt.mylog.info(f'Host halo for star {my_pid} found.')

    ofn = 'star_hosts.yaml'
    with open(ofn, 'r') as f:
        host_data = yaml.load(f, Loader=yaml.FullLoader)

    output_data = dict([(my_star, star_hosts[my_star])
                        for my_star in star_hosts
                        if 'tree_id' in star_hosts[my_star]])
    output_data.update(host_data)
    with open('star_hosts.yaml', mode='w') as f:
        yaml.dump(output_data, stream=f)
