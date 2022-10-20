import numpy as np
import os
import yaml
import yt
import ytree

from yt.extensions.p2p.misc import reunit

def _relative_distance(field, data):
    ptype = field.name[0]
    return data[ptype, "particle_radius"] / \
        data[ptype, "virial_radius"]

if __name__ == "__main__":
    ds = yt.load('pop3/DD0560.h5')
    es = yt.load('simulation.h5')
    efns = es.data['filename'].astype(str)
    etimes = es.data['time'].to('Myr')
    eredshifts = es.data['redshift']

    afn = "merger_trees/p2p_nd_rs/p2p_nd_rs.h5"
    a = ytree.load(afn)
    a.ytds.add_field(
        ("halos", "relative_distance"),
        function=_relative_distance,
        units="", sampling_type="local")

    data_dir = "."
    with open(os.path.join(data_dir, 'simulation.yaml'), 'r') as f:
        sim_data = yaml.load(f, Loader=yaml.FullLoader)

    cts = ds.data['pop3', 'creation_time'].to('Myr')
    star_ids = ds.data['pop3', 'particle_index'].d.astype(int)

    star_hosts = {}
    for i in range(cts.size):
        ifn = np.where(etimes > cts[i])[0][0]
        my_fn = efns[ifn]
        my_z = eredshifts[ifn]
        z_high = (my_z + eredshifts[ifn-1]) / 2
        z_low = (my_z + eredshifts[ifn+1]) / 2

        my_pid = int(star_ids[i])
        star_hosts[my_pid] = {'creation_time': str(cts[i])}
        my_star = star_hosts[my_pid]

        yt.mylog.info(f'Finding host halo for star {my_pid}.')

        sfn = os.path.join('pop3', f"{my_fn.split('/')[-1]}.h5")
        sds = yt.load(sfn)
        istar = np.where(sds.data['pop3', 'particle_index'].d.astype(int) == my_pid)[0][0]
        my_star_pos = sds.data['pop3', 'particle_position'][istar]
        my_star_pos = reunit(a.ytds, my_star_pos, "unitary")

        ad = a.ytds.all_data()
        ad.set_field_parameter("center", my_star_pos)
        selection = a.get_yt_selection(below=[("relative_distance", 1.0, ""),
                                              ("redshift", float(z_high), "")],
                                       above=[("redshift", float(z_low), "")],
                                       data_source=ad)
        nodes = list(a.get_nodes_from_selection(selection))
        del selection, ad

        if len(nodes) > 0:
            yt.mylog.info(f"Found {len(nodes)} nodes for star {my_pid}.")
            my_star["arbor"] = a.filename
            my_star["tree_ids"] = [node.tree_id for node in nodes]
            my_star["_arbor_indices"] = [int(node.root._arbor_index) for node in nodes]
        else:
            yt.mylog.info(f'Host halo for star {my_pid} not found.')

    with open('star_hosts_rs.yaml', mode='w') as f:
        yaml.dump(star_hosts, stream=f)
