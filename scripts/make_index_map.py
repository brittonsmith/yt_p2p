"""
Create a yaml file with a map between the Snap_idx consistent-trees field
and the Enzo snapshot name.
"""

import numpy as np
import os
import sys
import yaml
import yt
import ytree

if __name__ == "__main__":
    es = yt.load('simulation.h5')
    fns = es.data['filename'].astype(str)[::-1]
    redshifts = es.data['redshift'][::-1]

    a = ytree.load(sys.argv[1])
    idx_min = np.array([t['forest', 'Snap_idx'].min() for t in a])
    tree = a[np.argmin(idx_min)]

    tidx = tree['forest', 'Snap_idx']
    uidx = np.flip(np.sort(np.unique(tidx)))
    my_nodes = [tree.get_node('forest', np.where(tidx == i)[0][0]) for i in uidx]

    map_file = 'simulation.yaml'
    if os.path.exists(map_file):
        with open(map_file, 'r') as f:
            file_map = yaml.load(f, Loader=yaml.FullLoader)
        file_map.reverse()
        istart = np.where(uidx == file_map[-1]['Snap_idx'])[0][0] + 1
    else:
        file_map = []
        istart = 0

    max_offset = fns.size - uidx.max()
    for inode, node in enumerate(my_nodes):
        if inode < istart:
            continue
        if node['phantom']:
            continue

        if file_map:
            ifile = np.where(file_map[-1]['filename'] == fns)[0][0] + 1
        else:
            ifile = 0

        for i in range(ifile, min(ifile+max_offset, fns.size)):
            hdsfn = 'rockstar_halos/halos_%s.0.bin' % os.path.basename(fns[i])
            hds = yt.load(hdsfn)

            hid = hds.r['all', 'particle_identifier'].astype(int) == int(node['halo_id'])
            if not hid.any():
                continue

            hmass = hds.r['all', 'particle_mass'][hid].to('Msun')
            nmass = node['mass'].to('Msun')

            yt.mylog.info(f"Looking for node {node['uid']} ({node['halo_id']}) in {fns[i]}.")

            diff = np.abs(hmass - nmass) / max(hmass, nmass)
            yt.mylog.info(f'Diff: {diff}.')
            if diff < 0.5:
                file_map.append(
                    {'filename': str(fns[i]),
                     'Snap_idx': int(node['Snap_idx'])})
                yt.mylog.info(f"Found node {node['uid']} ({node['halo_id']}) in {fns[i]}.")
                print ("")
                break
            print ("")

    file_map.reverse()

    idx = np.array([m['Snap_idx'] for m in file_map])
    tofix = np.where(np.diff(idx) > 1)[0]

    fns = fns[::-1]

    for spot in tofix:
        isnap = file_map[spot]['Snap_idx']
        isnapp1 = file_map[spot+1]['Snap_idx']
        snap_diff = isnapp1 - isnap

        ifile = np.where(file_map[spot]['filename'] == fns)[0][0]
        ifilep1 = np.where(file_map[spot+1]['filename'] == fns)[0][0]
        file_diff = ifilep1 - ifile

        if snap_diff == file_diff:
            for isn, ifi in zip(range(isnap+1, isnapp1),
                                range(ifile+1, ifilep1)):
                file_map.append({'filename': str(fns[ifi]), 'Snap_idx': isn})

    file_map.sort(key=lambda a: a['Snap_idx'])

    with open(map_file, 'w') as f:
        yaml.dump(file_map, stream=f)
