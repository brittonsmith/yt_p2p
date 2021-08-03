from collections import defaultdict
import numpy as np
import sys
import yt

if __name__ == "__main__":
    fns = [line.strip() for line in open('pfs.dat', 'r').readlines()]

    parameters = ['RefineRegionLeftEdge',
                  'RefineRegionRightEdge']

    data = defaultdict(list)
    for fn in fns:
        ds = yt.load(fn)
        data['filename'].append(fn)
        data['time'].append(ds.current_time.to('Myr'))
        data['redshift'].append(ds.current_redshift)
        for par in parameters:
            data[par].append(ds.parameters[par])

    data['filename'] = np.array(data['filename'])
    data['time'] = ds.arr(data['time'])
    data['redshift'] = np.array(data['redshift'])
    data['RefineRegionLeftEdge'] = ds.arr(data['RefineRegionLeftEdge'], 'unitary')
    data['RefineRegionRightEdge'] = ds.arr(data['RefineRegionRightEdge'], 'unitary')

    yt.save_as_dataset(ds, filename='simulation.h5',  data=data)
