from collections import defaultdict
import glob
import numpy as np
import os
import sys
import yt

if __name__ == "__main__":
    if os.path.exists("pfs.dat"):
        fns = [line.strip() for line in open('pfs.dat', 'r').readlines()]
    else:
        fns = sorted(glob.glob("DD????/DD????"))

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
