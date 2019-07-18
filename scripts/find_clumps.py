"""
Clump finding.

Usage: python find_clumps.py <dataset> [halo catalog dataset]

The optional halo catalog dataset should be the one with a
single halo, used to make profiles of the collapsed region.
"""
import numpy as np
import os
import sys
import yt

from yt.data_objects.level_sets.api import \
    Clump, \
    find_clumps
from yt.funcs import ensure_dir

import yt.extensions.p2p.clumps
from yt.extensions.p2p.misc import \
    iterate_center_of_mass, \
    reunit

if __name__ == "__main__":
    ds = yt.load(sys.argv[1])
    if len(sys.argv) > 2:
        hds = yt.load(sys.argv[2])
    else:
        data_dir = os.path.dirname(os.path.abspath(ds.directory))
        hfn = os.path.join(data_dir, 'halo_catalogs/profile_catalogs',
                           '%s.0.h5' % ds.parameter_filename)
        hds = yt.load(hfn)

    center = reunit(ds, hds.r['particle_position'][0], 'unitary')
    radius = 0.1 * reunit(ds, hds.r['virial_radius'][0], 'unitary')

    yt.mylog.info('Finding clumps within %s.' % radius.to('pc'))

    data_source = ds.sphere(center, radius)
    field = ("gas", "density")
    step = 2.0
    c_min = 10**np.floor(np.log10(data_source[field]).min()  )
    c_max = 10**np.floor(np.log10(data_source[field]).max()+1)
    master_clump = Clump(data_source, field)
    output_dir = "clumps/"
    ensure_dir(output_dir)

    master_clump.add_validator(
        "future_bound",
        use_thermal_energy=True,
        truncate=True,
        include_cooling=True)

    master_clump.add_info_item("center_of_mass")
    master_clump.add_info_item("min_number_density")
    master_clump.add_info_item("max_number_density")
    master_clump.add_info_item("jeans_mass")

    find_clumps(master_clump, c_min, c_max, step)

    fn = master_clump.save_as_dataset(
        filename=output_dir,
        fields=["density", "particle_mass"])

    leaf_clumps = master_clump.leaves
    pdir = os.path.join(output_dir, 'projections')
    ensure_dir(pdir)
    inner_radius = ds.quan(100, 'AU')

    units = 'pc'
    for i, sphere in enumerate(
            iterate_center_of_mass(data_source, inner_radius,
                                   com_kwargs={'use_gas': True,
                                               'use_particles': True})):
        width = 2 * sphere.radius
        region = ds.box(sphere.center-1.05*width/2,
                        sphere.center+1.05*width/2)

        if width.to('pc') < 0.01:
            units = 'AU'

        for ax in 'xyz':
            p = yt.ProjectionPlot(
                ds, ax, ('gas', 'number_density'), weight_field=('gas', 'density'),
                center=sphere.center, width=width, data_source=region)
            p.set_axes_unit(units)
            p.annotate_clumps(leaf_clumps)
            p.save(os.path.join(pdir, "%03d" % i))
