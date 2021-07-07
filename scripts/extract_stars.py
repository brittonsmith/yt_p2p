"""
I use this to create a dataset containing just the Pop III
star particles and their associated fields.
"""

import os
import sys
import yt
yt.enable_parallelism()

from yt.extensions.p2p import \
    add_p2p_particle_filters

if __name__ == "__main__":
    es = yt.load(sys.argv[1])
    fns = es.data['filename'].astype(str)[::-1]
    data_dir = es.directory

    for fn in yt.parallel_objects(fns, njobs=-1, dynamic=True):
        ds = yt.load(os.path.join(data_dir, fn))

        output_file = os.path.join("pop3", f"{ds.basename}.h5")
        if os.path.exists(output_file):
            continue

        add_p2p_particle_filters(ds)
        region = ds.box(ds.parameters["RefineRegionLeftEdge"],
                        ds.parameters["RefineRegionRightEdge"])

        fields=["particle_mass", "particle_index", "particle_type",
                "particle_position_x", "particle_position_y", "particle_position_z",
                "particle_velocity_x", "particle_velocity_y", "particle_velocity_z",
                "creation_time", "metallicity_fraction"]
        data = dict((field, region[('pop3', field)]) for field in fields)
        ftypes = dict((field, 'pop3') for field in fields)

        yt.save_as_dataset(
            ds, filename=output_file,
            data=data, field_types=ftypes)
