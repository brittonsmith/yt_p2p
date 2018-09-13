"""
I use this to create a dataset containing just the Pop III
star particles and their associated fields.
"""

import sys
import yt

from yt.data_objects.particle_filters import \
     particle_filter

@particle_filter('pop3', ['particle_mass', 'particle_type', 'creation_time'])
def p3(pfilter, data):
        return ((data['particle_type'] == 5) & (data['particle_mass'].in_units('Msun') < 1e-10)) \
             | ((data['particle_type'] == 1) & (data['creation_time'] > 0) & \
                (data['particle_mass'].in_units('Msun') > 1)) \
             | ((data['particle_type'] == 5) & (data['particle_mass'].in_units('Msun') > 1e-3))

@particle_filter("star", requires=["creation_time"],
                filtered_type="io")
def _enzo_2_star(pfilter, data):
    return data["creation_time"] > 0

if __name__ == "__main__":
    ds = yt.load(sys.argv[1])
    ds.add_particle_filter("pop3")
    region = ds.box(ds.parameters["RefineRegionLeftEdge"],
                    ds.parameters["RefineRegionRightEdge"])

    fields=["particle_mass", "particle_index", "particle_type",
            "particle_position_x", "particle_position_y", "particle_position_z",
            "particle_velocity_x", "particle_velocity_y", "particle_velocity_z",
            "creation_time", "metallicity_fraction"]
    region.save_as_dataset(fields=[("pop3", field) for field in fields])
