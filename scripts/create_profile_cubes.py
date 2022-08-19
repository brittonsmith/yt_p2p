"""
Create profile cubes for all halos.
"""

import glob
import os
from yt.extensions.p2p.model_profiles import create_profile_cube

if __name__ == "__main__":
    star_ids = [int(my_dir[my_dir.rfind("_")+1:])
                for my_dir in glob.glob("star_minihalos/star_*")]
    for star_id in star_ids:
        if star_id in skip:
            continue
        create_profile_cube(star_id, data_dir="star_minihalos",
                            output_dir="star_cubes")
    # create_profile_cube(334267082, data_dir="star_minihalos",
    #                     output_dir="star_cubes")

    # create_profile_cube(None, data_dir="target_halos/tree_43523709",
    #                     output_dir="target_cubes/tree_43523709")
