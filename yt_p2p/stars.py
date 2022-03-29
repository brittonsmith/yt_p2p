"""
Functions relating to star particles formed in the simulations.



"""

#-----------------------------------------------------------------------------
# Copyright (c) Britton Smith <brittonsmith@gmail.com>.  All rights reserved.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

import numpy as np
import yaml

def get_star_data(filename, remove_doubles=True):
    with open(filename, "r") as f:
        star_data = yaml.load(f, Loader=yaml.FullLoader)

    star_cts = np.array([float(star["creation_time"].split()[0])
                         for star in star_data.values()])
    star_ids = np.array(list(star_data.keys()))

    if remove_doubles:
        asct = star_cts.argsort()
        star_cts = star_cts[asct]
        star_ids = star_ids[asct]
        double_ids = star_ids[np.where(np.diff(star_cts) < 1e-3)[0]+1]
        for double_id in double_ids:
            del star_data[double_id]

    return star_data
