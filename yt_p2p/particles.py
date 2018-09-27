"""
particle stuff



"""

#-----------------------------------------------------------------------------
# Copyright (c) Britton Smith <brittonsmith@gmail.com>.  All rights reserved.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

from yt.data_objects.particle_filters import \
    add_particle_filter
from yt.utilities.logger import ytLogger as mylog

def _pop3(pfilter, data):
    return ((data['particle_type'] == 5) & (data['particle_mass'].in_units('Msun') < 1e-10)) \
        | ((data['particle_type'] == 1) & (data['creation_time'] > 0) & \
           (data['particle_mass'].in_units('Msun') > 1)) \
        | ((data['particle_type'] == 5) & (data['particle_mass'].in_units('Msun') > 1e-3))

add_particle_filter(
    "pop_3", function=_pop3, filtered_type="all",
    requires=["particle_type", "creation_time", "particle_mass"])

def add_p2p_particle_filters(ds):
    pfilters = ["pop_3"]:
    for pfilter in pfilters:
        if not ds.add_particle_filter(pfilter):
            mylog.warn("Failed to add filter: %s." % pfilter)
