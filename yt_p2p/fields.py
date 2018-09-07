"""
Pop2Prime fields.



"""

#-----------------------------------------------------------------------------
# Copyright (c) Britton Smith <brittonsmith@gmail.com>.  All rights reserved.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

def _metallicity3(field, data):
    return data["enzo", "SN_Colour"] / data["gas", "density"]

def add_p2p_fields(ds):
    ds.add_field("metallicity3", function=_metallicity3,
                 units="Zsun", sampling_type="cell")
