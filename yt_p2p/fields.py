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

import numpy as np

def _metallicity3(field, data):
    return data["enzo", "SN_Colour"] / data["gas", "density"]

def _metal3_mass(field, data):
    return data["enzo", "SN_Colour"] * data["index", "cell_volume"]

def _metallicity3_min7(field, data):
    field_data = data["enzo", "SN_Colour"] / data["gas", "density"]
    field_data.convert_to_units("")
    min_Z = data.ds.quan(1.e-7, "Zsun").in_units("")
    field_data[field_data < min_Z] = 0.5 * min_Z
    return field_data

def _tangential_velocity_magnitude(field, data):
    return np.sqrt(data["gas", "velocity_spherical_theta"]**2 +
                   data["gas", "velocity_spherical_phi"]**2)

def _HD_H2(field, data):
    return data["gas", "HD_density"] / data["gas", "H2_density"]

def _dark_matter_mass(field, data):
    return data["gas", "dark_matter_density"] * data["index", "cell_volume"]

def _vortical_time(field, data):
    return 1. / data["gas", "vorticity_magnitude"]

def _vortical_dynamical_ratio(field, data):
    return data["gas", "vortical_time"] / data["gas", "dynamical_time"]

def _vortical_cooling_ratio(field, data):
    return data["gas", "vortical_time"] / data["gas", "cooling_time"]

def _cooling_dynamical_ratio(field, data):
    return data["gas", "cooling_time"] / data["gas", "dynamical_time"]

def add_p2p_fields(ds):
    # use the value of solar metallicity in the dataset
    ds.unit_registry.modify('Zsun', ds.parameters['SolarMetalFractionByMass'])
    ds.add_field("metallicity3",
                 function=_metallicity3,
                 units="Zsun", sampling_type="cell")
    ds.add_field(("gas", "metal3_mass"),
                 function=_metal3_mass,
                 units="g", sampling_type="cell")
    ds.add_field(("gas", "metallicity3_min7"),
                 function=_metallicity3_min7,
                 units="Zsun", sampling_type="cell")
    ds.add_field(("gas", "vortical_time"),
                 function=_vortical_time,
                 units="s", sampling_type="cell")
    ds.add_field(("gas", "dark_matter_mass"),
                 function=_dark_matter_mass,
                 units="g", sampling_type="cell")
    ds.add_field(("gas", "vortical_dynamical_ratio"),
                 function=_vortical_dynamical_ratio,
                 units="", sampling_type="cell")
    ds.add_field(("gas", "vortical_cooling_ratio"),
                 function=_vortical_cooling_ratio,
                 units="", sampling_type="cell")
    ds.add_field(("gas", "cooling_dynamical_ratio"),
                 function=_cooling_dynamical_ratio,
                 units="", sampling_type="cell")
    ds.add_field(("gas", "HD_H2_ratio"),
                 function=_HD_H2,
                 units="", sampling_type="cell")
    ds.add_field(("gas", "tangential_velocity_magnitude"),
                 function=_tangential_velocity_magnitude,
                 units="cm/s", sampling_type="cell")
