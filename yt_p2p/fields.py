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

from yt.utilities.exceptions import \
    YTFieldNotFound
from yt.utilities.physical_constants import G

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

def _total_metal_density(field, data):
    field_data = np.zeros_like(data["gas", "density"])
    fields = [("enzo", "Metal_Density"),
              ("enzo", "SN_Colour")]
    for field in fields:
        if field in data.ds.field_list:
            field_data += data[field]
    return field_data

def _total_metallicity(field, data):
    return data["gas", "total_metal_density"] / \
        data["gas", "density"]

def _total_dynamical_time(field, data):
    """
    sqrt(3 pi / (16 G rho))
    """
    return np.sqrt(3.0 * np.pi / (16.0 * data["gas", "matter_density"] * G))

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

def add_p2p_field(ds, name, function=None, units='', sampling_type='cell'):
    try:
        ds.add_field(name, function=function,
                     units=units, sampling_type=sampling_type)
    except YTFieldNotFound:
        pass

def add_p2p_fields(ds):
    # use the value of solar metallicity in the dataset
    ds.unit_registry.modify('Zsun', ds.parameters['SolarMetalFractionByMass'])

    if ("gas", "metallicity3") in ds.field_info:
        return

    add_p2p_field(ds, ("gas", "metallicity3"),
                  function=_metallicity3,
                  units="Zsun", sampling_type="cell")
    add_p2p_field(ds, ("gas", "metal3_mass"),
                  function=_metal3_mass,
                  units="g", sampling_type="cell")
    add_p2p_field(ds, ("gas", "metallicity3_min7"),
                  function=_metallicity3_min7,
                  units="Zsun", sampling_type="cell")
    add_p2p_field(ds, ("gas", "total_metal_density"),
                  function=_total_metal_density,
                  units="g/cm**3", sampling_type="cell")
    add_p2p_field(ds, ("gas", "total_metallicity"),
                  function=_total_metallicity,
                  units="Zsun", sampling_type="cell")
    add_p2p_field(ds, ("gas", "total_dynamical_time"),
                  function=_total_dynamical_time,
                  units="s", sampling_type="cell")
    add_p2p_field(ds, ("gas", "vortical_time"),
                  function=_vortical_time,
                  units="s", sampling_type="cell")
    add_p2p_field(ds, ("gas", "dark_matter_mass"),
                  function=_dark_matter_mass,
                  units="g", sampling_type="cell")
    add_p2p_field(ds, ("gas", "vortical_dynamical_ratio"),
                  function=_vortical_dynamical_ratio,
                  units="", sampling_type="cell")
    add_p2p_field(ds, ("gas", "vortical_cooling_ratio"),
                  function=_vortical_cooling_ratio,
                  units="", sampling_type="cell")
    add_p2p_field(ds, ("gas", "cooling_dynamical_ratio"),
                  function=_cooling_dynamical_ratio,
                  units="", sampling_type="cell")
    add_p2p_field(ds, ("gas", "HD_H2_ratio"),
                  function=_HD_H2,
                  units="", sampling_type="cell")
    add_p2p_field(ds, ("gas", "tangential_velocity_magnitude"),
                  function=_tangential_velocity_magnitude,
                  units="cm/s", sampling_type="cell")
