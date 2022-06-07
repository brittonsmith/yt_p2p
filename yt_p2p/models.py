"""
Functions for creating collapse models based on radial profiles.
"""

import numpy as np

from pygrackle import FluidContainer

from pygrackle.yt_fields import \
    _get_needed_fields, \
    _field_map

from yt.utilities.physical_constants import me, mp

def _specific_thermal_energy(field, data):
    ftype = field.name[0]
    return data[ftype, "thermal_energy"]

# Grackle fluid container preparation

def prepare_model(mds, rds, start_time, profile_indices, fc=None):
    profile_indices = np.asarray(profile_indices)
    time_data = mds.data["time"]
    time_index = np.abs(time_data - start_time).argmin()

    if ('data', 'specific_thermal_energy') not in mds.field_list:
        mds.add_field(('data', 'specific_thermal_energy'),
                      sampling_type="local",
                      function=_specific_thermal_energy,
                      units="erg/g")

    efields = ["hydrostatic_pressure",
               "dark_matter",
               "gas_density",
               "gas_mass_enclosed",
               "dark_matter_mass_enclosed",
               "metallicity",
               "turbulent_velocity",
               "H2_p0_dissociation_rate",
               "H_p0_ionization_rate",
               "He_p0_ionization_rate",
               "He_p1_ionization_rate",
               "photo_gamma",
               "used_bins"]
    external_data = {}

    field_list = [field for field in mds.field_list if field[1] != "time"]
    if ('data', 'specific_thermal_energy') not in field_list:
        field_list.append(('data', 'specific_thermal_energy'))
    mass_data = {field: mds.data[field][time_index:, profile_indices]
                 for field in field_list}

    if fc is None:
        fc = FluidContainer(mds.grackle_data, profile_indices.size)

    fields = _get_needed_fields(fc.chemistry_data)
    for gfield in fields:
        yfield, units = _field_map[gfield]
        pfield = ("data", yfield[1])

        # set fluid container initial conditions
        fc[gfield][:] = mass_data[pfield][0].to(units)

        # get external data to be used in fluid container
        if pfield[1] in efields:
            efields.pop(efields.index(pfield[1]))
            external_data[gfield] = rds.data[pfield][time_index:].to(units).d

    # get extra solely external data fields
    for efield in efields:
        yfield, units = _field_map[efield]
        external_data[efield] = rds.data[yfield][time_index:].to(units).d

    external_data["used_bins"] = external_data["used_bins"].astype(bool)
    external_data["radial_bins"] = rds.data["data", "radius"].to("code_length").d
    rb = external_data["radial_bins"]
    external_data["radius"] = rb[:-1] * np.sqrt(rb[1:] / rb[:-1])

    if 'de' in fc:
        fc['de'] *= (mp/me).d

    mass_data["time"] = time_data[time_index:] - time_data[time_index]
    external_data["time"] = mass_data["time"].to("s").d / \
      fc.chemistry_data.time_units

    return fc, external_data, mass_data
