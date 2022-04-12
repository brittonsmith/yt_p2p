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

def prepare_model(cds, start_time, profile_index, fc=None):
    time_data = cds.data["time"]
    time_index = np.abs(time_data - start_time).argmin()

    if ('data', 'specific_thermal_energy') not in cds.field_list:
        cds.add_field(('data', 'specific_thermal_energy'),
                      sampling_type="local",
                      function=_specific_thermal_energy,
                      units="erg/g")

    efields = ["external_pressure",
               "dark_matter",
               "metallicity",
               "turbulent_velocity",
               "H2_p0_dissociation_rate",
               "H_p0_ionization_rate",
               "He_p0_ionization_rate",
               "He_p1_ionization_rate",
               "photo_gamma"]
    external_data = {}

    field_list = [field for field in cds.field_list if field[1] != "time"]
    if ('data', 'specific_thermal_energy') not in field_list:
        field_list.append(('data', 'specific_thermal_energy'))
    field_data = dict((field, cds.data[field][time_index:, profile_index])
                      for field in field_list)

    if fc is None:
        fc = FluidContainer(cds.grackle_data, 1)

    fields = _get_needed_fields(fc.chemistry_data)
    for gfield in fields:
        yfield, units = _field_map[gfield]
        pfield = ("data", yfield[1])

        fc[gfield][:] = field_data[pfield][0].to(units)
        # get external data to be used in fluid container
        if pfield[1] in efields:
            efields.pop(efields.index(pfield[1]))
            external_data[gfield] = field_data[pfield].to(units).d

    # get extra solely external data fields
    for efield in efields:
        yfield, units = _field_map[efield]
        external_data[efield] = field_data[yfield].to(units).d

    if 'de' in fc:
        fc['de'] *= (mp/me).d

    field_data["time"] = time_data[time_index:] - time_data[time_index]
    external_data["time"] = field_data["time"].to("s").d / \
      fc.chemistry_data.time_units

    return fc, external_data, field_data
