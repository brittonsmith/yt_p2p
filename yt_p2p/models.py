"""
Functions for creating collapse models based on radial profiles.
"""

import numpy as np
import os
import yt

from pygrackle import FluidContainer
from pygrackle.one_zone import MinihaloModel1D
from pygrackle.utilities.physical_constants import mass_hydrogen_cgs
from pygrackle.yt_fields import \
     prepare_grackle_data, \
    _get_needed_fields, \
    _field_map

from yt.frontends.enzo.data_structures import EnzoDataset
from yt.utilities.physical_constants import me, mp

yt.mylog.setLevel(40)

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


def initialize_model_set(star_id, grackle_pars,
                         data_dir="star_cubes"):
    mass_cube_fn = os.path.join(data_dir, f"star_{star_id}_mass.h5")
    radius_cube_fn = os.path.join(data_dir, f"star_{star_id}_radius.h5")

    mds = yt.load(mass_cube_fn)
    rds = yt.load(radius_cube_fn)

    mass_field = "gas_mass_enclosed"
    peak_field = "bonnor_ebert_ratio"
    time_index = -1

    i_max = mds.data['data', peak_field][time_index].argmax()
    mass_enclosed = mds.data['data', mass_field][time_index]
    m_peak = mass_enclosed[i_max]
    m_min = m_peak / 10
    m_max = 2 * m_peak
    model_indices = np.where((m_min <= mass_enclosed) & (mass_enclosed <= m_max))[0]
    print (f"Mass peak: {mass_enclosed[i_max]}")
    print (f"Mass range: {mass_enclosed[model_indices[0]]} to "
           f"{mass_enclosed[model_indices[-1]]} "
           f"({model_indices.size} pts)")

    prepare_grackle_data(mds, sim_type=EnzoDataset, parameters=grackle_pars)

    # First time where density values available for all points in model
    density = mds.data['data', 'density']
    density_time_index = np.where((density[:, model_indices[0]] > 0) &
                                  (density[:, model_indices[-1]] > 0))[0][0]

    # First time where hydrostatic pressure within 10% of gas pressure
    # This is extremely important to getting good model behavior!
    prat = mds.data["data", "pressure"][:, model_indices] / \
        mds.data["data", "hydrostatic_pressure"][:, model_indices]
    pressure_time_index = np.where(prat.max(axis=1) < 1.1)[0][0]

    first_time_index = max(density_time_index, pressure_time_index)

    data_time = mds.data['data', 'time'].to('Myr')
    start_time = data_time[first_time_index]
    print (f"Model starting at {start_time} ({first_time_index}).")

    model_parameters = {}
    model_parameters["include_pressure"] = True
    model_parameters["safety_factor"] = 0.01
    model_parameters["include_turbulence"] = True
    model_parameters["event_trigger_fields"] = "all"

    mds.cosmology.omega_baryon = 0.0449
    model_parameters["cosmology"] = mds.cosmology
    model_parameters["unit_registry"] = mds.unit_registry

    model_data = mds, rds, start_time, model_indices

    return model_data, model_parameters


def create_model(model_data, model_parameters, metallicity=None):

    mds, rds, start_time, model_indices = model_data

    my_fc, external_data, full_data = prepare_model(
        mds, rds, start_time, model_indices)

    if metallicity is not None:
        external_data['metallicity'][:] = mds.quan(metallicity, 'Zsun').to('').d

    run_parameters = model_parameters.copy()
    data_time = mds.data['data', 'time'].to('Myr')
    creation_time = mds.parameters["creation_time"]
    relative_creation_time = creation_time - start_time
    run_parameters["star_creation_time"] = relative_creation_time.to("s").d / \
      my_fc.chemistry_data.time_units

    run_parameters["final_time"] = (data_time[-1] - start_time).to("s").d / \
      my_fc.chemistry_data.time_units
    run_parameters["gas_mass"] = \
      full_data['data', 'gas_mass_enclosed'].to('code_mass').d[0]
    run_parameters["initial_radius"] = \
      full_data['data', 'radius'].to('code_length').d[0]
    run_parameters["max_density"] = 1e7 * mass_hydrogen_cgs / \
      my_fc.chemistry_data.density_units

    run_parameters["external_data"] = external_data

    model = MinihaloModel1D(
        my_fc,
        **run_parameters,
    )
    model.verbose = 2
    return model
