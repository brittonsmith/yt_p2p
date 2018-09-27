import numpy as np
import os

from yt.data_objects.level_sets.api import \
    add_validator
from yt.funcs import mylog
from yt.utilities.lib.misc_utilities import \
    gravitational_binding_energy
from yt.utilities.physical_constants import \
    gravitational_constant_cgs as G

def _future_bound(
        clump,
        use_thermal_energy=True,
        use_particles=False,
        truncate=True,
        include_cooling=True,
        include_contraction=True,
        allow_negative_thermal=False):
    """
    True if clump is gravitationally bound, optionally including thermal pressure,
    radiative losses over a free-fall time, and adiabatic contraction.
    """

    num_threads = int(os.environ.get('OMP_NUM_THREADS', 1))

    if clump["gas", "cell_mass"].size <= 1:
        mylog.info("Clump has only one cell.")
        return False

    bulk_velocity = clump.quantities.bulk_velocity(
        use_particles=False)

    kinetic = 0.5 * (clump["gas", "cell_mass"] *
        ((bulk_velocity[0] - clump["gas", "velocity_x"])**2 +
         (bulk_velocity[1] - clump["gas", "velocity_y"])**2 +
         (bulk_velocity[2] - clump["gas", "velocity_z"])**2)).sum()

    if use_particles:
        kinetic += 0.5 * (clump["all", "particle_mass"] *
            ((bulk_velocity[0] - clump["all", "particle_velocity_x"])**2 +
             (bulk_velocity[1] - clump["all", "particle_velocity_y"])**2 +
             (bulk_velocity[2] - clump["all", "particle_velocity_z"])**2)).sum()

    mylog.info("Kinetic energy: %e erg." %
               kinetic.in_units("erg"))

    if use_thermal_energy:
        cooling_loss = clump.data.ds.quan(0.0, "erg")
        contraction = clump.data.ds.quan(0.0, "erg")
        thermal = (clump["gas", "cell_mass"] *
                   clump["gas", "thermal_energy"]).sum()
        mylog.info("Thermal energy: %e erg." %
                   thermal.in_units("erg"))

        if include_contraction:
            contraction = (clump.data.ds.gamma - 1.) * \
              (clump["gas", "cell_mass"] *
               clump["gas", "thermal_energy"]).sum()
            mylog.info("Adiabatic contraction: %e erg." %
                       contraction.in_units("erg"))

        if include_cooling:
            # divide by sqrt(2) since t_ff = t_dyn / sqrt(2)
            cooling_loss = \
                (clump["gas", "cell_mass"] *
                 clump["gas", "dynamical_time"] *
                 clump["gas", "thermal_energy"] /
                 clump["gas", "cooling_time"]).sum() / np.sqrt(2)
            mylog.info("Cooling loss: %e erg." %
                       cooling_loss.in_units("erg"))

        thermal += contraction - np.abs(cooling_loss)

        if not allow_negative_thermal:
            # do not allow cooling losses to make thermal energy negative
            thermal = max(thermal, clump.data.ds.quan(0.0, "erg"))
        kinetic += thermal
        kinetic = max(kinetic, clump.data.ds.quan(0.0, "erg"))

    mylog.info("Available energy: %e erg." %
               kinetic.in_units("erg"))

    if use_particles:
        m = np.concatenate([clump["gas", "cell_mass"].in_cgs(),
                            clump["all", "particle_mass"].in_cgs()])
        px = np.concatenate([clump["index", "x"].in_cgs(),
                             clump["all", "particle_position_x"].in_cgs()])
        py = np.concatenate([clump["index", "y"].in_cgs(),
                             clump["all", "particle_position_y"].in_cgs()])
        pz = np.concatenate([clump["index", "z"].in_cgs(),
                             clump["all", "particle_position_z"].in_cgs()])
    else:
        m = clump["gas", "cell_mass"].in_cgs()
        px = clump["index", "x"].in_cgs()
        py = clump["index", "y"].in_cgs()
        pz = clump["index", "z"].in_cgs()

    potential = clump.data.ds.quan(
        G * gravitational_binding_energy(
            m, px, py, pz,
            truncate, (kinetic / G).in_cgs(),
            num_threads=num_threads),
        kinetic.in_cgs().units)

    mylog.info("Potential energy: %e erg." %
               potential.to('erg'))

    return potential >= kinetic

add_validator("future_bound", _future_bound)
