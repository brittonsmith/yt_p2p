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
        truncate=True,
        include_cooling=True):
    """
    True if clump has negative total energy. This considers gas kinetic
    energy, thermal energy, and radiative losses over a free-fall time
    against gravitational potential energy of the gas and collisionless
    particle system.
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

    mylog.info("Kinetic energy: %e erg." %
               kinetic.in_units("erg"))

    if use_thermal_energy:
        cooling_loss = clump.data.ds.quan(0.0, "erg")
        thermal = (clump["gas", "cell_mass"] *
                   clump["gas", "thermal_energy"]).sum()
        mylog.info("Thermal energy: %e erg." %
                   thermal.in_units("erg"))

        if include_cooling:
            # divide by sqrt(2) since t_ff = t_dyn / sqrt(2)
            cooling_loss = \
                (clump["gas", "cell_mass"] *
                 clump["gas", "dynamical_time"] *
                 clump["gas", "thermal_energy"] /
                 clump["gas", "cooling_time"]).sum() / np.sqrt(2)
            mylog.info("Cooling loss: %e erg." %
                       cooling_loss.in_units("erg"))

        thermal -= np.abs(cooling_loss)
        kinetic += thermal
        kinetic = max(kinetic, clump.data.ds.quan(0.0, "erg"))

    mylog.info("Available energy: %e erg." %
               kinetic.in_units("erg"))

    m = np.concatenate([clump["gas", "cell_mass"].in_cgs(),
                        clump["all", "particle_mass"].in_cgs()])
    px = np.concatenate([clump["index", "x"].in_cgs(),
                         clump["all", "particle_position_x"].in_cgs()])
    py = np.concatenate([clump["index", "y"].in_cgs(),
                         clump["all", "particle_position_y"].in_cgs()])
    pz = np.concatenate([clump["index", "z"].in_cgs(),
                         clump["all", "particle_position_z"].in_cgs()])

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
