import numpy as np

from yt.data_objects.level_sets.api import \
    add_clump_info
from yt.utilities.physical_constants import \
    G, kboltz, mh

def _min_number_density(clump):
    min_n = clump.data["gas", "number_density"].min().in_units("cm**-3")
    return "Min number density: %.6e cm^-3.", min_n

add_clump_info("min_number_density", _min_number_density)

def _max_number_density(clump):
    max_n = clump.data["gas", "number_density"].max().in_units("cm**-3")
    return "Max number density: %.6e cm^-3.", max_n

add_clump_info("max_number_density", _max_number_density)

def _jeans_mass(clump):
    temperature = clump.data.quantities.weighted_average_quantity(
        ("gas", "temperature"), ("gas", "cell_mass"))
    density = clump.data.quantities.weighted_average_quantity(
        ("gas", "density"), ("index", "cell_volume"))
    mu = clump.data.quantities.weighted_average_quantity(
        ("gas", "mean_molecular_weight"), ("gas", "cell_mass"))

    MJ_constant = (((5.0 * kboltz) / (G * mh))**(1.5)) * \
        (3.0 / (4.0 * np.pi))**(0.5)
    u = MJ_constant * \
        ((temperature / mu)**(1.5)) * \
        (density**(-0.5))

    return "Jeans mass: %.6e Msun.", u.in_units("Msun")

add_clump_info("jeans_mass", _jeans_mass)
