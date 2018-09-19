"""
I used this to locate the halo in which the metal-enriched collapse
occured by filtering out halos with a relatively low peak gas density.
For halos that made it through the filter, I make projections so I
can inspect manually.
"""

import h5py
import numpy as np
import os
import sys
import yt
yt.enable_parallelism()

from yt.extensions.astro_analysis.halo_analysis.api import \
    HaloCatalog, \
    add_quantity

from yt.extensions.p2p import \
    add_p2p_fields
import yt.extensions.p2p.halo_catalog_callbacks

def _max_gas_density(halo):
    return halo.data_object["gas", "density"].max()
add_quantity("max_gas_density", _max_gas_density)

if __name__ == "__main__":
    dds = yt.load(sys.argv[1])
    add_p2p_fields(dds)

    if len(sys.argv) > 2:
        hds = yt.load(sys.argv[2])
    else:
        hds = yt.load("rockstar_halos_2Myr/halos_%s.0.bin" %str(dds))

    ad = hds.all_data()
    cr = ad.cut_region(["obj['particle_mass'].to('Msun') > 1e4"])

    hc = HaloCatalog(halos_ds=hds, data_ds=dds,
                     data_source=cr,
                     output_dir="halo_catalogs/location_catalogs/%s" % dds.basename)
    hc.add_callback("sphere", factor=1.)
    hc.add_quantity("max_gas_density")
    hc.add_filter("quantity_value", "max_gas_density", ">", 1e-16, "g/cm**3")
    hc.add_callback("sphere_projection", ["density", "temperature", "metallicity3"],
                    weight_field="density", axes="xyz", output_dir="sphere_projections")
    hc.add_callback("sphere_projection", ["dark_matter_density"],
                    weight_field="dark_matter_density", axes="xyz", output_dir="sphere_projections")
    hc.create(dynamic=False)
