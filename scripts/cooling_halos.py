"""
Calculate the poorly thought-out "cooling metallicity", i.e., the metallicity
at which cooling time is less than dynamical time. See grackle_fields.py for
field definition code.
"""

import numpy as np
import os
import sys
import yt
yt.enable_parallelism()

from yt.extensions.astro_analysis.halo_analysis.api import \
    add_callback, \
    add_filter, \
    HaloCatalog
from yt.extensions.astro_analysis.halo_analysis.halo_callbacks import \
    periodic_distance

from yt.extensions.p2p import \
    add_p2p_fields
from yt.extensions.p2p.grackle_fields import \
    add_grackle_fields
from yt.extensions.p2p.halo_catalog_callbacks import \
    halo_data_center, \
    halo_data_radius

grackle_pars = {
    "use_grackle": 1,
    "primordial_chemistry": 3,
    "metal_cooling": 1,
    "with_radiative_cooling": 1,
    "grackle_data_file": "cloudy_metals_2008_3D.h5",
    "H2_self_shielding": 0,
    "use_radiative_transfer": 1
}

def cooling_metallicity_callback(halo):
    sp = halo.data_object
    gas_mass = sp['gas', 'cell_mass']
    total_gas_mass = gas_mass.sum()
    halo.quantities['gas_mass'] = total_gas_mass

    for suf in ['', '_diss']:
        field = 'cooling_metallicity%s' % suf
        Zcool = sp['gas', field]
        can_cool = ~np.isnan(Zcool)
        Zave = (Zcool[can_cool] * gas_mass[can_cool]).sum() / \
            gas_mass[can_cool].sum()
        halo.quantities[field] = Zave
        halo.quantities['uncoolable_mass%s' % suf] = \
            gas_mass[~can_cool].sum()

add_callback('cooling_metallicities', cooling_metallicity_callback)

def in_refine_region(halo):
    dds = halo.halo_catalog.data_ds
    left = dds.parameters['RefineRegionLeftEdge']
    right = dds.parameters['RefineRegionRightEdge']

    p = halo_data_center(halo).d
    r = halo_data_radius(halo).d

    inside = (((p-r) >= left) & ((p+r) <= right)).all()
    return inside
add_filter('in_refine_region', in_refine_region)

if __name__ == "__main__":
    es = yt.load('p2p_nd.h5')
    fns = es.data['filename'].astype(str)

    data_dir = '/disk12/brs/pop2-prime/firstpop2_L2-Seed3_large/cc_512_no_dust_continue'
    halos_dir = 'rockstar_halos_2Myr'

    factor = 1.0
    output_dir = 'halo_catalogs/cooling_metals_%.1f_rvir' % factor

    for fn in yt.parallel_objects(fns, njobs=2):
        fnkey = os.path.basename(fn)
        hcfn = os.path.join(output_dir, f"{fn}.0.h5")
        if os.path.exists(hcfn):
            continue

        hdsfn = os.path.join(halos_dir, f"halos_{fnkey}.0.bin")
        if not os.path.exists(hdsfn):
            continue

        dds = yt.load(os.path.join(data_dir, fn))
        add_p2p_fields(dds)
        add_grackle_fields(dds, parameters=grackle_pars)

        hds = yt.load(hdsfn)

        hc = HaloCatalog(data_ds=dds, halos_ds=hds,
                         output_dir=os.path.join(output_dir, str(dds)))
        hc.add_filter('in_refine_region')
        hc.add_filter('quantity_value', 'particle_mass', '>=', 1e3, 'Msun')
        hc.add_callback('sphere', factor=factor)
        hc.add_callback('cooling_metallicities')
        for quantity in ['gas_mass', 'cooling_metallicity', 'uncoolable_mass',
                         'cooling_metallicity_diss', 'uncoolable_mass_diss']:
            hc.quantities.append(quantity)
        hc.add_callback('delete_attribute', 'data_object')
        hc.create(dynamic=True)
