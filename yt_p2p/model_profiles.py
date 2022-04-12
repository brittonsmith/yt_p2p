"""
Functions for preparing profile data for the Minihalo model.
"""

import numpy as np
import os

from unyt import uvstack

import yt
from yt.funcs import ensure_dir
from yt.utilities.physical_constants import G

from yt.extensions.p2p.models import \
    find_peaks, \
    rebin_profiles, \
    get_datasets, \
    load_model_profiles
from yt_p2p.stars import get_star_data

def create_profile_cube(star_id, output_dir="star_cubes"):
    ### Bonnor-Ebert Mass constant
    a = 1.67
    b = (225 / (32 * np.sqrt(5 * np.pi))) * a**-1.5

    ensure_dir(output_dir)

    star_data = get_star_data("star_hosts.yaml")
    my_star = star_data[star_id]
    creation_time = my_star["creation_time"]

    profiles = load_model_profiles(star_id)
    profile_data = []
    time_data = []

    pbar = yt.get_pbar("Gathering profiles", len(profiles)-1)
    for i, profile_dict in enumerate(profiles):
        pbar.update(i)
        npds = profile_dict["None"]
        vpds = profile_dict["cell_volume"]
        mpds = profile_dict["cell_mass"]

        time_data.append(npds.current_time)

        x_bins = npds.profile.x_bins
        used = npds.data['data', 'used'].d.astype(bool)

        m_gas = npds.profile['data', 'cell_mass'].to('Msun')[used]
        gas_mass_enclosed = m_gas.cumsum()
        m_dm = npds.profile['data', 'dark_matter_mass'].to('Msun')[used]
        dark_matter_mass_enclosed = m_dm.cumsum()
        total_mass_enclosed = gas_mass_enclosed + dark_matter_mass_enclosed

        gas_density_volume = vpds.data['data', 'density'][used]
        dark_matter_density_volume = vpds.data['data', 'dark_matter_density'][used]
        volume_weight = vpds.data['data', 'weight'][used]

        r = npds.data['data', 'radius'][used].to('pc')
        dr = np.diff(x_bins)[used]

        profile_datum = {
            "gas_mass_enclosed": gas_mass_enclosed,
            "total_mass_enclosed": total_mass_enclosed,
            "dark_matter_mass_enclosed": dark_matter_mass_enclosed,
            "gas_density_volume_weighted": gas_density_volume,
            "dark_matter_density_volume_weighted": dark_matter_density_volume,
            "cell_volume_weight": volume_weight,
            "radius": r,
            "dr": dr
        }

        current_time = npds.current_time.to("Myr")
        if current_time < creation_time:
            v_turb = mpds.data['standard_deviation', 'velocity_magnitude'][used]
            profile_datum["turbulent_velocity"] = v_turb

            p = mpds.data['data', 'pressure'][used]
            cs = mpds.data['data', 'sound_speed'][used]
            cs_eff = np.sqrt(cs**2 + v_turb**2)
            m_BE = (b * (cs**4 / G**1.5) * p**-0.5)
            profile_datum["bonnor_ebert_ratio"] = (gas_mass_enclosed / m_BE).to("")

            exclude_fields = ['x', 'x_bins', 'radius', 'weight']
            pfields = [field for field in mpds.field_list
                       if field[0] == 'data' and
                       field[1] not in exclude_fields]
            profile_datum.update(
                {field: mpds.data[field][used] for field in pfields})
            profile_datum["cell_mass_weight"] = mpds.data["data", "weight"]

        profile_data.append(profile_datum)

    time_data = npds.arr(time_data)

    fn = os.path.join(output_dir, f"star_{star_id}_radius.h5")
    create_cube(npds, fn, profile_data, tdata=time_data, bin_field="radius")

    fn = os.path.join(output_dir, f"star_{star_id}_mass.h5")
    create_cube(npds, fn, profile_data, tdata=time_data, bin_field="gas_mass_enclosed")

def create_cube(ds, filename, pdata, tdata=None, bin_field="radius", bin_density=5):

    rdata = rebin_profiles(pdata, bin_field, 5)

    pcube = {}
    for field in rdata[0]:
        datum = []
        for pdatum in rdata:
            if field not in pdatum:
                continue
            datum.append(pdatum[field])
        pcube[field] = uvstack(datum)
    
    if tdata is not None:
        pcube["time"] = tdata

    yt.save_as_dataset(ds, filename=filename, data=pcube)
