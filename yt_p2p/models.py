"""
Functions for creating collapse models based on radial profiles.
"""

from collections import defaultdict
import numpy as np
import os

from pygrackle import FluidContainer

from pygrackle.yt_fields import \
    _get_needed_fields, \
    _field_map
from yt.utilities.logger import ytLogger

from yt.funcs import get_pbar
from yt.loaders import load as yt_load
from yt.utilities.physical_constants import me, mp

from yt_p2p.stars import get_star_data

from ytree.utilities.logger import log_level

def get_datasets(fns):
    pbar = get_pbar('Loading datasets', len(fns))
    dss = []
    for i, fn in enumerate(fns):
        if fn is None:
            ds = None
        else:
            ds = yt_load(fn, minimal_fields=True)
        dss.append(ds)
        pbar.update(i+1)
    pbar.finish()
    return dss

_model_es = None
_model_stars = None
def load_model_profiles(star_id,
                        data_dir="star_minihalos",
                        stars_fn="star_hosts.yaml",
                        sim_fn="simulation.h5"):

    global _model_es
    if _model_es is None:
        _model_es = yt_load(sim_fn)
    es = _model_es
    times = es.data["time"].to("Myr")
    fns = es.data["filename"].astype(str)

    global _model_stars
    if _model_stars is None:
        _model_stars = get_star_data(stars_fn)
    star_data = _model_stars

    ct = star_data[star_id]["creation_time"]

    dsfns = defaultdict(list)
    weights = ["None", "cell_volume", "cell_mass"]

    for t, fn in zip(times, fns):
        if t <= ct:
            pdir = "icom_gas_position"
            to_get = ["None", "cell_volume", "cell_mass"]
        else:
            pdir = "icom_all_position"
            to_get = ["None", "cell_volume"]

        exists = 0
        these_fns = {}
        for weight in weights:
            if weight in to_get:
                basename = f"{os.path.basename(fn)}_profile_weight_field_{weight}.h5"
                my_fn = os.path.join(
                    data_dir, f"star_{star_id}", "profiles", pdir, basename)
                if os.path.exists(my_fn):
                    exists += 1
            else:
                my_fn = None

            these_fns[weight] = my_fn

        if len(to_get) == exists:
            for weight in weights:
                dsfns[weight].append(these_fns[weight])

    # dict of lists
    with log_level(40, mylog=ytLogger):
        my_dss = {weight: get_datasets(dsfns[weight]) for weight in weights}
    # list of dicts
    rval = [{weight: my_dss[weight][i] for weight in weights}
            for i in range(len(my_dss[weights[0]]))]

    return rval

# Profile rebinning functions

def rebin_profile(profile, bin_field, new_bins):
    """
    Rebin profile data using linear interpolation.
    """
    ibins = np.digitize(new_bins, profile[bin_field]) - 1
    do = (ibins >= 0) & (ibins < profile[bin_field].size - 1)

    l_new_bins = np.log10(new_bins)
    l_old_bins = np.log10(profile[bin_field])
    new_profile = {bin_field: profile[bin_field].units * new_bins}
    for field, data in profile.items():
        if field in [bin_field, ('data', 'used')]:
            continue

        new_data = data.units * np.zeros(new_bins.size)
        new_profile[field] = new_data
        if data.nonzero()[0].size == 0:
            continue

        l_data = np.log10(data.clip(1e-100, np.inf))
        slope = (l_data[ibins[do]+1] - l_data[ibins[do]]) / \
            (l_old_bins[ibins[do]+1] - l_old_bins[ibins[do]])
        new_l_data = slope * (l_new_bins[do] - l_old_bins[ibins[do]]) + \
          l_data[ibins[do]]
        new_data[do] = np.power(10, new_l_data)

    return new_profile

def rebin_profiles(profile_data, bin_field, bin_density):
    """
    Rebin a list of profiles with a different field.
    """
    bmin, bmax = calc_global_binning(profile_data, bin_field)
    n_bins = int(np.round((bmax - bmin) * bin_density)) + 1
    p_bins = np.logspace(bmin, bmax, n_bins, endpoint=True)

    new_profiles = []
    for profile in profile_data:
        new_profiles.append(rebin_profile(profile, bin_field, p_bins))

    return new_profiles

def default_func(func, tracked_val, new_vals):
    if tracked_val is None:
        return func(new_vals)
    else:
        return func(tracked_val, func(new_vals))

def track_global_binning(pdata, bin_data):
    if not bin_data:
        for key in ['max', 'min']:
            bin_data[key] = None

    bin_data['min'] = default_func(min, bin_data['min'], pdata)
    bin_data['max'] = default_func(max, bin_data['max'], pdata)

def calc_global_binning(data, field, log=True, rounding=True, nonzero=True):
    binning = {}
    for datum in data:

        my_datum = datum[field]
        if nonzero:
            my_datum = my_datum[my_datum.nonzero()[0]]
        if log:
            my_datum = np.log10(my_datum)

        track_global_binning(my_datum, binning)
    if rounding:
        binning['min'] = np.floor(binning['min'])
        binning['max'] = np.ceil(binning['max'])
    return binning['min'], binning['max']

# Peak finding functions

def udwshed(x, within=5):
    """
    Upside-down watershed. Use a watershed algorithm to find local maxima.
    """
    asx = np.flip(x.argsort())
    basins = np.full(x.size, -1, dtype=np.int8)
    nb = 0
    for i in asx:
        for ib in range(nb):
            diff = np.abs(i - np.where(basins == ib)[0]).min()
            if diff > within:
                continue
            basins[i] = ib
            break
        if basins[i] < 0:
            basins[i] = nb
            nb += 1
    return basins

def basin_peak_ids(x):
    basin = udwshed(x)
    peaks = []
    for i in range(basin.max()+1):
        ib = np.where(basin == i)[0]
        peaks.append(ib[x[ib].argmax()])
    return np.array(peaks)

def find_peaks(ds, bin_field, peak_field, time_index):
    peak_data = ds.data[peak_field][time_index]
    bin_data = ds.data[bin_field][time_index]

    peak_used = peak_data > 0
    peak_data = peak_data[peak_used]
    bin_data = bin_data[peak_used]

    i_peaks = basin_peak_ids(peak_data)
    i_peaks = i_peaks[bin_data[i_peaks] < 1e4]

    # if peak is above 1, include the innermost coordinate that is above 1
    if peak_data.max() > 1:
        i_peaks = np.append(i_peaks, np.where(peak_data > 1)[0].min())

    if i_peaks.size == 0:
        i_peaks = np.append(i_peaks, np.argmax(peak_data))

    # the actual indices in the profile_all arrays
    i_peaks = np.where(peak_used)[0][i_peaks]
    i_peaks = np.unique(i_peaks)
    i_peaks.sort()
    return i_peaks

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
