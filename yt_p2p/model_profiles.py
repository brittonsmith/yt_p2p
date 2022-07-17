"""
Functions for preparing profile data for the Minihalo model.
"""

from collections import defaultdict
import numpy as np
import os
from scipy.interpolate import interp1d
from unyt import uvstack

import yt
from yt.funcs import ensure_dir, get_pbar
from yt.loaders import load as yt_load
from yt.utilities.logger import ytLogger
from yt.utilities.physical_constants import G

from ytree.utilities.logger import log_level

from yt_p2p.stars import get_star_data

def get_profile_datasets(fns):
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
        my_dss = {weight: get_profile_datasets(dsfns[weight]) for weight in weights}
    # list of dicts
    rval = [{weight: my_dss[weight][i] for weight in weights}
            for i in range(len(my_dss[weights[0]]))]

    return rval

# Profile rebinning functions

_lin_fields = ("velocity_x", "velocity_y", "velocity_z")

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
        if isinstance(field, tuple):
            fname = field[1]
        else:
            fname = field

        log_field = fname not in _lin_fields

        if field == bin_field:
            continue

        new_data = data.units * np.zeros(new_bins.size)
        new_profile[field] = new_data
        if data.nonzero()[0].size == 0:
            continue

        if log_field:
            l_data = np.log10(data.clip(1e-100, np.inf))
        else:
            l_data = data.copy()

        slope = (l_data[ibins[do]+1] - l_data[ibins[do]]) / \
            (l_old_bins[ibins[do]+1] - l_old_bins[ibins[do]])
        new_l_data = slope * (l_new_bins[do] - l_old_bins[ibins[do]]) + \
          l_data[ibins[do]]

        if log_field:
            new_data[do] = np.power(10, new_l_data)
        else:
            new_data[do] = new_l_data.copy()

    return new_profile

def rebin_profiles(profile_data, bin_field, bin_density):
    """
    Rebin a list of profiles with a different field.
    """

    yt.mylog.info(f"Rebinning profiles by {bin_field}.")

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
    if 'bin_density' not in bin_data:
        bin_data['bin_density'] = (pdata.size - 1) / (pdata.max() - pdata.min())

def calc_global_binning(data, field, log=True, rounding=True, nonzero=True,
                        bin_density=False):
    binning = {}
    for datum in data:

        if field is None:
            my_datum = datum.x_bins
        else:
            my_datum = datum[field]

        if nonzero:
            my_datum = my_datum[my_datum.nonzero()[0]]
        if log:
            my_datum = np.log10(my_datum)

        track_global_binning(my_datum, binning)

    if rounding:
        binning['min'] = np.floor(binning['min'])
        binning['max'] = np.ceil(binning['max'])

    if bin_density:
        bin_density = int(np.round(binning['bin_density']))
        return binning['min'], binning['max'], bin_density

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

    # if peak is above 1, include the inner/outermost coordinate above 1
    if peak_data.max() > 1:
        i_peaks = np.append(i_peaks, np.where(peak_data > 1)[0].min())
        i_peaks = np.append(i_peaks, np.where(peak_data > 1)[0].max())

    if i_peaks.size == 0:
        i_peaks = np.append(i_peaks, np.argmax(peak_data))

    # the actual indices in the profile_all arrays
    i_peaks = np.where(peak_used)[0][i_peaks]
    i_peaks = np.unique(i_peaks)
    i_peaks.sort()
    return i_peaks

def create_profile_cube(star_id, output_dir="star_cubes",
                        data_dir="star_minihalos"):
    ### Bonnor-Ebert Mass constant
    a = 1.67
    b = (225 / (32 * np.sqrt(5 * np.pi))) * a**-1.5

    ensure_dir(output_dir)

    star_data = get_star_data("star_hosts.yaml")
    my_star = star_data[star_id]
    creation_time = my_star["creation_time"]

    profiles = load_model_profiles(star_id, data_dir=data_dir)
    profile_data = []
    time_data = []

    profile_cube = {}
    bmin, bmax, bden = calc_global_binning(
        [p["None"].profile for p in profiles], None,
        bin_density=True)
    dx = 1 / bden
    nbins = int((bmax - bmin) * bden)

    pbar = yt.get_pbar("Gathering profiles", len(profiles))
    for i, profile_dict in enumerate(profiles):
        pbar.update(i+1)
        npds = profile_dict["None"]
        vpds = profile_dict["cell_volume"]
        mpds = profile_dict["cell_mass"]

        time_data.append(npds.current_time)

        x_bins = npds.profile.x_bins
        used = npds.data['data', 'used'].d.astype(bool)

        m_gas = npds.profile['data', 'cell_mass'].to('Msun')
        gas_mass_enclosed = m_gas.cumsum()
        m_dm = npds.profile['data', 'dark_matter_mass'].to('Msun')
        dark_matter_mass_enclosed = m_dm.cumsum()
        total_mass_enclosed = gas_mass_enclosed + dark_matter_mass_enclosed

        gas_density_volume = vpds.data['data', 'density']
        dark_matter_density_volume = vpds.data['data', 'dark_matter_density']
        volume_weight = vpds.data['data', 'weight']

        profile_datum = {
            "gas_mass_enclosed": gas_mass_enclosed,
            "total_mass_enclosed": total_mass_enclosed,
            "dark_matter_mass_enclosed": dark_matter_mass_enclosed,
            "gas_density_volume_weighted": gas_density_volume,
            "dark_matter_density": dark_matter_density_volume,
            "cell_volume_weight": volume_weight,
            "used": npds.data['data', 'used'],
        }

        current_time = npds.current_time.to("Myr")
        if current_time < creation_time:
            v_turb = mpds.data['standard_deviation', 'velocity_magnitude']
            profile_datum["turbulent_velocity"] = v_turb

            p = mpds.data['data', 'pressure']
            cs = mpds.data['data', 'sound_speed']
            m_BE = (b * (cs**4 / G**1.5) * p**-0.5)
            profile_datum["bonnor_ebert_ratio"] = (gas_mass_enclosed / m_BE).to("")

            dr = np.diff(x_bins)
            r = mpds.data['data', 'radius']
            rho = mpds.data['data', 'density']
            dP_hyd = (G * total_mass_enclosed * rho * dr / r**2)[used]
            P_hyd1 = np.flip(np.flip(dP_hyd).cumsum()).to(p.units)
            P_hydro = np.zeros_like(p)
            P_hydro[used] = P_hyd1
            profile_datum["hydrostatic_pressure"] = P_hydro

            cs_eff = np.sqrt(cs**2 + v_turb**2)
            t_cs = (2 * r / cs_eff).to("Myr")
            profile_datum["sound_crossing_time"] = t_cs

            exclude_fields = ['x', 'x_bins', 'used', 'weight', 'dark_matter_density']
            pfields = [field for field in mpds.field_list
                       if field[0] == 'data' and
                       field[1] not in exclude_fields]
            profile_datum.update(
                {field: mpds.data[field] for field in pfields})
            profile_datum["cell_mass_weight"] = mpds.data["data", "weight"]

        cstart = int((np.log10(x_bins[0]) - bmin) / dx)
        cend = cstart + x_bins.size - 1
        for field, values in profile_datum.items():
            if field not in profile_cube:
                profile_cube[field] = np.zeros((len(profiles), nbins)) * values.units

            profile_cube[field][i, cstart:cend][used] = values[used]

        # Filter out unused profile values.
        # That's the way I made rebin_profile and I don't feel like changing it.
        for field in profile_datum:
            profile_datum[field] = profile_datum[field][used]
        profile_data.append(profile_datum)

    cube_time = npds.arr(time_data)
    extra_data = {"time": cube_time}
    extra_attrs = {"creation_time": creation_time}

    fn = os.path.join(output_dir, f"star_{star_id}_mass.h5")
    create_rebinned_cube(npds, fn, profile_data, extra_data=extra_data,
                         extra_attrs=extra_attrs)

    # include radius as just a 1d array since it is constant over time
    cube_radius = np.logspace(bmin, bmax, nbins+1) * npds.profile.x.units
    extra_data["radius"] = cube_radius
    del profile_cube["data", "radius"]
    profile_cube.update(extra_data)
    fn = os.path.join(output_dir, f"star_{star_id}_radius.h5")
    yt.save_as_dataset(npds, filename=fn, data=profile_cube,
                       extra_attrs=extra_attrs)

def time_interpolate(data, tdata, bin_field):
    """
    Fix holes in data by interpolating in the time dimension.
    """

    yt.mylog.info(f"Fixing data with interpolation.")

    ikwargs = {"kind": "linear", "fill_value": np.nan, "bounds_error": False}

    used = data["used"].d.astype(int)
    empty = np.where(used == 0)
    rows = np.unique(empty[1])

    for field in data:
        if field in [bin_field, "used"]:
            continue

        datum = data[field]

        for row in rows:
            rused = used[:, row]
            if not rused.any():
                continue

            my_good = (rused == 1)[:datum.shape[0]]
            if my_good.sum() < 2:
                continue

            igood = np.where(my_good)[0]
            ibad = np.where(~my_good)[0]

            tgood = tdata[igood].d
            tbad = tdata[ibad].d

            if isinstance(field, tuple):
                fname = field[1]
            else:
                fname = field
            log_field = fname not in _lin_fields

            if log_field:
                my_y = np.log10(datum[:, row].clip(1e-100, np.inf))
            else:
                my_y = datum[:, row].copy()

            ygood = my_y[igood]
            f1 = interp1d(tgood, ygood, **ikwargs)
            yfix = f1(tbad)

            fixed = ~np.isnan(yfix)
            yfixed = yfix[fixed]
            if log_field:
                yfixed = np.power(10, yfixed)

            ireplace = ibad[fixed]
            datum[ireplace, row] = yfixed
            data["used"][ireplace, row] = 2

def create_rebinned_cube(ds, filename, pdata,
                         extra_data=None, extra_attrs=None,
                         bin_field="gas_mass_enclosed", bin_density=10):

    rdata = rebin_profiles(pdata, bin_field, bin_density)

    pcube = {}
    for field in rdata[0]:
        datum = []
        for pdatum in rdata:
            if field not in pdatum:
                continue
            datum.append(pdatum[field])
        pcube[field] = uvstack(datum)

    time_interpolate(pcube, extra_data["time"], bin_field)
    
    if extra_data is not None:
        pcube.update(extra_data)

    yt.save_as_dataset(ds, filename=filename, data=pcube,
                       extra_attrs=extra_attrs)
