from more_itertools import always_iterable
import numpy as np

from yt.data_objects.profiles import \
    create_profile
from yt.units.yt_array import \
    YTArray

def my_profile(dobj, bin_fields, profile_fields, n_bins=None, extrema=None, logs=None, units=None,
               weight_field="cell_mass", accumulation=False, fractional=False, bin_density=None):
    r"""
    Create 1, 2, or 3D profiles.

    Store profile data in a dictionary associated with the halo object.

    Parameters
    ----------
    dobj : data container
        The object to use for profiling.
    bin_fields : list of strings
        The binning fields for the profile.
    profile_fields : string or list of strings
        The fields to be profiled.
    n_bins : int or list of ints
        The number of bins in each dimension.  If None, 32 bins for
        each bin are used for each bin field.
        Default: 32.
    extrema : dict of min, max tuples
        Minimum and maximum values of the bin_fields for the profiles.
        The keys correspond to the field names. Defaults to the extrema
        of the bin_fields of the dataset. If a units dict is provided, extrema
        are understood to be in the units specified in the dictionary.
    logs : dict of boolean values
        Whether or not to log the bin_fields for the profiles.
        The keys correspond to the field names. Defaults to the take_log
        attribute of the field.
    units : dict of strings
        The units of the fields in the profiles, including the bin_fields.
    weight_field : string
        Weight field for profiling.
        Default : "cell_mass"
    accumulation : bool or list of bools
        If True, the profile values for a bin n are the cumulative sum of
        all the values from bin 0 to n.  If -True, the sum is reversed so
        that the value for bin n is the cumulative sum from bin N (total bins)
        to n.  If the profile is 2D or 3D, a list of values can be given to
        control the summation in each dimension independently.
        Default: False.
    fractional : If True the profile values are divided by the sum of all
        the profile data such that the profile represents a probability
        distribution function.

    """

    ds = dobj.ds

    if isinstance(bin_fields[0], str):
        bin_fields = [bin_fields]

    if bin_density is None:
        if n_bins is None:
            n_bins = dict((bin_field, 32) for bin_field in bin_fields)
    else:
        if n_bins is None:
            n_bins = {}
        elif not isinstance(n_bins, dict):
            raise RuntimeError("Can only specify n_bins or bin_density, but not both.")

        if extrema is None:
            extrema = {}
        if len(extrema) != len(bin_fields):
            exs = dobj.quantities.extrema(bin_fields)
            if isinstance(exs, YTArray):
                exs = [exs]

        for bin_field, ex in zip(bin_fields, exs):
            if bin_field in n_bins:
                continue
            if units is not None and bin_field in units:
                ex.convert_to_units(units[bin_field])
            if logs is None:
                my_log = True
            else:
                my_log = logs.get(bin_field, True)

            if extrema.get(bin_field, None) is None:
                if my_log:
                    if ex[0] <= 0:
                        fd = dobj[bin_field]
                        if units is not None and bin_field in units:
                            fd.convert_to_units(units[bin_field])
                        ex[0] = fd[fd > 0].min()
                        del fd
                    mi = 10**np.floor(np.log10(ex[0]))
                    ma = 10**np.ceil(np.log10(ex[1]))
                else:
                    mi = np.floor(ex[0])
                    ma = np.ceil(ex[1])
                extrema[bin_field] = (mi, ma)

            my_ex = extrema[bin_field]
            if my_log:
                my_n_bins = int(np.log10(my_ex[1] / my_ex[0]) * bin_density)
            else:
                my_n_bins = int((my_ex[1] - my_ex[0]) * bin_density)
            n_bins[bin_field] = my_n_bins

    if isinstance(n_bins, dict):
        n_bins = tuple([n_bins[bin_field] for bin_field in bin_fields])
    
    bin_fields = list(always_iterable(bin_fields))
    prof = create_profile(
        dobj, bin_fields, profile_fields, n_bins=n_bins,
        extrema=extrema, logs=logs, units=units, weight_field=weight_field,
        accumulation=accumulation, fractional=fractional)

    return prof

### Profile rebinning functions
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
        new_l_data = slope * (l_new_bins[do] - l_old_bins[ibins[do]]) + l_data[ibins[do]]
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

def track_global_binning(pdata, bin_data, nonzero=True):
    if not bin_data:
        for key in ['max', 'min']:
            bin_data[key] = None
    if nonzero:
        my_pdata = pdata[pdata.nonzero()[0]]
    else:
        my_pdata = pdata

    bin_data['min'] = default_func(min, bin_data['min'], my_pdata)
    bin_data['max'] = default_func(max, bin_data['max'], my_pdata)

def calc_global_binning(data, field, log=True, rounding=True, nonzero=True):
    binning = {}
    for datum in data:
        if log:
            my_datum = np.log10(datum[field])
        else:
            my_datum = datum[field]
        track_global_binning(my_datum, binning, nonzero=nonzero)
    if rounding:
        binning['min'] = np.floor(binning['min'])
        binning['max'] = np.ceil(binning['max'])
    return binning['min'], binning['max']
