import numpy as np
from scipy.optimize import brentq
from yt.fields.field_detector import \
    FieldDetector
from pygrackle import \
    add_grackle_fields, \
    FluidContainer, \
    chemistry_data
from pygrackle.yt_fields import \
    _data_to_fc, \
    _get_needed_fields

from yt.config import ytcfg
from yt.funcs import \
    get_pbar, \
    DummyProgressBar

def _calculate_cooling_metallicity_fast(field, data, fc):
    gfields = _get_needed_fields(fc.chemistry_data)
    if field.name[1].endswith('tdt'):
        tdfield = 'total_dynamical_time'
    else:
        tdfield = 'dynamical_time'
    td = data['gas', tdfield].to('code_time').d
    flatten = len(td.shape) > 1
    if flatten:
        td = td.flatten()

    fc.chemistry_data.metal_cooling_only = 0
    fc.chemistry_data.metal_cooling = 0
    fc.calculate_cooling_time()
    ct_0 = fc['cooling_time'] + td
    calc = ct_0 > 0
    lct_0 = np.log(ct_0[calc])
    z = data.ds.arr(np.zeros(ct_0.size), '')

    # if not isinstance(data, FieldDetector):
    #     breakpoint()

    fc.chemistry_data.metal_cooling = 1
    fc.chemistry_data.metal_cooling_only = 1

    z1 = 1e-4
    lz1 = np.log(z1)
    fc['metal'][:] = z1 * fc['density']
    fc.calculate_cooling_time()
    lct1 = np.log(-fc['cooling_time'])

    z2 = 2 * z1
    lz2 = np.log(z2)
    fc['metal'][:] = z2 * fc['density']
    fc.calculate_cooling_time()
    lct2 = np.log(-fc['cooling_time'])

    slope = ((lct2 - lct1) / (z2 - z1))[calc]
    z[calc] = np.exp((lct_0 - lct1[calc]) / slope + lz1)
    return z

def _cooling_metallicity_fast(field, data):
    fc = _data_to_fc(data)
    return _calculate_cooling_metallicity_fast(field, data, fc)

def _cooling_metallicity_diss_fast(field, data):
    fc = _data_to_fc(data)
    if fc.chemistry_data.primordial_chemistry > 1:
        fc['HI'] += fc['H2I'] + fc['H2II']
        fc['H2I'][:] = 0
        fc['H2II'][:] = 0
    if fc.chemistry_data.primordial_chemistry > 2:
        fc['HI'] += fc['HDI'] / 3
        fc['DI'] += 2 * fc['HDI'] / 3
        fc['HDI'][:] = 0
    return _calculate_cooling_metallicity_fast(field, data, fc)

def _calculate_cooling_metallicity(field, data, fc):
    gfields = _get_needed_fields(fc.chemistry_data)
    if field.name[1].endswith('tdt'):
        tdfield = 'total_dynamical_time'
    else:
        tdfield = 'dynamical_time'
    td = data['gas', tdfield].to('code_time').d
    flatten = len(td.shape) > 1
    if flatten:
        td = td.flatten()
    fc_mini = FluidContainer(data.ds.grackle_data, 1)

    fc.calculate_cooling_time()

    def cdrat(Z, my_td):
        fc_mini['metal'][:] = Z * fc_mini['density']
        fc_mini.calculate_cooling_time()
        return my_td + fc_mini['cooling_time'][0]

    field_data = data.ds.arr(np.zeros(td.size), '')
    if isinstance(data, FieldDetector):
        return field_data

    if field_data.size > 200000:
        my_str = "Reticulating splines"
        if ytcfg.getboolean("yt","__parallel"):
            my_str = "P%03d %s" % \
                (ytcfg.getint("yt", "__global_parallel_rank"),
                 my_str)
        pbar = get_pbar(my_str, field_data.size, parallel=True)
    else:
        pbar = DummyProgressBar()
    for i in range(field_data.size):
        pbar.update(i)
        if td[i] + fc['cooling_time'][i] > 0:
            continue
        for mfield in gfields:
            fc_mini[mfield][:] = fc[mfield][i]
        success = False
        if i > 0 and field_data[i-1] > 0:
            try:
                field_data[i] = brentq(
                    cdrat, 0.1*field_data[i-1], 10*field_data[i-1],
                    args=(td[i]), xtol=1e-6)
                success = True
            except:
                pass
        if not success:
            bds = np.logspace(-2, 2, 5)
            for bd in bds:
                try:
                    field_data[i] = brentq(cdrat, 1e-6, bd, args=(td[i]), xtol=1e-6)
                    success = True
                    break
                except:
                    continue
            if not success:
                field_data[i] = np.nan
                # field_data[i] = 0. # hack for imaging
    pbar.finish()

    if flatten:
        field_data = field_data.reshape(data.ActiveDimensions)
    return field_data

def _cooling_metallicity(field, data):
    fc = _data_to_fc(data)
    return _calculate_cooling_metallicity(field, data, fc)

def _cooling_metallicity_diss(field, data):
    fc = _data_to_fc(data)
    if fc.chemistry_data.primordial_chemistry > 1:
        fc['HI'] += fc['H2I'] + fc['H2II']
        fc['H2I'][:] = 0
        fc['H2II'][:] = 0
    if fc.chemistry_data.primordial_chemistry > 2:
        fc['HI'] += fc['HDI'] / 3
        fc['DI'] += 2 * fc['HDI'] / 3
        fc['HDI'][:] = 0
    return _calculate_cooling_metallicity(field, data, fc)

def add_p2p_grackle_fields(ds, parameters=None):
    add_grackle_fields(ds, parameters=parameters)

    for suf in ['', '_tdt']:
        ds.add_field("cooling_metallicity%s" % suf,
                     function=_cooling_metallicity,
                     units="Zsun", sampling_type="cell")
        ds.add_field("cooling_metallicity_diss%s" % suf,
                     function=_cooling_metallicity_diss,
                     units="Zsun", sampling_type="cell")
        ds.add_field("cooling_metallicity_fast%s" % suf,
                     function=_cooling_metallicity_fast,
                     units="Zsun", sampling_type="cell")
        ds.add_field("cooling_metallicity_diss_fast%s" % suf,
                     function=_cooling_metallicity_diss_fast,
                     units="Zsun", sampling_type="cell")
