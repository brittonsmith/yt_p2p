from yt.fields.field_detector import \
    FieldDetector
from yt.utilities.physical_constants import \
    me, mp
from pygrackle import \
    FluidContainer, \
    chemistry_data

_parameter_map = {
    "use_grackle": "use_grackle",
    "Gamma": "Gamma",
    "primordial_chemistry": "MultiSpecies",
    "metal_cooling": "MetalCooling",
    "h2_on_dust": "H2FormationOnDust",
    "cmb_temperature_floor": "CMBTemperatureFloor",
    "three_body_rate": "ThreeBodyRate",
    "cie_cooling": "CIECooling",
    "h2_optical_depth_approximation": "H2OpticalDepthApproximation",
    "photoelectric_heating": "PhotoelectricHeating",
    "photoelectric_heating_rate": "PhotoelectricHeatingRate",
    "NumberOfTemperatureBins": "NumberOfTemperatureBins",
    "CaseBRecombination": "CaseBRecombination",
    "TemperatureStart": "TemperatureStart",
    "TemperatureEnd": "TemperatureEnd",
    "NumberOfDustTemperatureBins": "NumberOfDustTemperatureBins",
    "DustTemperatureStart": "DustTemperatureStart",
    "DustTemperatureEnd": "DustTemperatureEnd",
    "HydrogenFractionByMass": "HydrogenFractionByMass",
    "DeuteriumToHydrogenRatio": "DeuteriumToHydrogenRatio",
    "SolarMetalFractionByMass": "SolarMetalFractionByMass",
    "UVbackground_redshift_on": "RadiationRedshiftOn",
    "UVbackground_redshift_off": "RadiationRedshiftOff",
    "UVbackground_redshift_fullon": "RadiationRedshiftFullOn",
    "UVbackground_redshift_drop": "RadiationRedshiftDropOff",
    "use_radiative_transfer": "RadiativeTransfer",
    "radiative_transfer_coupled_rate_solver": "RadiativeTransferCoupledRateSolver",
    "radiative_transfer_hydrogen_only": "RadiativeTransferReadParameters",
    "with_radiative_cooling": "with_radiative_cooling",
    "use_volumetric_heating_rate": "use_volumetric_heating_rate",
    "use_specific_heating_rate": "use_specific_heating_rate",
    "self_shielding_method": "self_shielding_method",
    "H2_self_shielding": "H2_self_shielding",
    "grackle_data_file": "grackle_data_file",
    "UVbackground": "UVbackground",
    "Compton_xray_heating": "Compton_xray_heating",
    "LWbackground_intensity": "LWbackground_intensity",
    "LWbackground_sawtooth_suppression": "LWbackground_sawtooth_suppression"
}

_field_map = {
    'density': (('gas', 'density'), 'density_units'),
    'HI': (('gas', 'H_p0_density'), 'density_units'),
    'HII': (('gas', 'H_p1_density'), 'density_units'),
    'HM': (('gas', 'H_m1_density'), 'density_units'),
    'HeI': (('gas', 'He_p0_density'), 'density_units'),
    'HeII': (('gas', 'He_p1_density'), 'density_units'),
    'HeIII': (('gas', 'He_p2_density'), 'density_units'),
    'H2I': (('gas', 'H2_p0_density'), 'density_units'),
    'H2II': (('gas', 'H2_p1_density'), 'density_units'),
    'DI': (('gas', 'D_p0_density'), 'density_units'),
    'DII': (('gas', 'D_p1_density'), 'density_units'),
    'HDI': (('gas', 'HD_p0_density'), 'density_units'),
    'de': (('gas', 'El_density'), 'density_units'),
    'metal': (('gas', 'metal_density'), 'density_units'),
    'dust': (('gas', 'dust_density'), 'density_units'),
    'x-velocity': (('gas', 'velocity_x'), 'velocity_units'),
    'y-velocity': (('gas', 'velocity_y'), 'velocity_units'),
    'z-velocity': (('gas', 'velocity_z'), 'velocity_units'),
    'energy': (('gas', 'thermal_energy'), 'energy_units')
}

def _data_to_fc(data, size=None):
    if size is None:
        size = data['gas', 'density'].size
    fc = FluidContainer(data.ds.grackle_data, size)

    flatten = len(data['gas', 'density'].shape) > 1

    for gfield, (yfield, units) in _field_map.items():
        if yfield not in data.ds.field_info:
            continue
        conv = getattr(fc.chemistry_data, units)
        fdata = data[yfield] / conv
        if flatten:
            fdata = fdata.flatten()
        fc[gfield][:] = fdata

    if 'de' in fc:
        fc['de'] *= (mp/me)

    return fc

def prepare_grackle_data(ds):
    my_chemistry = chemistry_data()
    for gpar, dpar in _parameter_map.items():
        val = ds.parameters.get(dpar)
        if val is None:
            continue
        if isinstance(val, str):
            sval = bytearray(val, 'utf-8')
            setattr(my_chemistry, gpar, sval)
        else:
            setattr(my_chemistry, gpar, val)

    my_chemistry.comoving_coordinates = ds.cosmological_simulation
    my_chemistry.density_units = (ds.mass_unit / ds.length_unit**3).in_cgs().d
    my_chemistry.length_units = ds.length_unit.in_cgs().d
    my_chemistry.time_units = ds.time_unit.in_cgs().d
    my_chemistry.a_units = 1 / (1 + ds.parameters.get('CosmologyInitialRedshift', 0))
    my_chemistry.a_value = 1 / (1 + ds.current_redshift) / my_chemistry.a_units
    my_chemistry.velocity_units = ds.velocity_unit.in_cgs().d
    my_chemistry.initialize()
    ds.grackle_data = my_chemistry

_grackle_fields = {
    'cooling_time': 'code_time',
    'dust_temperature': 'K',
    'gamma': '',
    'mean_molecular_weight': '',
    'pressure': 'code_mass * code_velocity**2 / code_length**3',
    'temperature': 'K',
    }

def _grackle_field(field, data):
    gfield = field.name[1][len("grackle_"):]
    units = _grackle_fields[gfield]

    if not hasattr(data.ds, "grackle_data"):
        raise RuntimeError("Grackle has not been initialized.")

    fc = _data_to_fc(data)
    func = "calculate_%s" % gfield
    getattr(fc, func)()

    fdata = fc[gfield]
    if hasattr(data, 'ActiveDimensions'):
        fdata = fdata.reshape(data.ActiveDimensions)

    return fdata * data.ds.quan(1, units).in_cgs()

def add_grackle_fields(ds):
    prepare_grackle_data(ds)
    for field, units in _grackle_fields.items():
        fname = "grackle_%s" % field
        funits = str(ds.quan(1, units).in_cgs().units)
        ds.add_field(fname, function=_grackle_field,
                     sampling_type="cell", units=funits)
