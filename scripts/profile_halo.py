"""
I used this to do all of the radial, density, etc. profiles of
the target halo.

The iterative_center_of_mass callback can be uncommented to
create a series of images as we zoom in on the center of mass
from the halo scale down to the collapsed object.

For each simulation, the target halo was:

cc_512_collapse_solar_dust    DD0182  87497
cc_512_collapse_solar_no_dust DD0185  66768
cc_512_no_dust_continue       DD0560  41732
cc_512_no_dust_continue_2     DD0532 161449
cc_512_continue_collapse_dust DD0564  50010
"""
import numpy as np
import os
import sys
import yt

from yt.extensions.astro_analysis.halo_analysis.api import \
    add_callback, \
    add_recipe, \
    HaloCatalog
from yt.extensions.astro_analysis.halo_analysis.halo_callbacks import \
    periodic_distance

from yt.extensions.p2p import \
    add_p2p_fields
from yt.extensions.p2p.halo_catalog_callbacks import \
    get_my_sphere

def iterative_center_of_mass(halo, inner_radius,
                             radius_field="virial_radius", step_ratio=0.9):
    if step_ratio <= 0.0 or step_ratio >= 1.0:
        raise RuntimeError(
            "iterative_center_of_mass: step_ratio must be between 0 and 1.")

    dds = halo.halo_catalog.data_ds
    center_orig = halo_data_center(halo)
    radius_orig = halo_data_radius(halo, radius_field="virial_radius")
    sphere = get_my_sphere(halo, radius_field="virial_radius")

    my_units = "pc"
    yt.mylog.info("Halo %d: radius: %s, center: %s." %
                  (halo.quantities["particle_identifier"],
                   sphere.radius.in_units(my_units), sphere.center))
    i = 0
    try:
        while sphere.radius > inner_radius:
            new_center = sphere.quantities.center_of_mass(
                use_gas=True, use_particles=False)
            sphere = sphere.ds.sphere(
                new_center, step_ratio * sphere.radius)
            region = dds.box(sphere.center-1.05*sphere.radius,
                             sphere.center+1.05*sphere.radius)
            for ax in "xyz":
                p = yt.ProjectionPlot(
                    dds, ax, ["density", "metallicity3", "temperature"],
                    weight_field="density", center=sphere.center,
                    width=(2*sphere.radius), data_source=region)
                if sphere.radius < sphere.ds.quan(0.1, "pc"):
                    my_units = "AU"
                p.set_axes_unit(my_units)
                p.set_cmap("density", "algae")
                p.set_cmap("temperature", "gist_heat")
                p.set_cmap("metallicity3", "kamae")
                p.save("sphere_center_box/%s_%03d" % (str(dds), i))
            i+=1
            yt.mylog.info("Radius: %s, center: %s." %
                          (sphere.radius.in_units(my_units), sphere.center))
    except:
        yt.mylog.info("Reached minimum radius.")
        pass

    distance = periodic_distance(
        center_orig.in_units("unitary").v,
        new_center.in_units("unitary").v)
    distance = dds.quan(distance, "unitary")
    yt.mylog.info("Recentering halo %d %f pc away." %
                  (halo.quantities["particle_identifier"],
                   distance.in_units("pc")))

    set_halo_center(halo, new_center)
    del sphere
    yt.mylog.info("Original center: %s." % center_orig.to("unitary"))
    yt.mylog.info("     New center: %s." % new_center.to("unitary"))
add_callback("iterative_center_of_mass", iterative_center_of_mass)

def my_2d_radial_profile(hc, output_dir, my_field, field_units,
                         log=True):
    hc.add_callback(
        "profile", [("index", "radius"), ("gas", my_field)],
        ["cell_mass"], weight_field=None, bin_density=20,
        logs={("gas", my_field): log, ("index", "radius"): True},
        units={("gas", my_field): field_units, ("index", "radius"): "pc"})
    hc.add_callback("save_object_as_dataset", "profiles_object",
                    output_dir=output_dir, filename=my_field)
    hc.add_callback("delete_attribute", "profiles")
    hc.add_callback("delete_attribute", "profiles_variance")
    hc.add_callback("delete_attribute", "profiles_object")
add_recipe("my_2d_radial_profile", my_2d_radial_profile)

def my_2d_density_profile(hc, output_dir, my_field,
                          log=True):
    hc.add_callback(
        "profile", [("gas", "number_density"),
                    ("gas", my_field)], ["cell_mass"],
        weight_field=None, bin_density=20)
    hc.add_callback("save_object_as_dataset", "profiles_object",
                    output_dir=output_dir, filename=my_field)
    hc.add_callback("delete_attribute", "profiles")
    hc.add_callback("delete_attribute", "profiles_variance")
    hc.add_callback("delete_attribute", "profiles_object")
add_recipe("my_2d_density_profile", my_2d_density_profile)

def run_halo_catalog(ddsfn, hdsfn, halo_id):
    dds = yt.load(ddsfn)
    add_p2p_fields(dds)

    hds = yt.load(hdsfn)

    hc = HaloCatalog(
        halos_ds=hds, data_ds=dds,
        output_dir="halo_catalogs/profile_catalogs/%s" % dds.basename)

    hc.add_filter("quantity_value", "particle_identifier", "==", halo_id, "")

    # my_radius = dds.quan(100.0, "AU")
    my_radius = dds.quan(1.0, "pc")
    # hc.add_callback("iterative_center_of_mass", my_radius)
    hc.add_callback("field_max_center", "density")

    # calculate virial mass
    hc.add_recipe("calculate_virial_quantities",
                  [('index', 'radius'),
                   ('gas', 'cell_mass'),
                   ('gas', 'dark_matter_mass'),
                   ('gas', 'matter_mass')])

    hc.add_callback("sphere")

    # 1D mass-weighted radial profiles
    hc.add_callback(
        "profile", ("index", "radius"),
        [('index', 'radius'),
         ('gas', 'cell_mass'),
         ('gas', 'dark_matter_mass'),
         ('gas', 'matter_mass')],
        units={("index", "radius"): "pc"},
        accumulation=True,
        weight_field=None, bin_density=20)
    hc.add_callback("save_object_as_dataset", "profiles_object",
                    output_dir="profiles", filename="virial_profiles")
    hc.add_callback("delete_attribute", "profiles")
    hc.add_callback("delete_attribute", "profiles_variance")
    hc.add_callback("delete_attribute", "profiles_object")

    # hc.add_callback("set_inner_bulk_velocity", my_radius)
    hc.add_callback("set_field_max_bulk_velocity", "density")
    hc.add_callback("set_inner_angular_momentum_vector", my_radius)

    # 1D mass-weighted radial profiles
    hc.add_callback(
        "profile", ("index", "radius"),
        [("gas", "metal3_mass"),
         ("gas", "cell_mass"),
         ("gas", "dark_matter_mass"),
         ("gas", "velocity_magnitude"),
         ("gas", "velocity_spherical_radius"),
         ("gas", "velocity_spherical_theta"),
         ("gas", "velocity_spherical_phi"),
         ("gas", "sound_speed"),
         ("gas", "specific_angular_momentum_magnitude"),
         ("gas", "specific_angular_momentum_x"),
         ("gas", "specific_angular_momentum_y"),
         ("gas", "specific_angular_momentum_z"),
         ("gas", "vortical_time"),
         ("gas", "dynamical_time"),
         ("gas", "cooling_time"),
         ("gas", "density"),
         ("gas", "temperature")],
        units={("index", "radius"): "pc"},
        weight_field="cell_mass", bin_density=20)
    hc.add_callback("save_object_as_dataset", "profiles_object",
                    output_dir="profiles", filename="profiles")
    hc.add_callback("delete_attribute", "profiles")
    hc.add_callback("delete_attribute", "profiles_variance")
    hc.add_callback("delete_attribute", "profiles_object")

    # 2D density profiles
    nprofs = ["H2_fraction",
              "HD_fraction",
              "HD_H2_ratio",
              "temperature"]
    for my_field in nprofs:
        hc.add_recipe("my_2d_density_profile", "density_profiles",
                      my_field)

    # 2D radial profiles
    rprofs = [("density", 'g/cm**3'),
              ("metallicity3_min7", 'Zsun')]
    for my_field, field_units in rprofs:
        hc.add_recipe("my_2d_radial_profile", "radial_profiles",
                      my_field, field_units)

    # 2D radial/timescale profiles
    tprofs = [("vortical_dynamical_ratio", ""),
              ("vortical_cooling_ratio",   ""),
              ("cooling_dynamical_ratio",  ""),
              ("vortical_time", "yr"),
              ("dynamical_time","yr"),
              ("cooling_time",  "yr")]
    for my_field, field_units in tprofs:
        hc.add_recipe("my_2d_radial_profile", "timescale_profiles",
                      my_field, field_units)

    # 2D radial/velocity profiles
    my_field = "cell_mass"
    hc.add_callback("profile", [("index", "radius")], [("gas", my_field)],
                    weight_field=None, bin_density=20,
                    logs={("index", "radius"): True},
                    units={("index", "radius"): "pc"})
    hc.add_callback("save_object_as_dataset", "profiles_object",
                    output_dir="velocity_profiles", filename=my_field)
    hc.add_callback("delete_attribute", "profiles")
    hc.add_callback("delete_attribute", "profiles_variance")
    hc.add_callback("delete_attribute", "profiles_object")
    vprofs = ["velocity_magnitude",
              "velocity_spherical_radius",
              "velocity_spherical_theta",
              "velocity_spherical_phi",
              "tangential_velocity_magnitude",
              "sound_speed"]
    for my_field in vprofs:
        hc.add_recipe("my_2d_radial_profile", "velocity_profiles",
                      my_field, 'km/s', log=False)

    hc.create()

if __name__ == "__main__":
    ddsfn = sys.argv[1]
    hdsfn = sys.argv[2]

    datasets = {
        'cc_512_collapse_solar_dust':    {'DD0182':  87497},
        'cc_512_collapse_solar_no_dust': {'DD0185':  66768},
        'cc_512_no_dust_continue':       {'DD0560':  41732},
        'cc_512_no_dust_continue_2':     {'DD0532':  161449},
        'cc_512_continue_collapse_dust': {'DD0564':   50010}}

    if len(sys.argv) > 3:
        halo_id = int(sys.argv[3])
    else:
        curdir = os.path.basename(os.path.abspath(os.curdir))
        ds_key = os.path.basename(ddsfn)

        if curdir not in datasets:
            raise RuntimeError("Can't find %s in dataset collection." % curdir)
        my_sim = datasets[curdir]
        if ds_key not in my_sim:
            raise RuntimeError("Can't find %s in %s simulation." % (ds_key, curdir))
        halo_id = my_sim[ds_key]
        yt.mylog.info("Running profiles for halo %06d." % halo_id)

    run_halo_catalog(ddsfn, hdsfn, halo_id)
