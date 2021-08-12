import copy
import numpy as np
import os
import time
import yaml
import yt
yt.enable_parallelism()
import ytree

from pygrackle.yt_fields import \
    prepare_grackle_data, \
    _get_needed_fields, \
    _field_map as _grackle_map

from yt.frontends.enzo.data_structures import EnzoDataset

from ytree.analysis import \
    AnalysisPipeline, \
    add_operation, \
    add_recipe

from yt.extensions.p2p.misc import \
    sphere_icom, \
    reunit
from yt.extensions.p2p.profiles import my_profile
from yt.extensions.p2p.tree_analysis_operations import yt_dataset
from yt.extensions.p2p import add_p2p_fields

def decorate_plot(node, p):
    p.set_axes_unit("pc")
    title = f"z = {node['redshift']:.2f}, M = {node['mass'].to('Msun'):.2g}"
    p.annotate_title(title)

def icom_sphere(node):
    ds = node.ds

    radius = reunit(ds, node["virial_radius"], "unitary")
    if "icom_gas_position_x" in node.arbor.field_list:
        center = reunit(ds, node["icom_gas_position"], "unitary")
    else:
        center = reunit(ds, node["position"], "unitary")
        sphere = ds.sphere(center, radius)
        center = sphere_icom(sphere, 4*sphere["gas", "dx"].min(),
                             com_kwargs=dict(use_particles=False, use_gas=True))

    sphere = ds.sphere(center, radius)
    node.sphere = sphere

add_operation("icom_sphere", icom_sphere)

def my_yt_dataset(node, data_dir, es):
    if "Snap_idx" in node.arbor.field_list:
        yt_dataset(node, data_dir)
    else:
        efns = es.data["filename"].astype(str)
        etimes = reunit(node.arbor, es.data["time"].to("Myr"), "Myr")
        ifn = np.abs(etimes - node["time"]).argmin()
        dsfn = os.path.join(data_dir, efns[ifn])
        node.ds = yt.load(dsfn)

    node.ds.grackle_fields = copy.deepcopy(es.grackle_fields)
    add_p2p_fields(node.ds)

add_operation("my_yt_dataset", my_yt_dataset)

def region_projections(node, output_dir="."):
    ds = node.ds
    sphere = node.sphere

    left  = sphere.center - 1.05 * sphere.radius
    right = sphere.center + 1.05 * sphere.radius
    region = ds.box(left, right)

    pfields = {"density": "algae",
               "temperature": "gist_heat",
               "metallicity3": "kamae",
               "H2_p0_fraction": "cool"}

    for ax in "xyz":
        do_fields = \
            [field for field in pfields
             if not os.path.exists(
                 os.path.join(output_dir, f"{str(ds)}_Projection_{ax}_{field}_density.png"))]

        if do_fields:
            p = yt.ProjectionPlot(
                ds, ax, do_fields, weight_field="density",
                center=sphere.center, width=2*sphere.radius,
                data_source=region)
            for field, cmap in pfields.items():
                p.set_cmap(field, cmap)
            decorate_plot(node, p)
            p.save(output_dir + "/")

    do_axes = \
        [ax for ax in "xyz"
         if not os.path.exists(
             os.path.join(output_dir, f"{str(ds)}_Particle_{ax}_particle_mass.png"))]
    for ax in do_axes:
        p = yt.ParticleProjectionPlot(
            ds, ax, ("all", "particle_mass"),
            center=sphere.center, width=2*sphere.radius,
            data_source=region)
        p.set_unit(("all", "particle_mass"), "Msun")
        p.set_cmap(("all", "particle_mass"), "turbo")
        decorate_plot(node, p)
        p.save(output_dir + "/")

add_operation("region_projections", region_projections)

def sphere_radial_profiles(node, fields, weight_field=None, output_dir="."):
    if weight_field is None:
        weight_name = "None"
    else:
        if not isinstance(weight_field, tuple) and len(weight_field) != 2:
            raise ValueError("weight_field must be a tuple of length 2.")
        weight_name = weight_field[1]

    fn = os.path.join(output_dir, f"{str(node.ds)}_profile_weight_field_{weight_name}.h5")
    if os.path.exists(fn):
        return

    data_source = node.sphere
    profile = my_profile(
        data_source,
        ("index", "radius"), fields,
        units={("index", "radius"): "pc"},
        weight_field=weight_field,
    accumulation=False, bin_density=20)
    profile.save_as_dataset(filename=fn)

add_operation("sphere_radial_profiles", sphere_radial_profiles)

def mass_weighted_profiles(node, output_dir="."):

    enzo_fields = [field for field in node.ds.field_list
                   if field[0] == "enzo"]
    fields = node.ds.grackle_fields + \
        [("gas", "density"),
         ('gas', 'dark_matter_density'),
         ('gas', 'matter_density'),
         ("gas", "temperature"),
         ('gas', 'entropy'),
         ("gas", "cooling_time"),
         ("gas", "dynamical_time"),
         ("gas", "total_dynamical_time"),
         ("gas", "sound_speed"),
         ("gas", "pressure"),
         ("gas", "H2_p0_fraction"),
         ("gas", "metallicity3")]

    sphere_radial_profiles(node, fields, weight_field=("gas", "cell_mass"),
                           output_dir=output_dir)

add_operation("mass_weighted_profiles", mass_weighted_profiles)

def volume_weighted_profiles(node, output_dir="."):
    fn = os.path.join(output_dir, f"{str(node.ds)}_profile_weight_field_volume.h5")
    if os.path.exists(fn):
        return

    data_source = node.sphere
    profile = my_profile(
        data_source,
        ("index", "radius"),
        [("gas", "density"),
         ("gas", "dark_matter_density"),
         ("gas", "matter_density")],
        units={("index", "radius"): "pc"},
        weight_field=("index", "cell_volume"),
    accumulation=False, bin_density=20)
    profile.save_as_dataset(filename=fn)

add_operation("volume_weighted_profiles", volume_weighted_profiles)

def unweighted_profiles(node, output_dir="."):
    fn = os.path.join(output_dir, f"{str(node.ds)}_profile_weight_field_None.h5")
    if os.path.exists(fn):
        return

    data_source = node.sphere
    profile = my_profile(
        data_source,
        ("index", "radius"),
        [("gas", "cell_mass"),
         ("gas", "dark_matter_mass"),
         ("gas", "matter_mass")],
        units={("index", "radius"): "pc"},
        weight_field=None,
    accumulation=False, bin_density=20)
    profile.save_as_dataset(filename=fn)

add_operation("unweighted_profiles", unweighted_profiles)

def write_done_file(node, filename, output_dir="."):
    fn = os.path.join(output_dir, filename)
    if yt.is_root():
        with open(fn, mode='w') as f:
            f.write(f"{time.time()}\n")

add_operation("write_done_file", write_done_file)

def file_not_there(node, filename, output_dir="."):
    fn = os.path.join(output_dir, filename)
    return not os.path.exists(fn)

add_operation("file_not_there", file_not_there)

def my_analysis(pipeline, data_dir="."):
    pipeline.add_operation("my_yt_dataset", data_dir, es)
    pipeline.add_operation("icom_sphere")
    pipeline.add_operation("region_projections", output_dir="projections")
    pipeline.add_operation("mass_weighted_profiles", output_dir=".")
    pipeline.add_operation(
        "sphere_radial_profiles",
        [("gas", "density"),
         ("gas", "dark_matter_density"),
         ("gas", "matter_density")],
        weight_field=("gas", "cell_volume"), output_dir=".")
    pipeline.add_operation(
        "sphere_radial_profiles",
        [("gas", "cell_mass"),
         ("gas", "dark_matter_mass"),
         ("gas", "matter_mass")],
        weight_field=None, output_dir=".")
    pipeline.add_operation("delattrs", ["sphere", "ds"])

add_recipe("my_analysis", my_analysis)

if __name__ == "__main__":
    output_data_dir = "star_minihalo_profiles"

    grackle_pars = {
        "use_grackle": 1,
        "primordial_chemistry": 3,
        "metal_cooling": 1,
        "with_radiative_cooling": 1,
        "grackle_data_file": "cloudy_metals_2008_3D.h5",
        "H2_self_shielding": 0,
        "use_radiative_transfer": 1
    }

    es = yt.load("simulation.h5")
    prepare_grackle_data(es, parameters=grackle_pars, sim_type=EnzoDataset, initialize=False)
    es.grackle_fields = [_grackle_map[field][0] for field in _get_needed_fields(es.grackle_data)]

    data_dir = "/disk12/brs/pop2-prime/firstpop2_L2-Seed3_large/cc_512_no_dust_continue"

    with open("star_hosts.yaml", "r") as f:
        star_data = yaml.load(f, Loader=yaml.FullLoader)

    star_cts = np.array([float(star["creation_time"].split()[0])
                         for star in star_data.values()])
    star_ids = np.array(list(star_data.keys()))
    asct = star_cts.argsort()
    star_cts = star_cts[asct]
    star_ids = star_ids[asct]
    double_ids = star_ids[np.where(np.diff(star_cts) < 1e-3)[0]+1]
    for double_id in double_ids:
        del star_data[double_id]

    for star_id, star_info in star_data.items():
        a = ytree.load(star_info["arbor"])
        if "icom_gas_position_x" in a.field_list:
            a.add_vector_field("icom_gas_position")

        output_dir = os.path.join(output_data_dir, f"star_{star_id}")
        done_file = os.path.join(output_dir, "done")
        if os.path.exists(done_file):
            continue

        ap = AnalysisPipeline(output_dir=output_dir)
        ap.add_recipe("my_analysis", data_dir=data_dir)

        my_tree = a[star_info["_arbor_index"]]
        # find the halo where the star formed
        form_node = my_tree.get_node("forest", star_info["tree_id"])
        t_halo = form_node["time"].to("Myr")
        # get all nodes going back at least 50 Myr or until progenitor is 1e4 Msun
        nodes = [node for node in form_node["prog"]
                 if (t_halo - node["time"] < a.quan(50, "Myr") or
                     node["mass"] >= a.quan(1e4, "Msun"))]

        if yt.is_root():
            yt.mylog.info(f"Profiling {str(form_node)} for past "
                          f"{str(nodes[0]['time'] - nodes[-1]['time'])}.")

        for i in yt.parallel_objects(range(len(nodes)), dynamic=True):
            ap.process_target(nodes[i])

        if yt.is_root():
            with open(done_file, mode='w') as f:
                f.write(f"{time.time()}\n")
