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

from ytree.analysis import AnalysisPipeline
from yt.extensions.p2p.misc import \
    sphere_icom, \
    reunit
from yt.extensions.p2p.profiles import my_profile
from yt.extensions.p2p.tree_analysis_operations import \
    yt_dataset, \
    delattrs, \
    garbage_collect
from yt.extensions.p2p import add_p2p_fields

pfields = {
    ("gas", "density"): "algae",
    ("gas", "temperature"): "gist_heat",
    ("gas", "metallicity3"): "kamae",
    ("gas", "H2_p0_fraction"): "cool"
}

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

    node.sphere = ds.sphere(center, radius)

def sphere_radial_profiles(node, fields, weight_field=None, output_dir=".",
                           profile_kwargs=None):
    if weight_field is None:
        weight_name = "None"
    else:
        if not isinstance(weight_field, tuple) and len(weight_field) != 2:
            raise ValueError("weight_field must be a tuple of length 2.")
        weight_name = weight_field[1]

    fn = os.path.join(output_dir, f"{str(node.ds)}_profile_weight_field_{weight_name}.h5")
    if os.path.exists(fn):
        return

    pkwargs = {"accumulation": False, "bin_density": 20}
    if profile_kwargs is not None:
        pkwargs.update(profile_kwargs)

    data_source = node.sphere
    profile = my_profile(
        data_source,
        ("index", "radius"), fields,
        units={("index", "radius"): "pc"},
        weight_field=weight_field,
        **pkwargs)
    profile.save_as_dataset(filename=fn)
    del profile

def my_analysis(pipeline, grackle_fields, data_dir="."):
    pipeline.add_operation(yt_dataset, data_dir)
    pipeline.add_operation(icom_sphere)
    # pipeline.add_operation(region_projections, pfields, output_dir="projections")
    pipeline.add_operation(
        sphere_radial_profiles,
        grackle_fields + \
        [("gas", "density"),
         ('gas', 'dark_matter_density'),
         ('gas', 'matter_density'),
         ("gas", "temperature"),
         ('gas', 'entropy'),
         ("gas", "cooling_time"),
         ("gas", "dynamical_time"),
         ("gas", "total_dynamical_time"),
         ("gas", "velocity_magnitude"),
         ("gas", "sound_speed"),
         ("gas", "pressure"),
         ("gas", "H2_p0_fraction"),
         ("gas", "metallicity3")],
        weight_field=("gas", "cell_mass"),
        output_dir=".")

    pipeline.add_operation(
        sphere_radial_profiles,
        [("gas", "density"),
         ("gas", "dark_matter_density"),
         ("gas", "matter_density")],
        weight_field=("gas", "cell_volume"),
        output_dir=".")

    pipeline.add_operation(
        sphere_radial_profiles,
        [("gas", "cell_mass"),
         ("gas", "dark_matter_mass"),
         ("gas", "matter_mass")],
        weight_field=None,
        output_dir=".")

    pipeline.add_operation(delattrs, ["sphere", "ds"], always_do=True)
    pipeline.add_operation(garbage_collect, 60, always_do=True)

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
    grackle_fields = [_grackle_map[field][0] for field in _get_needed_fields(es.grackle_data)]

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
        ap.add_recipe(my_analysis, grackle_fields, data_dir=data_dir)

        my_tree = a[star_info["_arbor_index"]]

        # find the halo where the star formed
        form_node = my_tree.get_node("forest", star_info["tree_id"])

        # get all nodes going back at least 50 Myr or until progenitor is 1e4 Msun
        pmass = form_node["prog", "mass"]
        t_halo = form_node["time"]
        tbefore = t_halo - form_node["prog", "time"]
        mmin = a.quan(1e4, "Msun")
        tprior = a.quan(50, "Myr")
        tstart = tbefore[(pmass > mmin) | (tbefore < tprior)][-1]

        if yt.is_root():
            yt.mylog.info(f"Profiling {form_node} for past {tstart.to('Myr')}.")

        for node in ytree.parallel_tree_nodes(form_node, group="prog", dynamic=True):
            if t_halo - node["time"] > tprior and node["mass"] < mmin:
                continue

            ap.process_target(node)

        if yt.is_root():
            with open(done_file, mode='w') as f:
                f.write(f"{time.time()}\n")
