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
from yt.extensions.p2p.stars import get_star_data
from yt.extensions.p2p.tree_analysis_operations import \
    get_progenitor_line, \
    sphere_radial_profiles, \
    yt_dataset, \
    delattrs, \
    node_sphere, \
    garbage_collect

def my_analysis(pipeline, grackle_fields, data_dir="."):
    pipeline.add_operation(yt_dataset, data_dir)

    center_fields = ["icom_gas_position", "icom_all_position"]

    for field in center_fields:
        pipeline.add_operation(
            node_sphere, center_field=field)

        if field in ["icom_gas_position"]:
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
                output_dir=f"profiles/{field}")

        pipeline.add_operation(
            sphere_radial_profiles,
            [("gas", "density"),
             ("gas", "dark_matter_density"),
             ("gas", "matter_density")],
            weight_field=("gas", "cell_volume"),
            output_dir=f"profiles/{field}")

        pipeline.add_operation(
            sphere_radial_profiles,
            [("gas", "cell_mass"),
             ("gas", "dark_matter_mass"),
             ("gas", "matter_mass")],
            weight_field=None,
            output_dir=f"profiles/{field}")

        pipeline.add_operation(delattrs, ["sphere"], always_do=True)

if __name__ == "__main__":
    output_data_dir = "star_minihalos"

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

    star_data = get_star_data("star_hosts.yaml")

    for star_id, star_info in star_data.items():
        a = ytree.load(star_info["arbor"])
        for field in ["icom_gas_position", "icom_all_position"]:
            if f"{field}_x" in a.field_list:
                a.add_vector_field(field)

        output_dir = os.path.join(output_data_dir, f"star_{star_id}")
        done_file = os.path.join(output_dir, "done")
        if os.path.exists(done_file):
            continue

        ap = AnalysisPipeline(output_dir=output_dir)
        ap.add_recipe(my_analysis, grackle_fields, data_dir=data_dir)
        ap.add_operation(delattrs, ["ds"], always_do=True)
        ap.add_operation(garbage_collect, 60, always_do=True)

        my_tree = a[star_info["_arbor_index"]]

        # find the halo where the star formed
        form_node = my_tree.get_node("forest", star_info["tree_id"])
        nodes = list(get_progenitor_line(form_node))

        # get all nodes going back at least 50 Myr or until progenitor is 1e4 Msun
        m_min = a.quan(1e4, "Msun")
        tprior = a.quan(50, "Myr")

        if yt.is_root():
            yt.mylog.info(f"Profiling star {star_id}.")

        for node in ytree.parallel_tree_nodes(form_node, nodes=nodes, dynamic=True):
            if form_node["time"] - node["time"] > tprior and node["mass"] < m_min:
                continue

            ap.process_target(node)

        if yt.is_root():
            with open(done_file, mode='w') as f:
                f.write(f"{time.time()}\n")
