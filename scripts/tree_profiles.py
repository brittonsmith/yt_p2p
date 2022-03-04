import numpy as np
import os
import time
import yaml
import yt
yt.enable_parallelism()
import ytree

from ytree.analysis import AnalysisPipeline

from yt.extensions.p2p.tree_analysis_operations import \
    yt_dataset, \
    delattrs, \
    garbage_collect, \
    node_profile, \
    node_sphere

def set_inner_bulk_velocity(node):
    sphere = node.sphere
    ds = sphere.ds
    ex_dx = sphere.quantities.extrema(("index", "dx"))
    sp_small = ds.sphere(sphere.center, 4*ex_dx[0])
    vb = sp_small.quantities.bulk_velocity(use_gas=True, use_particles=False)
    yt.mylog.info(f"Setting bulk velocity using {sp_small.radius.to('pc')} sphere as {vb.to('km/s')}.")
    sphere.set_field_parameter("bulk_velocity", vb)

def velocity_profiles(ap, output_dir="."):
    """
    All the 1D and 2D profiles to make the velocity profiles figure.
    """

    x_bin_field = ("index", "radius")
    profile_fields = [("gas", "cell_mass")]
    weight_field = None
    pkwargs = {
        "accumulation": False,
        "bin_density": 20
    }
    
    _fields = [
        "velocity_magnitude",
        "velocity_spherical_radius",
        "velocity_spherical_theta",
        "velocity_spherical_phi",
        "tangential_velocity_magnitude",
        "sound_speed"
    ]
    velocity_fields = [("gas", field) for field in _fields]

    ap.add_operation(node_sphere, center_field="icom_gas_position")
    ap.add_operation(set_inner_bulk_velocity)

    for field in velocity_fields:
        my_kwargs = pkwargs.copy()
        my_kwargs["logs"] = {x_bin_field: True, field: False}
        my_kwargs["units"] = {x_bin_field: "pc", field: "km/s"}

        ap.add_operation(
            node_profile, [x_bin_field, field],
            profile_fields, weight_field,
            profile_kwargs=my_kwargs, output_dir=output_dir)

    # 1D mass vs. radius profiles to get circular velocity
    mass_fields = [
        ("gas", "cell_mass"),
        ("gas", "dark_matter_mass"),
        ("gas", "matter_mass")
    ]
    my_kwargs = pkwargs.copy()
    my_kwargs["logs"] = {x_bin_field: True}
    my_kwargs["units"] = {x_bin_field: "pc"}

    ap.add_operation(
        node_profile, [x_bin_field], mass_fields, weight_field,
        profile_kwargs=my_kwargs, output_dir=output_dir)

if __name__ == "__main__":
    output_data_dir = "minihalo_analysis"
    data_dir = "."
    
    a = ytree.load("merger_trees/target_halos/target_halos.h5")
    if "icom_gas_position_x" in a.field_list:
        a.add_vector_field("icom_gas_position")

    for tree in a:
        output_dir = os.path.join(output_data_dir, f"node_{tree.uid}")

        ap = AnalysisPipeline(output_dir=output_dir)
        ap.add_operation(yt_dataset, data_dir)

        ap.add_recipe(velocity_profiles, output_dir="profiles")

        ap.add_operation(delattrs, ["sphere", "ds"], always_do=True)
        ap.add_operation(garbage_collect, 60, always_do=True)

        ap.process_target(tree)
