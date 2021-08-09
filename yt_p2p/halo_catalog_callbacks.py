"""
HaloCatalog callbacks



"""

#-----------------------------------------------------------------------------
# Copyright (c) Britton Smith <brittonsmith@gmail.com>.  All rights reserved.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

import numpy as np
import os
import yt

from yt.extensions.astro_analysis.halo_analysis import \
    add_callback
from yt.extensions.astro_analysis.halo_analysis.halo_catalog.halo_callbacks import \
    periodic_distance

def sphere_projection(halo, fields, weight_field=None, axes="xyz", output_dir=".",
                      sphere=None):
    if sphere is None:
        sphere = getattr(halo, 'data_object')
    if sphere is None:
        raise RuntimeError('No sphere provided.')

    yt.mylog.info(f"Projecting halo {int(halo.quantities['particle_identifier']):d}.")
    for axis in axes:
        plot = yt.ProjectionPlot(halo.halo_catalog.data_ds, axis, fields,
                                 weight_field=weight_field, data_source=sphere,
                                 center=sphere.center,
                                 width=(2*sphere.radius))
        plot.set_axes_unit("pc")
        plot.annotate_title(f"M = {halo.quantities['particle_mass'].in_units('Msun')}.")
        plot.save(os.path.join(output_dir, f"halo_{int(halo.quantities['particle_identifier']):06d}"))
add_callback("sphere_projection", sphere_projection)

def iterative_center_of_mass(halo, inner_radius,
                             radius_field="virial_radius", step_ratio=0.9,
                             use_gas=True, use_particles=False):
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
            old_center = sphere.center
            new_center = sphere.quantities.center_of_mass(
                use_gas=use_gas, use_particles=use_particles)
            sphere = sphere.ds.sphere(
                new_center, step_ratio * sphere.radius)

            diff = periodic_distance(
                old_center.in_units("unitary").v,
                new_center.in_units("unitary").v)
            diff = dds.quan(diff, "unitary")

            # region = dds.box(sphere.center-1.05*sphere.radius,
            #                  sphere.center+1.05*sphere.radius)
            # for ax in "xyz":
            #     p = yt.ProjectionPlot(
            #         dds, ax, ["density", "metallicity3", "temperature"],
            #         weight_field="density", center=sphere.center,
            #         width=(2*sphere.radius), data_source=region)
            #     if sphere.radius < sphere.ds.quan(0.1, "pc"):
            #         my_units = "AU"
            #     p.set_axes_unit(my_units)
            #     p.set_cmap("density", "algae")
            #     p.set_cmap("temperature", "gist_heat")
            #     p.set_cmap("metallicity3", "kamae")
            #     p.save("sphere_center_box/%s_%03d" % (str(dds), i))
            i+=1
            yt.mylog.info(
                "Radius: %s, center: %s, diff: %s." %
                (sphere.radius.in_units(my_units), sphere.center, diff))
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

def halo_data_radius(halo, radius_field="virial_radius"):
    dds = halo.halo_catalog.data_ds
    return dds.quan(halo.quantities[radius_field].to("unitary"))

def halo_data_center(halo):
    dds = halo.halo_catalog.data_ds
    return dds.arr(
        [halo.quantities["particle_position_%s" % axis].to("unitary")
         for axis in "xyz"])

def set_halo_center(halo, new_center):
    for i, axis in enumerate("xyz"):
        halo.quantities["particle_position_%s" % axis] = new_center[i]

def get_my_sphere(halo, radius_field="virial_radius"):
    dds = halo.halo_catalog.data_ds
    sphere = getattr(halo, "data_object", None)
    if sphere is None:
        center_orig = halo_data_center(halo)
        radius_orig = halo_data_radius(halo, radius_field="virial_radius")
        sphere = dds.sphere(center_orig, radius_orig)
    return sphere

def my_sphere(halo, radius):
    dds = halo.halo_catalog.data_ds
    center = halo_data_center(halo)
    sphere = dds.sphere(center, radius)
    halo.data_object = sphere
add_callback("my_sphere", my_sphere)

def set_inner_bulk_velocity(halo, inner_radius):
    dds = halo.halo_catalog.data_ds
    center = halo_data_center(halo)
    sphere = dds.sphere(center, inner_radius)
    bulk_velocity = sphere.quantities.bulk_velocity()
    yt.mylog.info("Halo %06d: Setting bulk velocity to %s.",
                  halo.quantities["particle_identifier"], bulk_velocity)
    halo.data_object.set_field_parameter("bulk_velocity", bulk_velocity)
add_callback("set_inner_bulk_velocity", set_inner_bulk_velocity)

def set_field_max_bulk_velocity(halo, field, radius_field="virial_radius"):
    dds = halo.halo_catalog.data_ds
    sphere = get_my_sphere(halo, radius_field="virial_radius")

    max_vals = sphere.quantities.sample_at_max_field_values(
        field, ['velocity_%s' % ax for ax in "xyz"])
    bulk_velocity = dds.arr(max_vals[1:])
    yt.mylog.info("Halo %06d: Setting bulk velocity to %s.",
                  halo.quantities["particle_identifier"], bulk_velocity)
    halo.data_object.set_field_parameter("bulk_velocity", bulk_velocity)
    halo.data_object.clear_data()
add_callback("set_field_max_bulk_velocity", set_field_max_bulk_velocity)

def set_inner_angular_momentum_vector(halo, inner_radius):
    dds = halo.halo_catalog.data_ds
    center = halo_data_center(halo)
    sphere = dds.sphere(center, inner_radius)
    angular_momentum_vector = sphere.quantities.angular_momentum_vector()
    normal = angular_momentum_vector / np.sqrt((angular_momentum_vector**2).sum())
    yt.mylog.info("Halo %06d: Setting angular momentum vector to %s.",
                  halo.quantities["particle_identifier"], normal)
    halo.data_object.set_field_parameter("normal", normal)
    halo.data_object.clear_data()
add_callback("set_inner_angular_momentum_vector", set_inner_angular_momentum_vector)

def field_max_center(halo, field, radius_field="virial_radius"):
    dds = halo.halo_catalog.data_ds
    center_orig = halo_data_center(halo)
    sphere = get_my_sphere(halo, radius_field="virial_radius")

    max_vals = sphere.quantities.max_location(field)
    new_center = dds.arr(max_vals[1:])

    distance = periodic_distance(
        center_orig.in_units("unitary").v,
        new_center.in_units("unitary").v)
    distance = dds.quan(distance, "unitary")
    yt.mylog.info("Recentering halo %d %f pc away." %
                  (halo.quantities["particle_identifier"],
                   distance.in_units("pc")))

    set_halo_center(halo, new_center)
    del sphere
add_callback("field_max_center", field_max_center)
