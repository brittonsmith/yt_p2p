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

from yt.extensions.astro_analysis.halo_analysis.api import \
    add_callback
from yt.extensions.astro_analysis.halo_analysis.halo_callbacks import \
    periodic_distance

def sphere_projection(halo, fields, weight_field=None, axes="xyz", output_dir="."):
    yt.mylog.info("Projecting halo %d." % halo.quantities["particle_identifier"])
    for axis in axes:
        plot = yt.ProjectionPlot(halo.halo_catalog.data_ds, axis, fields,
                                 weight_field=weight_field, data_source=halo.data_object, 
                                 center=halo.data_object.center,
                                 width=(2* halo.data_object.radius))
        plot.set_axes_unit("pc")
        plot.annotate_title("M = %s." % halo.quantities["particle_mass"].in_units("Msun"))
        plot.save(os.path.join(halo.halo_catalog.output_dir, output_dir,
                               "halo_%06d" % (halo.quantities['particle_identifier'])))
add_callback("sphere_projection", sphere_projection)

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
