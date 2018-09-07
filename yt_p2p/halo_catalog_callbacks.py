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

import os
import yt

from yt.extensions.astro_analysis.halo_analysis.api import \
    add_callback

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
