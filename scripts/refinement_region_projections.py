import gc
import numpy as np
import os
import yt
yt.enable_parallelism()

from yt.funcs import ensure_dir
from yt.extensions.p2p import add_p2p_fields
from yt.extensions.p2p.misc import reunit

if __name__ == "__main__":
    es = yt.load("simulation.h5")
    fns = es.data["filename"].astype(str)
    rleft = es.data["RefineRegionLeftEdge"]
    rright = es.data["RefineRegionRightEdge"]

    center = es.arr([0.5]*3, "unitary")
    width = es.quan(0.15, "unitary")

    data_dir = "/disk14/brs/pop2-prime/firstpop2_L2-Seed3_large/cc_512_no_dust_continue"
    output_dir = "projections"
    ensure_dir(output_dir)

    fields = [
        ("gas", "density"),
        ("gas", "temperature"),
        ("gas", "metallicity3")
    ]
    wfield = ("gas", "density")
    pfields = [
        ("io", "particle_mass")
    ]

    for fn in yt.parallel_objects(fns, dynamic=True):
        ds = yt.load(os.path.join(data_dir, fn))

        ax = "x"
        ofn = os.path.join(output_dir, f"{ds.basename}_{ax}.h5")
        if os.path.exists(ofn):
            continue
        
        add_p2p_fields(ds)

        my_c = reunit(ds, center, "unitary")
        my_w = reunit(ds, width, "unitary")
        region = ds.box(my_c - 1.05 * my_w / 2,
                        my_c + 1.05 * my_w / 2)

        p = yt.ProjectionPlot(
            ds, ax, fields, weight_field=wfield,
            center=my_c, width=my_w, data_source=region)
        data = {field[1]: p.frb[field] for field in fields}
        if yt.is_root():
            p.save("projection_images/")
        del p

        p = yt.ParticleProjectionPlot(
            ds, ax, pfields,
            center=my_c, width=my_w, data_source=region)
        data.update({field[1]: p.frb[field] for field in pfields})
        if yt.is_root():
            p.save("projection_images/")
        del p

        if yt.is_root():
            extra_attrs = {"center": my_c, "width": my_w}
            yt.save_as_dataset(ds, filename=ofn, data=data, extra_attrs=extra_attrs)

        region.clear_data()
        del region
        del ds
        val = gc.collect()
        yt.mylog.info(f"Removed {val} garbages!")
