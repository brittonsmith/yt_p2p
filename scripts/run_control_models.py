"""
Run and plot a control model.
"""

import numpy as np
import os
import yaml
import yt
from yt.funcs import ensure_dir

from yt.extensions.p2p.models import \
    initialize_model_set, \
    create_model, \
    calculate_final_time

if __name__ == "__main__":
    star_ids = [
        334267081,
        334267082,
        334267090,
        334267093,
        334267099,
        334267102,
        334267086
    ]
    
    data_dir = "star_cubes"
    grackle_pars = {
        "use_grackle": 1,
        "primordial_chemistry": 3,
        "metal_cooling": 1,
        "with_radiative_cooling": 1,
        "grackle_data_file": "cloudy_metals_2008_3D.h5",
        "h2_on_dust": 1,
        "H2_self_shielding": 0,
        "use_radiative_transfer": 1,
        "radiative_transfer_coupled_rate_solver": 0
    }

    base_output_dir = "minihalo_models/control_runs"
    ensure_dir(base_output_dir)
    models_fn = "models.yaml"

    for star_id in star_ids:
        model_data, model_parameters = initialize_model_set(
            star_id, grackle_pars,
            data_dir=data_dir)

        if os.path.exists(models_fn):
            with open(models_fn, 'r') as f:
                models = yaml.load(f, Loader=yaml.FullLoader)
            if models is None:
                models = {}
        else:
            models = {}

        calculate_final_time(models, star_id, models_fn,
                             model_data, model_parameters)
        model_parameters["final_time"] = models[star_id]["final_time"]

        filekey = f"star_{star_id}"
        filename = os.path.join(base_output_dir, f"{filekey}.h5")
        if os.path.exists(filename):
            continue

        model, full_data = create_model(model_data, model_parameters,
                                        get_external_data=True)

        ed_filename = os.path.join(base_output_dir, f"{filekey}_external_data.h5")
        mds = model_data[0]
        start_time = model_data[2]
        extra_attrs = {"start_time": start_time}
        yt.save_as_dataset(mds, filename=ed_filename, data=full_data,
                           extra_attrs=extra_attrs)

        model.name = filekey
        model.evolve()
        model.save_as_dataset(filename=filename)
