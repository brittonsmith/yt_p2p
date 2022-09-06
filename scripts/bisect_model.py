"""
Compute critical collapse metallicity with bisection for all good halos.
"""

import numpy as np
import os
from scipy.optimize import bisect
from scipy.stats import linregress
import yaml
from yt.funcs import ensure_dir

from yt.extensions.p2p.models import \
    initialize_model_set, \
    create_model, \
    calculate_final_time

def evaluate_model(lZ, star_id, model_data, model_parameters):
    my_metallicity = 10**lZ

    filekey = f"model_lZ_{lZ:.6f}"

    model = create_model(model_data, model_parameters,
                        metallicity=my_metallicity)
    model.name = f"star_{star_id}/{filekey}"
    model.evolve()
    data = model.finalize_data()

    gas_mass = data["gas_mass"]
    be_mass = data["bonnor_ebert_mass"]

    # ratio = (gas_mass / be_mass).max(axis=1)
    imax = (gas_mass / be_mass[-1]).argmax()
    ratio = gas_mass[imax] / be_mass[:, imax]
    if ratio[-1] >= 1:
        print ("This model collapsed.")
        return 1
    if ratio[-1] / ratio.min() >= 10:
        print ("This model achieved 10x more instability.")
        return 1

    time = data["time"].to("Myr")
    t_until = time[-1] - time
    # evaluate slope for final 10 Myr
    my_filter = t_until < 20
    my_x = time[my_filter]
    my_y = np.log10(ratio[my_filter])

    fit = linregress(my_x, my_y)
    if fit.slope > 0.01:
        print ("This model achieved critical slope.")
        return 1

    print ("This model remained stable.")
    return -1


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
        "primordial_chemistry": 1,
        "metal_cooling": 1,
        "with_radiative_cooling": 1,
        "grackle_data_file": "cloudy_metals_2008_3D.h5",
        "h2_on_dust": 1,
        "H2_self_shielding": 0,
        "use_radiative_transfer": 1,
        "radiative_transfer_coupled_rate_solver": 0
    }

    models_fn = "models.yaml"
    base_output_dir = "minihalo_models/metallicity_grids"
    ensure_dir(base_output_dir)

    tolerance = 1e-3

    for star_id in star_ids:
        if os.path.exists(models_fn):
            with open(models_fn, 'r') as f:
                models = yaml.load(f, Loader=yaml.FullLoader)
            if models is None:
                models = {}
        else:
            models = {}

        if star_id not in models:
            models[star_id] = {}
        my_model = models[star_id]

        if "solutions" not in my_model:
            my_model["solutions"] = {}
        my_solutions = my_model["solutions"]

        my_tol = f"{tolerance:g}"
        if my_tol in my_solutions:
            continue

        output_dir = os.path.join(base_output_dir, f"star_{star_id}")
        ensure_dir(output_dir)

        model_data, model_parameters = initialize_model_set(
            star_id, grackle_pars,
            data_dir=data_dir)

        calculate_final_time(models, star_id, models_fn,
                             model_data, model_parameters)
        model_parameters["final_time"] = models[star_id]["final_time"]

        my_root = bisect(evaluate_model, -5, -3, xtol=tolerance,
                         args=(star_id, model_data, model_parameters))

        my_metallicity=10**my_root
        filekey = f"model_lZ_{my_root:.6f}"
        filename = os.path.join(output_dir, f"{filekey}.h5")

        model = create_model(model_data, model_parameters,
                             metallicity=my_metallicity)
        model.name = f"star_{star_id}/{filekey}"
        model.evolve()
        model.save_as_dataset(filename=filename)

        my_solutions[my_tol] = {"filename": filename, "value": my_root}

        with open(models_fn, mode="w") as f:
            yaml.dump(models, stream=f)
