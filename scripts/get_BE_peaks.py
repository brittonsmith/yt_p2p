import numpy as np
import os
import yaml
import yt

from unyt import uvstack
from yt.extensions.p2p.stars import get_star_data
from yt.utilities.physical_constants import G

if __name__ == "__main__":
    star_ids = [
        334267081,
        334267082,
        334267083,
        334267086,
        334267090,
        334267093,
        334267099,
        334267102,
        334267111,
    ]

    star_data = get_star_data("star_hosts.yaml")

    with open("bwo_info.yaml", "r") as f:
        bwo_info = yaml.load(f, Loader=yaml.FullLoader)

    for i, star_id in enumerate(star_ids):
        if star_id not in bwo_info:
            continue

        my_star = star_data[star_id]
        filename = os.path.join("star_cubes", f"star_{star_id}_radius.h5")
        pds = yt.load(filename)
        profile_data = pds.data

        creation_time = my_star["creation_time"]
        times = profile_data["data", "time"].to("Myr")
        ilast = np.where(times < creation_time)[0][-1]

        used = profile_data["data", "used"][ilast].d.astype(bool)
        m_gas_enc = profile_data["data", "gas_mass_enclosed"][ilast, used].to("Msun")

        p = profile_data["data", "pressure"][ilast, used]
        p_hyd = profile_data["data", "hydrostatic_pressure"][ilast, used]

        a = 1.67
        b = (225 / (32 * np.sqrt(5 * np.pi))) * a**-1.5
        cs = profile_data["data", "sound_speed"][ilast, used]
        p_max = uvstack([p, p_hyd]).max(axis=0)
        m_BE = (b * (cs**4 / G**1.5) * p_max**-0.5).to("Msun")

        m_ratio = m_gas_enc / m_BE

        bwo_info[star_id]["Bonnor_Ebert_peak"] = str(m_gas_enc[m_ratio.argmax()])

    with open("bwo_info.yaml", "w") as f:
        yaml.dump(bwo_info, stream=f)
