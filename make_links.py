import glob
import os
import sys

if __name__ == "__main__":
    sim_dir = sys.argv[1]
    link_dir = sys.argv[2]

    sim_dds = glob.glob(os.path.join(sim_dir, 'DD????'))
    sim_ids = [int(ds[-4:]) for ds in sim_dds]
    sim_start = min(sim_ids)

    for i in range(sim_start):
        ds = f"DD{i:04d}"
        link_path = os.path.abspath(os.path.join(link_dir, ds))
        if os.path.exists(link_path):
            sim_path = os.path.join(sim_dir, ds)
            print (f"Linking {sim_path} to {link_path}.")
            os.symlink(link_path, sim_path)
        else:
            print (f"Skipping {link_path}.")
            continue
