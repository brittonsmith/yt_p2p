import glob
import os
import sys

here_dir = os.getcwd()
source_dir = os.path.abspath(sys.argv[1])
data_dir = "rockstar_halos"
snap_dirs = sorted(glob.glob("DD????"))
os.makedirs(data_dir)

for i, snap_dir in enumerate(snap_dirs):
    if not os.path.islink(snap_dir):
        stop_index = i - 1
        print (f"Stopping at {snap_dirs[stop_index]}.")
        break

for i, snap_dir in enumerate(snap_dirs[:stop_index]):
    out_file = f"out_{i}.list"
    source_file = os.path.join(source_dir, data_dir, out_file)
    here_file = os.path.join(here_dir, data_dir, out_file)
    print (f"Linking {source_file} to {here_file}.")
    os.symlink(source_file, here_file)

    data_files = glob.glob(os.path.join(source_dir, data_dir, f"halos_{snap_dir}*"))
    for data_file in data_files:
        filename = os.path.basename(data_file)
        here_file = os.path.join(here_dir, data_dir, filename)
        os.symlink(data_file, here_file)
