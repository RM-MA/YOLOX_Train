import os
from pathlib import Path, PurePath
import argparse
import shutil
import time


def add_prefix(src_dir, dst_dir, prefix):
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)

    i = 0

    for src_file in src_path.iterdir():

        if src_file.stem.find("draw") != -1:
            continue

        print(src_file.stem)
        print(src_file.suffix)

        dst_file = dst_path / f'{prefix}{src_file.stem}{src_file.suffix}'
        print(f"src path = {src_file}")
        print(f"dst path = {dst_file}")

        # copy
        shutil.copy(src=src_file, dst=dst_file)

        i += 1

    print(f"all {i} files!")
    time.sleep(2)



def main():
    src_dir_list = [
        # '/home/www/datasets/power/robomaster',
        # '/home/www/datasets/power/王锋蓝',
        # '/home/www/datasets/power/王锋红',
        # '/home/www/datasets/power/hz_plus',
        # '/home/www/datasets/power/hz',
        # '/home/www/datasets/power/datasets',
        # '/home/www/datasets/power/markPan',
        '/home/www/datasets/new_power/gan',
        '/home/www/datasets/new_power/p_red',
        '/home/www/datasets/new_power/p_blue',
        '/home/www/datasets/new_power/robomaster',
        '/home/www/datasets/new_power/blue',
        '/home/www/datasets/new_power/red',
    ]
    dst_dir = '/home/www/datasets/new_power/all'

    for i, src_dir in enumerate(src_dir_list):
        add_prefix(src_dir=src_dir,
                   dst_dir=dst_dir,
                   prefix=f'{i}_')


if __name__ == '__main__':
    main()
