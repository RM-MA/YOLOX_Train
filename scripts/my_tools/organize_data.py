import os
from pathlib import Path, PurePath
import argparse

# 整理数据


parser = argparse.ArgumentParser()
parser.add_argument("--root_dir", default="label", type=str, help="输入根路径.")
parser.add_argument("--save_path", default="output", type=str, help="输出根路径.")
parser.add_argument("--prefix", type=str, help="添加前缀.")

arg = parser.parse_args()


def main():
    src_dir = arg.root_dir
    out_dir = arg.save_path
    prefix = arg.prefix





if __name__ == "__main__":
    main()
