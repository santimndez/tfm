import os
import shutil

import argparse
parser = argparse.ArgumentParser(usage=
"""
usage: filter_dataset.py [-h] -a DATASET_A -b DATASET_B -o OUTPUT_DATASET
Este script une dos datasets YOLO-seg.
""")

parser.add_argument("-a", "--dataset_a", help="Ruta al dataset A", required=True)
parser.add_argument("-b", "--dataset_b", help="Ruta al dataset B", required=True)
parser.add_argument("-o", "--output_path", help="Ruta en la que se guardará dataset unido. Si no se proporciona se utilizará la del dataset A", default=None)
parser.add_argument("-m", "--move", help="Mover los archivos en lugar de copiarlos", action='store_true')
args = parser.parse_args()

subsets = ['train', 'val', 'test']
datasize_a = 0
for subset in subsets:
    path_a = os.path.join(args.dataset_a, 'images', subset)
    datasize_a += len([name for name in os.listdir(path_a) if os.path.isfile(os.path.join(path_a, name)) and name.endswith('.jpg')])

if args.output_path is not None:
    # Copy dataset A to output path
    if args.move:
        shutil.move(args.dataset_a, args.output_path)
    else:
        shutil.copytree(args.dataset_a, args.output_path)
else:
    args.output_path = args.dataset_a

# Add datasize_a to the name of dataset B files and move them to output path
folders = ['images/', 'labels/', 'masks/']
extensions = ['jpg', 'txt', 'png']
for subset in subsets:
    for folder, extension in zip(folders, extensions):
        path_b = os.path.join(args.dataset_b, folder, subset)
        path_out = os.path.join(args.output_path, folder, subset)
        for name in os.listdir(path_b):
            if os.path.isfile(os.path.join(path_b, name)):
                index = int(name.split('.')[0]) + datasize_a
                namein = os.path.join(path_b, name)
                nameout = os.path.join(path_out, f'{index:05d}.{extension}')
                if args.move:
                    shutil.move(namein, nameout)
                else:
                    shutil.copy(namein, nameout)