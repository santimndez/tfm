import pandas as pd 
import numpy as np
import os
import shutil
from sklearn.model_selection import train_test_split

import argparse
parser = argparse.ArgumentParser(usage=
"""
usage: filter_dataset.py [-h] -s SELECTED_FRAMES -d DATASET_PATH -o OUTPUT_PATH
Este script filtra un dataset YOLO11seg basándose en un archivo de frames seleccionados.
""")

parser.add_argument("-s", "--selected_frames", help="Archivo con los frames seleccionados en formato con dos columnas: indice y seleccionado o no (0/1) separadas por un espacio.", required=True)
parser.add_argument("-d", "--dataset_path", help="Ruta al dataset original", required=True)
parser.add_argument("-o", "--output_path", help="Ruta al dataset filtrado", default='./filtered_dataset/')
args = parser.parse_args()

selected_frames_path = args.selected_frames
dataset_path = args.dataset_path
output_path = args.output_path
df_selected = pd.read_csv(selected_frames_path, header=None, sep=' ')
selected = np.array(df_selected[1])

j = 0
folders = ['images/', 'labels/', 'masks/']
extensions = ['jpg', 'txt', 'png']

for folder in folders:
    os.makedirs(os.path.join(output_path, folder), exist_ok=True)

for i in range(len(selected)):
    if selected[i]:
        for folder, extension in zip(folders, extensions):
            namein = os.path.join(dataset_path, folder, f'{i:05d}.{extension}')
            nameout = os.path.join(output_path, folder, f'{j:05d}.{extension}')
            if os.path.exists(namein):
                shutil.copy(namein, nameout)
        j += 1

datasize = j

shutil.copy(os.path.join(dataset_path, 'data.yaml'), os.path.join(output_path, 'data.yaml'))


train_split = 0.8
val_split = 0.1
test_split = 0.1

random_state = 42
class_id = 32

# Divisiones
train_idx, temp_idx = train_test_split(range(datasize), test_size=1-train_split, random_state=random_state)
if val_split>0:
    val_idx, test_idx = train_test_split(temp_idx, test_size=(1-train_split-val_split)/(1-train_split), random_state=random_state+1)
else:
    val_idx = []
    test_idx = temp_idx

subsets = {'train': train_idx, 'val': val_idx, 'test': test_idx}

# data.yaml
with open(os.path.join(output_path, 'data.yaml'), "w") as f:
    f.write(f"""path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n {class_id}: ball""")

for subset, indices in subsets.items():
    for folder, extension in zip(folders, extensions):
        os.makedirs(os.path.join(output_path, folder, subset), exist_ok=True)
    for idx in indices:
        for folder, extension in zip(folders, extensions):
            namein = os.path.join(output_path, folder, f'{idx:05d}.{extension}')
            nameout = os.path.join(output_path, folder, subset, f'{idx:05d}.{extension}')
            if os.path.exists(namein):
                shutil.move(namein, nameout)