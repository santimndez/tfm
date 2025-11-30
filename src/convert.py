import numpy as np
import pandas as pd
import os

def convert_points(input_file: str, output_file: str) -> None:
    """
    Convierte un archivo de puntos seleccionado en un formato diferente.

    Args:
        input_file (str): Ruta al archivo de entrada con puntos seleccionados.
        output_file (str): Ruta al archivo de salida para guardar los puntos convertidos.
    """
    # Leer el archivo de entrada
    df = pd.read_csv(input_file, sep='\s+')

    # Verificar que las columnas esperadas estén presentes
    expected_columns = {'frame', 'x', 'y', 'isin'}
    if not expected_columns.issubset(df.columns):
        raise ValueError(f"El archivo de entrada debe contener las columnas: {expected_columns}")

    df = df.sort_values(by='frame')

    ann_frame_idx_list = list(df.groupby('frame').groups.keys())
    ann_obj_id = 32
    points_list = [np.array(df[df['frame'] == frame_idx][['x', 'y']].values, dtype = np.float32) for frame_idx in ann_frame_idx_list]
    labels_list = [np.array([isin for isin in df[df['frame'] == frame_idx]['isin']], dtype=np.int32) for frame_idx in ann_frame_idx_list]

    # Escribir el codigo python correspondiente a la inicialización de las variables

    # Si el archivo de salida es None, imprimir por pantalla
    if output_file is None:
        print(f"ann_frames_idx_list = {ann_frame_idx_list}")
        print(f"ann_obj_id = {ann_obj_id}")
        print("points_list = [")
        first = True
        for points in points_list:
            if first:
                first = False
            else:
                print(",")
            print(f"    np.array({points.tolist()}, dtype=np.float32)", end="")
                
        print("]")
        print("labels_list = [")
        first = True
        for labels in labels_list:
            if first:
                first = False
            else:
                print(",")
            print(f"    np.array({labels.tolist()}, dtype=np.int32)", end="")
        print("]")
        return
    else:
        with open(output_file, 'w') as f:
            f.write(f"ann_frame_idx_list = {ann_frame_idx_list}\n")
            f.write(f"ann_obj_id = {ann_obj_id}\n")
            f.write("points_list = [\n")
            first = True
            for points in points_list:
                if first:
                    first = False
                else:
                    f.write(",\n")
                f.write(f"    np.array({points.tolist()}, dtype=np.float32)")
            f.write("]\n")
            f.write("labels_list = [\n")
            first = True
            for labels in labels_list:
                if first:
                    first = False
                else:
                    f.write(",\n")
                f.write(f"    np.array({labels.tolist()}, dtype=np.int32)")
            f.write("]\n")