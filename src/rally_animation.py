import argparse
import numpy as np
import cv2 as cv
from ultralytics import YOLO
import pandas as pd
import time

from utils.tt_animation import ball_animation
PROFILING = False

# Argumentos de la línea de comandos
parser = argparse.ArgumentParser(description='Crea una animación para ver la trayectoria de la pelota previamente calculada')
parser.add_argument('-i', '--input', metavar='video_files', type=str, nargs=2, help='Rutas de los dos vídeos a sincronizar')
parser.add_argument('-o', '--output', metavar='output_file', type=str, help='Ruta donde se guardará el vídeo con la animación', default=None)
parser.add_argument('-p', '--positions', metavar='ball_positions', type=str, nargs=2, help='Ruta a los archivos csv con las posiciones de la pelota en cada frame. Si no se proporciona, se deberá proporcionar el parámetro --model', default=None)
parser.add_argument('--offset', metavar='offset', type=float, help='Desfase en segundos a aplicar al primer vídeo (positivo para retrasarlo, negativo para adelantarlo). Si no se proporciona, se estimará automáticamente', default=None)
parser.add_argument('-vfr', action='store_true', help='Indica si los vídeos son de frame rate variable (VFR) y se deben usar las marcas de tiempo para la sincronización')
parser.add_argument('-t', '--trajectory', help='Ruta al archivo con la trayectoria de la pelota en formato csv', default=None)
args = parser.parse_args()


df = pd.read_csv(args.trajectory, header=None).values
X = df[:, :-1].T
t = df[:, -1]

ball_animation(X, t, out=args.output, fps=30)