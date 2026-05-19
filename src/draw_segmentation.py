import argparse
import os
from utils.ball_detection import draw_segmentation, draw_masks
from ultralytics import YOLO
import time

parser = argparse.ArgumentParser(description='Obtiene y dibuja las máscaras de segmentación sobre un vídeo.')
parser.add_argument('-i', '--input', type=str, metavar='input_video', help='Ruta al archivo de vídeo de entrada.')
parser.add_argument('-o', '--output', type=str, metavar='output_video', help='Ruta al archivo de vídeo de salida.')
parser.add_argument('-m', '--model', type=str, metavar='model_path', help='Ruta al archivo del modelo de segmentación.')
parser.add_argument('--masks', type=str, metavar='masks_folder', help='Ruta a la carpeta con las máscaras de segmentación.', default=None)

args = parser.parse_args()

tic = time.time()
if args.masks is not None:
    draw_masks(args.input, args.masks, args.output)
else:
    model = YOLO(args.model, task='segment', verbose=False)
    draw_segmentation(args.input, model, args.output)
toc = time.time()
print(f"Máscaras de segmentación dibujadas en {args.output} ({toc - tic:.2f} s)")

