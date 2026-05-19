import cv2 as cv
import os
from utils.ball_detection import get_ball_positions_from_masks, save_ball_positions
import argparse

parser = argparse.ArgumentParser(description='Obtener las posiciones de la pelota en un vídeo a partir de las máscaras generadas por SAM3 u otro modelo de segmentación y guardarlas en archivos CSV.')
parser.add_argument('-m', '--masks', metavar='masks_folder', help='Carpeta con las máscaras generadas por SAM3 para cada vídeo. Debe contener archivos de imagen con el formato 00001.png, 00002.png, etc.')
parser.add_argument('-o', '--output', metavar='output_file', help='Archivo donde se guardarán las posiciones de la pelota en formato CSV.')
parser.add_argument('-v', '--video', metavar='video_file', help='Archivo de vídeo de entrada.')
args = parser.parse_args()

masks_folder = args.masks
video = args.video
output_file = args.output

# Get width and height of the video
cap = cv.VideoCapture(video)
w, h = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
nframes = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
cap.release()

print("Ancho del video:", w)
print("Alto del video:", h)
print("Número de frames del video:", nframes)

# Rename files
# for i in range(1, nframes + 1):
#     old_name = os.path.join(masks_folder, f"mask_{i:06d}.png")
#     new_name = os.path.join(masks_folder, f"{i:06d}.png")
#     os.rename(old_name, new_name)

os.makedirs(os.path.dirname(output_file), exist_ok=True)
positions = get_ball_positions_from_masks(masks_folder, w, h, nframes)
save_ball_positions(output_file, positions)
print("Posiciones de la pelota guardadas en:", output_file)