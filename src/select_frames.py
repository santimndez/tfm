import cv2 as cv
import argparse
import numpy as np
import os
from convert import convert_points

parser = argparse.ArgumentParser(usage=
"""
usage: select_points.py [-h] -i INPUT [-f FRAME] [-o OUTPUT] [-a]

Este script permite seleccionar frames de un vídeo y guardar el resultado en un archivo de texto.
Es útil para seleccionar los frames válidos de un conjunto de entrenamiento obtenido mediante destilación de un modelo profesor como SAM2.
Controles: 
    - Tecla 's': seleccionar el frame actual
    - Tecla 'w': rechazar el frame actual
    - Tecla 'a': retroceder un frame
    - Tecla 'd': avanzar un frame
    - Tecla 'r': retroceder 10 frames
    - Tecla 'f': avanzar 10 frames
    - Tecla 'q': salir del programa
    - Tecla 'i': hacer zoom in
    - Tecla 'o': hacer zoom out
Formato del archivo de salida:
    columnas: frame valid
    frame es el número de frame y valid es 1 si el frame es válido y 0 si no.
""")
parser.add_argument("-i", "--input", help="Ruta al vídeo de entrada o carpeta con los frames en formato jpg y nombre 00001.jpg, 00002.jpg, etc.", required=True)
parser.add_argument("-f", "--frame", help="Frame inicial para la selección", type=int, default=0)
parser.add_argument("-o", "--output", help="Archivo de salida para guardar los puntos", default='./frames.txt')
parser.add_argument("-a", "--all", help="Inicialmente selecciona todos los frames", action='store_true')

args = parser.parse_args()

isfolder = os.path.isdir(args.input)
zoom = 1.0

if not isfolder:
    def get_frame_count(input):
        video = cv.VideoCapture(input)
        frame_count = int(video.get(cv.CAP_PROP_FRAME_COUNT))
        video.release()
        return frame_count
    
    frame_count = get_frame_count(args.input)

    def extract_frame(input, nframe=0, putText=True):
        global frame_count, zoom
        # Cargar el video
        video = cv.VideoCapture(input)
        video.set(cv.CAP_PROP_POS_FRAMES, nframe)
        success, frame = video.read()
        if zoom != 1.0: 
            width, height = int(frame.shape[1] * zoom), int(frame.shape[0] * zoom)
            frame = cv.resize(frame, (width, height), interpolation=cv.INTER_CUBIC)
        if success and putText:
            cv.putText(frame, f"Frame {nframe}/{frame_count}", (10, 30), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

        # Liberar el video      
        video.release()

        return frame if success else None
else:
    frame_count = len([name for name in os.listdir(args.input) if os.path.isfile(os.path.join(args.input, name)) and name.endswith('.jpg')])
    def extract_frame(input, nframe=0, putText=True):
        global frame_count, zoom
        # Leer la imagen
        frame = cv.imread(os.path.join(input, f"{nframe:05d}.jpg"))
        if zoom != 1.0: 
            width, height = int(frame.shape[1] * zoom), int(frame.shape[0] * zoom)
            frame = cv.resize(frame, (width, height), interpolation=cv.INTER_CUBIC)
        if frame is not None and putText:
            cv.putText(frame, f"Frame {nframe}/{frame_count}", (10, 30), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
        return frame if frame is not None else None

selected_frames = np.zeros(frame_count, dtype=np.uint8) if not args.all else np.ones(frame_count, dtype=np.uint8)
frame = extract_frame(args.input, args.frame)

if frame is not None:
    cv.namedWindow("window")
    cv.imshow("window", frame)
    def manejador_tecla(key):
        global frame, selected_frames, zoom
        if key & 0xFF == ord('s') or key & 0xFF == ord('w') or key & 0xFF == ord('d'):  # Avanzar un frame
            if key & 0xFF == ord('s'):
                selected_frames[args.frame] = 1
            elif key & 0xFF == ord('w'):
                selected_frames[args.frame] = 0
            args.frame = (args.frame + 1) % frame_count
            frame = extract_frame(args.input, args.frame)
            frame = cv.circle(frame, (10, 60), 10, (0, 255, 0), -1) if selected_frames[args.frame] == 1 else cv.circle(frame, (10, 60), 10, (0, 0, 255), -1)
            cv.imshow("window", frame)
        elif key & 0xFF == ord('a'):  # Retroceder un frame
            args.frame = (frame_count + args.frame - 1) % frame_count
            frame = extract_frame(args.input, args.frame)
            frame = cv.circle(frame, (10, 60), 10, (0, 255, 0), -1) if selected_frames[args.frame] == 1 else cv.circle(frame, (10, 60), 10, (0, 0, 255), -1)
            cv.imshow("window", frame)
        elif key & 0xFF == ord('f'):  # Avanzar 10 frames
            args.frame = (args.frame + 10) % frame_count
            frame = extract_frame(args.input, args.frame)
            frame = cv.circle(frame, (10, 60), 10, (0, 255, 0), -1) if selected_frames[args.frame] == 1 else cv.circle(frame, (10, 60), 10, (0, 0, 255), -1)
            cv.imshow("window", frame)
        elif key & 0xFF == ord('r'):  # Retroceder 10 frames
            args.frame = (frame_count + args.frame - 10) % frame_count
            frame = extract_frame(args.input, args.frame)
            frame = cv.circle(frame, (10, 60), 10, (0, 255, 0), -1) if selected_frames[args.frame] == 1 else cv.circle(frame, (10, 60), 10, (0, 0, 255), -1)
            cv.imshow("window", frame)
        elif key & 0xFF == ord('i'):  # Zoom in
            zoom += 0.1
            frame = extract_frame(args.input, args.frame)
            cv.imshow("window",  frame)
        elif key & 0xFF == ord('o'):  # Zoom out
            zoom = max(0.1, zoom - 0.1)
            frame = extract_frame(args.input, args.frame)
            cv.imshow("window",  frame)
    key = cv.waitKey(0)
    while key & 0xFF != ord('q'):
        manejador_tecla(key)
        key = cv.waitKey(0)

    # Guardar los frames seleccionados en un archivo
    with open(args.output, 'w') as f:
        for i in range(frame_count):
            f.write(f"{i} {selected_frames[i]}\n")
    cv.destroyAllWindows()

else:
    print(f"Error al leer el frame {args.frame} del video {args.input}")