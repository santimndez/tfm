"""
Este script permite seleccionar puntos en un frame seleccionado de un vídeo y mostrar sus coordenadas haciendo clic.
Es útil para obtener las coordenadas de la bola y usarlas para el prompt de SAM2.
"""

import cv2 as cv
import argparse
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Ruta al vídeo de entrada o carpeta con los frames en formato jpg y nombre 00001.jpg, 00002.jpg, etc.", required=True)
parser.add_argument("-f", "--frame", help="Frame seleccionado del vídeo", type=int, default=0)
parser.add_argument("-o", "--output", help="Archivo de salida para guardar los puntos", default='./points.txt')

args = parser.parse_args()

isfolder = os.path.isdir(args.input)

if not isfolder:
    def get_frame_count(input):
        video = cv.VideoCapture(input)
        frame_count = int(video.get(cv.CAP_PROP_FRAME_COUNT))
        video.release()
        return frame_count
    
    frame_count = get_frame_count(args.input)

    def extract_frame(input, nframe=0, putText=True):
        global frame_count
        # Cargar el video
        video = cv.VideoCapture(input)
        video.set(cv.CAP_PROP_POS_FRAMES, nframe)
        success, frame = video.read()
        if success and putText:
            cv.putText(frame, f"Frame {nframe}/{frame_count}", (10, 30), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

        # Liberar el video      
        video.release()

        return frame if success else None
else:
    frame_count = len([name for name in os.listdir(args.input) if os.path.isfile(os.path.join(args.input, name)) and name.endswith('.jpg')])
    def extract_frame(input, nframe=0, putText=True):
        global frame_count
        # Leer la imagen
        frame = cv.imread(os.path.join(input, f"{nframe+1:05d}.jpg"))
        if frame is not None and putText:
            cv.putText(frame, f"Frame {nframe}/{frame_count}", (10, 30), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
        return frame if frame is not None else None

frame = extract_frame(args.input, args.frame)

if frame is not None:
    point = np.array([-1, -1])
    points = []
    isin = True
    cv.namedWindow("window")
    cv.imshow("window", frame)
    def manejador(event, x, y, flags, param):
        global point, frame, isin
        output = frame
        if event == cv.EVENT_LBUTTONDOWN or event == cv.EVENT_MBUTTONDOWN:
            point = np.array([x, y])
            isin = (event == cv.EVENT_LBUTTONDOWN)
            # print(f"({point[0]}, {point[1]})")
            output = frame.copy()
            cv.circle(output, (point[0], point[1]), 2, (0, 255, 0) if isin else (0, 0, 255), 3)
            cv.putText(output, f"({point[0]}, {point[1]})", (point[0], point[1]), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
            cv.imshow("window", output)

    def manejador_tecla(key):
        global point, frame, isin
        if key & 0xFF == ord('s'):  # Escribir el punto seleccionado
            if point[0] >= 0 and point[1] >= 0:
                print(f"{args.frame}: ({point[0]}, {point[1]}) {'1' if isin else '0'}")
                points.append((args.frame, point[0], point[1], 1 if isin else 0))
        elif key & 0xFF == ord('a'):  # Retroceder un frame
            args.frame = (frame_count + args.frame - 1) % frame_count
            frame = extract_frame(args.input, args.frame)
            point = np.array([-1, -1])
            cv.imshow("window", frame)
        elif key & 0xFF == ord('d'):  # Avanzar un frame
            args.frame = (args.frame + 1) % frame_count
            frame = extract_frame(args.input, args.frame)
            point = np.array([-1, -1])
            cv.imshow("window", frame)
        elif key & 0xFF == ord('f'):  # Avanzar 10 frames
            args.frame = (args.frame + 10) % frame_count
            frame = extract_frame(args.input, args.frame)
            point = np.array([-1, -1])
            cv.imshow("window", frame)
        elif key & 0xFF == ord('r'):  # Retroceder 10 frames
            args.frame = (frame_count + args.frame - 10) % frame_count
            frame = extract_frame(args.input, args.frame)
            point = np.array([-1, -1])
            cv.imshow("window", frame)
    cv.setMouseCallback("window", manejador)
    key = cv.waitKey(0)
    while key & 0xFF != ord('q'):
        manejador_tecla(key)
        key = cv.waitKey(0)

    # Guardar los puntos en un archivo
    if len(points) > 0:
        with open(args.output, 'w') as f:
            f.write("frame x y isin\n")
            for p in points:
                f.write(f"{p[0]} {p[1]} {p[2]} {p[3]}\n")
        print(f"Puntos guardados en {args.output}")
    cv.destroyAllWindows()

else:
    print(f"Error al leer el frame {args.frame} del video {args.input}")