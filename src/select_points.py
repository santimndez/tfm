"""
Este script permite seleccionar puntos en un frame seleccionado de un vídeo y mostrar sus coordenadas haciendo clic.
Es útil para obtener las coordenadas de la bola y usarlas para el prompt de SAM2.
"""

import cv2 as cv
import argparse
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Ruta al vídeo de entrada", required=True)
parser.add_argument("-f", "--frame", help="Frame seleccionado del vídeo", type=int, default=0)

args = parser.parse_args()

def extract_frame(input, nframe=0):
    # Cargar el video
    video = cv.VideoCapture(input)
    
    video.set(cv.CAP_PROP_POS_FRAMES, nframe)
    success, frame = video.read()
    
    # Liberar el video      
    video.release()
    
    return frame if success else None

frame = extract_frame(args.input, args.frame)

if frame is not None:
    point = np.array([-1, -1])
    cv.namedWindow("window")
    cv.imshow("window", frame)
    def manejador(event, x, y, flags, param):
        global point, frame
        if event == cv.EVENT_LBUTTONDOWN:
            point = np.array([x, y])
            print(f"({point[0]}, {point[1]})")
            output = frame.copy()
            cv.circle(output, (point[0], point[1]), 2, (0, 0, 255), 3)
            cv.putText(output, f"({point[0]}, {point[1]})", (point[0], point[1]), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
            cv.imshow("window", output)
    # Create opencv window
    cv.setMouseCallback("window", manejador)
    cv.waitKey(0)

    cv.destroyAllWindows()
else:
    print(f"Error al leer el frame {args.frame} del video {args.input}")