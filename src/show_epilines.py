import argparse
import numpy as np
import cv2 as cv
from ultralytics import YOLO
import pandas as pd
import time

from utils.binocular import *
from utils.ball_detection import *
from utils.sync import *

# Argumentos de la línea de comandos
parser = argparse.ArgumentParser(description="""Muestra la detección de la pelota en dos vídeos y las líneas epipolares correspondientes.
Controles: 
    - Tecla 'a': retroceder un frame en el vídeo 1 (izquierda)
    - Tecla 's': avanzar un frame en el vídeo 1 (izquierda)
    - Tecla 'd': retroceder un frame en el vídeo 2 (derecha)
    - Tecla 'f': avanzar un frame en el vídeo 2 (derecha)
    - Tecla 'z': retroceder 10 frames en el vídeo 1 (izquierda)
    - Tecla 'x': avanzar 10 frames en el vídeo 1 (izquierda)
    - Tecla 'c': retroceder 10 frames en el vídeo 2 (derecha)
    - Tecla 'v': avanzar 10 frames en el vídeo 2 (derecha)
    - Tecla 'p': para dibujar polilínea con los puntos considerados al calcular la correspondencia local
    - Tecla 'q': salir del programa
""")
parser.add_argument('-i', '--input', metavar='video_files', type=str, nargs=2, help='Rutas de los dos vídeos a sincronizar')
parser.add_argument('-m', '--model', metavar='segmentation_model', type=str, help='Ruta del archivo del modelo de segmentación YOLO de la pelota. Si no se proporciona, se deberá proporcionar el parámetro --positions', default=None)
parser.add_argument('-k', metavar='camera_matrix', type=str, nargs=2, help='Ruta a los archivos con la matriz de cada cámara en formato .npy', default=None)
parser.add_argument('-r', '--ref_points', metavar='reference_points', type=str, help='Ruta al archivo con los puntos de referencia de cada vídeo (esquinas de la mesa) en formato csv con dos columnas separadas por tabulador', default=None)
parser.add_argument('-p', '--positions', metavar='ball_positions', type=str, nargs=2, help='Ruta a los archivos csv con las posiciones de la pelota en cada frame. Si no se proporciona, se deberá proporcionar el parámetro --model', default=None)
parser.add_argument('-c', '--correspondences', type=str, help='Ruta al archivo csv de donde se obtienen las correspondencias de frames', default=None)
args = parser.parse_args()

video_files = args.input
zoom = 1.0  # Factor de zoom inicial

K = [None, None]  # Matrices de cámara
distortion = [None, None]  # Coeficientes de distorsión

if args.k is None: # Matriz de cámara por defecto
    camera_matrix = np.array([[1.69281160e+03, 0.00000000e+00, 6.81281341e+02],
                              [0.00000000e+00, 1.68036637e+03, 9.03147730e+02],
                              [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    K = [camera_matrix, camera_matrix]
    distortion = [np.array([0.13225064, -0.27888641, -0.00721176,  0.03960687, -0.17568582])] * 2
else:
    for i in range(2):
        with open(args.k[i], 'r') as file:
            lines = file.readlines()
            K[i] = np.loadtxt(''.join(lines[:3]))
            distortion[i] = np.loadtxt(lines[3])

# Esquinas de la mesa en el sistema de coordenadas de la cámara en cm
TABLE_WIDTH = 152.5
TABLE_LENGTH = 274.0
TABLE_HEIGHT = 76.0
NET_HEIGHT = 15.25
NET_EXTRA_WIDTH = 15.25
BALL_RADIUS = 2.01

refs = np.array([[0, 0, 0], 
                 [TABLE_WIDTH, 0, 0], 
                 [TABLE_WIDTH, TABLE_LENGTH, 0], 
                 [0, TABLE_LENGTH, 0]], dtype=np.float32) # Esquinas en orden antihorario
points = pd.read_csv(args.ref_points, header=None).values.reshape((2, 4, 2)).astype(np.float32)

fps = [cv.VideoCapture(video_files[0]).get(cv.CAP_PROP_FPS),
       cv.VideoCapture(video_files[1]).get(cv.CAP_PROP_FPS)]

M = [np.zeros((3, 4)), np.zeros((3, 4))]  # Inicializar matrices de proyección
R = [np.zeros((3, 3)), np.zeros((3, 3))]  # Inicializar matrices de rotación
t = [np.zeros((3, 1)), np.zeros((3, 1))]  # Inicializar vectores de traslación
C = [np.zeros((3, 1)), np.zeros((3, 1))]  # Inicializar centros de cámara

M[0], R[0], t[0], C[0] = get_projection_matrix(refs, points[0, :], K[0], np.zeros((5,)))  # Matriz de proyección del vídeo 1
M[1], R[1], t[1], C[1] = get_projection_matrix(refs, points[1, :], K[1], np.zeros((5,)))  # Matriz de proyección del vídeo 2

F = get_fundamental_matrix(M[0], M[1], C[0])  # Matriz fundamental

# Obtener la trayectoria de la pelota en cada vídeo
if args.model is None and args.positions is None:
    raise ValueError('Debe proporcionar un modelo de segmentación (--model) o las posiciones de la pelota en cada frame (--positions)')

model = YOLO(args.model, task='segment', verbose=False) if args.model is not None else None

if args.model is not None: # Obtener las posiciones de la pelota usando el modelo de segmentación
    tic = time.time()
    homog_points1 = get_ball_positions(video_files[0], model).T
    pd.DataFrame(homog_points1.T).to_csv('positions_video1.csv', index=False, header=None)
    toc = time.time()
    print(f'Trayectoria de la pelota en el vídeo 1 guardada en positions_video1.csv')
    print(f'Tiempo de obtención de posiciones del vídeo 1: {toc - tic:.2f} segundos')
    print(f'Pelota detectada: {np.sum(homog_points1[2, :])}/{homog_points1.shape[1]} frames')

    tic = time.time()
    homog_points2 = get_ball_positions(video_files[1], model).T
    pd.DataFrame(homog_points2.T).to_csv('positions_video2.csv', index=False, header=None)
    toc = time.time()
    print(f'Trayectoria de la pelota en el vídeo 2 guardada en positions_video2.csv')
    print(f'Tiempo de obtención de posiciones del vídeo 2: {toc - tic:.2f} segundos')
    print(f'Pelota detectada: {np.sum(homog_points2[2, :])}/{homog_points2.shape[1]} frames')
else:   # Cargar las posiciones de la pelota
    homog_points1 = read_ball_positions(args.positions[0]).T
    homog_points2 = read_ball_positions(args.positions[1]).T

epilines = (F @ homog_points1).T
epilines1 = (F.T @ homog_points2).T

def get_frame_count(input):
    video = cv.VideoCapture(input)
    frame_count = int(video.get(cv.CAP_PROP_FRAME_COUNT))
    video.release()
    return frame_count

def draw_polyline(frame, points, color=(0, 255, 255), thickness=2):
    for i in range(points.shape[1]-1):
        pt1 = (int(round(points[0, i])), int(round(points[1, i])))
        pt2 = (int(round(points[0, i+1])), int(round(points[1, i+1])))
        frame = cv.line(frame, pt1, pt2, color, thickness)
    for i in range(points.shape[1]):
        pt = (int(round(points[0, i])), int(round(points[1, i])))
        frame = cv.circle(frame, pt, 3, color, -1)
    return frame

def extract_frame(input, nframe=0, frame_count=100, point=None, line=None, putText=True, mframe=None, distance=None, poly=None):
    global zoom
    # Cargar el video
    video = cv.VideoCapture(input)
    # video.set(cv.CAP_PROP_POS_FRAMES, nframe)
    # success, frame = video.read()
    for _ in range(nframe + 1):
        success, frame = video.read()
        if not success:
            break
    if success and putText:
        frame = cv.putText(frame, f"Frame {nframe}/{frame_count}", (10, 30), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
        if mframe is not None:
            frame = cv.putText(frame, f"->{mframe}", (410, 30), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
        if distance is not None:
            frame = cv.putText(frame, f"d={distance:.2f} px", (910, 30), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
    if success and point is not None:
        frame = cv.circle(frame, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1)
    if success and line is not None:
        frame = draw_line(frame, line, color=(255, 0, 0), thickness=2)
    if poly is not None:
        frame = draw_polyline(frame, poly[:2, :], color=(0, 255, 255), thickness=2)
    if zoom != 1.0: 
        width, height = int(frame.shape[1] * zoom), int(frame.shape[0] * zoom)
        frame = cv.resize(frame, (width, height), interpolation=cv.INTER_CUBIC)
    # Liberar el video      
    video.release()

    return frame if success else None

frame_count = [get_frame_count(args.input[0]), get_frame_count(args.input[1])]
frame = [extract_frame(args.input[0], 0, frame_count[0]), extract_frame(args.input[1], 0, frame_count[1])]
idx = [0, 0]  # Índices de los frames actuales
poly = False  # Para mostrar polígono considerado al calcular la correspondencia local
thres = 10
fps_ratio = fps[1] / fps[0]
offset = -40 # Desfase a priori en frames (entero)

if frame[0] is not None and frame[1] is not None:
    cv.namedWindow("window")
    cv.imshow("window", np.hstack((frame[0], frame[1])))
    if args.correspondences is None:
        def manejador_tecla(key):
            global frame, idx, frame_count, zoom
            # Vídeo 1
            if key & 0xFF == ord('s'):  # Avanzar un frame en el vídeo 1
                idx[0] = (idx[0] + 1) % frame_count[0]
            elif key & 0xFF == ord('a'):  # Retroceder un frame en el vídeo 1
                idx[0] = (idx[0] - 1 + frame_count[0]) % frame_count[0]
            elif key & 0xFF == ord('x'):  # Avanzar 10 frames en el vídeo 1
                idx[0] = (idx[0] + 10) % frame_count[0]
            elif key & 0xFF == ord('z'):  # Retroceder 10 frames en el vídeo 1
                idx[0] = (idx[0] - 10 + frame_count[0]) % frame_count[0]
            # Vídeo 2
            if key & 0xFF == ord('f'):  # Avanzar un frame en el vídeo 2
                idx[1] = (idx[1] + 1) % frame_count[1]
            elif key & 0xFF == ord('d'):  # Retroceder un frame en el vídeo 1
                idx[1] = (idx[1] - 1 + frame_count[1]) % frame_count[1]
            elif key & 0xFF == ord('v'):  # Avanzar 10 frames en el vídeo 1
                idx[1] = (idx[1] + 10) % frame_count[1]
            elif key & 0xFF == ord('c'):  # Retroceder 10 frames en el vídeo 1
                idx[1] = (idx[1] - 10 + frame_count[1]) % frame_count[1]
            # Zoom
            elif key & 0xFF == ord('i'):  # Zoom in
                zoom += 0.1
            elif key & 0xFF == ord('o'):  # Zoom out
                zoom = max(0.1, zoom - 0.1)
            if key & 0xFF in [ord('a'), ord('s'), ord('d'), ord('f'), ord('z'), ord('x'), ord('c'), ord('v'), ord('i'), ord('o')]:
                point0 = homog_points1[:, idx[0]] if homog_points1[2, idx[0]] != 0 else None
                epiline0 = epilines1[idx[1], :] if homog_points2[2, idx[1]] != 0 else None
                epiline = epilines[idx[0], :] if homog_points1[2, idx[0]] != 0 else None
                point1 = homog_points2[:, idx[1]] if homog_points2[2, idx[1]] != 0 else None
                distance = distance_point_to_line(homog_points2[:, idx[1]][:, np.newaxis], epiline[np.newaxis, :])[0] if (epiline is not None and point1 is not None) else None
                frame[0] = extract_frame(args.input[0], idx[0], frame_count[0], point=point0, line=epiline0, distance=distance)
                frame[1] = extract_frame(args.input[1], idx[1], frame_count[1], point=point1, line=epiline)
                cv.imshow("window", np.hstack((frame[0], frame[1])))
    else:
        df = pd.read_csv(args.correspondences, na_values=['nan'])
        # get dictionary of correspondences
        correspondences = dict(zip(df['frame'], df['offset']))
        def manejador_tecla(key):
            global frame, idx, frame_count, zoom, correspondences, poly, fps_ratio, offset, thres
            # Vídeo 1
            if key & 0xFF == ord('s'):  # Avanzar un frame en el vídeo 1
                idx[0] = (idx[0] + 1) % frame_count[0]
            elif key & 0xFF == ord('a'):  # Retroceder un frame en el vídeo 1
                idx[0] = (idx[0] - 1 + frame_count[0]) % frame_count[0]
            elif key & 0xFF == ord('x'):  # Avanzar 10 frames en el vídeo 1
                idx[0] = (idx[0] + 10) % frame_count[0]
            elif key & 0xFF == ord('z'):  # Retroceder 10 frames en el vídeo 1
                idx[0] = (idx[0] - 10 + frame_count[0]) % frame_count[0]
            
            corr = idx[0] + correspondences.get(idx[0], np.nan)
            if key & 0xFF in [ord('a'), ord('s'), ord('x'), ord('z')]:
                if ~np.isnan(corr) and int(corr)>=0 and int(corr)<frame_count[1]:
                    idx[1] = int(round(corr))
            # Vídeo 2
            if key & 0xFF == ord('f'):  # Avanzar un frame en el vídeo 2
                idx[1] = (idx[1] + 1) % frame_count[1]
            elif key & 0xFF == ord('d'):  # Retroceder un frame en el vídeo 1
                idx[1] = (idx[1] - 1 + frame_count[1]) % frame_count[1]
            elif key & 0xFF == ord('v'):  # Avanzar 10 frames en el vídeo 1
                idx[1] = (idx[1] + 10) % frame_count[1]
            elif key & 0xFF == ord('c'):  # Retroceder 10 frames en el vídeo 1
                idx[1] = (idx[1] - 10 + frame_count[1]) % frame_count[1]
            # Zoom
            elif key & 0xFF == ord('i'):  # Zoom in
                zoom += 0.1
            elif key & 0xFF == ord('o'):  # Zoom out
                zoom = max(0.1, zoom - 0.1)
            elif key & 0xFF == ord('p'):  # Toggle polilínea
                poly = not poly
            if key & 0xFF in [ord('a'), ord('s'), ord('d'), ord('f'), ord('z'), ord('x'), ord('c'), ord('v'), ord('i'), ord('o')] \
                    or (key & 0xFF == ord('p') and poly):
                point0 = homog_points1[:, idx[0]] if homog_points1[2, idx[0]] != 0 else None
                epiline0 = epilines1[idx[1], :] if homog_points2[2, idx[1]] != 0 else None
                epiline = epilines[idx[0], :] if homog_points1[2, idx[0]] != 0 else None
                point1 = homog_points2[:, idx[1]] if homog_points2[2, idx[1]] != 0 else None
                distance = distance_point_to_line(homog_points2[:, idx[1]][:, np.newaxis], epiline[np.newaxis, :])[0] if (epiline is not None and point1 is not None) else None
                polyline0 = None
                if poly:
                    polyline0 = np.zeros((3, 2*thres+1))
                    valid = np.zeros((2*thres+1,), dtype=bool)
                    for d in range(-thres, thres+1):
                        polyline0[:, d+thres] = homog_points1[:, idx[0] + d] if idx[0] + d >=0 and idx[0] + d < homog_points1.shape[1] else np.zeros((3,))
                        valid[d+thres] = homog_points1[2, idx[0] + d] > 0 if idx[0] + d >=0 and idx[0] + d < homog_points1.shape[1] else False
                    polyline0 = polyline0[:2, valid] if np.sum(valid)>0 else None
                frame[0] = extract_frame(args.input[0], idx[0], frame_count[0], point=point0, line=epiline0, mframe=corr, distance=distance, poly=polyline0)
                polyline = None
                if poly:
                    polyline = np.zeros((3, 2*thres+1))
                    valid = np.zeros((2*thres+1,), dtype=bool)
                    corr_idx = int(round(idx[0] * fps_ratio + offset))
                    for d in range(-thres, thres+1):
                        polyline[:, d+thres] = homog_points2[:, corr_idx + d] if corr_idx + d >=0 and corr_idx + d < homog_points2.shape[1] else np.zeros((3,))
                        valid[d+thres] = homog_points2[2, corr_idx + d] > 0 if corr_idx + d >=0 and corr_idx + d < homog_points2.shape[1] else False
                    polyline = polyline[:2, valid] if np.sum(valid)>0 else None
                frame[1] = extract_frame(args.input[1], idx[1], frame_count[1], point=point1, line=epiline, poly=polyline)
                cv.imshow("window", np.hstack((frame[0], frame[1])))

    key = cv.waitKey(0)
    while key & 0xFF != ord('q'):
        manejador_tecla(key)
        key = cv.waitKey(0)
else:
    print(f"Error al leer el frame {args.frame} del video {args.input}")
