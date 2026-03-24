import argparse
import numpy as np
import cv2 as cv
from ultralytics import YOLO
import pandas as pd
import time

from utils.binocular import *
from utils.ball_detection import *
from utils.sync import *
from utils.trajectory_segmentation import *

# Argumentos de la línea de comandos
parser = argparse.ArgumentParser(description="""Muestra la detección de la pelota en dos vídeos y las líneas epipolares correspondientes.
Controles: 
    - Tecla 'a': retroceder un frame en el vídeo (izquierda)
    - Tecla 's': avanzar un frame en el vídeo (izquierda)
    - Tecla 'z': retroceder 10 frames en el vídeo (izquierda)
    - Tecla 'x': avanzar 10 frames en el vídeo (izquierda)
    - Tecla 'p': para dibujar polilínea con los puntos considerados al calcular la correspondencia local
    - Tecla 'q': salir del programa
""")

parser.add_argument('-i', '--input', metavar='video_file', type=str, help='Ruta del vídeo')
parser.add_argument('-m', '--model', metavar='segmentation_model', type=str, help='Ruta del archivo del modelo de segmentación YOLO de la pelota. Si no se proporciona, se deberá proporcionar el parámetro --positions', default=None)
parser.add_argument('-k', metavar='camera_matrix', type=str, help='Ruta al archivo con la matriz de cámara en formato .npy', default=None)
parser.add_argument('-r', '--ref_points', metavar='reference_points', type=str, help='Ruta al archivo con los puntos de referencia de cada vídeo (esquinas de la mesa) en formato csv con dos columnas separadas por tabulador', default=None)
parser.add_argument('-p', '--positions', metavar='ball_positions', type=str, help='Ruta al archivo csv con las posiciones de la pelota en cada frame. Si no se proporciona, se deberá proporcionar el parámetro --model', default=None)
parser.add_argument('-s', '--segments', type=str, help='Ruta al archivo csv con la segmentación de la trayectoria', default=None)
args = parser.parse_args()

video_files = args.input
zoom = 1.0  # Factor de zoom inicial

K = None  # Matrices de cámara
distortion = np.zeros((5,))  # Coeficientes de distorsión

if args.k is None: # Matriz de cámara por defecto
    camera_matrix = np.array([[1.69281160e+03, 0.00000000e+00, 6.81281341e+02],
                              [0.00000000e+00, 1.68036637e+03, 9.03147730e+02],
                              [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    K = camera_matrix
    distortion = np.array([0.13225064, -0.27888641, -0.00721176,  0.03960687, -0.17568582])
else:
    with open(args.k, 'r') as file:
        lines = file.readlines()
        K = np.fromstring(''.join(lines[:3]), sep=' ').reshape((3, 3))
        distortion = np.fromstring(lines[3], sep=' ')

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

fps = cv.VideoCapture(args.input).get(cv.CAP_PROP_FPS)

M = np.zeros((3, 4))  # Inicializar matriz de proyección
R = np.zeros((3, 3))  # Inicializar matriz de rotación
t = np.zeros((3, 1))  # Inicializar vector de traslación
C = np.zeros((3, 1))  # Inicializar centro de cámara

M, R, t, C = get_projection_matrix(refs, points[0, :], K, distortion)  # Matriz de proyección del vídeo

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
else:   # Cargar las posiciones de la pelota
    homog_points1 = read_ball_positions(args.positions).T

# Obtener la segmentación de la pelota en el vídeo
segments = None

if args.segments is not None:
    segments = pd.read_csv(args.segments, header=None).values
else:
    timestamps = get_timestamps(args.input) # if args.vfr else np.arange(homog_points1.shape[1]) / fps[0]
    tic = time.time()
    homog_points1_scaled = homog_points1[:2, homog_points1[2, :] != 0].T / np.array(get_frame_shape(video_files[0]))
    segments = dp_segmenter(timestamps[homog_points1[2, :] != 0], homog_points1_scaled)
    toc = time.time()
    print(f'Segmentación de la trayectoria ({toc - tic:.2f} segundos)')

def get_frame_count(input):
    video = cv.VideoCapture(input)
    frame_count = int(video.get(cv.CAP_PROP_FRAME_COUNT))
    video.release()
    return frame_count

# Append element to segments array
segments = np.append(segments, [get_frame_count(args.input)])

def draw_polyline(frame, points, color=(0, 255, 255), thickness=2, radius=3):
    if points.ndim == 1:
        points = points[:, np.newaxis]
    if thickness!=0:
        for i in range(points.shape[1]-1):
            pt1 = (int(round(points[0, i])), int(round(points[1, i])))
            pt2 = (int(round(points[0, i+1])), int(round(points[1, i+1])))
            frame = cv.line(frame, pt1, pt2, color, thickness)
    for i in range(points.shape[1]):
        pt = (int(round(points[0, i])), int(round(points[1, i])))
        frame = cv.circle(frame, pt, radius, color, -1)
    return frame

def transform_frame(frame, nframe=0, frame_count=100, point=None, line=None, putText=True, mframe=None, distance=None, poly=None, intersection=None, rebounds=None):
    global zoom
    res = frame.copy()
    if putText:
        res = cv.putText(res, f"Frame {nframe}/{frame_count}", (10, 30), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
        if mframe is not None:
            res = cv.putText(res, f"->{mframe}", (410, 30), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
        if distance is not None:
            res = cv.putText(res, f"d={distance:.2f} px", (910, 30), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
    if line is not None:
        res = draw_line(res, line, color=(255, 0, 0), thickness=2)
    if point is not None:
        res = cv.circle(res, (int(point[0]), int(point[1])), 7, (0, 255, 0), -1)
    if poly is not None:
        res = draw_polyline(res, poly[:2, :], color=(0, 255, 255), thickness=2)
    if rebounds is not None:
        res = draw_polyline(res, rebounds[:2, :], color=(255, 0, 255), thickness=0, radius=5)
    if intersection is not None:
        res = draw_polyline(res, intersection[:2, :], color=(0, 0, 255), thickness=0)
    if zoom != 1.0: 
        width, height = int(res.shape[1] * zoom), int(res.shape[0] * zoom)
        res = cv.resize(res, (width, height), interpolation=cv.INTER_CUBIC)

    return res

video = cv.VideoCapture(args.input)
frame_count = int(video.get(cv.CAP_PROP_FRAME_COUNT))
idx = 0  # Índice del frame actual
old_idx = 0  # Índices de los frames anteriores
frame = None
_, frame = video.read()

poly = False  # Para mostrar polígono considerado al calcular la correspondencia local
thres = 10
fps_ratio = 1 # fps[1] / fps[0]
offset = -40 # Desfase a priori en frames (entero)

def get_frame():
    # Obtiene el frame del vídeo correspondiente. Evita repetir la carga del vídeo.
    global idx, video, frame_count, frame, old_idx
    nframe = (idx + frame_count) % frame_count
    if nframe>old_idx:
        for f in range(old_idx+1, nframe+1):
            success, frame = video.read()
            if not success:
                return None
    elif nframe<old_idx:
        video = cv.VideoCapture(args.input)
        for f in range(nframe+1):
            success, frame = video.read()
            if not success:
                return None
    idx = nframe
    old_idx = nframe
    return frame

def get_tframe(point=None, line=None, putText=True, mframe=None, distance=None, poly=None, intersection=None, rebounds=None):
    # Obtiene el frame transformado del vídeo correspondiente
    temp_frame = get_frame()
    if temp_frame is not None:
        return transform_frame(temp_frame, idx, frame_count, point, line, putText, mframe, distance, poly, intersection, rebounds)
    else:
        return None

if frame is not None:
    cv.namedWindow("window")
    tframe0 = get_tframe()
    cv.imshow("window", tframe0)
    if args.segments is not None:
        def manejador_tecla(key):
            global frame, idx, zoom, frame_count, correspondences, poly, fps_ratio, offset, thres
            # Vídeo 1
            if key & 0xFF == ord('s'):  # Avanzar un frame en el vídeo 1
                idx = (idx + 1) % frame_count
            elif key & 0xFF == ord('a'):  # Retroceder un frame en el vídeo 1
                idx = (idx - 1 + frame_count) % frame_count
            elif key & 0xFF == ord('x'):  # Avanzar 10 frames en el vídeo 1
                idx = (idx + 10) % frame_count
            elif key & 0xFF == ord('z'):  # Retroceder 10 frames en el vídeo 1
                idx = (idx - 10 + frame_count) % frame_count
            # Zoom
            elif key & 0xFF == ord('i'):  # Zoom in
                zoom += 0.1
            elif key & 0xFF == ord('o'):  # Zoom out
                zoom = max(0.1, zoom - 0.1)
            elif key & 0xFF == ord('p'):  # Toggle polilínea
                poly = not poly
            if key & 0xFF in [ord('a'), ord('s'), ord('d'), ord('f'), ord('z'), ord('x'), ord('c'), ord('v'), ord('i'), ord('o')] \
                    or (key & 0xFF == ord('p') and poly):
                point0 = homog_points1[:, idx] if homog_points1[2, idx] != 0 else None
                polyline0, intersection0, rebounds0 = None, None, None
                rebounds = None
                if poly: # Obtener la polilínea del segmento actual
                    nseg = np.searchsorted(segments, idx, side='right')
                    r = segments[nseg+1] if nseg+1 < len(segments) else homog_points1.shape[1]
                    l = segments[nseg-2] if nseg-2 >= 0 else 0
                    polyline0 = np.zeros((3, r-l))
                    valid = np.zeros((r-l,), dtype=bool)
                    for d in range(r-l):
                        polyline0[:, d] = homog_points1[:, l+d] if l+d >=0 and l+d < homog_points1.shape[1] else np.zeros((3,))
                        valid[d] = homog_points1[2, l+d] > 0 if l+d >=0 and l+d < homog_points1.shape[1] else False
                    rebounds0 = []
                    if valid[segments[nseg-1]-l]:
                        rebounds0.append(polyline0[:, segments[nseg-1]-l])
                    if valid[segments[nseg]-l]:
                        rebounds0.append(polyline0[:, segments[nseg]-l])
                    rebounds = np.vstack(rebounds0).T if len(rebounds0)>0 else None
                    polyline0 = polyline0[:, valid] if np.sum(valid)>0 else None
                tframe0 = get_tframe(point=point0, line=None, mframe=None, distance=None, poly=polyline0, rebounds=rebounds)
                cv.imshow("window", tframe0)

    key = cv.waitKey(0)
    while key & 0xFF != ord('q'):
        manejador_tecla(key)
        key = cv.waitKey(0)
else:
    print(f"Error al leer el frame {args.frame} del video {args.input}")

video.release()
cv.destroyAllWindows()