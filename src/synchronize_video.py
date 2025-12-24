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
parser = argparse.ArgumentParser(description='Sincronizar dos vídeos estéreo minimizando el error de reproyección de la segmentación de la pelota')
parser.add_argument('-i', '--input', metavar='video_files', type=str, nargs=2, help='Rutas de los dos vídeos a sincronizar')
parser.add_argument('-o', '--output', metavar='output_file', type=str, help='Ruta del archivo de salida', default=None)
parser.add_argument('-m', '--model', metavar='segmentation_model', type=str, help='Ruta del archivo del modelo de segmentación YOLO de la pelota. Si no se proporciona, se deberá proporcionar el parámetro --positions', default=None)
parser.add_argument('-s', '--separated', action='store_true', help='Si se activa se guardará el primer vídeo desplazado en lugar de un vídeo compuesto over-under')
parser.add_argument('-k', metavar='camera_matrix', type=str, nargs=2, help='Ruta a los archivos con la matriz de cada cámara en formato .npy', default=None)
parser.add_argument('-r', '--ref_points', metavar='reference_points', type=str, help='Ruta al archivo con los puntos de referencia de cada vídeo (esquinas de la mesa) en formato csv con dos columnas separadas por tabulador', default=None)
parser.add_argument('--max_offset', metavar='max_offset', type=int, help='Desfase máximo a considerar en frames (por defecto 600)', default=600)
parser.add_argument('-p', '--positions', metavar='ball_positions', type=str, nargs=2, help='Ruta a los archivos csv con las posiciones de la pelota en cada frame. Si no se proporciona, se deberá proporcionar el parámetro --model', default=None)
parser.add_argument('--segment', action='store_true', help='Marca el centro de la pelota detectada en cada frame del vídeo de salida')
parser.add_argument('--offset', metavar='offset', type=float, help='Desfase en frames a aplicar al primer vídeo (positivo para retrasarlo, negativo para adelantarlo). Si no se proporciona, se estimará automáticamente', default=None)
parser.add_argument('--save_correspondences', type=str, help='Ruta donde se guardarán las correspondencias de frames obtenidas en la refinación del desfase')
args = parser.parse_args()

video_files = args.input

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

# Estimar el desfase entre los dos vídeos
if args.offset is not None:
    offset = args.offset
    fps_ratio = fps[1] / fps[0]
    refined_offset_seconds = offset / fps[0]
    print(f'Usando desfase proporcionado: {offset} frames')
else:
    tic = time.time()
    offset, offset_loss = estimate_offset(F, homog_points1[:, :1800], homog_points2[:, :1800], args.max_offset)
    offset = -offset    # Cambiar el signo para que sea negativo si hay que adelantar el primer vídeo
    toc = time.time()
    print(f'Calculando desfases {args.max_offset} - Tiempo: {toc - tic:.2f} segundos')
    print(f'Desfase estimado: {offset} frames, pérdida: {offset_loss[-offset + args.max_offset]}')
    # Refinar el desfase estimado
    tic = time.time()
    refined_offset, fps_ratio = refine_offset(F, homog_points1, homog_points2, offset, fps_ratio=fps[1]/fps[0], thres=10, save_correspondences=args.save_correspondences)
    toc = time.time()
    refined_offset_seconds = refined_offset / fps[0]
    print(f'Refinamiento del desfase - Tiempo: {toc - tic:.2f} segundos')
    print(f'Desfase refinado: {refined_offset:.2f} frames ({refined_offset_seconds:.2f} s), ratio fps: {fps_ratio:.6f}')
    print(f'FPS vídeo 1: {fps[0]}, vídeo 2: {fps[1]}')
    offset = int(round(refined_offset))

# Crear el vídeo sincronizado
if args.segment:
    # Dibujar la trayectoria de la pelota en cada vídeo
    draw_trajectory(video_files[0], homog_points1.T, 'video1_with_trajectory.mp4')
    draw_trajectory(video_files[1], homog_points2.T, 'video2_with_trajectory.mp4')
    video_files = ['video1_with_trajectory.mp4', 'video2_with_trajectory.mp4']

# Crear el vídeo con los dos vídeos sincronizados
if args.output is not None:
    tic = time.time()
    if False and args.offset is not None:
        adjust_video_offset(video_files[0], video_files[1], int(round(offset)), args.output, args.separated)  
    else:
        adjust_video_offset_ffmpeg(video_files[0], video_files[1], refined_offset_seconds, args.output, args.separated)
    toc = time.time()
    print(f'El vídeo final ha sido creado en: {args.output} ({toc - tic:.2f} segundos)')

# Calcular la posición 3D de la pelota y guardarla en un archivo CSV
homog_points1, homog_points2 = sync_positions(homog_points1, homog_points2, offset, fps_ratio)
trajectory = triangulate_ball(homog_points1, homog_points2, M[0], M[1])
pd.DataFrame(trajectory.T).to_csv('ball_trajectory_3D.csv', index=False, header=None)
print(f'Trayectoria 3D de la pelota guardada en ball_trajectory_3D.csv')

# Rellenar los huecos en la trayectoria mediante interpolación lineal
# first = trajectory.shape[1]
# last = -1
# for i in range(trajectory.shape[1]):
#     if trajectory[3, i] > 0.0:
#         first = i
#         break
# for i in range(trajectory.shape[1]-1, -1, -1):
#     if trajectory[3, i] > 0.0:
#         last = i
#         break
# trajectory[:, first:last+1] = interpolate_missing_positions(trajectory[:, first:last+1], trajectory[3, first:last+1] > 0)
trajectory = interpolate_missing_positions(trajectory, trajectory, trajectory[3, :] > 0)

# Crea una animación (vídeo) 3D de la trayectoria de la pelota sobre la mesa
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax.plot(trajectory[0, :], trajectory[1, :], trajectory[2, :], label='Trayectoria de la pelota')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Trayectoria 3D de la pelota')
ax.set_xlim3d(-TABLE_LENGTH/2, 1.5*TABLE_LENGTH)
ax.set_ylim3d(-TABLE_LENGTH/2, 1.5*TABLE_LENGTH)
ax.set_zlim3d(-TABLE_HEIGHT, 3*TABLE_HEIGHT)


# Dibujar superficie de la mesa
xx, yy = np.meshgrid([0, TABLE_WIDTH], [0, TABLE_LENGTH])
zz = np.full_like(xx, 0)
ax.plot_surface(xx, yy, zz, color='blue', alpha=0.5)

# Dibujar líneas de la mesa
ax.plot([0, TABLE_WIDTH], [0, 0], [0, 0], color='white')
ax.plot([TABLE_WIDTH, TABLE_WIDTH], [0, TABLE_LENGTH], [0, 0], color='white')
ax.plot([TABLE_WIDTH, 0], [TABLE_LENGTH, TABLE_LENGTH], [0, 0], color='white')
ax.plot([0, 0], [TABLE_LENGTH, 0], [0, 0], color='white')
ax.plot([TABLE_WIDTH/2, TABLE_WIDTH/2], [0, TABLE_LENGTH], [0, 0], color='white')

# Dibujar la red
x_net = np.linspace(-NET_EXTRA_WIDTH, TABLE_WIDTH + NET_EXTRA_WIDTH, 40)
z_net = np.linspace(0,  NET_HEIGHT, 10)
XX, ZZ = np.meshgrid(x_net, z_net)
YY = np.full_like(XX, TABLE_LENGTH/2)

# Dibujar la red como una malla
ax.plot_surface(XX, YY, ZZ, color='black', alpha=0.3)

# Punto de la pelota
ball, = ax.plot([], [], [], 'o', color='orange', markersize=5)

def update(frame, ball, trajectory):
    x = trajectory[0, frame]
    y = trajectory[1, frame]
    z = trajectory[2, frame]
    ball.set_data([x], [y])
    ball.set_3d_properties([z])
    return ball,

ani = FuncAnimation(fig, update, frames=trajectory.shape[1], fargs = (ball, trajectory), interval=16, blit=False, repeat=True)

# ani.save('ball_trajectory_3D.mp4', writer='ffmpeg', fps=60)
plt.show()
