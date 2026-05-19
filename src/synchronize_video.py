import argparse
import numpy as np
import cv2 as cv
from ultralytics import YOLO
import pandas as pd
import time

from utils.binocular import *
from utils.ball_detection import *
from utils.sync import *
from utils.filter import estimate_trajectory
from utils.tt_animation import ball_animation
from utils.trajectory_segmentation import dp_segmenter, DEFAULT_LOSS
PROFILING = False

# Argumentos de la línea de comandos
parser = argparse.ArgumentParser(description='Sincronizar dos vídeos estéreo minimizando el error de reproyección de la segmentación de la pelota')
parser.add_argument('-i', '--input', metavar='video_files', type=str, nargs=2, help='Rutas de los dos vídeos a sincronizar')
parser.add_argument('-o', '--output', metavar='output_folder', type=str, help='Ruta de la carpeta en la que se guardarán archivos de salida. Por defecto, será output_DATETIME, con DATETIME la fecha y hora actuales', default=None)
parser.add_argument('-m', '--model', metavar='segmentation_model', type=str, help='Ruta del archivo del modelo de segmentación YOLO de la pelota. Si no se proporciona, se deberá proporcionar el parámetro --positions', default=None)
parser.add_argument('-k', metavar='camera_matrix', type=str, nargs=2, help='Ruta a los archivos con la matriz de cada cámara en formato .npy', default=None)
parser.add_argument('-r', '--ref_points', metavar='reference_points', type=str, help='Ruta al archivo con los puntos de referencia de cada vídeo (esquinas de la mesa) en formato csv con dos columnas separadas por tabulador', default=None)
parser.add_argument('--max_offset', metavar='max_offset', type=int, help='Desfase máximo a considerar en frames (por defecto 600)', default=600)
parser.add_argument('-p', '--positions', metavar='ball_positions', type=str, nargs=2, help='Ruta a los archivos csv con las posiciones de la pelota en cada frame. Si no se proporciona, se deberá proporcionar el parámetro --model', default=None)
parser.add_argument('--offset', metavar='offset', type=float, help='Desfase en segundos a aplicar al primer vídeo (positivo para retrasarlo, negativo para adelantarlo). Si no se proporciona, se estimará automáticamente', default=None)
parser.add_argument('-s', '--sync', action='store_true', help='Si se activa se guardará un vídeo sincronizado over-under')
parser.add_argument('--segment', action='store_true', help='Si se activa se guardarán los vídeos segmentados con la pelota detectada')
parser.add_argument('--separated', action='store_true', help='Si se activa se guardará el primer vídeo desplazado en lugar de un vídeo compuesto over-under')
# parser.add_argument('-d', '--save_detection', action='store_true', help='Si se activa se guardarán las posiciones de la pelota detectadas en cada frame en archivos CSV')
parser.add_argument('--save_correspondences', metavar='correspondences', type=str, help='Ruta donde se guardarán las correspondencias de frames obtenidas en la refinación del desfase', default=None)
parser.add_argument('-vfr', action='store_true', help='Indica si los vídeos son de frame rate variable (VFR) y se deben usar las marcas de tiempo para la sincronización')
args = parser.parse_args()

video_files = args.input

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
aux = pd.read_csv(args.ref_points, header=None).values.reshape((2, 8, 2)).astype(np.float32)
points = aux[:, :4, :]
vertical_points = aux[:, 4:, :]

camera_center = [np.array(get_frame_shape(video))/2 for video in args.input]
K = [None, None]  # Matrices de cámara
distortion = [None, None]  # Coeficientes de distorsión

if args.k is None: 
    # Matriz de cámara por defecto
    camera_matrix = np.array([[1.69281160e+03, 0.00000000e+00, 6.81281341e+02],
                              [0.00000000e+00, 1.68036637e+03, 9.03147730e+02],
                              [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    K = [camera_matrix, camera_matrix]
    distortion = [np.array([0.13225064, -0.27888641, -0.00721176,  0.03960687, -0.17568582])] * 2
    # K = [get_camera_matrix(points[i, :, :], np.roll(points[i, :, :], 1, axis=0), vertical_points[i, :, :], camera_center[i]) for i in range(2)]
    # distortion = [np.zeros((5,))] * 2
else:
    for i in range(2):
        with open(args.k[i], 'r') as file:
            lines = file.readlines()
            K[i] = np.fromstring(''.join(lines[:3]), sep=' ').reshape((3, 3))
            distortion[i] = np.fromstring(lines[3], sep=' ')

print("Matrices de calibración:")
for i in range(2):
    print(f"Vídeo {i + 1}:")
    print("Matriz de cámara:")
    print(K[i])
    print("Coeficientes de distorsión:")
    print(distortion[i])
    print("")

# Calcular fps
videos = [cv.VideoCapture(video) for video in args.input]
fps = [video.get(cv.CAP_PROP_FPS) for video in videos]
for video in videos:
    video.release()

M = [np.zeros((3, 4)), np.zeros((3, 4))]  # Inicializar matrices de proyección
R = [np.zeros((3, 3)), np.zeros((3, 3))]  # Inicializar matrices de rotación
t = [np.zeros((3, 1)), np.zeros((3, 1))]  # Inicializar vectores de traslación
C = [np.zeros((3, 1)), np.zeros((3, 1))]  # Inicializar centros de cámara

M[0], R[0], t[0], C[0] = get_projection_matrix(refs, points[0, :], K[0], np.zeros((5,)))  # Matriz de proyección del vídeo 1
M[1], R[1], t[1], C[1] = get_projection_matrix(refs, points[1, :], K[1], np.zeros((5,)))  # Matriz de proyección del vídeo 2

F = get_fundamental_matrix(M[0], M[1], C[0])  # Matriz fundamental

# Crear carpeta output
if args.output is None:
    args.output = f'output_{ time.strftime("%Y%m%d-%H%M%S")}'
os.makedirs(args.output, exist_ok=True)
filenames = [os.path.splitext(os.path.basename(video))[0] for video in args.input] # Nombre de los vídeos de entrada sin extensión

# Obtener la trayectoria de la pelota en cada vídeo
if args.model is None and args.positions is None:
    raise ValueError('Debe proporcionar un modelo de segmentación (--model) o las posiciones de la pelota en cada frame (--positions)')

if args.model is not None: # Obtener las posiciones de la pelota usando el modelo de segmentación
    tic = time.time()
    model = YOLO(args.model, task='segment', verbose=False)
    homog_points1 = get_ball_positions(video_files[0], model).T
    toc = time.time()
    print(f'Tiempo de obtención de trayectoria de la pelota en el vídeo 1: {toc - tic:.2f} segundos')
    print(f'Pelota detectada: {np.sum(homog_points1[2, :])}/{homog_points1.shape[1]} frames')

    tic = time.time()
    homog_points2 = get_ball_positions(video_files[1], model).T
    toc = time.time()
    print(f'Tiempo de obtención de trayectoria de la pelota en el vídeo 2: {toc - tic:.2f} segundos')
    print(f'Pelota detectada: {np.sum(homog_points2[2, :])}/{homog_points2.shape[1]} frames')

    # Guardar detección
    detection_output = [os.path.join(args.output, f'positions_{filenames[i]}.csv') for i in range(len(filenames))]
    pd.DataFrame(homog_points1.T).to_csv(detection_output[0], index=False, header=None)
    pd.DataFrame(homog_points2.T).to_csv(detection_output[1], index=False, header=None)
    print(f'Las detecciones han sido guardadas en: {detection_output}')

elif args.positions is not None:   # Cargar las posiciones de la pelota
    homog_points1 = read_ball_positions(args.positions[0]).T
    homog_points2 = read_ball_positions(args.positions[1]).T    

# Estimar el desfase entre los dos vídeos
timestamps1 = get_timestamps(args.input[0]) if args.vfr else np.arange(homog_points1.shape[1]) / fps[0]
timestamps2 = get_timestamps(args.input[1]) if args.vfr else np.arange(homog_points2.shape[1]) / fps[1]

if args.offset is not None:
    refined_offset_seconds = args.offset
    fps_ratio = fps[1] / fps[0]
    print(f'Usando desfase proporcionado: {refined_offset_seconds} s')
else:
    tic = time.time()
    offset, offset_loss = estimate_offset(F, homog_points1[:, :1800], homog_points2[:, :1800], args.max_offset)
    offset = -offset    # Cambiar el signo para que sea negativo si hay que adelantar el primer vídeo
    offset_seconds = offset / fps[1]
    # offset_seconds = refine_offset_timestamps(F, homog_points1[:, :1800], homog_points2[:, :1800], timestamps1[:1800], timestamps2[:1800], 0, thres=args.max_offset)
    # offset_loss = np.nan * np.ones((2 * args.max_offset + 1,))
    # offset = int(offset_seconds*fps[0])

    toc = time.time()
    print(f'Calculando desfases {args.max_offset} - Tiempo: {toc - tic:.2f} segundos')
    print(f'Desfase estimado: {offset} frames ({offset_seconds:.4f} s), pérdida: {offset_loss[-offset + args.max_offset]:.2f}')
    # Refinar el desfase estimado
    tic = time.time()
    if args.vfr: # Si los vídeos son de frame rate variable se utilizan los timestamps
        fps_ratio = fps[1] / fps[0]
        refined_offset_seconds = refine_offset_timestamps(F, homog_points1, homog_points2, timestamps1, timestamps2, offset_seconds, thres=10, save_correspondences=args.save_correspondences)
        refined_offset = refined_offset_seconds * fps[1]
    else:
        # refined_offset, fps_ratio = refine_offset_correlation(F, homog_points1, homog_points2, offset, fps_ratio=fps[1]/fps[0], thres=10, save_correspondences=args.save_correspondences)
        refined_offset, fps_ratio = refine_offset(F, homog_points1, homog_points2, offset, fps_ratio=fps[1]/fps[0], thres=10, save_correspondences=args.save_correspondences)
        # refined_offset, fps_ratio = refine_offset(F, homog_points1, homog_points2, offset, fps_ratio=1.0, thres=10, save_correspondences=args.save_correspondences)
        refined_offset_seconds = refined_offset / fps[1]
    toc = time.time()
    print(f'Refinamiento del desfase - Tiempo: {toc - tic:.2f} segundos')
    print(f'Desfase refinado: {refined_offset:.2f} frames ({refined_offset_seconds:.4f} s), ratio fps: {fps_ratio:.6f}')

print(f'FPS vídeo 1: {fps[0]}, vídeo 2: {fps[1]}, ratio: {fps[1]/fps[0]:.6f}')

# Guardar vídeo sincronizado
if args.sync:
    output_video = os.path.join(args.output, 'sync.mp4')
    tic = time.time()
    adjust_video_offset_ffmpeg(video_files[0], video_files[1], refined_offset_seconds, output_video, args.separated)
    toc = time.time()
    print(f'El vídeo sincronizado ha sido creado en: {output_video} ({toc - tic:.2f} segundos)')

# Guardar segmentación
if args.segment:
    # Dibujar la trayectoria de la pelota en cada vídeo
    segmentation_output = [os.path.join(args.output, f'segmented_{filenames[i]}.mp4') for i in range(len(filenames))]
    tic = time.time()
    draw_trajectory(video_files[0], homog_points1.T, segmentation_output[0])
    draw_trajectory(video_files[1], homog_points2.T, segmentation_output[1])
    toc = time.time()
    print(f'La segmentación ha sido guardada en: {segmentation_output} ({toc - tic:.2f} segundos)')

APPLY_FILTERING = True
TRAJECTORY_SEGMENTATION = False

trajectory_output = os.path.join(args.output, '3D_ball_trajectory.csv')
if not APPLY_FILTERING:
    # Calcular la posición 3D de la pelota y guardarla en un archivo CSV
    homog_points1, homog_points2 = sync_positions(homog_points1, homog_points2, offset, fps_ratio=1.0)
    trajectory = triangulate_ball(homog_points1, homog_points2, M[0], M[1])
    pd.DataFrame(trajectory.T).to_csv(trajectory_output, index=False, header=None)
    print(f'Trayectoria 3D de la pelota guardada en {trajectory_output}')

    trajectory = interpolate_missing_positions(trajectory, trajectory, trajectory[3, :] > 0)

    # Crea una animación (vídeo) 3D de la trayectoria de la pelota sobre la mesa
    ball_animation(trajectory)
else:
    # Segmentación de la trayectoria
    if TRAJECTORY_SEGMENTATION:
        tic = time.time()
        # Escalar la trayectoria
        homog_points1_scaled = homog_points1[:2, homog_points1[2, :] != 0].T / np.array(get_frame_shape(video_files[0]))
        homog_points2_scaled = homog_points2[:2, homog_points2[2, :] != 0].T / np.array(get_frame_shape(video_files[1]))
        segments = dp_segmenter(timestamps1[homog_points1[2, :] != 0], homog_points1_scaled)
        segments2 = dp_segmenter(timestamps2[homog_points2[2, :] != 0], homog_points2_scaled)
        toc = time.time()
        print(f'Segmentación de la trayectoria ({toc - tic:.2f} segundos)')
        print(f'Segmentos obtenidos: {len(segments)} en el vídeo 1, {len(segments2)} en el vídeo 2')
        # Guardar los segmentos en un archivo CSV
        # pd.DataFrame(segments).to_csv(os.path.join(args.output, f"ball_trajectory_segments_L{DEFAULT_LOSS}.csv"), index=False, header=None)
        # pd.DataFrame(segments2).to_csv(os.path.join(args.output, f"ball_trajectory_segments2_L{DEFAULT_LOSS}.csv"), index=False, header=None)
        # print(f"Segmentos guardados en: {os.path.join(args.output, f'ball_trajectory_segments_L{DEFAULT_LOSS}.csv')} y {os.path.join(args.output, f'ball_trajectory_segments2_L{DEFAULT_LOSS}.csv')}")
    
    # Estimar la trayectoria de la pelota utilizando un filtro UKF
    timestamps1 += refined_offset_seconds # Aplicar el offset a los timestamps
    
    if PROFILING: # Probar solo un trozo de vídeo
        homog_points1 = homog_points1[:, (timestamps1 >= 10) & (timestamps1 <= 12)]
        homog_points2 = homog_points2[:, (timestamps2 >= 10) & (timestamps2 <= 12)]
        timestamps1 = timestamps1[(timestamps1 >= 10) & (timestamps1 <= 12)]-10
        timestamps2 = timestamps2[(timestamps2 >= 10) & (timestamps2 <= 12)]-10

    smoothed_X, t, rebounds, bounces, _, X = estimate_trajectory(homog_points1, homog_points2, timestamps1, timestamps2, M[0], M[1], get_frame_shape(video_files[0]), get_frame_shape(video_files[1]))

    print("Estimaciones UKF")
    print(X[:, :100].T)
    print("Suavizado")
    print(smoothed_X[:, :100].T)   
    print("Rebotes")
    print(rebounds)

    pd.DataFrame(np.hstack((smoothed_X.T, t[:, np.newaxis]))).to_csv(trajectory_output, index=False, header=None)
    print(f'Trayectoria 3D de la pelota guardada en {trajectory_output}')

    ball_animation(smoothed_X, t)