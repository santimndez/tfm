import subprocess
import argparse
import numpy as np
import cv2 as cv
from ultralytics import YOLO
import pandas as pd
import time

def homog(points):# Convertir a coordenadas homogéneas (vector vertical)
    return np.vstack((points, np.ones((points.shape[1], 1))))

def inhomog(homog_points):  # Convertir a coordenadas cartesianas (vector vertical)
    return homog_points[:2, :] / homog_points[2, :][np.newaxis, :]

def line_p2p(h1, h2):
    # Calcular la ecuación de la recta que pasa por dos puntos dados en coordenadas homogéneas
    return np.cross(h1, h2).T

def get_epipolar_point(C, P):
    # Get epipolar point in camera 2 from camera 1 center C and projection matrix P of camera 2
    return P @ C

def distance_point_to_line(homog_point, line):
    distances = np.abs((line @ homog_point) / np.linalg.norm(line[:, :2], axis=1))
    return np.nan_to_num(distances, nan=0.0, posinf=0.0, neginf=0.0)

def skew(v):
    # Matriz antisimétrica a partir de un vector
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def get_projection_matrix(refs, points, camera_matrix, dist_coefs, ):
    # Calcular la matriz de la cámara usando solvePnP
    ok, rvec, tvec = cv.solvePnP(refs, points, camera_matrix, dist_coefs, flags=cv.SOLVEPNP_IPPE)
    if ok: # Refinamiento
        rvec, tvec = cv.solvePnPRefineVVS(refs, points, camera_matrix, dist_coefs, rvec, tvec)
    
    R, _ = cv.Rodrigues(rvec)
    M = camera_matrix @ np.hstack((R, tvec))

    C = np.linalg.solve(R, tvec)  # Centro de la cámara

    return M, R, tvec, C

def get_fundamental_matrix(M1, M2, C1):
    # Calcular la matriz fundamental F a partir de las matrices de proyección
    # F transforma puntos de la imagen 1 en líneas epipolares en la imagen 2
    P1 = M1[:, :3]
    C1 = -np.linalg.inv(P1) @ M1[:, 3]
    e2 = M2 @ np.hstack((C1, 1))
    F = skew(e2) @ M2 @ np.linalg.pinv(M1)
    # F = np.cross(e2, M2 @ np.linalg.pinv(M1))
    return F

# Argumentos de la línea de comandos
parser = argparse.ArgumentParser(description='Sincronizar dos vídeos estéreo minimizando el error de reproyección de la segmentación de la pelota')
parser.add_argument('-i', '--input', metavar='video_files', type=str, nargs=2, help='Rutas de los dos vídeos a sincronizar')
parser.add_argument('-o', '--output', metavar='output_file', type=str, help='Ruta del archivo de salida')
parser.add_argument('-m', '--model', metavar='segmentation_model', type=str, help='Ruta del archivo del modelo de segmentación YOLO de la pelota. Si no se proporciona, se deberá proporcionar el parámetro --positions', default=None)
parser.add_argument('-s', '--separated', action='store_true', help='Si se activa se guardará el primer vídeo desplazado en lugar de un vídeo compuesto over-under')
parser.add_argument('-k', metavar='camera_matrix', type=str, nargs=2, help='Ruta a los archivos con la matriz de cada cámara en formato .npy', default=None)
parser.add_argument('-r', '--ref_points', metavar='reference_points', type=str, help='Ruta al archivo con los puntos de referencia de cada vídeo (esquinas de la mesa) en formato csv con dos columnas separadas por tabulador', default=None)
parser.add_argument('--max_offset', metavar='max_offset', type=int, help='Desfase máximo a considerar en frames (por defecto 600)', default=600)
parser.add_argument('-p', '--positions', metavar='ball_positions', type=str, nargs=2, help='Ruta a los archivos csv con las posiciones de la pelota en cada frame. Si no se proporciona, se deberá proporcionar el parámetro --model', default=None)
parser.add_argument('--segment', action='store_true', help='Marca el centro de la pelota detectada en cada frame del vídeo de salida')
args = parser.parse_args()

video_files = args.input
output_file = args.output

if args.k is None: # Matriz de cámara por defecto
    camera_matrix = np.array([[1.69281160e+03, 0.00000000e+00, 6.81281341e+02],
                              [0.00000000e+00, 1.68036637e+03, 9.03147730e+02],
                              [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    K = [camera_matrix, camera_matrix]

# Esquinas de la mesa en el sistema de coordenadas de la cámara en cm
refs = np.array([[0, 0, 0], 
                 [152.5, 0, 0], 
                 [152.5, 274, 0], 
                 [0, 274, 0]], dtype=np.float32) 
points = pd.read_csv(args.ref_points, sep='\t', header=None).values.reshape((2, 4, 2)).astype(np.float32)

M = [np.zeros((3, 4)), np.zeros((3, 4))]  # Inicializar matrices de proyección
R = [np.zeros((3, 3)), np.zeros((3, 3))]  # Inicializar matrices de rotación
t = [np.zeros((3, 1)), np.zeros((3, 1))]  # Inicializar vectores de traslación
C = [np.zeros((3, 1)), np.zeros((3, 1))]  # Inicializar centros de cámara

M[0], R[0], t[0], C[0] = get_projection_matrix(refs, points[0, :], K[0], np.zeros((5,)))  # Matriz de proyección del vídeo 1
M[1], R[1], t[1], C[1] = get_projection_matrix(refs, points[1, :], K[1], np.zeros((5,)))  # Matriz de proyección del vídeo 2

F = get_fundamental_matrix(M[0], M[1], C[0])  # Matriz fundamental

def draw_trajectory(video_file, positions, output_file):
    cap = cv.VideoCapture(video_file)
    fps = cap.get(cv.CAP_PROP_FPS)
    w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(output_file, fourcc, fps, (w, h))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if positions[frame_idx, 2] > 0:
            x = int(positions[frame_idx, 0])
            y = int(positions[frame_idx, 1])
            cv.circle(frame, (x, y), 5, (0, 0, 255), -1)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

def adjust_video_offset(video_file_1, video_file_2, offset, output_file, separated=False):
    # Crea un vídeo sincronizado usando OpenCV
    cap1 = cv.VideoCapture(video_file_1)
    cap2 = cv.VideoCapture(video_file_2)
    fps = cap1.get(cv.CAP_PROP_FPS)
    w1 = int(cap1.get(cv.CAP_PROP_FRAME_WIDTH))
    h1 = int(cap1.get(cv.CAP_PROP_FRAME_HEIGHT))
    w2 = int(cap2.get(cv.CAP_PROP_FRAME_WIDTH))
    h2 = int(cap2.get(cv.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(output_file, fourcc, fps, (max(w1, w2) if not separated else w1, h1+h2 if not separated else h1))

    # Ajusta el desfase
    if offset<0:    # El primer vídeo empieza antes
        delay_frames = -int(offset)
        for _ in range(delay_frames):
            ret1, frame1 = cap1.read()
            if not ret1:
                break
            if separated:
                out.write(frame1)
            else:
                frame2 = np.zeros((h2, w2, 3), dtype=np.uint8)
                combined_frame = cv.vconcat([frame1, frame2])
                out.write(combined_frame)
    else:   # El segundo vídeo empieza antes
        delay_frames = int(offset)
        for _ in range(delay_frames):
            ret2, frame2 = cap2.read()
            if not ret2:
                break
            if separated:
                out.write(frame1)
            else:
                frame1 = np.zeros((h1, w1, 3), dtype=np.uint8)
                combined_frame = cv.vconcat([frame1, frame2])
                out.write(combined_frame)
    # Escribir los frames restantes
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 and not ret2:
            break
        if not ret1:
            frame1 = np.zeros((h1, w1, 3), dtype=np.uint8)
        if not ret2:
            frame2 = np.zeros((h2, w2, 3), dtype=np.uint8)
        if separated:
            out.write(frame1)
        else:
            combined_frame = cv.vconcat([frame1, frame2])
            out.write(combined_frame)
    cap1.release()
    cap2.release()
    out.release()

# Función para ajustar la sincronización de los vídeos según el desfase
def adjust_video_offset_ffmpeg(video_file_1, video_file_2, offset, output_file, separated=False):

    if not separated:
        # Usamos ffmpeg para ajustar el desfase y juntar los vídeos
        # El comando ffmpeg para desplazar el segundo vídeo
        command = [
            'ffmpeg',
            '-i', video_file_1,   # Primer vídeo
            '-itsoffset', str(offset),  # Desfase en segundos
            '-i', video_file_2,   # Segundo vídeo
            '-filter_complex', '[0:v][1:v]vstack=inputs=2[v]',  # Superponer los vídeos verticalmente
            '-map', '[v]',  # Mapear el vídeo compuesto
            '-y',            # Sobrescribir archivo de salida si ya existe        
            '-r', str(cv.VideoCapture(video_file_1).get(cv.CAP_PROP_FPS)), # Usar fps del primer vídeo
            '-vsync', '2',  # Evitar la duplicación de frames
            '-c:v', 'libx264',  # Codec de vídeo
            '-c:a', 'aac',      # Codec de audio
            '-y',               # Sobrescribir archivo de salida si ya existe
            output_file         # Archivo de salida
        ]
        subprocess.run(command)
    else:
        # Usamos ffmpeg para ajustar el desfase y guardar el primer vídeo desplazado
        command = [
            'ffmpeg',
            '-i', video_file_1,         # Primer vídeo
            '-itsoffset', str(offset),  # Desfase en segundos
            '-y',                       # Sobrescribir archivo de salida si ya existe        
            '-r', '60',                 # str(cv.VideoCapture(video_file_1).get(cv.CAP_PROP_FPS)), # Usar fps del primer vídeo
            '-vsync', '2',              # Evitar la duplicación de frames
            '-c:v', 'libx264',          # Codec de vídeo
            '-c:a', 'aac',              # Codec de audio
            '-y',                       # Sobrescribir archivo de salida si ya existe
            output_file                 # Archivo de salida
        ]
        subprocess.run(command)

def get_ball_positions(video_file, model):
    """ Obtener las posiciones de la pelota en cada frame del vídeo usando el modelo de segmentación.
    Devuelve una matriz de tamaño (num_frames, 3) con las coordenadas homogéneas de la pelota en cada frame
    o (0, 0, 0) si no se detecta la pelota en ese frame.
    """

    # Capturar el vídeo
    cap = cv.VideoCapture(video_file)
    w, h = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    ball_positions = np.zeros((int(cap.get(cv.CAP_PROP_FRAME_COUNT)), 3))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Realizar la segmentación
        results = model(frame, verbose=False)
        masks = results[0].masks
        masks = masks.data.cpu().numpy() if masks is not None else []
        if len(masks) > 0:
            # Suponiendo que la pelota es el mayor objeto segmentado
            mask_idx = np.argmax(np.sum(masks, axis=(1, 2)))
            mask = masks[mask_idx].squeeze().astype(np.uint8)
            mh, mw = mask.shape
            moments = cv.moments(mask)
            if moments["m00"] != 0:
                cX = int(w * moments["m10"] / (moments["m00"] * mw))
                cY = int(h * moments["m01"] / (moments["m00"] * mh))
                ball_positions[frame_idx, 0] = cX
                ball_positions[frame_idx, 1] = cY
                ball_positions[frame_idx, 2] = 1  # Coordenada homogénea
            else:
                ball_positions[frame_idx, 0] = 0
                ball_positions[frame_idx, 1] = 0
                ball_positions[frame_idx, 2] = 0  # Coordenada homogénea
        else:
            ball_positions[frame_idx, 0] = 0
            ball_positions[frame_idx, 1] = 0
            ball_positions[frame_idx, 2] = 0  # Coordenada homogénea
        
        frame_idx += 1

    cap.release()
    return ball_positions

def read_ball_positions(file_path):
    """ Leer las posiciones de la pelota desde un archivo CSV.
    Devuelve una matriz de tamaño (num_frames, 3) con las coordenadas homogéneas de la pelota en cada frame.
    """
    return pd.read_csv(file_path).values

def estimate_offset(F, homog_points1, homog_points2, max_offset=600):
    # Ampliar homog_points al mismo tamaño añadiendo ceros al final
    max_length = max(homog_points1.shape[1], homog_points2.shape[1])
    if homog_points1.shape[1] < max_length:
        pad_width = max_length - homog_points1.shape[1]
        homog_points1 = np.hstack((homog_points1, np.zeros((3, pad_width))))
    if homog_points2.shape[1] < max_length:
        pad_width = max_length - homog_points2.shape[1]
        homog_points2 = np.hstack((homog_points2, np.zeros((3, pad_width))))

    epilines = (F @ homog_points1).T

    offset_loss = np.zeros((2*max_offset+1,))
    
    for o in range(-max_offset, max_offset+1):
        # Desplazar las posiciones de la segunda cámara
        shifted_points = np.roll(homog_points2, o, axis=1)

        # Marcar como no válidos los puntos desplazados fuera del rango (la distancia contará 0)
        shifted_points[:, :max(0, o)] = 0                           
        shifted_points[:, shifted_points.shape[1]-max(0, -o):] = 0
        
        # Suma de las distancias de cada punto a la recta epipolar de la otra cámara proyectada
        offset_loss[max_offset+o] = np.sum(distance_point_to_line(shifted_points, epilines))
        offset_loss[max_offset+o] /= np.sum(epilines[:, 2] * shifted_points[2, :] > 0)  # Normalizar por el número de puntos válidos
    
    # TODO: Usar padding y FFT para calcular la distancia de forma más eficiente
    
    #  Devolver el desfase estimado
    return np.argmin(offset_loss) - max_offset, offset_loss

# Obtener la trayectoria de la pelota en cada vídeo
if args.model is None and args.positions is None:
    raise ValueError('Debe proporcionar un modelo de segmentación (--model) o las posiciones de la pelota en cada frame (--positions)')

model = YOLO(args.model, task='segment', verbose=False) if args.model is not None else None

tic = time.time()
homog_points1 = get_ball_positions(video_files[0], model).T if args.model is not None else read_ball_positions(args.positions[0]).T
toc = time.time()
print(f'Tiempo de obtención de posiciones del vídeo 1: {toc - tic:.2f} segundos')
print(f'Pelota detectada: {np.sum(homog_points1[2, :])}/{homog_points1.shape[1]} frames')

tic = time.time()
homog_points2 = get_ball_positions(video_files[1], model).T if args.model is not None else read_ball_positions(args.positions[1]).T
toc = time.time()
print(f'Tiempo de obtención de posiciones del vídeo 2: {toc - tic:.2f} segundos')
print(f'Pelota detectada: {np.sum(homog_points2[2, :])}/{homog_points2.shape[1]} frames')

# Estimar el desfase entre los dos vídeos
tic = time.time()
offset, offset_loss = estimate_offset(F, homog_points1, homog_points2, args.max_offset)
toc = time.time()
print(f'Calculando desfases {args.max_offset} - Tiempo: {toc - tic:.2f} segundos')
print(f'Desfase estimado: {offset} frames, pérdida: {offset_loss[offset + args.max_offset]}')

# Crear el vídeo sincronizado
if args.segment:
    # Dibujar la trayectoria de la pelota en cada vídeo
    draw_trajectory(video_files[0], homog_points1.T, 'video1_with_trajectory.mp4')
    draw_trajectory(video_files[1], homog_points2.T, 'video2_with_trajectory.mp4')
    video_files = ['video1_with_trajectory.mp4', 'video2_with_trajectory.mp4']

tic = time.time()
adjust_video_offset(video_files[0], video_files[1], -offset, output_file, args.separated)    # Crear el vídeo con los dos vídeos sincronizados
toc = time.time()
print(f'El vídeo final ha sido creado en: {output_file} ({toc - tic:.2f} segundos)')
