import subprocess
import argparse
import numpy as np
import cv2 as cv
from ultralytics import YOLO
import pandas as pd
import time

def homog(points):# Convertir a coordenadas homogéneas (vector vertical)
    return np.vstack((points, np.ones((1, points.shape[1]))))

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
parser.add_argument('-o', '--output', metavar='output_file', type=str, help='Ruta del archivo de salida', default=None)
parser.add_argument('-m', '--model', metavar='segmentation_model', type=str, help='Ruta del archivo del modelo de segmentación YOLO de la pelota. Si no se proporciona, se deberá proporcionar el parámetro --positions', default=None)
parser.add_argument('-s', '--separated', action='store_true', help='Si se activa se guardará el primer vídeo desplazado en lugar de un vídeo compuesto over-under')
parser.add_argument('-k', metavar='camera_matrix', type=str, nargs=2, help='Ruta a los archivos con la matriz de cada cámara en formato .npy', default=None)
parser.add_argument('-r', '--ref_points', metavar='reference_points', type=str, help='Ruta al archivo con los puntos de referencia de cada vídeo (esquinas de la mesa) en formato csv con dos columnas separadas por tabulador', default=None)
parser.add_argument('--max_offset', metavar='max_offset', type=int, help='Desfase máximo a considerar en frames (por defecto 600)', default=600)
parser.add_argument('-p', '--positions', metavar='ball_positions', type=str, nargs=2, help='Ruta a los archivos csv con las posiciones de la pelota en cada frame. Si no se proporciona, se deberá proporcionar el parámetro --model', default=None)
parser.add_argument('--segment', action='store_true', help='Marca el centro de la pelota detectada en cada frame del vídeo de salida')
parser.add_argument('--offset', metavar='offset', type=int, help='Desfase en frames a aplicar al primer vídeo (positivo para retrasarlo, negativo para adelantarlo). Si no se proporciona, se estimará automáticamente', default=None)
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
    # El offset en frames se aplica al primer vídeo (positivo para retrasarlo, negativo para adelantarlo)
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
                cX = w * moments["m10"] / (moments["m00"] * mw) # No es necesario que sea entero
                cY = h * moments["m01"] / (moments["m00"] * mh)
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

if args.model is not None: # Obtener las posiciones de la pelota usando el modelo de segmentación
    tic = time.time()
    homog_points1 = get_ball_positions(video_files[0], model).T
    pd.DataFrame(homog_points1.T).to_csv('positions_video1.csv', index=False)
    toc = time.time()
    print(f'Tiempo de obtención de posiciones del vídeo 1: {toc - tic:.2f} segundos')
    print(f'Pelota detectada: {np.sum(homog_points1[2, :])}/{homog_points1.shape[1]} frames')

    tic = time.time()
    homog_points2 = get_ball_positions(video_files[1], model).T
    pd.DataFrame(homog_points2.T).to_csv('positions_video2.csv', index=False)
    toc = time.time()
    print(f'Tiempo de obtención de posiciones del vídeo 2: {toc - tic:.2f} segundos')
    print(f'Pelota detectada: {np.sum(homog_points2[2, :])}/{homog_points2.shape[1]} frames')
else:   # Cargar las posiciones de la pelota
    homog_points1 = read_ball_positions(args.positions[0]).T
    homog_points2 = read_ball_positions(args.positions[1]).T

# Estimar el desfase entre los dos vídeos
if args.offset is not None:
    offset = args.offset
    print(f'Usando desfase proporcionado: {offset} frames')
else:
    tic = time.time()
    offset, offset_loss = estimate_offset(F, homog_points1, homog_points2, args.max_offset)
    offset = -offset
    toc = time.time()
    print(f'Calculando desfases {args.max_offset} - Tiempo: {toc - tic:.2f} segundos')
    print(f'Desfase estimado: {offset} frames, pérdida: {offset_loss[offset + args.max_offset]}')

# Crear el vídeo sincronizado
if args.segment:
    # Dibujar la trayectoria de la pelota en cada vídeo
    draw_trajectory(video_files[0], homog_points1.T, 'video1_with_trajectory.mp4')
    draw_trajectory(video_files[1], homog_points2.T, 'video2_with_trajectory.mp4')
    video_files = ['video1_with_trajectory.mp4', 'video2_with_trajectory.mp4']

# Crear el vídeo con los dos vídeos sincronizados
if args.output is not None:
    tic = time.time()
    adjust_video_offset(video_files[0], video_files[1], offset, args.output, args.separated)   
    toc = time.time()
    print(f'El vídeo final ha sido creado en: {args.output} ({toc - tic:.2f} segundos)')

# Calcular la posición 3D de la pelota y guardarla en un archivo CSV

nframes = min(homog_points1.shape[1]-max(0, offset), homog_points2.shape[1]-max(0, -offset))
homog_points1 = homog_points1[:, max(0, offset):max(0, offset) + nframes]
homog_points2 = homog_points2[:, max(0, -offset):max(0, -offset) + nframes]

# Obtener la trayectoria 3D de la pelota por triangulación
trajectory = cv.triangulatePoints(M[0], M[1], homog_points1[:2, :], homog_points2[:2, :])
trajectory /= trajectory[3, :][np.newaxis, :]  # Convertir a coordenadas cartesianas
trajectory = np.nan_to_num(trajectory, nan=0.0, posinf=0.0, neginf=0.0)
trajectory[:, homog_points1[2, :] * homog_points2[2, :] == 0] = 0  # Marcar como no válidos los puntos no detectados en alguna cámara
pd.DataFrame(trajectory.T).to_csv('ball_trajectory_3D.csv', index=False)

# Rellenar los huecos en la trayectoria mediante interpolación lineal
prev = -1
last = -1
for i in range(trajectory.shape[1]):
    if trajectory[3, i] == 0.0:
        if prev == -1:
            prev = i
    else:
        if prev != -1:
            if last != -1:
                trajectory[:, prev:i] = trajectory[:, last][:, np.newaxis] + ((trajectory[:, i]-trajectory[:, last])[:, np.newaxis] @ np.arange(1, i-last)[np.newaxis, :]/(i-last))
            else:
                trajectory[:, prev:i] = trajectory[:, i][:, np.newaxis]
            prev = -1
        last = i

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
ax.set_xlim3d(-TABLE_WIDTH/2, 1.5*TABLE_WIDTH)
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

ani = FuncAnimation(fig, update, frames=trajectory.shape[1], fargs = (ball, trajectory), interval=32, blit=False, repeat=True)

# ani.save('ball_trajectory_3D.mp4', writer='ffmpeg', fps=60)
plt.show()
