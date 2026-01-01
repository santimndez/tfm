import cv2 as cv
import numpy as np
import subprocess
from scipy.interpolate import make_interp_spline
from .binocular import distance_point_to_line, line_p2p
import pandas as pd

def get_frame_shape(video_file):
    """
    Obtiene el tamaño de frame (w, h) de un vídeo.
    :param video_file: Ruta al archivo de vídeo.
    :return: Tupla con el tamaño de frame (w, h).
    """
    cap = cv.VideoCapture(video_file)
    w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return w, h

def adjust_video_offset(video_file_1, video_file_2, offset, output_file, separated=False):
    """
    Sincroniza dos vídeos usando OpenCV. 
    El offset en frames se aplica al primer vídeo (positivo para retrasarlo, negativo para adelantarlo).
    Si separated es True, solo se guarda el primer vídeo desplazado.
    """
    cap1 = cv.VideoCapture(video_file_1)
    cap2 = cv.VideoCapture(video_file_2)
    fps = cap1.get(cv.CAP_PROP_FPS)
    fps2 = cap2.get(cv.CAP_PROP_FPS)
    # print(f'FPS vídeo 1: {fps}, vídeo 2: {fps2}')
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
def adjust_video_offset_ffmpeg(video_file_1, video_file_2, offset, output_file, separated=False, verbose=False):
    """
    Sincroniza dos vídeos usando ffmpeg. 
    El offset en segundos se aplica al primer vídeo (positivo para retrasarlo, negativo para adelantarlo).
    Si separated es True, solo se guarda el primer vídeo desplazado.
    """
    if not separated:
        # Usamos ffmpeg para ajustar el desfase y juntar los vídeos
        filter_complex = (
            f'[0:v]setpts=PTS-STARTPTS+({offset})/TB[v0];'
            '[1:v]setpts=PTS-STARTPTS[v1];'
            '[v0][v1]vstack=inputs=2[v]'
        )
        # El comando ffmpeg para desplazar el segundo vídeo
        command = [
            'ffmpeg',
            '-i', video_file_1,   # Primer vídeo
            '-i', video_file_2,   # Segundo vídeo
            '-filter_complex', filter_complex,  # Superponer los vídeos verticalmente
            '-map', '[v]',  # Mapear el vídeo compuesto
            '-map', '0:a?',      # Audio del primer vídeo (si existe)
            # '-r', str(cv.VideoCapture(video_file_1).get(cv.CAP_PROP_FPS)), # Usar fps del primer vídeo
            # '-vsync', '2',  # Evitar la duplicación de frames
            '-fps_mode', 'vfr',
            '-c:v', 'libx264',  # Codec de vídeo
            '-c:a', 'aac',      # Codec de audio
            '-y',               # Sobrescribir archivo de salida si ya existe
            output_file         # Archivo de salida
        ]
    else:
        # Usamos ffmpeg para ajustar el desfase y guardar el primer vídeo desplazado
        filter_complex = f'[0:v]setpts=PTS-STARTPTS+({offset})/TB[v]'
        command = [
            'ffmpeg',
            '-i', video_file_1,         # Primer vídeo
            '-filter_complex', filter_complex,  # Desplazar el vídeo
            '-y',                       # Sobrescribir archivo de salida si ya existe        
            # '-r', '60',                 # str(cv.VideoCapture(video_file_1).get(cv.CAP_PROP_FPS)), # Usar fps del primer vídeo
            # '-vsync', '2',              # Evitar la duplicación de frames
            '-c:v', 'libx264',          # Codec de vídeo
            '-c:a', 'aac',              # Codec de audio
            '-y',                       # Sobrescribir archivo de salida si ya existe
            output_file                 # Archivo de salida
        ]
    # Verbosity: global options
    if not verbose:
        command[1:1] = ['-hide_banner', '-loglevel', 'error', '-nostdin']
        run_kwargs = {'stdout': subprocess.DEVNULL, 'stderr': subprocess.DEVNULL, 'check': True}
    else:
        # verbose: dejar que ffmpeg imprima a stdout/stderr y no suprimir errores (ayuda en depuración)
        command[1:1] = ['-nostdin']
        run_kwargs = {'check': True}

    subprocess.run(command, **run_kwargs)

# Función para ajustar la sincronización de los vídeos según el desfase
def adjust_video_offset_ffmpeg(video_file_1, video_file_2, offset, output_file, separated=False, verbose=False):
    """
    Sincroniza dos vídeos usando ffmpeg. 
    El offset en segundos se aplica al primer vídeo (positivo para retrasarlo, negativo para adelantarlo).
    Si separated es True, solo se guarda el primer vídeo desplazado.
    :param verbose: si True muestra la salida de ffmpeg, si False reduce la verbosidad.
    """
    # Construir filter_complex
    if not separated:
        filter_complex = (
            f'[0:v]setpts=PTS-STARTPTS+({offset})/TB[v0];'
            '[1:v]setpts=PTS-STARTPTS[v1];'
            '[v0][v1]vstack=inputs=2[v]'
        )
        command = [
            'ffmpeg',
            '-y',
            '-i', video_file_1,
            '-i', video_file_2,
            '-filter_complex', filter_complex,
            '-map', '[v]',
            '-map', '0:a?',
            '-fps_mode', 'vfr',
            '-c:v', 'libx264',
            '-c:a', 'aac',
            output_file
        ]
    else:
        filter_complex = f'[0:v]setpts=PTS-STARTPTS+({offset})/TB[v]'
        command = [
            'ffmpeg',
            '-y',
            '-i', video_file_1,
            '-filter_complex', filter_complex,
            '-c:v', 'libx264',
            '-c:a', 'aac',
            output_file
        ]

    # Ajustar verbosidad: insertar opciones globales justo después de 'ffmpeg'
    if not verbose:
        command[1:1] = ['-hide_banner', '-loglevel', 'error', '-nostdin']
        run_kwargs = {'stdout': subprocess.DEVNULL, 'stderr': subprocess.DEVNULL, 'check': True}
    else:
        # verbose: dejar que ffmpeg imprima a stdout/stderr y no suprimir errores (ayuda en depuración)
        command[1:1] = ['-nostdin']
        run_kwargs = {'check': True}

    subprocess.run(command, **run_kwargs)


def estimate_offset(F, homog_points1, homog_points2, max_offset=600):
    """
    Estima el desfase (offset) entre dos secuencias de puntos 2D usando la matriz fundamental F.
    El offset es de la secuencia 2 respecto a la secuencia 1: positivo si la cámara 2 va adelantada respecto a la cámara 1.
    El offset se estima como el que minimiza la suma de las distancias de homog_points2 a las líneas epipolares correspondientes a homog_points1.
    :param F: Matriz fundamental entre las dos cámaras. Transforma puntos de la imagen 1 en líneas epipolares en la imagen 2.
    :param homog_points1: Puntos 2D en coordenadas homogéneas de la cámara 1 (3xN).
    :param homog_points2: Puntos 2D en coordenadas homogéneas de la cámara 2 (3xM).
    :param max_offset: Desfase máximo a considerar (en número de frames).
    :return: Desfase estimado (en número de frames) y array con la pérdida para cada desfase.
    """
    max_offset = min(max_offset, homog_points1.shape[1]-1, homog_points2.shape[1]-1)
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
        offset_loss[max_offset+o] = np.sum(np.abs(np.nan_to_num(distance_point_to_line(shifted_points, epilines), nan=0.0, posinf=0.0, neginf=0.0)))
        offset_loss[max_offset+o] /= np.sum(epilines[:, 2] * shifted_points[2, :] > 0)  # Normalizar por el número de puntos válidos
        
    #  Devolver el desfase estimado
    return np.argmin(offset_loss) - max_offset, offset_loss

def intersect_line_polyline(line, polyline, all=False):
    """
    Calcula la intersección entre una recta y una polilínea en coordenadas homogéneas.
    :param line: Ecuación de la recta en coordenadas homogéneas (3,).
    :param polyline: Vértices de la polilínea en coordenadas homogéneas (3, N).
    :param t: Parámetro con los valores de tiempo asociados a cada vértice de la polilínea (N,). 
              Debe ser una secuencia estrictamente monótona.
              Si no se proporciona, se asume t = [0, 1, ..., N-1].
    :param all: Si es True, se buscan todas las intersecciones. Si es False, se busca solo la más cercana al centro de la polilínea.
    :return: (I, idx, t), con:
             Punto de intersección I en coordenadas homogéneas (3,).
             Índice idx del segmento de la polilínea.
             Parámetro t en [0, 1] que indica la posición relativa en el segmento.
             I = polyline[:, idx] * (1 - t) + polyline[:, idx + 1] * t.
             Si no hay intersección, devuelve None.
    """

    distance_sign = np.sign(line @ polyline)
    sign_changes = np.where(np.diff(distance_sign))[0]
    if len(sign_changes) == 0:
        return None  # No hay intersección
    if not all:
        # Get sign change closest to the middle of the polyline
        mid_idx = polyline.shape[1] / 2
        idx = sign_changes[np.argmin(np.abs(sign_changes - mid_idx))]
    else:
        idx = sign_changes
    
    # Get intersection point
    p1 = polyline[:, idx]
    p2 = polyline[:, idx + 1]
    line_segment = line_p2p(p1, p2)
    intersection = np.cross(line, line_segment, axisc=0)
    intersection /= intersection[2]
    t = (np.linalg.norm(intersection[:2] - p1[:2], axis=0) / np.linalg.norm(p2[:2] - p1[:2], axis=0))
    return intersection, idx, t

def intersect_line_polycurve(line, polyline, t=None, all=False):
    """
    Calcula la intersección entre una recta y una curva poligonal en coordenadas homogéneas.
    :param line: Ecuación de la recta en coordenadas homogéneas (3,).
    :param polyline: Vértices de la polilínea en coordenadas homogéneas (3, N).
    :param t: Parámetro con los valores de tiempo asociados a cada vértice de la polilínea (N,). 
              Debe ser una secuencia estrictamente monótona.
              Si no se proporciona, se asume t = [0, 1, ..., N-1].
    :return: (I, u), con:
             Punto de intersección I en coordenadas homogéneas (3,).
             Parámetro temporal u en el que se alcanza la intersección en la curva poligonal.
             I = polyline(u) = polyline[:, idx] * (1 - v) + polyline[:, idx + 1] * v, 
             con idx tal que t[idx]<=u<=t[idx+1], v = (u - t[idx]) / (t[idx+1]-t[idx]).
             Si no hay intersección, devuelve None.
    """
    aux = intersect_line_polyline(line, polyline, all=all)
    if aux is None:
        return None
    else:
        intersection, idx, v = aux
        u = t[idx]+ (t[idx+1]-t[idx])*v if t is not None else idx + v
        return intersection, u

def interpolate_missing_positions(homog_points, out=None, valid_mask=None):
    """
    Interpolar las posiciones faltantes en homog_points usando interpolación lineal.
    :param valid_mask: array booleano que indica qué posiciones son válidas (True) o faltantes (False).
    :param homog_points: array (D, N) de puntos en coordenadas homogéneas. Tener coordenada homogénea 1 en las posiciones válidas y 0 en las no válidas.
    :param out: array donde se guarda el resultado, permitiendo hacer la operación in-place (cuidado con el tipo del array out, la conversión puede afectar al resultado). 
    Devuelve homog_points con las posiciones faltantes interpoladas.
    """
    if valid_mask is None:
        valid_mask = homog_points[-1, :] > 0
    valid_indices = np.where(valid_mask)[0]
    missing_indices = np.where(~valid_mask)[0]
    if len(valid_indices) < 1: # < 2:
        return out  # No hay suficientes puntos para interpolar
    if out is None:
        out = homog_points.copy().astype(np.float64)
    for i in range(homog_points.shape[0]-1):  # Interpolar coordenadas no homogéneas
        out[i, missing_indices] = np.interp(missing_indices, valid_indices, homog_points[i, valid_indices])
    out[-1, missing_indices] = 1  # Establecer la coordenada homogénea a 1 para los puntos interpolados
    return out

def piecewise_linear_interpolate(t, points, t_new):
    """ 
    Interpolación lineal por tramos de una serie de puntos de dimensión D.
    :param t: array 1D con los tiempos originales de los puntos (N,).
    :param points: array (D, N) con los puntos a interpolar.
    :param t_new: array 1D con los nuevos tiempos donde interpolar (M,).
    :return: array (D, M) con los puntos interpolados en t_new.
    """
    spline = make_interp_spline(t, points.T, k=1)
    return spline(t_new).T    

def get_rebounds(points, thres=-0.5):
    """
    Obtiene cambios de dirección bruscos de los puntos 2D en coordenadas homogéneas.
    :param points: array (3xN) de puntos en coordenadas homogéneas.
    :param thres: umbral para detectar rebotes (entre 0 y -1) en función de la similitud coseno de la velocidad previa y posterior en cada instante.
    :return: array (N,) de rebotes.
    """
    v1_minus, v1_plus = np.zeros_like(points), np.zeros_like(points)
    v1_minus[:, :] = np.diff(points, prepend=0)
    v1_plus = np.roll(v1_minus, shift=-1, axis=1)
    v1_cos = np.einsum('ij,ij->j', v1_minus, v1_plus)/ (np.linalg.norm(v1_minus, axis=0) * np.linalg.norm(v1_plus, axis=0) + 1e-8)
    rebounds = np.where(v1_cos < thres)[0]
    return rebounds

def refine_offset(F, homog_points1, homog_points2, offset, fps_ratio=1.0, thres=10, min_points=2, save_correspondences=None):
    """
    Refina la estimación del desfase (offset) entre dos secuencias de puntos 2D usando la matriz fundamental F.
    El offset es de la secuencia 1 respecto a la secuencia 2: positivo si la cámara 1 va adelantada respecto a la cámara 2.
    :param F: Matriz fundamental entre las dos cámaras. Transforma puntos de la imagen 1 en líneas epipolares en la imagen 2.
    :param homog_points1: Puntos 2D en coordenadas homogéneas de la cámara 1 (3xN).
    :param homog_points2: Puntos 2D en coordenadas homogéneas de la cámara 2 (3xM).
    :param offset: Desfase inicial estimado (en número de frames).
    :param fps_ratio: Ratio de fps entre las dos cámaras (fps2 / fps1).
    :param thres: Número de frames a cada lado para buscar correspondencias locales.
    :param min_points: Número mínimo de puntos válidos para considerar una correspondencia local como válida.
                       Si no se proporciona, se 2. Como mínimo se tomará 2.
    :param save_correspondences: Ruta para guardar las correspondencias locales (opcional).
    :return: Nueva estimación del offset (en número de frames) y ratio de fps estimado.
    """
    epilines = (F @ homog_points1).T
    local_offset = np.zeros(homog_points1.shape[1], dtype=np.float32) * np.nan              # Array de offsets locales     
    permutation = np.array([thres + (-1)**(d % 2)*((d+1)//2) for d in range(2*thres+1)])    # permutation array for later use
                                                                                            # order [thres, thres-1, thres+1, thres-2, thres+2, ...]
    min_points = max(2, min_points)
    o = int(round(offset))
    count = 0
    # Estimar la correspondencia local para cada frame del vídeo 1
    start_idx = max(0, -o)+thres
    end_idx = min(homog_points1.shape[1], homog_points2.shape[1]-max(0, o))-thres
    for idx in range(start_idx, end_idx):
        if epilines[idx, 2] == 0: # Línea epipolar no existente
            # local_offset[idx] = np.nan
            continue
        # Calcular la intersección de la recta epipolar con el polígono de puntos válidos
        polyline = np.zeros((3,2*thres+1))
        valid = np.zeros((2*thres+1,), dtype=bool)
        corr_idx = int(round(idx * fps_ratio + offset))
        for d in range(-thres, thres+1):
            polyline[:, d+thres] = homog_points2[:, corr_idx + d] if corr_idx + d >=0 and corr_idx + d < homog_points2.shape[1] else np.zeros((3,))
            valid[d+thres] = homog_points2[2, corr_idx + d] > 0 if corr_idx + d >=0 and corr_idx + d < homog_points2.shape[1] else False
        nvalid = np.sum(valid)
        if nvalid < min_points: # No hay suficientes puntos válidos para interpolar
            local_offset[idx] = np.nan
            continue
        # if np.sum(valid[thres-1:thres+2]==0):  # No hay puntos válidos cerca del punto central: la correspondencia no es fiable
        #     local_offset[idx] = np.nan 
        #     continue
        aux = intersect_line_polycurve(epilines[idx, :], polyline=polyline[:, valid], t=np.arange(-thres, thres+1)[valid], all=True)
        
        if aux is None:  # Si no hay intersección, local offset es el punto más cercano a la epipolar
            distances = np.abs(distance_point_to_line(polyline, epilines[idx, :][np.newaxis, :]))
            distances[~valid] = np.inf          # Descartar puntos no válidos
            distances = distances[permutation]  # Reordenar las distancias para que los puntos centrales vayan primero
            argmin = np.argmin(distances)
            if distances[argmin]<10:   # Solo considerar la correspondencia si la distancia es pequeña
                local_offset[idx] =  permutation[argmin] - thres + corr_idx - idx
            continue
        
        intersection, t = aux

        # Si la intersección es única, la selecciona
        if intersection.shape[1] == 1:
            local_offset[idx] = t[0] + corr_idx - idx
            continue

        # Detectar rebotes o cambios de sentido de la pelota
        polyline0 = np.zeros((3, 2*thres+1))
        for d in range(-thres, thres+1):
            polyline0[:, d+thres] = homog_points1[:, corr_idx + d] if corr_idx + d >=0 and corr_idx + d < homog_points1.shape[1] else np.zeros((3,))
        rebounds0 = get_rebounds(polyline0) - max(0, idx-thres)
        rebounds1 = get_rebounds(polyline)
        rpos0 = np.sum(rebounds0>0)
        rprev0 = np.sum(rebounds0<0)
        r0 = 1 if 0 in rebounds0 else 0
        vt = np.ones_like(t, dtype=bool)
        for it in range(t.shape[0]): # Si el orden de rebotes en el tiempo es inconsistente, se descarta la intersección
            rpos = np.any(rebounds1>t[it])
            rprev = np.any(rebounds1<t[it])
            if rpos and not rprev \
                and rpos0==0 and rprev0+r0>0:
                    vt[it] = False
            if rprev and not rpos \
                and rprev0==0 and rpos0+r0>0:
                    vt[it] = False
            if rpos and rprev \
                and ((rpos0+r0>0 and rprev0==0) or (rpos0==0 and rprev0+r0>0)):
                    vt[it] = False
            # if not (rpos and rprev):
            #     vt[it] = not ((((rpos0>0) ^ rpos) or ((rprev0>0) ^ rprev)) \
            #         and ((rpos0+r0>0) ^ rpos or ((rprev0>0) ^ rprev)) \
            #         and ((rpos0>0) ^ rpos or ((rprev0+r0>0) ^ rprev)))
            # r0 es un comodín, puede contarse como rebote antes o después del frame actual del vídeo 1 para validar la intersección
        t = t[vt]
        
        # VELOCIDAD VERTICAL: NO TIENE SENTIDO PORQUE EN 2D NO SE CORRESPONDE CON LA VELOCIDAD VERTICAL EN 3D
        # Si hay varias intersecciones, se descartan las que tienen distinto signo de velocidad vertical
        # Velocidad vertical del punto de referencia
        # v1_minus = np.sign(homog_points1[1, idx] - homog_points1[1, idx-1] if idx>0 and homog_points1[2, idx-1] > 0 else 0)
        # v1_plus =  np.sign(homog_points1[1, idx+1] - homog_points1[1, idx] if idx<homog_points1.shape[1]-1 and homog_points1[2, idx+1] > 0 else 0)
        # if v1_minus*v1_plus > 0:  # Obtiene la velocidad vertical de los puntos de intersección
        #     v = np.zeros_like(t)
        #     for it in range(t.shape[0]):
        #         j = int(t[it])+thres
        #         # get previous and next index with valid polyline point
        #         pj = next((i for i in range(j, -1, -1) if valid[i]), -1)
        #         nj = next((i for i in range(j+1, polyline.shape[1]) if valid[i]), polyline.shape[1])
        #         v[it] = np.sign(polyline[1, nj] - polyline[1, pj]) if pj>=0 and nj < polyline.shape[1] else 0
        #     t = t[v*v1_minus>=0]
        
        if t.shape[0] == 0:  # Se han descartado todas las intersecciones
            count += 1
            continue

        # if intersection.shape[1] > 1:   # Heurística alternativa: Si hay más de una intersección, descartar la correspondencia por no ser fiable
        #     count += 1
        #     continue
        # local_offset[idx] = corr_idx + t[0] - idx

        # Si hay más de una intersección, escoger la más cercana al centro de la polilínea
        argt = np.argmin(np.abs(t))
        local_offset[idx] = t[argt] + corr_idx - idx

    # Linear regression of local_offset to estimate global offset and fps ratio
    valid_mask = ~np.isnan(local_offset)
    if np.sum(valid_mask) < 2:
        return offset  # No hay suficientes puntos para refinar
    A = np.vstack((np.arange(homog_points1.shape[1])[valid_mask], np.ones(np.sum(valid_mask)))).T
    b = np.arange(homog_points1.shape[1])[valid_mask] + local_offset[valid_mask]
    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    refined_offset = x[1]
    fps_ratio = x[0]
    
    # Show type of values in local_offset
    nan_count = np.sum(np.isnan(local_offset))
    int_count = np.sum(~np.isnan(local_offset) & (local_offset == local_offset.astype(np.int32)))
    float_count = np.sum(~np.isnan(local_offset) & (local_offset != local_offset.astype(np.int32)))
    total_count = len(local_offset)
    print(f'{nan_count}/{total_count} NaN values in local offset refinement')
    print(f'{int_count}/{total_count} integer values in local offset refinement')
    print(f'{float_count}/{total_count} floating point values in local offset refinement')
    print(f'{count}/{total_count} invalid values in local offset refinement by rebounds')
    if save_correspondences is not None:
        pd.DataFrame({'frame': np.arange(homog_points1.shape[1]), 'offset': local_offset}).to_csv(save_correspondences, index=False, na_rep='nan')
    return refined_offset, fps_ratio

def sync_positions(homog_points1, homog_points2, offset, fps_ratio):
    """ 
    Calcular las posiciones sincronizadas de la pelota en ambos vídeos.
    Modifica homog_points1 y homog_points2 in-place para que tengan la misma longitud y estén sincronizados.
    """
    o = int(round(offset))
    nframes = min(homog_points1.shape[1]-max(0, o), homog_points2.shape[1]-max(0, -o))
    homog_points1 = homog_points1[:, max(0, o):max(0, o) + nframes]
    # Si los FPS son iguales y offset es entero
    if fps_ratio == 1.0 and offset == o:
        homog_points2 = homog_points2[:, max(0, -o):max(0, -o) + nframes] 
    # Si los FPS son diferentes, interpolar las posiciones de la segunda cámara
    else:
        first = next((i for i in range(homog_points2.shape[1]) if homog_points2[2, i] > 0), homog_points2.shape[1]-1)    # Get index of first and last nonzero homog_points2
        last = next((i for i in range(homog_points2.shape[1]-1, -1, -1) if homog_points2[2, i] > 0), 0)
        t1 = (np.arange(first, last+1)+offset)/fps_ratio
        t2 = np.arange(max(0, o), max(0, o)+nframes)
        homog_points2[:, first:last+1] = interpolate_missing_positions(homog_points2[:, first:last+1])
        homog_points2 = piecewise_linear_interpolate(t1, homog_points2[:, first:last+1], t2)
    
    return homog_points1, homog_points2