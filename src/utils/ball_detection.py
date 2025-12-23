import numpy as np
import cv2 as cv
import pandas as pd
import os

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
            x = int(round(positions[frame_idx, 0]))
            y = int(round(positions[frame_idx, 1]))
            cv.circle(frame, (x, y), 5, (0, 0, 255), -1)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

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

def get_ball_positions_from_masks(masks_folder, w, h, nframes=-1):
    """ Obtener las posiciones de la pelota en cada frame del conjunto de máscaras guardadas en una carpeta.
    Devuelve una matriz de tamaño (num_frames, 3) con las coordenadas homogéneas de la pelota en cada frame
    o (0, 0, 0) si no se detecta la pelota en ese frame.
    """

    mask_files = [f for f in os.listdir(masks_folder) if f.endswith('.png')]
    if nframes==-1:
        frame_idx_list = [int(os.path.splitext(f)[0]) for f in mask_files]
        nframes = max(frame_idx_list) + 1
    ball_positions = np.zeros((nframes, 3))

    for mask_file in mask_files:
        frame_idx = int(os.path.splitext(mask_file)[0])
        mask_path = os.path.join(masks_folder, mask_file)
        mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
        if mask is not None:
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

    return ball_positions

def read_ball_positions(file_path):
    """ Leer las posiciones de la pelota desde un archivo CSV.
    Devuelve una matriz de tamaño (num_frames, 3) con las coordenadas homogéneas de la pelota en cada frame.
    """
    return pd.read_csv(file_path, header=None).values

def save_ball_positions(file_path, positions):
    """ Guardar las posiciones de la pelota en un archivo CSV.
    positions debe ser una matriz de tamaño (num_frames, 3) con las coordenadas homogéneas de la pelota en cada frame.
    """
    pd.DataFrame(positions).to_csv(file_path, index=False, header=None)

def draw_line(img, homog_line, color=(0, 0, 0), thickness=1, **args):
    """ Dibuja una línea dada en coordenadas homogéneas sobre la imagen. """
    a, b, c = homog_line
    h, w = img.shape[:2]
    if b != 0:
        y0 = int(round(-c / b))
        yw = int(round(-(a * w + c) / b))
        return cv.line(img, (0, y0), (w, yw), color, thickness, **args)
    else:
        if a != 0:
            x0 = int(round(-c / a))
            xh = int(round(-(b * h + c) / a))
            return cv.line(img, (x0, 0), (xh, h), color, thickness, **args)