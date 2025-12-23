import numpy as np
import cv2 as cv

def homog(points):# Convertir a coordenadas homogéneas (vector vertical)
    return np.vstack((points, np.ones((1, points.shape[1]))))

def inhomog(homog_points):  # Convertir a coordenadas cartesianas (vector vertical)
    return homog_points[:2, :] / homog_points[2, :][np.newaxis, :]

def line_p2p(h1, h2):
    # Calcular la ecuación de la recta que pasa por dos puntos dados en coordenadas homogéneas
    # Si h1 es un array de puntos (por columnas), entonces devuelve un array de rectas (por filas)
    return np.cross(h1, h2, axisa=0, axisb=0)

def get_epipolar_point(C, P):
    # Get epipolar point in camera 2 from camera 1 center C and projection matrix P of camera 2
    return P @ C

def distance_point_to_line(homog_point, line):
    "Signed distances from each point in homogeneous coordinates to the corresponding line in homogeneous coordinates"
    # distances = np.abs((line @ homog_point) / np.linalg.norm(line[:, :2], axis=1)) # Solo necesitamos la diagonal
    return np.einsum('ij,ji->i', line, homog_point) / np.linalg.norm(line[:, :2], axis=1)

def signed_distance_point_to_line(homog_point, line):
    "Signed distance from a point in homogeneous coordinates to every line in homogeneous coordinates"
    return (line @ homog_point) / np.linalg.norm(line[:, :2], axis=1)

def skew(v):
    # Matriz antisimétrica a partir de un vector unidimensional
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
    C = -np.linalg.solve(R, tvec)  # Centro de la cámara

    return M, R, tvec, C

def get_fundamental_matrix(M1, M2, C1):
    """
    Calcular la matriz fundamental F a partir de las matrices de proyección.
    F transforma puntos de la imagen 1 en líneas epipolares en la imagen 2.
    :param M1: Matriz de proyección de la cámara 1 (3x4)
    :param M2: Matriz de proyección de la cámara 2 (3x4)
    :param C1: Centro de la cámara 1 en coordenadas 3D (3x1)
    :return: Matriz fundamental F (3x3)
    """
    # C1 = -np.linalg.solve(M1[:, :3], M1[:, 3])
    e2 = M2 @ np.vstack((C1, np.ones((1, 1))))
    F = skew(e2.squeeze()) @ M2 @ np.linalg.pinv(M1)
    # F = np.cross(e2, M2 @ np.linalg.pinv(M1))
    # F /= F[2, 2]  # Normalize F
    return F


def triangulate_ball(homog_points1, homog_points2, M1, M2):
    """
    Obtener la trayectoria 3D de la pelota por triangulación
    :param homog_points1: puntos homogéneos de la pelota en la cámara 1 (3xN)
    :param homog_points2: puntos homogéneos de la pelota en la cámara 2 (3xN)
    :param M1: matriz de proyección de la cámara 1 (3x4)
    :param M2: matriz de proyección de la cámara 2 (3x4)
    :return: trayectoria 3D de la pelota en coordenadas homogéneas (4xN)
    """
    trajectory = cv.triangulatePoints(M1, M2, homog_points1[:2, :], homog_points2[:2, :])
    trajectory /= trajectory[3, :]  # Convertir a coordenadas homogéneas
    trajectory = np.nan_to_num(trajectory, nan=0.0, posinf=0.0, neginf=0.0)
    trajectory[:, homog_points1[2, :] * homog_points2[2, :] == 0] = 0  # Marcar como no válidos los puntos no detectados en alguna cámara
    return trajectory