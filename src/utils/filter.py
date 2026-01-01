from turtle import pos
import numpy as np
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt

# Esquinas de la mesa en el sistema de coordenadas de la cámara en cm
TABLE_WIDTH = 152.5
TABLE_LENGTH = 274.0
TABLE_HEIGHT = 76.0
NET_HEIGHT = 15.25
NET_EXTRA_WIDTH = 15.25
BALL_RADIUS = 2.01

# Physical constants
G = np.array([0, 0, -9.81])  # Gravity vector (m/s2)
KD = 3.8e-4                  # Drag coefficient (kg · s−4)
KM = 4.86e-6                 # Magnus coefficient (kg · s−4 · m−1)
COR = 0.85                   # Coefficient of restitution
MU = 0.3                     # Friction coefficient
M = 2.7e-3                   # Ball mass (kg)
R = BALL_RADIUS / 100        # Ball radius (m)

# Variables globales para la detección de rebotes
# bounce_counter = 0
# x_bounce = np.zeros((10, 4)) # La última fila es el momento del bote
# track_bounces=True

def get_alpha(v, w, MU=MU, COR=COR, R=R):
    return np.nan_to_num((MU * (1+COR) * np.abs(v[2])) / np.linalg.norm(v[:2]+w[:2]*np.array((-R, R))), nan=0.0)

def coulomb_friction(v, w, thres=0.4):
    """
    Computes the velocity and spin of the ball after the bounce using the Coulomb friction model described in TT3D article.
    :param v: Ball velocity before bounce (3,).
    :param w: Ball spin euler vector before bounce (3,).
    :return v_, w_: ball velocity and spin after bounce (3,).
    """
    alpha = get_alpha(v, w)
    alpha = min(alpha, thres)

    A = np.diag([1-alpha, 1-alpha, -COR])
    B = np.array([[0, alpha*R, 0],
                 [-alpha*R, 0, 0],
                 [0, 0, 0]])
    C = np.array([[0, -3*alpha/(2*R), 0],
                  [3*alpha/(2*R), 0, 0],
                  [0, 0, 0]])
    D = np.diag([1-3*alpha/2, 1-3*alpha/2, 1])

    return A @ v.T + B @ w.T, C @ v.T + D @ w.T

def hx(x, M1, M2):
    """
    Measurement function for the Unscented Kalman Filter.
    :param x: State vector (9,)
    :return: Measurement vector (4,)
    """
    pos = 100*x[:3, np.newaxis]  # Position in cm
    x1 = (M1 @ np.vstack((pos, 1)))
    x2 = (M2 @ np.vstack((pos, 1)))

    return np.hstack((x1[:2, 0]/x1[2, 0], x2[:2, 0]/x2[2, 0]))

def camera_measurement(x, M):
    """
    Measurement function of the first camera for the Unscented Kalman Filter.
    :param x: State vector (9,)
    :param M: Camera matrix
    :return: Measurement vector (2,)
    """
    pos = 100*x[:3, np.newaxis]  # Position in cm
    x1 = M @ np.vstack((pos, 1))
    return x1[:2, 0] / x1[2, 0]

def ball_ode(t, y, R=R, M=M, KD=KD, KM=KM, G=G):
    """
    ODE function for the ball motion.
    Ball ODE: m v' = KD||v||v + KM w × v + MG
    :param t: Time (s)
    :param x: State vector (9,)
    :return: Derivative of state vector (9,)
    """
    v = y[3:6]
    w = y[6:]

    dvdt = (KD * np.linalg.norm(v) * v + KM * np.cross(w, v)) / M + G
    dwdt = np.zeros(3)  # Assuming no change in spin

    return np.hstack([v, dvdt, dwdt])

def detect_bounce(t, y, terminal=True, direction=-1, R=R): # Length in meters
    """ Event function to detect ball falling to table height (possible bounces). """
    return y[2] - R

detect_bounce.terminal = True
detect_bounce.direction = -1

def fx(x, dt, R=R, TABLE_WIDTH=0.01*TABLE_WIDTH, TABLE_LENGTH=0.01*TABLE_LENGTH):
    """
    State transition function for the Unscented Kalman Filter.
    Based in the ODE of ball motion described in TT3D article.
    :param x: State vector (9,)
    :param dt: Time step size (s)
    :return: Predicted state vector (9,)
    """
    # global bounce_counter, x_bounce, track_bounces

    if dt <= 0.0:
        return x
    
    sol = solve_ivp(
        fun = ball_ode,     # ODE dy/dt = fun(t, y), y(t0) = y0
        t_span = (0, dt),
        y0 = x,
        t_eval = [dt],
        events=detect_bounce,
        method = 'RK45'  # Default method Runge-Kutta-Dormand-Prince of order 5(4)
    )

    if sol.status==1 and sol.t_events and sol.t_events[0].size > 0: # Possible bounce
        t_bounce = sol.t_events[0][0]
        y_bounce = sol.y_events[0][0]

        # Check bounce on the table
        if np.all(y_bounce[:2]>=0) and np.all(y_bounce[:2]<=[TABLE_WIDTH, TABLE_LENGTH]):
            if np.abs(y_bounce[2]-R) > 2*R:
                print(f"?{y_bounce}")
            # if track_bounces:  # Set global variables
            #     x_bounce[:-1, bounce_counter] = y_bounce.copy()
            #     x_bounce[-1, bounce_counter] = t_bounce
            #     bounce_counter += 1
            #     if x_bounce.shape[1] <= bounce_counter:
            #         x_bounce = np.hstack((x_bounce, np.zeros_like(x_bounce)))

            # Set initial values after the bounce
            y_bounce[3:6], y_bounce[6:] = coulomb_friction(y_bounce[3:6], y_bounce[6:]) # v and w after the bounce
            # y_bounce[2] = R # height after bounce is the radius of the ball

        sol = solve_ivp(
            fun=ball_ode,
            t_span=(t_bounce, dt),
            y0=y_bounce,
            t_eval=[dt],
            events=None,  # No more bounces are checked
            method='RK45'
        )

    return sol.y[:, -1] if sol.success and sol.status == 0 else x

def Ball_UKF(M1, M2):
    """
    Creates an Unscented Kalman Filter for the ball's state estimation.
    :param dt: Time step size (s)
    :param M1: Projection matrix of the first camera
    :param M2: Projection matrix of the second camera
    :return: Unscented Kalman Filter object
    """
    sigma_points = MerweScaledSigmaPoints(
        n=9,
        alpha=0.1,
        beta=2.0,
        kappa=0
    )

    UKF = UnscentedKalmanFilter(
        dim_x=9,    # State variables: pos (3), v (3), w (3)
        dim_z=4,    # Measurement variables: (x1, y1), (x2, y2) pixel coordinates
        dt=1,       # Time step size
        hx=lambda x, dt=None: hx(x, M1, M2),    # Measurement function
        fx=fx,      # State transition function
        points=sigma_points
        # TODO: revisar más parámetros de configuración
    )
    return UKF

# Predicción de la trayectoria de la pelota usando el UKF Ball_UKF

def getQ(dt, sigma_a=0.1):
    # TODO: Establecer Q con sentido en función de dt
    Q = np.zeros((9, 9))
    Q[0:3, 0:3] = sigma_a**2 * np.eye(3) * dt**4 / 4
    Q[3:6, 3:6] = sigma_a**2 * np.eye(3) * dt**2
    Q[6:9, 6:9] = sigma_a**2 * np.eye(3) * dt**2
    return Q

def update(UKF, pos2d, dt, M):
    UKF.update(pos2d, dt=dt, hx=lambda x, dt=None: camera_measurement(x, M))

def initialize_UKF(UKF, pos=np.array([0.005*TABLE_WIDTH, 0.005*TABLE_LENGTH, 0.02*NET_HEIGHT])):
    UKF.x[:3] = pos
    UKF.x[3:6] = np.zeros((3,)) # Velocidad inicial
    UKF.x[6:] = np.zeros((3,))  # Rotación inicial
    # Incertidumbre inicial
    UKF.P = np.diag([2.0, 2.0, 2.0,         # Posición: 4m
                     108.0, 108.0, 108.0,   # v: 108 m/s = 30 km/h 
                     100.0, 100.0, 100.0])  # w: 100 rad/s
    UKF.P *= UKF.P # Pasar de desviación típica a varianza

def estimate_trajectory(homog_points1, homog_points2, t_1, t_2, M1, M2, shape1, shape2, loglikelihood_thres=-1e5):
    """
    Estima la trayectoria de la pelota utilizando un UKF.
    :param homog_points1: Puntos de la primera cámara en coordenadas homogéneas. La 3ª coordenada es 0 si el punto es inválido y 1 si es válido.
    :param homog_points2: Puntos de la segunda cámara en coordenadas homogéneas.
    :param t_1: Instantes de tiempo correspondientes a los puntos de la primera cámara.
    :param t_2: Instantes de tiempo correspondientes a los puntos de la segunda cámara.
    :param M1: Matriz de proyección de la primera cámara.
    :param M2: Matriz de proyección de la segunda cámara.
    :param shape1: Tamaño (w, h) de la imagen de la primera cámara.
    :param shape2: Tamaño (w, h) de la imagen de la segunda cámara.
    :param likelihood_thres: Umbral de log-verosimilitud para considerar que ha habido un rebote y reiniciar el UKF.
    :return: Matriz de estados estimados (10, N), tiempos correpondientes (N,), índices de rebotes (no en la mesa), estados en los botes en la mesa, UKF final
    """
    global bounce_counter, x_bounce

    UKF = Ball_UKF(M1=M1, M2=M2)

    # weave homog_points1, t_1 and homog_points2, t2
    timestamps = np.hstack((t_1, t_2))
    points = np.hstack((homog_points1, homog_points2))
    camera_indices = np.hstack((np.zeros(len(t_1)), np.ones(len(t_2))))
    perm = np.argsort(timestamps)

    measurements = np.vstack((points, timestamps, camera_indices))
    measurements = measurements[:, perm]
    measurements = np.delete(measurements[:, measurements[2] != 0], 2, axis=0) # Remove invalid points
    t = measurements[2, :].copy()
    measurements[2, :] = np.diff(measurements[2, :], prepend=0) # Get time increments

    R = [np.diag(shape1)*0.02, np.diag(shape2)*0.02] # Ball detection noise for each camera in pixels

    initialize_UKF(UKF) # TODO: Posición inicial a partir de una o dos cámaras
    X = np.zeros((measurements.shape[1]+1, UKF.x.shape[0])) # Array of estimations
    X[0, :] = UKF.x.copy()
    covariances = np.zeros((measurements.shape[1]+1, UKF.P.shape[0], UKF.P.shape[1]))
    covariances[0, :, :] = UKF.P.copy()
    smoothed_x = np.zeros((measurements.shape[1], UKF.x.shape[0]))
    loglikelihoods = np.zeros((measurements.shape[1],))
    rebounds = []
    bounces = []
    last_rebound = 0

    i = 0
    while i < measurements.shape[1]:
        UKF.Q = getQ(dt=measurements[2, i], sigma_a=0.1)
        UKF.R = R[int(measurements[3, i])]
        UKF.predict(dt=measurements[2, i])
        update(UKF, pos2d=measurements[:2, i], dt=measurements[2, i], M=M2 if measurements[3, i] else M1)
        loglikelihoods[i] = UKF.log_likelihood
        if (UKF.log_likelihood<loglikelihood_thres and last_rebound+1 < i) or i==measurements.shape[1]-1:
            if False:
                # Plot loglikelihoods
                plt.plot(loglikelihoods[last_rebound:i+1])
                plt.axhline(y=loglikelihood_thres, color='r', linestyle='--')
                plt.xlabel('Measurement')
                plt.ylabel('Log-Likelihood')
                plt.title('UKF Log-Likelihood')
                plt.show()
            # Get speed and spin
            rebounds.append(i)
            print(f"REBOUND\nMeasure: {i}\nPosition: {X[i, :3]}\nTime: {t[i]}\nSpeed: {X[i, 3:6]}\nSpin: {X[i, 6:]} ({np.linalg.norm(X[i, 6:])/(2*np.pi)} rps)\nLikelihood: {UKF.log_likelihood}")
            # Recalculate trajectory with UKF backward pass
            try:
                # track_bounces = False
                smoothed_x[last_rebound:i, :], _, _ = UKF.rts_smoother(Xs=X[last_rebound+1:i+1, :], Ps=covariances[last_rebound+1:i+1, :, :], 
                                                    Qs=np.stack([getQ(dt=measurements[2, j], sigma_a=0.1) for j in range(last_rebound, i)], axis=0),
                                                    dts=measurements[2, last_rebound:i])
                # track_bounces = True
            except np.linalg.LinAlgError:
                print(f"np.linalg.LinAlgError on measure {i}")
                # print("Measurements")
                # print(measurements[2, last_rebound:i])
                # print("Estimates")
                # print(X[last_rebound+1:i+1, :])
                # print("Covariances")
                # print(covariances[last_rebound+1:i+1, :, :])
                smoothed_x[last_rebound:i, :] = X[last_rebound+1:i+1, :]
            # Reset UKF
            if i < measurements.shape[1]-1:
                UKF = Ball_UKF(M1=M1, M2=M2)
                initialize_UKF(UKF, pos=smoothed_x[i-1, :3])
                # print(f"Estimación inicial: {UKF.x}")
                last_rebound = i
                i -= 1  # Reprocess this measurement
        else:
            X[i+1, :] = UKF.x.copy()
            covariances[i+1, :, :] = UKF.P.copy()
            # for j in range(bounce_counter):
            #     bounces.append(x_bounce[:, j].copy())
            #     print(f"BOUNCE\nMeasure: {i}\nTime: {t[i-1] + x_bounce[-1, j]}\nSpeed: {x_bounce[3:6, j]}\nSpin: {x_bounce[6:9, j]} ({np.linalg.norm(x_bounce[6:9, j]/(2*np.pi))} rps)")

        # bounce_counter = 0
        if i%100==0:
            print(f"Processing measurement {i}")
        i += 1

    # Get predicted state
    # pos = UKF.x[:3]  # in meters
    # vel = UKF.x[3:6]  # in m/s
    # rot = UKF.x[6:]   # in radians    

    # print("Estimaciones UKF")
    # print(np.hstack((X[:100, :3], measurements[:, :100].T)))
    # print(rebounds)
    return smoothed_x.T, t, rebounds, bounces, UKF, X.T
