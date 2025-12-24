import numpy as np
import filterpy 
from scipy.integrate import solve_ivp

# Physical constants
G = np.array([0, 0, -9.81])  # Gravity vector (m/s²)
KD = 3.8e-4                  # Drag coefficient (kg · s−4)
KM = 4.86e-6                 # Magnus coefficient (kg · s−4 · m−1)

COR = 0.85                   # Coefficient of restitution
MU = 0.3                     # Friction coefficient

M = 2.7e-3                   # Ball mass (kg)
R = 2.01e-2                  # Ball radius (m)

# Esquinas de la mesa en el sistema de coordenadas de la cámara en cm
TABLE_WIDTH = 152.5
TABLE_LENGTH = 274.0
TABLE_HEIGHT = 76.0
NET_HEIGHT = 15.25
NET_EXTRA_WIDTH = 15.25
BALL_RADIUS = 2.01

def get_alpha(v, w, MU=MU, COR=COR, R=R):
    return (MU * (1+COR) * np.abs(v[2])) / np.linalg.norm(v[:2]+w[:2]*np.array((-R, R)))

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
    B = np.array([0, alpha*R, 0],
                 [-alpha*R, 0, 0],
                 [0, 0, 0])
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
    pos = x[:3].T  # Position

    x1 = M1 @ np.vstack((pos, 1))
    x2 = M2 @ np.vstack((pos, 1))

    return np.hstack([x1[:2, 0]/x1[2, 0], x2[:2, 0]/x2[2, 0]])

def ball_ode(t, y, R=R, M=M, KD=KD, KM=KM, G=G):
    """
    ODE function for the ball motion.
    Ball ODE: m v' = KD||v||v + KM w × v + MG
    :param t: Time (s)
    :param x: State vector (9,)
    :return: Derivative of state vector (9,)
    """
    pos = y[:3]
    v = y[3:6]
    w = y[6:]

    dvdt = (KD * np.linalg.norm(v) * v + KM * np.cross(w, v)) / M + G
    dwdt = np.zeros(3)  # Assuming no change in spin
    # if pos[2] <= R and pos[:2]>= 0 and pos[:2] <= [TABLE_WIDTH/2, TABLE_LENGTH/2]: # Ball bounces on the table
    #    v, w = coulomb_friction(v, w)
    return np.hstack([v, dvdt, dwdt])

def detect_bounce(t, y, terminal=False, direction=0, R=R, TABLE_WIDTH=TABLE_WIDTH, TABLE_LENGTH=TABLE_LENGTH):
    """ Event function to detect ball bounces on the table. """
    pos = y[:3]
    return 1 + (y[2]-R-1) * (y[:2]>=0 and y[:2]<=[TABLE_WIDTH/2, TABLE_LENGTH/2])

detect_bounce.terminal = True
detect_bounce.direction = -1

def fx(x, dt):
    """
    State transition function for the Unscented Kalman Filter.
    Based in the ODE of ball motion described in TT3D article.
    :param x: State vector (9,)
    :param dt: Time step size (s)
    :return: Predicted state vector (9,)
    """
    sol = solve_ivp(
        fun = ball_ode,     # ODE dy/dt = fun(t, y), y(t0) = y0
        t_span = (0, dt),
        y0 = x,
        t_eval = [dt],
        events=[detect_bounce],
        method = 'RK45'  # Default method Runge-Kutta-Dormand-Prince of order 5(4)
    )

    if not sol.success and sol.t_events: # A bounce has been detected
        t_bounce = sol.t_events[0][0]
        y_bounce = sol.y_events[0][0]
        # Get v and w after the bounce
        y_bounce[3:6], y_bounce[6:] = coulomb_friction(y_bounce[3:6], y_bounce[6:])

        sol = solve_ivp(
            fun=ball_ode,
            t_span=(t_bounce, dt),
            y0=y_bounce,
            t_eval=[dt],
            events=None,  # No more bounces are checked
            method='RK45'
        )

    y = sol.y
    success = sol.success

    return y[:, 0] if success else x

def Ball_UKF(dt, M1, M2):
    """
    Creates an Unscented Kalman Filter for the ball's state estimation.
    :param dt: Time step size (s)
    :param M1: Projection matrix of the first camera
    :param M2: Projection matrix of the second camera
    :return: Unscented Kalman Filter object
    """
    UKF = filterpy.kalman.UnscentedKalmanFilter(
        dim_x=9,    # State variables: pos (3), v (3), w (3)
        dim_z=4,    # Measurement variables: (x1, y1), (x2, y2) pixel coordinates
        dt=dt,      # Time step size
        hx=lambda x: hx(x, M1, M2),    # Measurement function
        fx=None     # State transition function
        # TODO: revisar más parámetros de configuración
    )
    return UKF

# TODO: Funcionalidad de predicción de la trayectoria de la pelota usando el UKF Ball_UKF