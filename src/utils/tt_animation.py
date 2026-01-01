import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import numpy as np
from .filter import TABLE_LENGTH, TABLE_WIDTH, TABLE_HEIGHT, NET_HEIGHT, NET_EXTRA_WIDTH

def update(frame, ball, trajectory):
    x = trajectory[0, frame]
    y = trajectory[1, frame]
    z = trajectory[2, frame]
    ball.set_data([x], [y])
    ball.set_3d_properties([z])
    return ball,

def update_with_time(frame, ball, trajectory, t):
    x = trajectory[0, frame]
    y = trajectory[1, frame]
    z = trajectory[2, frame]
    ball.set_data([x], [y])
    ball.set_3d_properties([z])
    dt = t[frame+1]-t[frame] if frame < len(t)-1 else 0.01
    plt.pause(dt)
    return ball,

def ball_animation(trajectory, t=None):
    """
    Crea una animación 3D de la trayectoria de la pelota sobre la mesa.
    :param trajectory: Trayectoria de la pelota en coordenadas cartesianas 3D (3 x N)
    :param t: Array de tiempos correspondientes a cada punto de la trayectoria (N,). Si es None, se usa un intervalo fijo entre puntos.
    """

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

    ani = FuncAnimation(fig, update if t is None else update_with_time, frames=trajectory.shape[1], 
                        fargs = (ball, trajectory) if t is None else (ball, trajectory, t), 
                        interval=16 if t is None else 0, blit=False, repeat=True)
    # ani.save('ball_trajectory_3D.mp4', writer='ffmpeg', fps=60)
    plt.show()
    # else:
    #     for i in range(trajectory.shape[1]-1):
    #         update(i, ball, trajectory)
    #         plt.draw()
    #         dt = t[i+1]-t[i]
    #         plt.pause(dt)
    #     update(trajectory.shape[1]-1, ball, trajectory)
    #     plt.draw()
    #     plt.pause(0.01)