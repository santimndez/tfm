from matplotlib import use
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import numpy as np
from .filter import TABLE_LENGTH, TABLE_WIDTH, TABLE_HEIGHT, NET_HEIGHT, NET_EXTRA_WIDTH, BALL_RADIUS

def ball_animation(trajectory, t=None, out=None, fps=60):
    """
    Crea una animación 3D de la trayectoria de la pelota sobre la mesa.
    :param trajectory: Trayectoria de la pelota en coordenadas cartesianas 3D (3 x N)
    :param t: Array de tiempos correspondientes a cada punto de la trayectoria (N,). Si es None, se usa un intervalo fijo entre puntos.
    :param out: Ruta del archivo de salida para guardar la animación (opcional).
    :param fps: Frames por segundo para la animación (opcional).
    """
    if out is not None:
        use("Agg")
    else:
        use("TkAgg")
    
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(121, projection='3d')      # Animación 3D: trayectoria de la pelota
    ax2d = fig.add_subplot(122)    # Animación 2D: velocidad de la pelota

    # fig.subplots_adjust(left=0, right=1, bottom=0, top=1) # Ajusta a la ventana

    # ANIMACIÓN 2D
    arrow = ax2d.quiver(0, 0, 0, 0, angles='xy', scale_units='xy', scale=1)
    ax2d.set_xlim(-200, 200)
    ax2d.set_ylim(-200, 200)
    ax2d.set_aspect('equal')
    speed_text = ax2d.text(0.5, 1.05, '', transform=ax2d.transAxes)

    # ANIMACIÓN 3D
    # ax.plot(trajectory[0, :], trajectory[1, :], trajectory[2, :], label='Trayectoria de la pelota')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Trayectoria 3D de la pelota')
    ax.set_xlim3d(-0.75*TABLE_WIDTH, 1.75*TABLE_WIDTH)
    ax.set_ylim3d(-TABLE_LENGTH/2, 1.5*TABLE_LENGTH)
    ax.set_zlim3d(-TABLE_HEIGHT, 2*TABLE_HEIGHT)
    ax.set_aspect('equal')
    ax.view_init(elev=20, azim=-80)
    # plt.tight_layout()

    # Dibujar líneas de la mesa
    ax.plot([0, TABLE_WIDTH],               [0, 0],                         [0, 0], color='black')
    ax.plot([TABLE_WIDTH, TABLE_WIDTH],     [0, TABLE_LENGTH],              [0, 0], color='black')
    ax.plot([TABLE_WIDTH, 0],               [TABLE_LENGTH, TABLE_LENGTH],   [0, 0], color='black')
    ax.plot([0, 0],                         [TABLE_LENGTH, 0],              [0, 0], color='black')
    ax.plot([TABLE_WIDTH/2, TABLE_WIDTH/2], [0, TABLE_LENGTH],              [0, 0], color='black')

    # Dibujar la red
    x_net = np.linspace(-NET_EXTRA_WIDTH, TABLE_WIDTH + NET_EXTRA_WIDTH, 10)
    z_net = np.linspace(0,  NET_HEIGHT, 5)
    XX, ZZ = np.meshgrid(x_net, z_net)
    YY = np.full_like(XX, TABLE_LENGTH/2)

    # Dibujar la red como una malla
    ax.plot_surface(XX, YY, ZZ, color='black', alpha=0.25)

    # Dibujar superficie de la mesa
    # xx, yy = np.meshgrid([0, TABLE_WIDTH], [0, TABLE_LENGTH])
    # zz = np.full_like(xx, 0)
    # ax.plot_surface(xx, yy, zz, color='blue', alpha=0.3)

    # Dibujar la pelota
    ball = ax.scatter(
        [trajectory[0, 0]*100],
        [trajectory[1, 0]*100],
        [trajectory[2, 0]*100],
        s=5,
        color='orange'
    )
    
    def update(frame):
        nonlocal ball, arrow, speed_text
        x = trajectory[0, frame]*100
        y = trajectory[1, frame]*100
        z = trajectory[2, frame]*100
        ball._offsets3d = ([x], [y], [z])

        if frame>0 and (bool(trajectory[1, frame]>TABLE_LENGTH/200) ^ bool(trajectory[1, frame-1]>TABLE_LENGTH/200)):
            spin_x = trajectory[7, frame] / (2*np.pi)
            spin_y = -trajectory[6, frame] / (2*np.pi)
            arrow.remove()
            arrow = ax2d.quiver(0, 0, spin_x, spin_y, angles='xy', scale_units='xy', scale=1)
            speed = np.linalg.norm(trajectory[3:6, frame])
            speed_text.set_text(f"{speed*3.6:.0f} km/h \n{np.linalg.norm(trajectory[6:9, frame]) / (2*np.pi):.1f} rps")

        return ball,arrow, speed_text,

    ani = FuncAnimation(fig, update, frames=trajectory.shape[1], interval=1000/fps, blit=False, repeat=False)

    if out is not None:
        ani.save(out, writer='ffmpeg', fps=fps)
    else:
        plt.show()