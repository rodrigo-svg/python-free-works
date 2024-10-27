import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def create_cone(x0, y0, z0, height, radius, resolution=50):
    """
    Crea un cono de luz desde un punto origen hasta una altura dada
    x0, y0, z0: coordenadas del origen del cono
    height: altura del cono
    radius: radio de la base del cono
    resolution: resolución de la malla
    """
    z = np.linspace(z0, z0 + height, resolution)
    theta = np.linspace(0, 2*np.pi, resolution)
    theta_grid, z_grid = np.meshgrid(theta, z)

    # El radio varía linealmente con la altura
    r_grid = radius * (z_grid - z0) / height

    x_grid = x0 + r_grid * np.cos(theta_grid)
    y_grid = y0 + r_grid * np.sin(theta_grid)

    return x_grid, y_grid, z_grid

# Configuración de la figura
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Parámetros de los conos
height = 10  # altura de los conos
radius = 3   # radio de la base de los conos
laser_positions = [(-4, 0), (0, 0), (4, 0)]  # posiciones (x, y) de los láseres
colors = ['red', 'green', 'blue']  # colores diferentes para cada cono

# Crear y graficar cada cono
for (x0, y0), color in zip(laser_positions, colors):
    X, Y, Z = create_cone(x0, y0, 0, height, radius)

    # Graficar superficie del cono con transparencia
    ax.plot_surface(X, Y, Z, alpha=0.3, color=color)

    # Graficar líneas verticales para mostrar el eje central del cono
    ax.plot([x0, x0], [y0, y0], [0, height], '--', color=color, alpha=0.5)

    # Graficar círculo en la base
    theta = np.linspace(0, 2*np.pi, 100)
    circle_x = x0 + radius * np.cos(theta)
    circle_y = y0 + radius * np.sin(theta)
    ax.plot(circle_x, circle_y, [height]*100, color=color)

# Configurar vista y etiquetas
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z (Altura)')
ax.set_title('Proyección de Conos de Luz Láser')

# Ajustar límites y vista
ax.set_box_aspect([1,1,1])
ax.view_init(elev=20, azim=45)

# Mostrar plano base
xx, yy = np.meshgrid(np.linspace(-8, 8, 10), np.linspace(-8, 8, 10))
ax.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.1, color='gray')

plt.show()

# Crear una vista desde arriba para ver las intersecciones
plt.figure(figsize=(8, 8))
for (x0, y0), color in zip(laser_positions, colors):
    circle = plt.Circle((x0, y0), radius, fill=False, color=color, alpha=0.5)
    plt.gca().add_artist(circle)
    plt.plot(x0, y0, 'o', color=color)  # Punto de origen del láser

plt.axis('equal')
plt.grid(True)
plt.title('Vista Superior - Intersección de los Conos')
plt.xlabel('X')
plt.ylabel('Y')

plt.show()
