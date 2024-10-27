import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def gaussian_beam(x, y, x0, intensity, width):
    """
    Calcula la intensidad de un haz gaussiano en un punto (x,y)
    x0: posición central del haz en el eje x
    intensity: intensidad máxima del haz
    width: ancho del haz
    """
    return intensity * np.exp(-2 * ((x - x0)**2 + y**2) / (width**2))

# Crear una malla de puntos para la visualización
x = np.linspace(-10, 10, 200)
y = np.linspace(-10, 10, 200)
X, Y = np.meshgrid(x, y)

# Parámetros de los láseres
laser_positions = [-3, 0, 3]  # Posiciones en x de los tres láseres
intensities = [1.0, 1.7, 2.2]  # Intensidades máximas
widths = [2.0, 2.0, 2.0]      # Anchos de los haces

# Calcular la intensidad total combinando los tres haces
total_intensity = np.zeros_like(X)
for pos, intensity, width in zip(laser_positions, intensities, widths):
    total_intensity += gaussian_beam(X, Y, pos, intensity, width)

# Crear la visualización
plt.figure(figsize=(15, 10))

# Gráfico 2D con mapa de calor
plt.subplot(121)
plt.imshow(total_intensity, extent=[-10, 10, -10, 10], origin='lower', cmap='hot')
plt.colorbar(label='Intensidad')
plt.title('Vista Superior de la Interacción de Láseres')
plt.xlabel('X')
plt.ylabel('Y')

# Gráfico 3D de superficie
ax = plt.subplot(122, projection='3d')
surf = ax.plot_surface(X, Y, total_intensity, cmap='hot')
plt.colorbar(surf, label='Intensidad')
ax.set_title('Vista 3D de la Interacción de Láseres')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Intensidad')

plt.tight_layout()
plt.show()

# Graficar un corte transversal
plt.figure(figsize=(10, 5))
plt.plot(x, total_intensity[len(y)//2, :])
plt.title('Corte Transversal de la Intensidad (Y = 0)')
plt.xlabel('X')
plt.ylabel('Intensidad')
plt.grid(True)
plt.show()
