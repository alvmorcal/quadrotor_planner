#!/usr/bin/env python3
#
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # para proyecciones 3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# -------------------------------
# Funciones de carga y utilidades
# -------------------------------

def load_map(json_file):
    """
    Carga el mapa desde el archivo JSON.
    Extrae los límites del entorno y almacena cada obstáculo con:
      - center: posición (pose) como np.array
      - size: dimensiones en X, Y, Z como np.array
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    bounds = data["bounds"]
    obstacles = []
    for obs in data["obstacles"]:
        obstacles.append({
            "name": obs["name"],
            "center": np.array(obs["pose"], dtype=float),
            "size": np.array(obs["size"], dtype=float)
        })
    return bounds, obstacles

def closest_point_on_box(point, center, size):
    """
    Calcula el punto más cercano en un AABB (caja alineada con los ejes)
    definido por 'center' y 'size', al punto 'point'.
    """
    half = size / 2.0
    lower = center - half
    upper = center + half
    return np.maximum(lower, np.minimum(point, upper))

# -------------------------------
# Funciones del campo potencial
# -------------------------------

def f_attract(pos, goal, m_goal=1.0):
    """
    Calcula la fuerza atractiva:
      F_attract = m_goal * (goal - pos) / ||goal - pos||
    Si pos coincide con goal, devuelve el vector cero.
    """
    diff = goal - pos
    norm = np.linalg.norm(diff)
    if norm == 0:
        return np.zeros(3)
    return m_goal * diff / norm

def f_repulsive(pos, obstacles, R_soi=3.0):
    """
    Calcula la fuerza repulsiva total en 'pos' producida por todos los obstáculos.
    
    Para cada obstáculo (definido como AABB), se calcula el punto Q más cercano en la caja a 'pos'.
    Se define un "radio efectivo" R_eff como la mitad de la dimensión mínima del obstáculo.
    
    Si la distancia d = ||pos - Q|| es menor que R_soi, se suma:
         F_rep = m_obs * (pos - Q) / d
    donde:
         m_obs = (R_soi - d) / (R_soi - R_eff)
    
    Se evita división por cero si d es muy pequeño.
    """
    force = np.zeros(3)
    for obs in obstacles:
        center = obs["center"]
        size = obs["size"]
        Q = closest_point_on_box(pos, center, size)
        diff = pos - Q
        d = np.linalg.norm(diff)
        R_eff = min(size) / 2.0  # aproximación del "radio efectivo" del obstáculo
        if d < R_soi:
            if d < 1e-6:
                d = 1e-6
            m_obs = (R_soi - d) / (R_soi - R_eff) if R_soi > R_eff else 1.0
            force += m_obs * (diff / d)
    return force

# -------------------------------
# Cálculo del campo potencial en 3D
# -------------------------------

def compute_potential_field_3d(X, Y, Z, goal, obstacles, m_goal=1.0, R_soi=3.0):
    """
    Para cada punto (x, y, z) de la malla, calcula el campo total:
        F_total = f_attract(pos, goal) + f_repulsive(pos, obstacles)
    Devuelve:
      - U, V, W: componentes de F_total en cada dirección.
      - M_force: magnitud de F_total en cada punto.
    """
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    W = np.zeros_like(Z)
    M_force = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            for k in range(X.shape[2]):
                pos = np.array([X[i, j, k], Y[i, j, k], Z[i, j, k]])
                F_att = f_attract(pos, goal, m_goal)
                F_rep = f_repulsive(pos, obstacles, R_soi)
                F_total = F_att + F_rep
                U[i, j, k] = F_total[0]
                V[i, j, k] = F_total[1]
                W[i, j, k] = F_total[2]
                M_force[i, j, k] = np.linalg.norm(F_total)
    return U, V, W, M_force

# -------------------------------
# Función para dibujar obstáculos en 3D
# -------------------------------

def plot_box(ax, center, size, color='r', alpha=0.3):
    """
    Dibuja un obstáculo como una caja (cuboid) en el eje 3D.
    """
    c = np.array(center)
    s = np.array(size) / 2.0
    corners = np.array([
        c + [ s[0],  s[1],  s[2]],
        c + [ s[0],  s[1], -s[2]],
        c + [ s[0], -s[1],  s[2]],
        c + [ s[0], -s[1], -s[2]],
        c + [-s[0],  s[1],  s[2]],
        c + [-s[0],  s[1], -s[2]],
        c + [-s[0], -s[1],  s[2]],
        c + [-s[0], -s[1], -s[2]],
    ])
    faces = [
        [corners[0], corners[1], corners[3], corners[2]],
        [corners[4], corners[5], corners[7], corners[6]],
        [corners[0], corners[1], corners[5], corners[4]],
        [corners[2], corners[3], corners[7], corners[6]],
        [corners[0], corners[2], corners[6], corners[4]],
        [corners[1], corners[3], corners[7], corners[5]],
    ]
    box = Poly3DCollection(faces, facecolors=color, edgecolors='k', alpha=alpha)
    ax.add_collection3d(box)

# -------------------------------
# Representación 3D del campo potencial
# -------------------------------

if __name__ == '__main__':
    # Cargar el mapa y los obstáculos desde el archivo JSON
    json_file = "world.json"  # Asegúrate de que esté en el mismo directorio
    bounds, obstacles = load_map(json_file)
    
    # Crear una malla 3D usando los límites del mapa
    # Puedes ajustar la resolución (nx, ny, nz) para mayor detalle o rapidez.
    nx = 5  # número de puntos en x
    ny = 5  # número de puntos en y
    nz = 5  # número de puntos en z
    x_vals = np.linspace(bounds["x"][0], bounds["x"][1], nx)
    y_vals = np.linspace(bounds["y"][0], bounds["y"][1], ny)
    z_vals = np.linspace(bounds["z"][0], bounds["z"][1], nz)
    X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')
    
    # Definir el objetivo (por ejemplo, en el centro del espacio en z)
    goal = np.array([0.0, 0.0, 10.0])
    
    # Parámetros del campo potencial
    m_goal = 1.0
    R_soi = 3.0
    
    # Calcular el campo potencial 3D en la malla
    U, V, W, M_force = compute_potential_field_3d(X, Y, Z, goal, obstacles, m_goal, R_soi)
    
    # Crear la figura 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Dibujar el campo vectorial usando quiver
    # Se normalizan los vectores para mejorar la visualización.
    ax.quiver(X, Y, Z, U, V, W, length=1.0, normalize=True, color='b', arrow_length_ratio=0.3)
    
    # Dibujar los obstáculos (cajas)
    for obs in obstacles:
        plot_box(ax, obs["center"], obs["size"], color='r', alpha=0.3)
    
    # Dibujar el objetivo
    ax.scatter(goal[0], goal[1], goal[2], color='m', s=100, label='Objetivo')
    
    # Configurar límites y etiquetas
    ax.set_xlim(bounds["x"])
    ax.set_ylim(bounds["y"])
    ax.set_zlim(bounds["z"])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Campo Potencial 3D")
    ax.legend()
    
    plt.show()
