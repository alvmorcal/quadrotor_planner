#!/usr/bin/env python3
#
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# -------------------------------
# Funciones de carga y utilidades
# -------------------------------

def load_map(json_file):
    """
    Carga el mapa desde el archivo JSON.
    Se extraen los límites del entorno y se almacena cada obstáculo con:
      - center: posición (pose) (como np.array)
      - size: dimensiones en X, Y, Z (como np.array)
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
    Calcula el punto más cercano en el AABB (caja alineada con los ejes) definido
    por 'center' y 'size', al punto 'point'.
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
    Si pos coincide con el goal, se devuelve el vector cero.
    """
    diff = goal - pos
    norm = np.linalg.norm(diff)
    if norm == 0:
        return np.zeros(3)
    return m_goal * diff / norm

def f_repulsive(pos, obstacles, R_soi=3.0):
    """
    Calcula la fuerza repulsiva total producida por los obstáculos.
    
    Para cada obstáculo (definido como AABB), se calcula el punto Q más cercano en la caja
    al punto 'pos'. Se define un "radio efectivo" R_eff como la mitad de la dimensión mínima
    del obstáculo.
    
    Si la distancia d = ||pos - Q|| es menor que R_soi, se añade el término:
    
         F_rep = m_obs * (pos - Q) / d
    
    donde:
         m_obs = (R_soi - d) / (R_soi - R_eff)
    
    (Se evita división por cero si d es muy pequeño).
    """
    force = np.zeros(3)
    for obs in obstacles:
        center = obs["center"]
        size = obs["size"]
        Q = closest_point_on_box(pos, center, size)
        diff = pos - Q
        d = np.linalg.norm(diff)
        # Se define un "radio efectivo" para el obstáculo (por ejemplo, para una pared delgada será pequeño)
        R_eff = min(size) / 2.0
        if d < R_soi:
            # Evitar división por cero
            if d < 1e-6:
                d = 1e-6
            m_obs = (R_soi - d) / (R_soi - R_eff) if R_soi > R_eff else 1.0
            force += m_obs * (diff / d)
    return force

# -------------------------------
# Cálculo del campo en una malla
# -------------------------------

def compute_potential_field(X, Y, z, goal, obstacles, m_goal=1.0, R_soi=3.0):
    """
    Calcula, para cada punto (x, y) de la malla (con z fijo),
    el campo potencial total F_total = f_attract + f_repulsive.
    
    Devuelve:
      - U: componente x de F_total
      - V: componente y de F_total
      - M: magnitud de F_total
    """
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    M_force = np.zeros_like(X)
    
    # Recorremos la malla (X e Y tienen la misma dimensión)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            pos = np.array([X[i, j], Y[i, j], z])
            F_att = f_attract(pos, goal, m_goal)
            F_rep = f_repulsive(pos, obstacles, R_soi)
            F_total = F_att + F_rep
            U[i, j] = F_total[0]
            V[i, j] = F_total[1]
            M_force[i, j] = np.linalg.norm(F_total)
    return U, V, M_force

# -------------------------------
# Representación del campo potencial
# -------------------------------

if __name__ == '__main__':
    # Cargar el mapa desde el archivo JSON
    json_file = "rrt_input.json"  # Asegúrate de que el archivo se encuentre en el directorio de trabajo
    bounds, obstacles = load_map(json_file)
    
    # Seleccionar la sección (slice) en z para visualizar el campo potencial
    z_slice = 0.0  # Por ejemplo, a mitad de la altura [0, 20]
    
    # Definir el punto objetivo. Para que la representación sea consistente en el slice,
    # elegimos un objetivo con coordenada z igual a z_slice.
    goal = np.array([8.0, -1.0, z_slice])
    
    # Crear una malla en el plano x-y usando los límites del mapa
    num_points = 30  # Resolución de la malla (puede ajustarse)
    x_vals = np.linspace(bounds["x"][0], bounds["x"][1], num_points)
    y_vals = np.linspace(bounds["y"][0], bounds["y"][1], num_points)
    X, Y = np.meshgrid(x_vals, y_vals)
    
    # Parámetros del campo
    m_goal = 1.0    # constante de atracción
    R_soi   = 3.0   # radio de influencia de la repulsión
    
    # Calcular el campo potencial sobre la malla (solo se consideran las componentes x e y)
    U, V, M_force = compute_potential_field(X, Y, z_slice, goal, obstacles, m_goal, R_soi)
    
    # Crear la figura
    plt.figure(figsize=(8, 6))
    
    # Dibujar un mapa de contornos de la magnitud del campo potencial
    contour = plt.contourf(X, Y, M_force, alpha=0.6, cmap='viridis')
    plt.colorbar(contour, label='Magnitud de la fuerza')
    
    # Superponer la representación vectorial (quiver) del campo (componentes x e y)
    plt.quiver(X, Y, U, V, color='white')
    
    # Dibujar los obstáculos que intersectan el slice en z
    ax = plt.gca()
    for obs in obstacles:
        center = obs["center"]
        size = obs["size"]
        z_min = center[2] - size[2] / 2.0
        z_max = center[2] + size[2] / 2.0
        # Solo se dibuja si el slice se encuentra dentro del obstáculo en z
        if z_slice >= z_min and z_slice <= z_max:
            x_min = center[0] - size[0] / 2.0
            y_min = center[1] - size[1] / 2.0
            rect = Rectangle((x_min, y_min), size[0], size[1],
                             linewidth=1, edgecolor='r', facecolor='none', alpha=0.8)
            ax.add_patch(rect)
    
    # Marcar el objetivo
    plt.plot(goal[0], goal[1], 'mo', markersize=8, label='Objetivo')
    
    # Configurar la gráfica
    plt.title("Campo Potencial en el plano x-y (z = {:.2f})".format(z_slice))
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.xlim(bounds["x"])
    plt.ylim(bounds["y"])
    plt.show()






