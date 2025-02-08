#!/usr/bin/env python3
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Rectangle

# -------------------------------
# Funciones de carga y utilidades
# -------------------------------

def load_map(json_file):
    """Carga el mapa desde el JSON y obtiene los límites y obstáculos."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data["bounds"], [
        {"name": obs["name"], "center": np.array(obs["pose"], dtype=float), "size": np.array(obs["size"], dtype=float)}
        for obs in data["obstacles"]
    ]

def closest_point_on_box(point, center, size):
    """Calcula el punto más cercano en un AABB al punto dado."""
    return np.maximum(center - size / 2.0, np.minimum(point, center + size / 2.0))

# -------------------------------
# Funciones del campo potencial
# -------------------------------

def f_attract(pos, goal, m_goal=1.0):
    """Calcula la fuerza atractiva hacia el objetivo."""
    diff = goal - pos
    norm = np.linalg.norm(diff)
    return np.zeros(3) if norm == 0 else m_goal * diff / norm

def f_repulsive(pos, obstacles, R_soi=3.0, k_rep=3):
    """Calcula la fuerza repulsiva generada por los obstáculos."""
    force = np.zeros(3)
    for obs in obstacles:
        Q = closest_point_on_box(pos, obs["center"], obs["size"])
        diff, d = pos - Q, np.linalg.norm(pos - Q)
        R_eff = min(obs["size"]) / 2.0  
        if d < R_soi and d > 1e-6:
            force += k_rep * ((R_soi - d) / (R_soi - R_eff)) * (diff / d) if R_soi > R_eff else 1.0
    return force

# -------------------------------
# Cálculo del campo potencial
# -------------------------------

def compute_potential_field(X, Y, Z, goal, obstacles, m_goal=1.0, R_soi=3.0, k_rep=2.0):
    """Calcula el campo potencial total en una malla 2D o 3D."""
    U, V, W, M_force = np.zeros_like(X), np.zeros_like(Y), np.zeros_like(Z), np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            for k in range(X.shape[2]):
                pos = np.array([X[i, j, k], Y[i, j, k], Z[i, j, k]])
                F_total = f_attract(pos, goal, m_goal) + f_repulsive(pos, obstacles, R_soi, k_rep)
                U[i, j, k], V[i, j, k], W[i, j, k], M_force[i, j, k] = *F_total, np.linalg.norm(F_total)
    
    return U, V, W, M_force

# -------------------------------
# Representación gráfica
# -------------------------------

def plot_2D(X, Y, M_force, U, V, obstacles, goal, z_slice):
    """Dibuja el campo potencial en 2D en el plano x-y a una altura z dada."""
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(X[:, :, 0], Y[:, :, 0], M_force[:, :, 0], cmap='viridis', alpha=0.7)
    plt.colorbar(contour, label='Magnitud de la fuerza')
    plt.quiver(X[:, :, 0], Y[:, :, 0], U[:, :, 0], V[:, :, 0], color='white', scale=30)

    ax = plt.gca()
    for obs in obstacles:
        z_min, z_max = obs["center"][2] - obs["size"][2] / 2, obs["center"][2] + obs["size"][2] / 2
        if z_min <= z_slice <= z_max:
            ax.add_patch(Rectangle((obs["center"][0] - obs["size"][0] / 2, obs["center"][1] - obs["size"][1] / 2),
                                   obs["size"][0], obs["size"][1], linewidth=1, edgecolor='r', facecolor='none'))

    plt.scatter(goal[0], goal[1], color='m', s=100, label='Objetivo')
    plt.xlabel("X"), plt.ylabel("Y"), plt.title(f"Campo Potencial 2D (z = {z_slice})")
    plt.legend(), plt.show()

def plot_3D(X, Y, Z, U, V, W, obstacles, goal):
    """Dibuja el campo potencial en 3D."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(X, Y, Z, U, V, W, length=0.5, normalize=True, color='b', arrow_length_ratio=0.3)

    for obs in obstacles:
        plot_box(ax, obs["center"], obs["size"])

    ax.scatter(goal[0], goal[1], goal[2], color='m', s=100, label='Objetivo')
    ax.set_xlabel("X"), ax.set_ylabel("Y"), ax.set_zlabel("Z"), ax.set_title("Campo Potencial 3D")
    ax.legend(), plt.show()

def plot_box(ax, center, size, color='r', alpha=0.3):
    """Dibuja un obstáculo en forma de caja en 3D."""
    c, s = np.array(center), np.array(size) / 2.0
    faces = [[c + s * np.array(offset) for offset in [(1,1,1),(1,1,-1),(1,-1,-1),(1,-1,1)]],
             [c + s * np.array(offset) for offset in [(1,1,1),(1,1,-1),(-1,1,-1),(-1,1,1)]],
             [c + s * np.array(offset) for offset in [(1,1,1),(1,-1,1),(-1,-1,1),(-1,1,1)]]]
    ax.add_collection3d(Poly3DCollection(faces, facecolors=color, edgecolors='k', alpha=alpha))

# -------------------------------
# Ejecución principal
# -------------------------------

if __name__ == '__main__':
    bounds, obstacles = load_map("rrt_input.json")
    mode = input("¿Quieres la representación en 2D o 3D? (escribe '2D' o '3D'): ").strip().upper()
    z_slice = float(input("Introduce la altura Z para la vista 2D: ")) if mode == "2D" else None
    goal = np.array([8, 8, 0])

    # Mayor resolución de la malla
    nx, ny, nz = 15, 15, 10  
    X, Y, Z = np.meshgrid(np.linspace(bounds["x"][0], bounds["x"][1], nx),
                          np.linspace(bounds["y"][0], bounds["y"][1], ny),
                          [z_slice] if z_slice else np.linspace(bounds["z"][0], bounds["z"][1], nz), indexing='ij')

    U, V, W, M_force = compute_potential_field(X, Y, Z, goal, obstacles)

    if mode == "2D":
        plot_2D(X, Y, M_force, U, V, obstacles, goal, z_slice)
    else:
        plot_3D(X, Y, Z, U, V, W, obstacles, goal)