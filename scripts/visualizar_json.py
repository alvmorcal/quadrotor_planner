import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_environment(json_file):
    """
    Carga un archivo JSON y visualiza los obstáculos en un espacio 3D.
    """
    # Cargar el archivo JSON
    with open(json_file, "r") as f:
        data = json.load(f)

    bounds = data["bounds"]
    obstacles = data["obstacles"]

    # Crear figura 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Dibujar obstáculos
    for obs in obstacles:
        x, y, z = obs["pose"]
        size_x, size_y, size_z = obs["size"]
        ax.bar3d(
            x - size_x / 2,  # Coordenada x del obstáculo
            y - size_y / 2,  # Coordenada y del obstáculo
            z - size_z / 2,  # Coordenada z del obstáculo
            size_x, size_y, size_z,  # Tamaño del cubo
            alpha=0.6, color="red"
        )

    # Configurar límites del espacio
    ax.set_xlim(bounds["x"])
    ax.set_ylim(bounds["y"])
    ax.set_zlim(bounds["z"])

    # Etiquetas
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Visualización del Entorno")

    # Mostrar gráfico
    plt.show()

# Ruta al archivo JSON
json_file = "/home/alvmorcal/robmov_ws/src/quadrotor_planner/scripts/rrt_input.json"
visualize_environment(json_file)
