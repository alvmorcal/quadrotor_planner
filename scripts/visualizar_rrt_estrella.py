import json
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import KDTree

class RRTStar:
    def __init__(self, start, goal, bounds, obstacles, step_size=0.5, max_iter=5000):
        self.start = start
        self.goal = goal
        self.bounds = bounds
        self.obstacles = obstacles
        self.step_size = step_size
        self.max_iter = max_iter
        self.tree = {tuple(start): None}
        self.kd_tree = KDTree([start])  # Inicializar KD-Tree

    def distance(self, p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def is_in_obstacle(self, point):
        safety_distance = 0.5
        for obs in self.obstacles:
            x, y, z = obs["pose"]
            size_x, size_y, size_z = obs["size"]
            if (x - size_x / 2 - safety_distance <= point[0] <= x + size_x / 2 + safety_distance and
                y - size_y / 2 - safety_distance <= point[1] <= y + size_y / 2 + safety_distance and
                z - size_z / 2 - safety_distance <= point[2] <= z + size_z / 2 + safety_distance):
                return True
        return False

    def sample_random_point(self):
        if random.random() < 0.1:  # 10% de probabilidad de sesgo hacia el objetivo
            return self.goal
        x = random.uniform(*self.bounds["x"])
        y = random.uniform(*self.bounds["y"])
        z = random.uniform(*self.bounds["z"])
        return [x, y, z]

    def add_node(self, node, parent):
        self.tree[node] = parent
        points = list(self.tree.keys())
        self.kd_tree = KDTree(points)  # Actualizar KD-Tree

    def nearest_node(self, point):
        _, idx = self.kd_tree.query(point)
        return tuple(self.kd_tree.data[idx])

    def steer(self, from_node, to_point):
        direction = np.array(to_point) - np.array(from_node)
        length = np.linalg.norm(direction)
        if length == 0:
            return tuple(from_node)
        direction = direction / length
        new_point = np.array(from_node) + direction * min(self.step_size, length)
        return tuple(new_point)

    def is_collision_free(self, from_node, to_node):
        line = np.linspace(np.array(from_node), np.array(to_node), num=max(2, int(self.distance(from_node, to_node) / self.step_size)))
        return all(not self.is_in_obstacle(point) for point in line)

    def find_path(self):
        for _ in range(self.max_iter):
            random_point = self.sample_random_point()
            nearest = self.nearest_node(random_point)

            new_node = self.steer(nearest, random_point)
            if not self.is_in_obstacle(new_node) and self.is_collision_free(nearest, new_node):
                self.add_node(new_node, nearest)

                if self.distance(new_node, self.goal) <= self.step_size:
                    self.add_node(tuple(self.goal), new_node)
                    return self.reconstruct_path()

        print("No se encontró un camino válido")
        return []

    def reconstruct_path(self):
        path = []
        current = tuple(self.goal)
        while current is not None:
            path.append(current)
            current = self.tree[current]
        return path[::-1]
    
    def smooth_path(self, path, max_checks=100):
        """
        Suaviza la ruta final eliminando puntos innecesarios mientras se mantiene libre de colisiones.
        :param path: La ruta original obtenida por RRT*
        :param max_checks: Número máximo de intentos para verificar un segmento
        :return: Una ruta suavizada
        """
        if not path:
            return []
        
        smoothed_path = [path[0]]  # Mantén el primer punto
        i = 0  # Índice del punto actual
        
        while i < len(path) - 1:
            next_point_found = False
            for j in range(len(path) - 1, i, -1):  # Busca hacia adelante en la ruta
                if self.is_collision_free(smoothed_path[-1], path[j]):
                    smoothed_path.append(path[j])  # Añade el punto más lejano accesible
                    i = j  # Avanza el índice
                    next_point_found = True
                    break
            
            if not next_point_found:
                break  # Si no se encuentra un punto válido, detén el suavizado
        
        return smoothed_path

def visualize_rrt(bounds, obstacles, path, tree):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlim(bounds["x"])
    ax.set_ylim(bounds["y"])
    ax.set_zlim(bounds["z"])

    for obs in obstacles:
        x, y, z = obs["pose"]
        dx, dy, dz = obs["size"]
        ax.bar3d(x - dx / 2, y - dy / 2, z - dz / 2, dx, dy, dz, alpha=0.5, color='red')

    for node, parent in tree.items():
        if parent is not None:
            ax.plot([node[0], parent[0]], [node[1], parent[1]], [node[2], parent[2]], 'gray', alpha=0.5)

    if path:
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], path[:, 2], 'blue', linewidth=2, label='Ruta')

    ax.set_title("RRT* Path Planning")
    ax.legend()
    plt.show()

if __name__ == "__main__":
    bounds = {
        "x": [-10, 10],
        "y": [-10, 10],
        "z": [0, 20]
    }

    with open("world.json", "r") as f:
        obstacles = json.load(f)["obstacles"]

    start = [
    float(input("(Origen) Ingrese X: ")), 
    float(input("(Origen) Ingrese Y: ")), 
    float(input("(Origen) Ingrese Z: "))
    ]
    goal = [
    float(input("(Meta)  Ingrese X: ")), 
    float(input("(Meta) Ingrese Y: ")), 
    float(input("(Meta) Ingrese Z: "))
    ]

    rrt_star = RRTStar(start, goal, bounds, obstacles)

    # Encontrar la trayectoria
    path = rrt_star.find_path()

    # Suavizar la trayectoria
    smoothed_path = rrt_star.smooth_path(path)

    # Visualizar el resultado
    visualize_rrt(bounds, obstacles, smoothed_path, rrt_star.tree)


