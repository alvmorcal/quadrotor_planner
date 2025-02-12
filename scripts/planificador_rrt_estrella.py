#!/usr/bin/env python3

import rospy
import json
import numpy as np
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import subprocess
from scipy.spatial import KDTree
import random
import os
import matplotlib
# Se establece un backend interactivo (por ejemplo, Qt5Agg)
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Asegura la proyección 3D
from datetime import datetime

# Activar modo interactivo de Matplotlib
plt.ion()

# =============================================================================
# Clase RRTStar
# -----------------------------------------------------------------------------
# Implementa el algoritmo RRT* para la búsqueda de rutas en 3D con obstáculos.
# =============================================================================
class RRTStar:
    def __init__(self, start, goal, bounds, obstacles, step_size=0.5, max_iter=5000):
        self.start = start
        self.goal = goal
        self.bounds = bounds
        self.obstacles = obstacles
        self.step_size = step_size
        self.max_iter = max_iter
        self.tree = {tuple(start): None}
        self.kd_tree = KDTree([start])

    def distance(self, p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def is_in_obstacle(self, point):
        safety_distance = 0.7
        for obs in self.obstacles:
            x, y, z = obs["pose"]
            size_x, size_y, size_z = obs["size"]
            if (x - size_x/2 - safety_distance <= point[0] <= x + size_x/2 + safety_distance and
                y - size_y/2 - safety_distance <= point[1] <= y + size_y/2 + safety_distance and
                z - size_z/2 - safety_distance <= point[2] <= z + size_z/2 + safety_distance):
                return True
        return False

    def sample_random_point(self):
        if random.random() < 0.1:
            return self.goal
        x = random.uniform(*self.bounds["x"])
        y = random.uniform(*self.bounds["y"])
        z = random.uniform(*self.bounds["z"])
        return [x, y, z]

    def add_node(self, node, parent):
        self.tree[node] = parent
        points = list(self.tree.keys())
        self.kd_tree = KDTree(points)

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
        line = np.linspace(np.array(from_node), np.array(to_node),
                           num=max(2, int(self.distance(from_node, to_node) / self.step_size)))
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
        rospy.logwarn("No se encontró un camino válido")
        return []

    def reconstruct_path(self):
        path = []
        current = tuple(self.goal)
        while current is not None:
            path.append(current)
            current = self.tree[current]
        return path[::-1]

    def smooth_path(self, path, max_checks=100):
        if not path:
            return []
        smoothed_path = [path[0]]
        i = 0
        while i < len(path) - 1:
            next_point_found = False
            for j in range(len(path) - 1, i, -1):
                if self.is_collision_free(smoothed_path[-1], path[j]):
                    smoothed_path.append(path[j])
                    i = j
                    next_point_found = True
                    break
            if not next_point_found:
                break
        return smoothed_path

# =============================================================================
# Clase DroneNavigator
# -----------------------------------------------------------------------------
# Gestiona la comunicación con ROS, la navegación del dron y la visualización
# interactiva de la trayectoria junto con el mapa (obstáculos).
# =============================================================================
class DroneNavigator:
    def __init__(self, bounds, obstacles):
        rospy.init_node('drone_navigator', anonymous=True)
        self.bounds = bounds
        self.obstacles = obstacles
        self.pose_pub = rospy.Publisher('/command/pose', PoseStamped, queue_size=10)
        self.pose_sub = rospy.Subscriber('/ground_truth/state', Odometry, self.pose_callback)
        self.current_pose = None
        self.rate = rospy.Rate(10)

    def pose_callback(self, msg):
        self.current_pose = msg.pose.pose.position

    def activate_motors(self):
        rospy.loginfo("Activando motores...")
        try:
            subprocess.call(["rosservice", "call", "/enable_motors", "true"])
            rospy.loginfo("Motores activados con éxito.")
        except Exception as e:
            rospy.logerr(f"Error al activar los motores: {e}")

    def get_start_position(self):
        rospy.loginfo("Esperando datos de la posición actual...")
        while self.current_pose is None and not rospy.is_shutdown():
            rospy.sleep(0.1)
        if self.current_pose is not None:
            start = (self.current_pose.x, self.current_pose.y, self.current_pose.z)
            rospy.loginfo(f"Inicio detectado: {start}")
            return start
        else:
            rospy.logerr("No se pudo obtener la posición actual. Usando (0, 0, 0) como inicio.")
            return (0, 0, 0)

    def move_to_waypoints(self, waypoints):
        rospy.loginfo("Iniciando movimiento hacia los waypoints...")
        proximity_radius = 0.3  # Radio de proximidad para considerar alcanzado un waypoint
        for waypoint in waypoints:
            if rospy.is_shutdown():
                break
            rospy.loginfo(f"Moviendo hacia waypoint: {waypoint}")
            while not rospy.is_shutdown():
                if self.current_pose is None:
                    rospy.logwarn("Esperando datos de la posición actual...")
                    rospy.sleep(0.1)
                    continue
                current_position = np.array([self.current_pose.x, self.current_pose.y, self.current_pose.z])
                waypoint_position = np.array(waypoint)
                distance = np.linalg.norm(current_position - waypoint_position)
                if distance < proximity_radius:
                    rospy.loginfo(f"Waypoint alcanzado: {waypoint}")
                    break
                # Se envía directamente la posición del waypoint como comando
                pose_msg = PoseStamped()
                pose_msg.header.frame_id = "world"
                pose_msg.header.stamp = rospy.Time.now()
                pose_msg.pose.position.x = waypoint[0]
                pose_msg.pose.position.y = waypoint[1]
                pose_msg.pose.position.z = waypoint[2]
                self.pose_pub.publish(pose_msg)
                rospy.sleep(0.1)
        if self.current_pose is not None:
            rospy.loginfo(f"Waypoints completados. Posición final: ({self.current_pose.x}, {self.current_pose.y}, {self.current_pose.z})")
        else:
            rospy.loginfo("Waypoints completados.")

    def display_route_plot(self, path, start, goal):
        """
        Genera y muestra de forma interactiva una figura 3D con:
          - La trayectoria seguida (línea azul con marcadores)
          - El punto de inicio (verde)
          - El punto de destino (rojo)
          - El mapa de obstáculos (cubo gris semitransparente)
        La ventana quedará abierta hasta que el usuario la cierre.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Graficar el mapa: Obstáculos
        for obs in self.obstacles:
            center = obs["pose"]
            size = obs["size"]
            x0 = center[0] - size[0] / 2.0
            y0 = center[1] - size[1] / 2.0
            z0 = center[2] - size[2] / 2.0
            ax.bar3d(x0, y0, z0, size[0], size[1], size[2],
                     color="gray", alpha=0.3, shade=True)

        # Graficar la trayectoria
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        zs = [p[2] for p in path]
        ax.plot(xs, ys, zs, label="Trayectoria", color="blue", marker="o", markersize=3)

        # Graficar el inicio y la meta
        ax.scatter([start[0]], [start[1]], [start[2]], color="green", s=100, label="Inicio")
        ax.scatter([goal[0]], [goal[1]], [goal[2]], color="red", s=100, label="Fin")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()

        # Mostrar la figura de forma interactiva (la ejecución se bloquea hasta que la ventana se cierra)
        plt.show()

    def plan_and_navigate(self):
        """
        Orquesta la planificación y navegación:
          1. Obtiene la posición inicial.
          2. Solicita el objetivo.
          3. Planifica la ruta con RRT* y la suaviza.
          4. Navega a través de los waypoints.
          5. Muestra el gráfico interactivo con la trayectoria y el mapa.
        """
        while not rospy.is_shutdown():
            start = self.get_start_position()
            try:
                goal = (
                    float(input("Ingrese la coordenada X del nuevo objetivo: ")),
                    float(input("Ingrese la coordenada Y del nuevo objetivo: ")),
                    float(input("Ingrese la coordenada Z del nuevo objetivo: "))
                )
            except ValueError:
                rospy.logerr("Valores ingresados inválidos. Intente nuevamente.")
                continue

            rospy.loginfo(f"Meta recibida: {goal}")
            rrt_star = RRTStar(start=start, goal=goal, bounds=self.bounds, obstacles=self.obstacles)
            rospy.loginfo("Planeando ruta con RRT*...")
            path = rrt_star.find_path()
            if not path:
                rospy.logerr("No se pudo encontrar una ruta válida. Intente con otro objetivo.")
                continue

            rospy.loginfo("Ruta encontrada. Suavizando...")
            smoothed_path = rrt_star.smooth_path(path)
            rospy.loginfo("Iniciando navegación hacia el objetivo...")
            self.move_to_waypoints(smoothed_path)
            rospy.loginfo("Objetivo alcanzado. Mostrando la trayectoria de forma interactiva...")
            self.display_route_plot(smoothed_path, start, goal)
            rospy.loginfo("Puede ingresar un nuevo objetivo.\n")

# =============================================================================
# Main del script
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    try:
        bounds = {
            "x": [-10, 10],
            "y": [-10, 10],
            "z": [0, 20]
        }
        with open("/home/alvmorcal/robmov_ws/src/quadrotor_planner/scripts/world.json", "r") as f:
            obstacles = json.load(f)["obstacles"]

        navigator = DroneNavigator(bounds, obstacles)
        navigator.activate_motors()
        navigator.plan_and_navigate()

    except rospy.ROSInterruptException:
        pass

