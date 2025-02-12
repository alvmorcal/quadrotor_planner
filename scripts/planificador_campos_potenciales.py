#!/usr/bin/env python3

import rospy
import json
import numpy as np
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import subprocess
import random
import os
import matplotlib
# Se usa un backend interactivo (asegúrate de tener instalado PyQt5)
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Asegura la proyección 3D
from datetime import datetime

# Activar el modo interactivo de Matplotlib
plt.ion()

# =============================================================================
# Clase DroneNavigator
# -----------------------------------------------------------------------------
# Esta clase se encarga de comunicarse con ROS (publicar y suscribirse a la 
# posición), activar los motores, leer la posición actual y realizar la 
# navegación basada en campos potenciales. Además, implementa la detección 
# de mínimos locales y genera un corte en el plano con el mapa de fuerzas.
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

    # =============================================================================
    # Método de navegación basado en campos potenciales con detección de mínimos locales
    # -----------------------------------------------------------------------------
    # Se calculan:
    #  - F_att: Fuerza atractiva hacia la meta.
    #  - F_rep: Fuerza repulsiva de cada obstáculo (considerando sus dimensiones).
    # Se actualiza la posición integrando la "velocidad" (proporcional a la fuerza total).
    # Si la fuerza neta es muy pequeña durante muchas iteraciones, se asume que hay un mínimo local.
    # En ese caso, se detiene el trayecto y se llama a display_force_field_slice para mostrar
    # un corte en el plano XY a la altura actual, con el mapa de distribución de fuerzas.
    # =============================================================================
    def potential_field_navigation(self, goal):
        # Parámetros de campos potenciales
        k_att = 1.0   # coeficiente de atracción
        k_rep = 2.0   # coeficiente de repulsión
        d0 = 2.0      # distancia de influencia de los obstáculos
        dt = 0.1      # paso de tiempo

        # Parámetros para detectar mínimos locales
        epsilon_force = 0.01       # si la norma de la fuerza total es menor que este valor
        local_min_threshold = 50   # número de iteraciones consecutivas con fuerza muy pequeña
        local_min_counter = 0

        trajectory = []  # almacena la trayectoria seguida
        goal_np = np.array(goal)

        while not rospy.is_shutdown():
            if self.current_pose is None:
                rospy.sleep(0.1)
                continue
            pos = np.array([self.current_pose.x, self.current_pose.y, self.current_pose.z])
            trajectory.append(pos.copy())

            # Verificar si se alcanzó la meta (se usa proximity_radius = 0.3)
            if np.linalg.norm(pos - goal_np) < 0.3:
                rospy.loginfo("Objetivo alcanzado.")
                break

            # Fuerza atractiva (potencial cuadrático)
            F_att = -k_att * (pos - goal_np)

            # Fuerza repulsiva: se suma la contribución de cada obstáculo
            F_rep_total = np.array([0.0, 0.0, 0.0])
            for obs in self.obstacles:
                obs_center = np.array(obs["pose"])
                size = np.array(obs["size"])
                min_corner = obs_center - size / 2.0
                max_corner = obs_center + size / 2.0
                # Se calcula el punto más cercano en el obstáculo (modelo caja)
                closest_point = np.clip(pos, min_corner, max_corner)
                diff = pos - closest_point
                d = np.linalg.norm(diff)
                if d < d0:
                    if d == 0:
                        d = 0.001
                        diff = pos - obs_center
                    F_rep = k_rep * (1.0/d - 1.0/d0) / (d**2) * (diff / d)
                    F_rep_total += F_rep

            F_total = F_att + F_rep_total
            norm_F_total = np.linalg.norm(F_total)

            # Detectar mínimo local: si la fuerza total es muy pequeña durante varias iteraciones
            if norm_F_total < epsilon_force:
                local_min_counter += 1
            else:
                local_min_counter = 0

            if local_min_counter > local_min_threshold:
                rospy.logwarn("Se ha detectado un mínimo local. Deteniendo el trayecto.")
                self.display_force_field_slice(goal, pos, k_att, k_rep, d0)
                break

            # Actualizar la posición integrando la "velocidad" (igual a la fuerza total)
            v = F_total
            new_pos = pos + v * dt

            # Publicar el nuevo comando de posición
            pose_msg = PoseStamped()
            pose_msg.header.frame_id = "world"
            pose_msg.header.stamp = rospy.Time.now()
            pose_msg.pose.position.x = new_pos[0]
            pose_msg.pose.position.y = new_pos[1]
            pose_msg.pose.position.z = new_pos[2]
            self.pose_pub.publish(pose_msg)

            rospy.sleep(dt)

        return trajectory

    # =============================================================================
    # Método para generar un corte en el plano XY (a la altura en la que se quedó estancado el UAV)
    # con el mapa de distribución de las fuerzas.
    # Se calcula la fuerza total (suma de atractiva y repulsiva) en una grilla de puntos en el plano,
    # y se muestra con un gráfico de vectores (quiver).
    # =============================================================================
    def display_force_field_slice(self, goal, pos_stuck, k_att, k_rep, d0):
        z_level = pos_stuck[2]
        x_min, x_max = self.bounds["x"]
        y_min, y_max = self.bounds["y"]
        grid_points = 20
        x_vals = np.linspace(x_min, x_max, grid_points)
        y_vals = np.linspace(y_min, y_max, grid_points)
        grid_x, grid_y = np.meshgrid(x_vals, y_vals)

        F_total_x = np.zeros_like(grid_x)
        F_total_y = np.zeros_like(grid_y)
        goal_np = np.array(goal)

        for i in range(grid_x.shape[0]):
            for j in range(grid_x.shape[1]):
                point = np.array([grid_x[i, j], grid_y[i, j], z_level])
                F_att = -k_att * (point - goal_np)
                F_rep_total = np.array([0.0, 0.0, 0.0])
                for obs in self.obstacles:
                    obs_center = np.array(obs["pose"])
                    size = np.array(obs["size"])
                    min_corner = obs_center - size / 2.0
                    max_corner = obs_center + size / 2.0
                    closest_point = np.clip(point, min_corner, max_corner)
                    diff = point - closest_point
                    d = np.linalg.norm(diff)
                    if d < d0:
                        if d == 0:
                            d = 0.001
                            diff = point - obs_center
                        F_rep = k_rep * (1.0/d - 1.0/d0) / (d**2) * (diff / d)
                        F_rep_total += F_rep
                F_total = F_att + F_rep_total
                F_total_x[i, j] = F_total[0]
                F_total_y[i, j] = F_total[1]

        plt.figure()
        plt.quiver(grid_x, grid_y, F_total_x, F_total_y, color='b')
        plt.title("Distribución de fuerzas en el plano XY a z = {:.2f}".format(z_level))
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()

    # =============================================================================
    # Método para mostrar de forma interactiva la trayectoria seguida y el mapa de obstáculos
    # =============================================================================
    def display_route_plot(self, trajectory, start, goal):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Graficar los obstáculos (mapa)
        for obs in self.obstacles:
            center = obs["pose"]
            size = obs["size"]
            x0 = center[0] - size[0] / 2.0
            y0 = center[1] - size[1] / 2.0
            z0 = center[2] - size[2] / 2.0
            ax.bar3d(x0, y0, z0, size[0], size[1], size[2],
                     color="gray", alpha=0.3, shade=True)

        trajectory = np.array(trajectory)
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                label="Trayectoria", color="blue", marker="o", markersize=3)

        ax.scatter([start[0]], [start[1]], [start[2]], color="green", s=100, label="Inicio")
        ax.scatter([goal[0]], [goal[1]], [goal[2]], color="red", s=100, label="Fin")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()
        plt.show()

    # =============================================================================
    # Método principal que orquesta el proceso:
    #  1. Obtiene la posición actual (inicio).
    #  2. Solicita al usuario el objetivo.
    #  3. Ejecuta la navegación basada en campos potenciales.
    #  4. Si se alcanza la meta o se detecta un mínimo local, muestra el gráfico interactivo.
    # =============================================================================
    def plan_and_navigate(self):
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
            rospy.loginfo("Iniciando navegación con campos potenciales...")
            trajectory = self.potential_field_navigation(goal)
            rospy.loginfo("Navegación completada. Mostrando la trayectoria de forma interactiva...")
            self.display_route_plot(trajectory, start, goal)
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


