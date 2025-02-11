#!/usr/bin/env python3

import rospy
import json
import numpy as np
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import subprocess
import matplotlib
# Usar backend interactivo (asegúrate de tener instalado PyQt5)
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Para proyección 3D
from matplotlib.patches import Rectangle
from datetime import datetime

# Activar el modo interactivo de Matplotlib
plt.ion()

# =============================================================================
# Clase DroneNavigator
# -----------------------------------------------------------------------------
# Se encarga de comunicarse con ROS, activar motores, leer la posición actual y
# navegar usando campos potenciales. Además, detecta mínimos locales y, en ese
# caso, muestra un corte en el plano XY con la distribución del campo potencial,
# calculado de forma similar al ejemplo que compartiste, mostrando tanto el
# objetivo como la ubicación actual del UAV.
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
    # =============================================================================
    def potential_field_navigation(self, goal):
        # Parámetros para la navegación (se usan otros para el cálculo del campo 2D)
        k_att = 1.0   # coeficiente de atracción (usado para la navegación)
        k_rep = 2.0   # coeficiente de repulsión (usado para la navegación)
        d0 = 2.0      # distancia de influencia de los obstáculos (navegación)
        dt = 0.1      # paso de tiempo

        # Parámetros para detectar mínimos locales
        epsilon_force = 0.01       # si la norma de la fuerza total es menor que este valor
        local_min_threshold = 50   # iteraciones consecutivas con fuerza muy pequeña
        local_min_counter = 0

        trajectory = []  # almacena la trayectoria seguida
        goal_np = np.array(goal)

        while not rospy.is_shutdown():
            if self.current_pose is None:
                rospy.sleep(0.1)
                continue

            pos = np.array([self.current_pose.x, self.current_pose.y, self.current_pose.z])
            trajectory.append(pos.copy())

            # Verificar si se alcanzó el objetivo (radio de proximidad = 0.3)
            if np.linalg.norm(pos - goal_np) < 0.3:
                rospy.loginfo("Objetivo alcanzado.")
                break

            # Fuerza atractiva (modelo cuadrático para la navegación)
            F_att = -k_att * (pos - goal_np)

            # Fuerza repulsiva: contribución de cada obstáculo (modelo caja)
            F_rep_total = np.array([0.0, 0.0, 0.0])
            for obs in self.obstacles:
                obs_center = np.array(obs["pose"])
                size = np.array(obs["size"])
                min_corner = obs_center - size / 2.0
                max_corner = obs_center + size / 2.0
                # Punto más cercano en el obstáculo
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

            # Detectar mínimo local
            if norm_F_total < epsilon_force:
                local_min_counter += 1
            else:
                local_min_counter = 0

            if local_min_counter > local_min_threshold:
                rospy.logwarn("Mínimo local detectado. Deteniendo el trayecto.")
                self.display_force_field_slice(goal, pos)
                break

            # Actualizar posición integrando la "velocidad" (igual a la fuerza total)
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
    # Método para mostrar un corte en el plano XY a la altura actual del UAV, con
    # el campo potencial calculado siguiendo el modelo del ejemplo:
    #
    #   - Se usa un mapa de contornos de la magnitud de la fuerza.
    #   - Se superpone el campo vectorial (flechas) calculado con los métodos:
    #       f_attract y f_repulsive (basados en la distribución del ejemplo).
    #   - Se dibujan los obstáculos (si intersecan el slice).
    #   - Se marca el objetivo (punto magenta) y la posición actual del UAV (punto azul).
    # =============================================================================
    def display_force_field_slice(self, goal, pos_stuck):
        # Parámetros para el cálculo del campo (según el ejemplo)
        m_goal = 1.0    # constante de atracción
        R_soi = 3.0     # radio de influencia de la repulsión

        z_level = pos_stuck[2]
        x_min, x_max = self.bounds["x"]
        y_min, y_max = self.bounds["y"]
        num_points = 30  # Resolución de la malla
        x_vals = np.linspace(x_min, x_max, num_points)
        y_vals = np.linspace(y_min, y_max, num_points)
        X, Y = np.meshgrid(x_vals, y_vals)

        # --- Definición de funciones locales (idénticas al ejemplo) ---
        def closest_point_on_box(point, center, size):
            half = size / 2.0
            lower = center - half
            upper = center + half
            return np.maximum(lower, np.minimum(point, upper))

        def f_attract(pos, goal, m_goal=1.0):
            diff = goal - pos
            norm = np.linalg.norm(diff)
            if norm == 0:
                return np.zeros(3)
            return m_goal * diff / norm

        def f_repulsive(pos, obstacles, R_soi=3.0):
            force = np.zeros(3)
            for obs in obstacles:
                center = np.array(obs["pose"])
                size = np.array(obs["size"])
                Q = closest_point_on_box(pos, center, size)
                diff = pos - Q
                d = np.linalg.norm(diff)
                R_eff = min(size) / 2.0
                if d < R_soi:
                    if d < 1e-6:
                        d = 1e-6
                    m_obs = (R_soi - d) / (R_soi - R_eff) if R_soi > R_eff else 1.0
                    force += m_obs * (diff / d)
            return force

        def compute_potential_field(X, Y, z, goal, obstacles, m_goal=1.0, R_soi=3.0):
            U = np.zeros_like(X)
            V = np.zeros_like(Y)
            M_force = np.zeros_like(X)
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

        # Calcular el campo potencial en la malla (usando z = z_level)
        goal_np = np.array(goal)
        U, V, M_force = compute_potential_field(X, Y, z_level, goal_np, self.obstacles, m_goal, R_soi)

        # --- Representación 2D ---
        plt.figure(figsize=(8, 6))
        # Mapa de contornos de la magnitud del campo
        contour = plt.contourf(X, Y, M_force, alpha=0.6, cmap='viridis')
        plt.colorbar(contour, label='Magnitud de la fuerza')
        # Superponer el campo vectorial (flechas)
        plt.quiver(X, Y, U, V, color='white')

        ax = plt.gca()
        # Dibujar los obstáculos que intersectan el slice (z = z_level)
        for obs in self.obstacles:
            center = np.array(obs["pose"])
            size = np.array(obs["size"])
            z_min_obs = center[2] - size[2] / 2.0
            z_max_obs = center[2] + size[2] / 2.0
            if z_level >= z_min_obs and z_level <= z_max_obs:
                x_obs = center[0] - size[0] / 2.0
                y_obs = center[1] - size[1] / 2.0
                rect = Rectangle((x_obs, y_obs), size[0], size[1],
                                 linewidth=1, edgecolor='r', facecolor='none', alpha=0.8)
                ax.add_patch(rect)

        # Marcar el objetivo (punto magenta) y la posición actual del UAV (punto azul)
        plt.plot(goal[0], goal[1], 'mo', markersize=8, label='Objetivo')
        plt.plot(pos_stuck[0], pos_stuck[1], 'bo', markersize=8, label='UAV')
        plt.title("Campo Potencial en el plano XY a z = {:.2f}".format(z_level))
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.show()

    # =============================================================================
    # Método para mostrar la trayectoria seguida y los obstáculos en 3D
    # =============================================================================
    def display_route_plot(self, trajectory, start, goal):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Graficar los obstáculos (como cajas)
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
        ax.scatter([goal[0]], [goal[1]], [goal[2]], color="red", s=100, label="Objetivo")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()
        plt.show()

    # =============================================================================
    # Método principal que orquesta el proceso:
    #   1. Obtiene la posición actual (inicio).
    #   2. Solicita al usuario el objetivo.
    #   3. Ejecuta la navegación basada en campos potenciales.
    #   4. Si se alcanza la meta o se detecta un mínimo local, muestra el gráfico interactivo.
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
            rospy.loginfo("Navegación completada. Mostrando la trayectoria en 3D...")
            self.display_route_plot(trajectory, start, goal)
            rospy.loginfo("Puede ingresar un nuevo objetivo.\n")


# =============================================================================
# Main del script
# =============================================================================
if __name__ == '__main__':
    try:
        bounds = {
            "x": [-10, 10],
            "y": [-10, 10],
            "z": [0, 20]
        }
        # Se asume que el archivo world.json contiene una clave "obstacles"
        with open("/home/alvmorcal/robmov_ws/src/quadrotor_planner/scripts/world.json", "r") as f:
            obstacles = json.load(f)["obstacles"]

        navigator = DroneNavigator(bounds, obstacles)
        navigator.activate_motors()
        navigator.plan_and_navigate()

    except rospy.ROSInterruptException:
        pass




