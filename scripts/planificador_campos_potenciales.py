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
# Se usa un backend interactivo; asegúrate de tener instaladas las dependencias (por ejemplo, PyQt5)
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Asegura la proyección 3D
from datetime import datetime

# Activar modo interactivo de Matplotlib
plt.ion()

# =============================================================================
# Clase PID (opcional en este caso, ya que el control se realiza directamente
# mediante la integración de las fuerzas potenciales; se deja aquí por estructura)
# =============================================================================
class PID:
    def __init__(self, kp, ki, kd, setpoint=0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.last_error = 0
        self.integral = 0

    def compute(self, current_value, dt):
        error = self.setpoint - current_value
        self.integral += error * dt
        derivative = (error - self.last_error) / dt if dt > 0 else 0
        self.last_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative

# =============================================================================
# Clase DroneNavigator
# -----------------------------------------------------------------------------
# Esta clase se encarga de comunicarse con ROS (publicar y suscribirse a la 
# posición), activar los motores, leer la posición actual, y realizar la 
# navegación basada en campos potenciales.
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
        # Los controladores PID no se usan en esta implementación, pero se dejan para mantener
        # una estructura similar al script anterior.
        self.pid_x = PID(2.0, 0.01, 0.8)
        self.pid_y = PID(2.0, 0.01, 0.8)
        self.pid_z = PID(3.0, 0.05, 1.2)

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
    # Método de navegación basado en campos potenciales
    # -----------------------------------------------------------------------------
    # Se definen:
    #  - F_att: fuerza atractiva hacia la meta.
    #  - F_rep: fuerza repulsiva de cada obstáculo (solo si se encuentra dentro del radio
    #           de influencia d0).
    # La posición se actualiza integrando la "velocidad" (proporcional a la fuerza total).
    # Se considera que el objetivo está alcanzado si la distancia a la meta es menor que 0.3.
    # =============================================================================
    def potential_field_navigation(self, goal):
        # Parámetros de los campos potenciales
        k_att = 1.0   # coeficiente de atracción
        k_rep = 2.0   # coeficiente de repulsión
        d0 = 2.0      # distancia de influencia de los obstáculos
        dt = 0.1      # paso de tiempo

        trajectory = []  # para guardar la trayectoria seguida
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

            # Fuerza atractiva (derivada de una función potencial cuadrática)
            F_att = -k_att * (pos - goal_np)

            # Fuerza repulsiva: suma sobre todos los obstáculos
            F_rep_total = np.array([0.0, 0.0, 0.0])
            for obs in self.obstacles:
                obs_center = np.array(obs["pose"])
                diff = pos - obs_center
                d = np.linalg.norm(diff)
                if d < d0 and d > 0:
                    # Fórmula clásica: F_rep = k_rep*(1/d - 1/d0)*(1/d^2)*(diff/d)
                    F_rep = k_rep * (1.0/d - 1.0/d0) / (d**2) * (diff / d)
                    F_rep_total += F_rep

            # Fuerza total (suma de atractiva y repulsiva)
            F_total = F_att + F_rep_total

            # Se utiliza la fuerza total como "velocidad" (podrías escalarla o saturarla)
            v = F_total

            # Actualizar la posición integrando la velocidad
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
    # Método para mostrar de forma interactiva el gráfico 3D
    # -----------------------------------------------------------------------------
    # Se muestra:
    #  - La trayectoria seguida (línea azul con marcadores)
    #  - El punto de inicio (verde) y el de meta (rojo)
    #  - El mapa de obstáculos (dibujados como cubos grises semitransparentes)
    # =============================================================================
    def display_route_plot(self, trajectory, start, goal):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Graficar obstáculos (mapa)
        for obs in self.obstacles:
            center = obs["pose"]
            size = obs["size"]
            x0 = center[0] - size[0] / 2.0
            y0 = center[1] - size[1] / 2.0
            z0 = center[2] - size[2] / 2.0
            ax.bar3d(x0, y0, z0, size[0], size[1], size[2],
                     color="gray", alpha=0.3, shade=True)

        # Graficar la trayectoria
        trajectory = np.array(trajectory)
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                label="Trayectoria", color="blue", marker="o", markersize=3)

        # Graficar inicio y meta
        ax.scatter([start[0]], [start[1]], [start[2]], color="green", s=100, label="Inicio")
        ax.scatter([goal[0]], [goal[1]], [goal[2]], color="red", s=100, label="Fin")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()

        # Mostrar la figura de forma interactiva (puedes mover, rotar, hacer zoom)
        plt.show()

    # =============================================================================
    # Método principal que orquesta el proceso:
    #  1. Obtiene la posición actual (inicio).
    #  2. Solicita al usuario el objetivo.
    #  3. Ejecuta la navegación basada en campos potenciales.
    #  4. Muestra el gráfico interactivo con la trayectoria y el mapa.
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

