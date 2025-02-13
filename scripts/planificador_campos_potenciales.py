#!/usr/bin/env python3

import rospy
import json
import numpy as np
from geometry_msgs.msg import TwistStamped  # Usamos TwistStamped para enviar velocidades
from nav_msgs.msg import Odometry
import subprocess
import matplotlib
# Usar backend interactivo (asegúrate de tener instalado PyQt5)
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Para la proyección 3D
from matplotlib.patches import Rectangle
from datetime import datetime
import threading

# Activar el modo interactivo de Matplotlib
plt.ion()

# =============================================================================
# Clase PID
# -----------------------------------------------------------------------------
# Implementa un controlador PID simple para cada eje.
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
# Se encarga de comunicarse con ROS, activar los motores, leer la posición
# actual y navegar usando campos potenciales. Además, se implementa un mecanismo
# para mantener al UAV en hover (publicando comandos de velocidad) mientras se 
# espera un nuevo objetivo o se calcula la ruta.
# =============================================================================
class DroneNavigator:
    def __init__(self, bounds, obstacles):
        rospy.init_node('drone_navigator', anonymous=True)
        self.bounds = bounds
        self.obstacles = obstacles
        # Publicador para comandos de velocidad (TwistStamped)
        self.vel_pub = rospy.Publisher('/command/twist', TwistStamped, queue_size=10)
        self.pose_sub = rospy.Subscriber('/ground_truth/state', Odometry, self.pose_callback)
        self.current_pose = None
        self.rate = rospy.Rate(10)

        # Inicializamos los controladores PID para cada eje.
        self.pid_x = PID(1.5, 0.02, 0.8)
        self.pid_y = PID(1.5, 0.02, 0.8)
        self.pid_z = PID(3.0, 0.05, 1.2)

        # Variable para controlar el hilo de hover
        self.hover_stop_event = None

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

    def maintain_hover(self, dt=0.05):
        """
        Mantiene el UAV en hover publicando continuamente comandos basados en 
        los controladores PID para mantener la posición inicial.
        Se supone que antes de llamar a esta función se fija la posición de hover.
        """
        # Fijar la posición de hover
        hover_position = (self.current_pose.x, self.current_pose.y, self.current_pose.z)
        self.pid_x.setpoint = hover_position[0]
        self.pid_y.setpoint = hover_position[1]
        self.pid_z.setpoint = hover_position[2]
        rospy.loginfo("Iniciando modo hover...")
        while not self.hover_stop_event.is_set() and not rospy.is_shutdown():
            vx = self.pid_x.compute(self.current_pose.x, dt)
            vy = self.pid_y.compute(self.current_pose.y, dt)
            vz = self.pid_z.compute(self.current_pose.z, dt)
            twist_msg = TwistStamped()
            twist_msg.header.stamp = rospy.Time.now()
            twist_msg.header.frame_id = "world"
            twist_msg.twist.linear.x = vx
            twist_msg.twist.linear.y = vy
            twist_msg.twist.linear.z = vz
            twist_msg.twist.angular.x = 0
            twist_msg.twist.angular.y = 0
            twist_msg.twist.angular.z = 0
            self.vel_pub.publish(twist_msg)
            rospy.sleep(dt)

    # =============================================================================
    # Navegación por campos potenciales
    #
    # En cada iteración:
    #   - Se calcula la fuerza atractiva (hacia el objetivo) y repulsiva (de los obstáculos).
    #   - Se define una posición deseada usando la integración (pos + F_total * dt).
    #   - Se actualizan los setpoints de cada PID y se calculan las velocidades.
    #   - Se publica un mensaje TwistStamped con esas velocidades.
    #
    # Si se alcanza el objetivo (dentro de un radio de 0.3) se finaliza.
    # =============================================================================
    def potential_field_navigation(self, goal):
        # Parámetros para la navegación por campos potenciales
        k_att = 1.5  # coeficiente de atracción
        k_rep = 2.5   # coeficiente de repulsión
        d0 = 4.0      # distancia de influencia de los obstáculos
        dt = 0.1      # intervalo de tiempo

        # Parámetros para la detección de mínimos locales
        epsilon_force = 0.3       # umbral para considerar que hay un mínimo local
        local_min_threshold = 15   # iteraciones consecutivas con fuerza muy pequeña
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
                stop_msg = TwistStamped()
                stop_msg.header.stamp = rospy.Time.now()
                stop_msg.header.frame_id = "world"
                stop_msg.twist.linear.x = 0
                stop_msg.twist.linear.y = 0
                stop_msg.twist.linear.z = 0
                self.vel_pub.publish(stop_msg)
                break

            # Calcular fuerza atractiva (modelo lineal)
            F_att = -k_att * (pos - goal_np)

            # Calcular fuerza repulsiva para cada obstáculo (modelo de "caja")
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

            # Detección de mínimo local
            if norm_F_total < epsilon_force:
                local_min_counter += 1
            else:
                local_min_counter = 0

            if local_min_counter > local_min_threshold:
                rospy.logwarn("Mínimo local detectado. Deteniendo la navegación.")
                self.display_force_field_slice(goal, pos)
                break

            # Se define la posición deseada (integración simple)
            desired_pos = pos + F_total * dt

            # Actualizar los setpoints de los PID
            self.pid_x.setpoint = desired_pos[0]
            self.pid_y.setpoint = desired_pos[1]
            self.pid_z.setpoint = desired_pos[2]

            # Calcular velocidades en cada eje usando el PID
            vx = self.pid_x.compute(pos[0], dt)
            vy = self.pid_y.compute(pos[1], dt)
            vz = self.pid_z.compute(pos[2], dt)

            twist_msg = TwistStamped()
            twist_msg.header.stamp = rospy.Time.now()
            twist_msg.header.frame_id = "world"
            twist_msg.twist.linear.x = vx
            twist_msg.twist.linear.y = vy
            twist_msg.twist.linear.z = vz
            twist_msg.twist.angular.x = 0
            twist_msg.twist.angular.y = 0
            twist_msg.twist.angular.z = 0
            self.vel_pub.publish(twist_msg)

            rospy.sleep(dt)

        return trajectory

    # =============================================================================
    # Método para representar en 2D un corte en el plano XY (a la altura actual del UAV)
    # con el campo potencial calculado.
    # =============================================================================
    def display_force_field_slice(self, goal, pos_stuck):
        m_goal = 1.0    # constante de atracción
        R_soi = 3.0     # radio de influencia de la repulsión

        z_level = pos_stuck[2]
        x_min, x_max = self.bounds["x"]
        y_min, y_max = self.bounds["y"]
        num_points = 30
        x_vals = np.linspace(x_min, x_max, num_points)
        y_vals = np.linspace(y_min, y_max, num_points)
        X, Y = np.meshgrid(x_vals, y_vals)

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

        goal_np = np.array(goal)
        U, V, M_force = compute_potential_field(X, Y, z_level, goal_np, self.obstacles, m_goal, R_soi)

        plt.figure(figsize=(8, 6))
        contour = plt.contourf(X, Y, M_force, alpha=0.6, cmap='viridis')
        plt.colorbar(contour, label='Magnitud de la fuerza')
        plt.quiver(X, Y, U, V, color='white')

        ax = plt.gca()
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
    # Método para representar la trayectoria seguida en 3D.
    # =============================================================================
    def display_route_plot(self, trajectory, start, goal):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Dibujar obstáculos
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
    #   1. Obtiene la posición actual (que se usará como hover).
    #   2. Inicia un hilo para mantener el hover mientras se espera un nuevo objetivo.
    #   3. Una vez ingresado el nuevo objetivo, se detiene el hover y se ejecuta la
    #      navegación por campos potenciales.
    #   4. Se muestra la trayectoria y se vuelve a esperar un nuevo objetivo.
    # =============================================================================
    def plan_and_navigate(self):
        while not rospy.is_shutdown():
            start = self.get_start_position()

            # Iniciar modo hover mientras se espera el nuevo objetivo
            self.hover_stop_event = threading.Event()
            hover_thread = threading.Thread(target=self.maintain_hover, args=(0.05,))
            hover_thread.start()

            try:
                goal = (
                    float(input("Ingrese la coordenada X del nuevo objetivo: ")),
                    float(input("Ingrese la coordenada Y del nuevo objetivo: ")),
                    float(input("Ingrese la coordenada Z del nuevo objetivo: "))
                )
            except ValueError:
                rospy.logerr("Valores ingresados inválidos. Intente nuevamente.")
                self.hover_stop_event.set()
                hover_thread.join()
                continue

            # Detener el modo hover antes de comenzar la navegación
            self.hover_stop_event.set()
            hover_thread.join()

            rospy.loginfo(f"Meta recibida: {goal}")

            # Actualizar los setpoints de los PID para fijar la posición actual (hover) mientras se planifica
            hover_position = (self.current_pose.x, self.current_pose.y, self.current_pose.z)
            self.pid_x.setpoint = hover_position[0]
            self.pid_y.setpoint = hover_position[1]
            self.pid_z.setpoint = hover_position[2]

            rospy.loginfo("Iniciando navegación por campos potenciales con control PID...")
            trajectory = self.potential_field_navigation(goal)
            rospy.loginfo("Navegación completada. Mostrando trayectoria en 3D...")
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
        with open("/home/alvmorcal/robmov_ws/src/quadrotor_planner/scripts/world.json", "r") as f:
            obstacles = json.load(f)["obstacles"]

        navigator = DroneNavigator(bounds, obstacles)
        navigator.activate_motors()
        navigator.plan_and_navigate()

    except rospy.ROSInterruptException:
        pass







