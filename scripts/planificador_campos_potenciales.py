#!/usr/bin/env python3
import rospy
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import subprocess
import random
import time

# -------------------------------
# Función auxiliar para calcular la longitud de un camino
# -------------------------------
def compute_path_length(path):
    total_length = 0.0
    for i in range(len(path)-1):
        total_length += np.linalg.norm(np.array(path[i+1]) - np.array(path[i]))
    return total_length

# -------------------------------
# Planificador basado en Campo Potencial
# -------------------------------

class PotentialFieldPlanner:
    def __init__(self, start, goal, obstacles, step_size=0.2, goal_threshold=0.3,
                 max_iter=10000, m_goal=1.0, R_soi=3.0, k_rep=2.0):
        """
        Parámetros:
          - start: punto de inicio (lista o tupla de 3 elementos)
          - goal: punto objetivo (lista o tupla de 3 elementos)
          - obstacles: lista de obstáculos (cada uno con "pose" y "size")
          - step_size: tamaño de paso en cada iteración (en metros)
          - goal_threshold: distancia mínima para considerar alcanzado el objetivo
          - max_iter: número máximo de iteraciones
          - m_goal: constante de atracción
          - R_soi: radio de influencia de la repulsión de cada obstáculo
          - k_rep: factor que amplifica la fuerza repulsiva
        """
        self.start = np.array(start, dtype=float)
        self.goal = np.array(goal, dtype=float)
        self.obstacles = obstacles
        self.step_size = step_size
        self.goal_threshold = goal_threshold
        self.max_iter = max_iter
        self.m_goal = m_goal
        self.R_soi = R_soi
        self.k_rep = k_rep

    @staticmethod
    def closest_point_on_box(point, center, size):
        """Calcula el punto más cercano en un AABB (caja alineada con los ejes) dado 'center' y 'size'."""
        half = np.array(size) / 2.0
        lower = np.array(center) - half
        upper = np.array(center) + half
        return np.maximum(lower, np.minimum(point, upper))

    def f_attract(self, pos):
        """Calcula la fuerza atractiva hacia el objetivo."""
        diff = self.goal - pos
        norm = np.linalg.norm(diff)
        if norm == 0:
            return np.zeros(3)
        return self.m_goal * diff / norm

    def f_repulsive(self, pos):
        """Calcula la fuerza repulsiva total de los obstáculos sobre el punto pos."""
        force = np.zeros(3)
        for obs in self.obstacles:
            center = np.array(obs["pose"], dtype=float)
            size = np.array(obs["size"], dtype=float)
            Q = self.closest_point_on_box(pos, center, size)
            diff = pos - Q
            d = np.linalg.norm(diff)
            R_eff = min(size) / 2.0  # radio efectivo aproximado del obstáculo
            if d < self.R_soi and d > 1e-6:
                factor = (self.R_soi - d) / (self.R_soi - R_eff) if self.R_soi > R_eff else 1.0
                force += self.k_rep * factor * (diff / d)
        return force

    def find_path(self):
        """Calcula la trayectoria desde el inicio hasta el objetivo basándose en el campo potencial.
           Devuelve (path, status) donde status puede ser:
             - "success": se alcanzó el objetivo.
             - "local_minimum": se encontró un mínimo local (la fuerza total es casi 0).
             - "max_iter": se alcanzó el número máximo de iteraciones.
        """
        pos = self.start.copy()
        path = [pos.copy()]
        for _ in range(self.max_iter):
            if np.linalg.norm(self.goal - pos) < self.goal_threshold:
                rospy.loginfo("Objetivo alcanzado por el campo potencial.")
                return path, "success"

            F_att = self.f_attract(pos)
            F_rep = self.f_repulsive(pos)
            F_total = F_att + F_rep

            norm_F = np.linalg.norm(F_total)
            if norm_F < 1e-3:
                rospy.logwarn("¡Advertencia! Mínimo local encontrado en el campo potencial.")
                return path, "local_minimum"

            # Sin normalización: el paso es proporcional a la fuerza total
            step = F_total * self.step_size
            pos = pos + step
            path.append(pos.copy())

        rospy.logwarn("Se alcanzó el número máximo de iteraciones sin llegar al objetivo.")
        return path, "max_iter"

# -------------------------------
# Controlador PID
# -------------------------------

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

# -------------------------------
# Nodo de Navegación del Dron
# -------------------------------

class DroneNavigator:
    def __init__(self, bounds, obstacles):
        rospy.init_node('drone_navigator', anonymous=True)
        self.bounds = bounds
        self.obstacles = obstacles
        self.pose_pub = rospy.Publisher('/command/pose', PoseStamped, queue_size=10)
        self.pose_sub = rospy.Subscriber('/ground_truth/state', Odometry, self.pose_callback)
        self.current_pose = None
        self.rate = rospy.Rate(10)
        # Controladores PID para X, Y, Z
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
            rospy.logerr("No se pudo obtener la posición actual. Usando (0,0,0) como inicio.")
            return (0, 0, 0)

    def move_to_waypoints(self, waypoints):
        rospy.loginfo("Iniciando movimiento hacia los waypoints...")
        for waypoint in waypoints:
            if rospy.is_shutdown():
                break

            # Configura los setpoints para cada eje
            self.pid_x.setpoint, self.pid_y.setpoint, self.pid_z.setpoint = waypoint

            while not rospy.is_shutdown():
                if self.current_pose is None:
                    rospy.logwarn("Esperando datos de la posición actual...")
                    rospy.sleep(0.1)
                    continue

                current_position = np.array([self.current_pose.x, self.current_pose.y, self.current_pose.z])
                distance = np.linalg.norm(current_position - np.array(waypoint))
                if distance < 0.3:
                    rospy.loginfo(f"Waypoint alcanzado: {waypoint}")
                    break

                dt = 0.1
                vx = self.pid_x.compute(self.current_pose.x, dt)
                vy = self.pid_y.compute(self.current_pose.y, dt)
                vz = self.pid_z.compute(self.current_pose.z, dt)

                pose_msg = PoseStamped()
                pose_msg.header.frame_id = "world"
                pose_msg.header.stamp = rospy.Time.now()
                pose_msg.pose.position.x = self.current_pose.x + vx * dt
                pose_msg.pose.position.y = self.current_pose.y + vy * dt
                pose_msg.pose.position.z = self.current_pose.z + vz * dt
                self.pose_pub.publish(pose_msg)
                rospy.sleep(dt)
        rospy.loginfo("Navegación completada.")
        rospy.loginfo(f"Posición final: ({self.current_pose.x}, {self.current_pose.y}, {self.current_pose.z})")

    def plan_and_navigate(self):
        start = self.get_start_position()
        goal = (
            float(input("Ingrese objetivo X: ")),
            float(input("Ingrese objetivo Y: ")),
            float(input("Ingrese objetivo Z: "))
        )
        rospy.loginfo(f"Meta recibida: {goal}")

        pf_planner = PotentialFieldPlanner(start=start, goal=goal, obstacles=self.obstacles,
                                           step_size=0.2, goal_threshold=0.3, max_iter=10000,
                                           m_goal=1.0, R_soi=3.0, k_rep=2.0)
        
        rospy.loginfo("Calculando trayectoria mediante campo potencial...")
        planning_start_time = time.time()
        path, status = pf_planner.find_path()
        planning_end_time = time.time()
        planning_duration = planning_end_time - planning_start_time

        # Notificamos la situación encontrada, pero se continúa la navegación hasta el último waypoint
        if status != "success":
            rospy.logwarn("Se detectó un mínimo local o se alcanzó el máximo de iteraciones. Navegando hasta ese punto.")

        if not path or len(path) < 2:
            rospy.logerr("No se pudo calcular una trayectoria válida.")
            return

        rospy.loginfo("Trayectoria calculada con {} puntos. Iniciando navegación...".format(len(path)))
        waypoints = [tuple(p) for p in path]
        navigation_start_time = time.time()
        self.move_to_waypoints(waypoints)
        navigation_end_time = time.time()
        navigation_duration = navigation_end_time - navigation_start_time
        total_time = planning_duration + navigation_duration

        # Ahora, tras haber finalizado la navegación, se envía el mensaje de advertencia si no se logró el éxito
        if status != "success":
            rospy.logwarn("¡Advertencia! Mínimo local alcanzado en el campo potencial.")

        path_length = compute_path_length(path)
        rospy.loginfo(f"Longitud del camino: {path_length:.2f} metros")
        rospy.loginfo(f"Tiempo total: {total_time:.2f} segundos (Planificación: {planning_duration:.2f}s, Navegación: {navigation_duration:.2f}s)")
        return path, goal, pf_planner, total_time

# -------------------------------
# Funciones de visualización en 3D
# -------------------------------

def plot_box(ax, center, size, color='r', alpha=0.3):
    """Dibuja un obstáculo en forma de cubo en 3D."""
    c = np.array(center)
    s = np.array(size) / 2.0
    faces = [
        [c + s * np.array(offset) for offset in [(1,1,1), (1,1,-1), (1,-1,-1), (1,-1,1)]],
        [c + s * np.array(offset) for offset in [(1,1,1), (1,1,-1), (-1,1,-1), (-1,1,1)]],
        [c + s * np.array(offset) for offset in [(1,1,1), (1,-1,1), (-1,-1,1), (-1,1,1)]]
    ]
    ax.add_collection3d(Poly3DCollection(faces, facecolors=color, edgecolors='k', alpha=alpha))

def plot_field_vectors_3d(planner, bounds, grid_resolution=5, scale=0.5):
    """
    Dibuja en 3D los vectores de atracción (verde) y repulsión (rojo)
    evaluados en una malla 3D sobre el entorno.
    """
    x_vals = np.linspace(bounds["x"][0], bounds["x"][1], grid_resolution)
    y_vals = np.linspace(bounds["y"][0], bounds["y"][1], grid_resolution)
    z_vals = np.linspace(bounds["z"][0], bounds["z"][1], grid_resolution)
    
    X_att, Y_att, Z_att, U_att, V_att, W_att = [], [], [], [], [], []
    X_rep, Y_rep, Z_rep, U_rep, V_rep, W_rep = [], [], [], [], [], []
    
    for x in x_vals:
        for y in y_vals:
            for z in z_vals:
                pos = np.array([x, y, z])
                f_att = planner.f_attract(pos)
                f_rep = planner.f_repulsive(pos)
                
                X_att.append(x)
                Y_att.append(y)
                Z_att.append(z)
                U_att.append(f_att[0])
                V_att.append(f_att[1])
                W_att.append(f_att[2])
                
                X_rep.append(x)
                Y_rep.append(y)
                Z_rep.append(z)
                U_rep.append(f_rep[0])
                V_rep.append(f_rep[1])
                W_rep.append(f_rep[2])
    
    return (np.array(X_att), np.array(Y_att), np.array(Z_att),
            np.array(U_att), np.array(V_att), np.array(W_att),
            np.array(X_rep), np.array(Y_rep), np.array(Z_rep),
            np.array(U_rep), np.array(V_rep), np.array(W_rep))

def plot_3D_path(obstacles, path, goal, planner, bounds, scale=0.5):
    """Dibuja la trayectoria del dron, los obstáculos y los vectores de fuerza en 3D."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Dibujar obstáculos
    for obs in obstacles:
        plot_box(ax, obs["pose"], obs["size"])
    
    # Dibujar la trayectoria
    path = np.array(path)
    ax.plot(path[:, 0], path[:, 1], path[:, 2], '-b', label="Trayectoria", linewidth=2)
    ax.scatter(path[0, 0], path[0, 1], path[0, 2], color="g", s=100, label="Inicio")
    ax.scatter(goal[0], goal[1], goal[2], color="m", s=100, label="Objetivo")
    
    # Dibujar los vectores de fuerza en 3D
    (X_att, Y_att, Z_att, U_att, V_att, W_att,
     X_rep, Y_rep, Z_rep, U_rep, V_rep, W_rep) = plot_field_vectors_3d(planner, bounds, grid_resolution=5, scale=scale)
    
    # Flechas de atracción (verde)
    ax.quiver(X_att, Y_att, Z_att, U_att, V_att, W_att, length=scale, color='g', normalize=True)
    # Flechas de repulsión (rojo)
    ax.quiver(X_rep, Y_rep, Z_rep, U_rep, V_rep, W_rep, length=scale, color='r', normalize=True)
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Campo Potencial, Trayectoria y Vectores de Fuerza (3D)")
    ax.legend()
    plt.show()

# -------------------------------
# Ejecución principal
# -------------------------------

if __name__ == '__main__':
    try:
        # Definir límites para la visualización
        bounds = {"x": [-10, 10], "y": [-10, 10], "z": [0, 20]}
        with open("rrt_input.json", "r") as f:
            obstacles = json.load(f)["obstacles"]
        
        navigator = DroneNavigator(bounds, obstacles)
        navigator.activate_motors()
        rospy.sleep(2)  # Esperar para obtener la posición inicial
        
        path, goal, pf_planner, elapsed_time = navigator.plan_and_navigate()
        
        # Solo se procede a la visualización si se obtuvo una trayectoria válida
        if path and len(path) > 1:
            scale = 0.5  # Escala para los vectores de fuerza
            rospy.logwarn("Cierre la representación 3D para detener el programa.")
            plot_3D_path(obstacles, path, goal, pf_planner, bounds, scale)
        
        # Finalizar el nodo ROS para cerrar el proceso
        rospy.signal_shutdown("Navegación completada.")
    
    except rospy.ROSInterruptException:
        pass
