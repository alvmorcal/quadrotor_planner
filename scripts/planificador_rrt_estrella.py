#!/usr/bin/env python3

import rospy
import json
import numpy as np
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import subprocess
from scipy.spatial import KDTree
import random

# =============================================================================
# Clase RRTStar
# -----------------------------------------------------------------------------
# Esta clase implementa el algoritmo RRT* para la búsqueda de rutas en 3D
# con obstáculos. Genera árboles de nodos aleatorios y los expande de manera
# iterativa para encontrar un camino libre de colisiones desde un inicio
# hasta una meta.
# =============================================================================
class RRTStar:
    def __init__(self, start, goal, bounds, obstacles, step_size=0.5, max_iter=5000):
        """
        Constructor de la clase RRTStar.

        :param start: Punto inicial (x, y, z).
        :param goal: Punto objetivo (x, y, z).
        :param bounds: Límites del espacio de búsqueda en formato dict {"x": (min, max), "y": (min, max), "z": (min, max)}.
        :param obstacles: Lista de obstáculos, cada uno con 'pose' (centro) y 'size' (dimensiones).
        :param step_size: Tamaño de paso usado para expandir el árbol.
        :param max_iter: Máximo número de iteraciones para intentar encontrar un camino.
        """
        self.start = start
        self.goal = goal
        self.bounds = bounds
        self.obstacles = obstacles
        self.step_size = step_size
        self.max_iter = max_iter

        # Diccionario para guardar cada nodo y su nodo padre: tree[node] = parent
        self.tree = {tuple(start): None}

        # KDTree para búsqueda eficiente del nodo más cercano
        self.kd_tree = KDTree([start])

    def distance(self, p1, p2):
        """
        Calcula la distancia euclidiana entre dos puntos 3D.
        """
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def is_in_obstacle(self, point):
        """
        Verifica si un punto se encuentra dentro (o muy cerca) de alguno de los obstáculos.
        Se define un safety_distance para evitar colisiones cercanas a los bordes.
        """
        safety_distance = 0.7
        for obs in self.obstacles:
            x, y, z = obs["pose"]
            size_x, size_y, size_z = obs["size"]
            # Chequear si el punto está dentro del volumen del obstáculo con un margen de seguridad
            if (x - size_x / 2 - safety_distance <= point[0] <= x + size_x / 2 + safety_distance and
                y - size_y / 2 - safety_distance <= point[1] <= y + size_y / 2 + safety_distance and
                z - size_z / 2 - safety_distance <= point[2] <= z + size_z / 2 + safety_distance):
                return True
        return False

    def sample_random_point(self):
        """
        Genera un punto aleatorio dentro de los límites. Hay un 10% de probabilidad
        de tomar directamente la meta (bias), para acelerar la convergencia.
        """
        if random.random() < 0.1:
            return self.goal
        x = random.uniform(*self.bounds["x"])
        y = random.uniform(*self.bounds["y"])
        z = random.uniform(*self.bounds["z"])
        return [x, y, z]

    def add_node(self, node, parent):
        """
        Agrega un nuevo nodo al árbol, y actualiza la estructura KDTree.
        :param node: Nodo a agregar.
        :param parent: Nodo padre en el árbol RRT*.
        """
        self.tree[node] = parent
        points = list(self.tree.keys())
        self.kd_tree = KDTree(points)

    def nearest_node(self, point):
        """
        Retorna el nodo del árbol más cercano a un punto dado,
        utilizando la KDTree para eficiencia.
        """
        _, idx = self.kd_tree.query(point)
        return tuple(self.kd_tree.data[idx])

    def steer(self, from_node, to_point):
        """
        Mueve un paso desde 'from_node' en dirección a 'to_point', 
        con un tamaño de paso máximo 'step_size'.
        """
        direction = np.array(to_point) - np.array(from_node)
        length = np.linalg.norm(direction)

        if length == 0:
            return tuple(from_node)
        # Normalizar dirección y avanzar step_size o la distancia restante si es menor
        direction = direction / length
        new_point = np.array(from_node) + direction * min(self.step_size, length)
        return tuple(new_point)

    def is_collision_free(self, from_node, to_node):
        """
        Verifica que el trayecto entre dos nodos no colisione con ningún obstáculo.
        Se revisan puntos intermedios a intervalos de step_size.
        """
        # Crear una línea con un número de pasos proporcional a la distancia
        line = np.linspace(np.array(from_node), np.array(to_node), num=max(2, int(self.distance(from_node, to_node) / self.step_size)))
        return all(not self.is_in_obstacle(point) for point in line)

    def find_path(self):
        """
        Intenta encontrar un camino desde self.start hasta self.goal en un máximo
        de max_iter iteraciones. Si se alcanza la meta, reconstruye la ruta.
        """
        for _ in range(self.max_iter):
            # Paso 1: Muestrear un punto aleatorio (con bias a la meta)
            random_point = self.sample_random_point()
            # Paso 2: Encontrar el nodo existente más cercano al punto muestreado
            nearest = self.nearest_node(random_point)
            # Paso 3: Steer: generar un nuevo nodo en dirección al punto aleatorio
            new_node = self.steer(nearest, random_point)

            # Paso 4: Verificar colisiones y, si es factible, agregar el nuevo nodo
            if not self.is_in_obstacle(new_node) and self.is_collision_free(nearest, new_node):
                self.add_node(new_node, nearest)

                # Paso 5: Verificar si estamos lo suficientemente cerca de la meta
                if self.distance(new_node, self.goal) <= self.step_size:
                    self.add_node(tuple(self.goal), new_node)
                    return self.reconstruct_path()

        # Si no se logra encontrar un camino, se imprime un mensaje
        rospy.logwarn("No se encontró un camino válido")
        return []

    def reconstruct_path(self):
        """
        Reconstruye el camino desde la meta hasta el inicio, usando el árbol de padres.
        """
        path = []
        current = tuple(self.goal)
        # Retroceder desde la meta hasta llegar a un padre nulo (inicio)
        while current is not None:
            path.append(current)
            current = self.tree[current]
        # Invertir la lista para que vaya desde inicio -> meta
        return path[::-1]

    def smooth_path(self, path, max_checks=100):
        """
        Realiza una 'suavización' del camino encontrado, saltando nodos intermedios
        si no hay colisiones en la línea recta que los conecta.
        """
        if not path:
            return []

        smoothed_path = [path[0]]
        i = 0

        # Se recorre la ruta y se buscan tramos directos sin colisión
        while i < len(path) - 1:
            next_point_found = False
            # Se intenta conectar directamente el punto actual con nodos siguientes
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
# Clase PID
# -----------------------------------------------------------------------------
# Controlador PID genérico para un solo eje. Se utiliza para cada eje X, Y, Z
# en el dron para ajustar las velocidades en función del error.
# =============================================================================
class PID:
    def __init__(self, kp, ki, kd, setpoint=0):
        """
        Constructor del controlador PID.

        :param kp: Ganancia proporcional.
        :param ki: Ganancia integral.
        :param kd: Ganancia derivativa.
        :param setpoint: Valor objetivo para el controlador.
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.last_error = 0
        self.integral = 0

    def compute(self, current_value, dt):
        """
        Calcula la salida del controlador PID dado el valor actual, el setpoint y el tiempo transcurrido (dt).

        :param current_value: Valor medido actual.
        :param dt: Intervalo de tiempo desde el último cálculo.
        :return: Acción de control calculada (velocidad, fuerza, etc.).
        """
        error = self.setpoint - current_value
        self.integral += error * dt
        derivative = (error - self.last_error) / dt if dt > 0 else 0
        self.last_error = error

        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        return output

# =============================================================================
# Clase DroneNavigator
# -----------------------------------------------------------------------------
# Clase principal que se encarga de inicializar el nodo de ROS, suscribirse a
# la posición del dron, publicar comandos, y orquestar la planificación con 
# RRT* y el movimiento con PID.
# =============================================================================
class DroneNavigator:
    def __init__(self, bounds, obstacles):
        """
        Constructor de la clase DroneNavigator.

        :param bounds: Límites del espacio de búsqueda (dict con min/max para x, y, z).
        :param obstacles: Lista de obstáculos con sus posiciones y tamaños.
        """
        # Inicializar nodo de ROS
        rospy.init_node('drone_navigator', anonymous=True)

        self.bounds = bounds
        self.obstacles = obstacles

        # Publicador de la pose o comando de navegación
        self.pose_pub = rospy.Publisher('/command/pose', PoseStamped, queue_size=10)
        # Suscriptor de la posición real del dron
        self.pose_sub = rospy.Subscriber('/ground_truth/state', Odometry, self.pose_callback)

        self.current_pose = None
        self.rate = rospy.Rate(10)

        # Se instancian controladores PID para cada eje
        self.pid_x = PID(2.0, 0.01, 0.8)
        self.pid_y = PID(2.0, 0.01, 0.8)
        self.pid_z = PID(3.0, 0.05, 1.2)

    def pose_callback(self, msg):
        """
        Callback para la suscripción de /ground_truth/state, actualiza la posición actual del dron.
        """
        self.current_pose = msg.pose.pose.position

    def activate_motors(self):
        """
        Activa los motores del dron mediante el servicio /enable_motors.
        """
        rospy.loginfo("Activando motores...")
        try:
            subprocess.call(["rosservice", "call", "/enable_motors", "true"])
            rospy.loginfo("Motores activados con éxito.")
        except Exception as e:
            rospy.logerr(f"Error al activar los motores: {e}")

    def get_start_position(self):
        """
        Espera a que se reciba la primera posición del dron para definir
        el punto de inicio. En caso de no recibirse, se asume (0, 0, 0).
        """
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
        """
        Función que itera sobre cada waypoint de la ruta encontrada y 
        aplica control PID hasta llegar a cada uno de ellos.
        """
        rospy.loginfo("Iniciando movimiento hacia los waypoints...")

        for i, waypoint in enumerate(waypoints):
            if rospy.is_shutdown():
                break

            # Definir un radio de proximidad (más pequeño para el último waypoint)
            proximity_radius = 0.3

            # Configurar los setpoints para cada eje
            self.pid_x.setpoint = waypoint[0]
            self.pid_y.setpoint = waypoint[1]
            self.pid_z.setpoint = waypoint[2]

            while not rospy.is_shutdown():
                # Verificar que se tenga información de la pose actual
                if self.current_pose is None:
                    rospy.logwarn("Esperando datos de la posición actual...")
                    rospy.sleep(0.1)
                    continue

                # Calcular la distancia al waypoint
                current_position = np.array([self.current_pose.x, self.current_pose.y, self.current_pose.z])
                waypoint_position = np.array(waypoint)
                distance = np.linalg.norm(current_position - waypoint_position)

                # Si estamos dentro del radio de proximidad, pasar al siguiente waypoint
                if distance < proximity_radius:
                    rospy.loginfo(f"Waypoint alcanzado: {waypoint}")
                    break

                # Calcular la salida PID para cada eje
                dt = 0.1
                vx = self.pid_x.compute(self.current_pose.x, dt)
                vy = self.pid_y.compute(self.current_pose.y, dt)
                vz = self.pid_z.compute(self.current_pose.z, dt)

                # Construir y publicar el mensaje de pose, avanzando vx*dt, vy*dt, vz*dt
                pose_msg = PoseStamped()
                pose_msg.header.frame_id = "world"
                pose_msg.header.stamp = rospy.Time.now()
                pose_msg.pose.position.x = self.current_pose.x + vx * dt
                pose_msg.pose.position.y = self.current_pose.y + vy * dt
                pose_msg.pose.position.z = self.current_pose.z + vz * dt
                self.pose_pub.publish(pose_msg)

                rospy.sleep(dt)

        rospy.loginfo(f"Waypoints completados. Posición final: ({self.current_pose.x}, {self.current_pose.y}, {self.current_pose.z})")

    def plan_and_navigate(self):
        """
        Orquesta el proceso de planificación y navegación:
         1. Obtiene la posición actual (inicio).
         2. Solicita al usuario el punto objetivo (goal).
         3. Genera la ruta con RRT* y la suaviza.
         4. Navega a través de los waypoints del camino resultante.
         5. Al finalizar, vuelve a solicitar un nuevo objetivo.
        """
        while not rospy.is_shutdown():
            # Obtener la posición de inicio (la posición actual del dron)
            start = self.get_start_position()

            # Leer la meta por consola
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

            # Instanciar el planeador RRTStar
            rrt_star = RRTStar(start=start, goal=goal, bounds=self.bounds, obstacles=self.obstacles)

            rospy.loginfo("Planeando ruta con RRT*...")
            path = rrt_star.find_path()

            # Verificar si se encontró una ruta
            if not path:
                rospy.logerr("No se pudo encontrar una ruta válida. Intente con otro objetivo.")
                continue

            rospy.loginfo("Ruta encontrada. Suavizando...")
            smoothed_path = rrt_star.smooth_path(path)

            rospy.loginfo("Iniciando navegación hacia el objetivo...")
            self.move_to_waypoints(smoothed_path)

            rospy.loginfo("Objetivo alcanzado. Puede ingresar un nuevo objetivo.\n")

# =============================================================================
# Main del script
# -----------------------------------------------------------------------------
# Configura los límites de búsqueda, lee los obstáculos de un archivo JSON,
# instancia el DroneNavigator, activa los motores y llama a la función de
# planificación y navegación.
# =============================================================================
if __name__ == '__main__':
    try:
        # Definir los límites (en x, y, z)
        bounds = {
            "x": [-10, 10],
            "y": [-10, 10],
            "z": [0, 20]
        }

        # Cargar obstáculos desde archivo JSON
        with open("/home/alvmorcal/robmov_ws/src/quadrotor_planner/scripts/world.json", "r") as f:
            obstacles = json.load(f)["obstacles"]

        # Crear instancia de DroneNavigator
        navigator = DroneNavigator(bounds, obstacles)

        # Activar motores
        navigator.activate_motors()

        # Iniciar el ciclo de planificación y navegación (se repetirá hasta que se interrumpa la ejecución)
        navigator.plan_and_navigate()

    except rospy.ROSInterruptException:
        pass


