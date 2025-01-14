#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from math import atan2, sin, cos
import subprocess
import numpy as np

class TrajectoryPlanner:
    def __init__(self):
        rospy.init_node('trajectory_planner', anonymous=True)

        # Publicadores
        self.pose_pub = rospy.Publisher('/command/pose', PoseStamped, queue_size=10)

        # Suscriptores
        self.pose_sub = rospy.Subscriber('/ground_truth/state', Odometry, self.pose_callback)

        # Estado actual del quadrotor
        self.current_pose = None
        self.current_orientation = None  # Cuaternión de orientación

        # Frecuencia de ejecución
        self.rate = rospy.Rate(10)  # Frecuencia general

    def pose_callback(self, msg):
        """Callback para actualizar la posición y orientación actuales del quadrotor."""
        self.current_pose = msg.pose.pose.position
        self.current_orientation = msg.pose.pose.orientation

    def activate_motors(self):
        """Activa los motores del quadrotor."""
        rospy.loginfo("Activando motores...")
        try:
            subprocess.call(["rosservice", "call", "/enable_motors", "true"])
            rospy.loginfo("Motores activados con éxito.")
        except Exception as e:
            rospy.logerr(f"Error al activar los motores: {e}")

    def calculate_yaw(self, target):
        """Calcula el yaw deseado para mirar hacia el siguiente waypoint."""
        if self.current_pose is None:
            return None

        delta_x = target[0] - self.current_pose.x
        delta_y = target[1] - self.current_pose.y
        return atan2(delta_y, delta_x)

    def yaw_to_quaternion(self, yaw):
        """Convierte un ángulo yaw (radianes) a un cuaternión."""
        return {
            "x": 0.0,
            "y": 0.0,
            "z": sin(yaw / 2.0),
            "w": cos(yaw / 2.0)
        }

    def generate_trajectory(self, start, goal, steps=50):
        """Genera una trayectoria suave entre dos puntos usando interpolación lineal."""
        t = np.linspace(0, 1, steps)
        x = np.interp(t, [0, 1], [start[0], goal[0]])
        y = np.interp(t, [0, 1], [start[1], goal[1]])
        z = np.interp(t, [0, 1], [start[2], goal[2]])
        trajectory = list(zip(x, y, z))
        rospy.loginfo(f"Trayectoria generada hacia {goal}")
        return trajectory

    def move_to_goal(self, goal):
        """Controlador para mover el quadrotor hacia un objetivo."""
        if self.current_pose is None:
            rospy.logwarn("Esperando datos de la posición actual...")
            return

        # Obtener posición inicial
        start = [self.current_pose.x, self.current_pose.y, self.current_pose.z]
        
        # Generar trayectoria
        trajectory = self.generate_trajectory(start, goal)
        rospy.loginfo(f"Iniciando movimiento hacia {goal}...")

        # Publicar cada waypoint
        for waypoint in trajectory:
            if rospy.is_shutdown():
                break

            # Calcular yaw deseado
            yaw_desired = self.calculate_yaw(waypoint)
            if yaw_desired is None:
                continue

            # Convertir yaw a cuaternión
            yaw_quat = self.yaw_to_quaternion(yaw_desired)

            # Publicar waypoint como posición deseada
            pose_msg = PoseStamped()
            pose_msg.header.frame_id = "world"
            pose_msg.header.stamp = rospy.Time.now()
            pose_msg.pose.position.x = waypoint[0]
            pose_msg.pose.position.y = waypoint[1]
            pose_msg.pose.position.z = waypoint[2]
            pose_msg.pose.orientation.x = yaw_quat["x"]
            pose_msg.pose.orientation.y = yaw_quat["y"]
            pose_msg.pose.orientation.z = yaw_quat["z"]
            pose_msg.pose.orientation.w = yaw_quat["w"]
            self.pose_pub.publish(pose_msg)
            self.rate.sleep()

        rospy.loginfo(f"Meta alcanzada en posición: {goal}")
        rospy.loginfo(f"Posición final: {self.current_pose.x}, {self.current_pose.y}, {self.current_pose.z}")

    def run(self):
        """Loop principal."""
        self.activate_motors()

        while not rospy.is_shutdown():
            rospy.loginfo("Esperando nuevo objetivo...")
            try:
                goal = [
                    float(input("Ingrese X: ")), 
                    float(input("Ingrese Y: ")), 
                    float(input("Ingrese Z: "))
                ]
                rospy.loginfo(f"Meta recibida: {goal}")
                self.move_to_goal(goal)
            except ValueError:
                rospy.logwarn("Por favor, ingrese valores numéricos válidos.")

if __name__ == '__main__':
    try:
        planner = TrajectoryPlanner()
        planner.run()
    except rospy.ROSInterruptException:
        pass






