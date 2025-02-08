#!/usr/bin/env python3

import rospy
import json
import numpy as np
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import subprocess

class CampoPotencialPlanner:
    def _init_(self):
        rospy.init_node('campo_potencial_planner', anonymous=True)

        # Cargar el mapa
        with open("rrt_input.json", "r") as f:
            mapa = json.load(f)
        self.bounds = mapa["bounds"]
        self.obstacles = mapa["obstacles"]
        
        self.pose_pub = rospy.Publisher('/command/pose', PoseStamped, queue_size=10)
        self.pose_sub = rospy.Subscriber('/ground_truth/state', Odometry, self.pose_callback)

        self.current_pose = None
        self.rate = rospy.Rate(10)
        
        self.k_att = 1.0  # Coeficiente de atracción
        self.k_rep = 2.0  # Coeficiente de repulsión
        self.R_soi = 3.0  # Radio de influencia de obstáculos
        self.step_size = 0.2  # Tamaño de paso
        self.goal_threshold = 0.2  # Umbral para considerar que se alcanzó la meta
        self.max_iter = 10000  # Límite de iteraciones
        
        rospy.loginfo("Nodo de planificación por campos potenciales iniciado.")
        self.solicitar_meta()

    def pose_callback(self, msg):
        self.current_pose = msg.pose.pose.position

    def solicitar_meta(self):
        rospy.loginfo("Introduce la meta (x, y, z):")
        x = float(input("x: "))
        y = float(input("y: "))
        z = float(input("z: "))
        self.goal = np.array([x, y, z])
        rospy.loginfo("Meta establecida en: {}".format(self.goal))
        self.planificar()

    def f_attract(self, pos):
        diff = self.goal - pos
        norm = np.linalg.norm(diff)
        return np.zeros(3) if norm == 0 else self.k_att * diff / norm

    def f_repulsive(self, pos):
        force = np.zeros(3)
        for obs in self.obstacles:
            center = np.array(obs["pose"])
            size = np.array(obs["size"]) / 2.0
            closest = np.maximum(center - size, np.minimum(pos, center + size))
            diff = pos - closest
            d = np.linalg.norm(diff)
            R_eff = min(size) / 2.0  
            if d < self.R_soi and d > 1e-6:
                m_obs = (self.R_soi - d) / (self.R_soi - R_eff) if self.R_soi > R_eff else 1.0
                force += self.k_rep * m_obs * (diff / d)
        return force

    def planificar(self):
        rospy.loginfo("Esperando datos de la posición actual...")
        while self.current_pose is None and not rospy.is_shutdown():
            rospy.sleep(0.1)
        
        start = np.array([self.current_pose.x, self.current_pose.y, self.current_pose.z])
        pos = start.copy()
        rate = rospy.Rate(10)
        
        for _ in range(self.max_iter):
            if np.linalg.norm(self.goal - pos) < self.goal_threshold:
                rospy.loginfo("Meta alcanzada.")
                return
            
            F_att = self.f_attract(pos)
            F_rep = self.f_repulsive(pos)
            F_total = F_att + F_rep
            
            norm_F = np.linalg.norm(F_total)
            if norm_F == 0:
                rospy.logwarn("Se detectó un mínimo local. Abortando...")
                return
            
            if norm_F > self.step_size:
                F_total = (F_total / norm_F) * self.step_size
            
            pos += F_total
            self.enviar_comando(pos)
            rate.sleep()
        
        rospy.logwarn("Se alcanzó el límite de iteraciones sin llegar a la meta.")

    def enviar_comando(self, posicion):
        pose_msg = PoseStamped()
        pose_msg.header.frame_id = "world"
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.pose.position.x = posicion[0]
        pose_msg.pose.position.y = posicion[1]
        pose_msg.pose.position.z = posicion[2]
        self.pose_pub.publish(pose_msg)
    
if __name__ == '__main__':
    try:
        planner = CampoPotencialPlanner()
    except rospy.ROSInterruptException:
        pass
