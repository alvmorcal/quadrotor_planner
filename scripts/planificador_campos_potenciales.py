import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import subprocess
import json

class PotentialFieldPlanner:
    def __init__(self, goal, obstacles, step_size=0.3, repulsive_scale=3.0, attractive_scale=1.5, R_soi=3.0):
        self.goal = np.array(goal)
        self.obstacles = obstacles
        self.step_size = step_size
        self.repulsive_scale = repulsive_scale
        self.attractive_scale = attractive_scale
        self.R_soi = R_soi

    def f_attract(self, position):
        diff = self.goal - position
        norm = np.linalg.norm(diff)
        return np.zeros(3) if norm == 0 else self.attractive_scale * diff / norm

    def closest_point_on_box(self, position, obs_pos, obs_size):
        half_size = np.array(obs_size) / 2.0
        min_corner = obs_pos - half_size
        max_corner = obs_pos + half_size
        return np.maximum(min_corner, np.minimum(position, max_corner))

    def f_repulsive(self, position):
        repulsive_force = np.zeros(3)
        for obs in self.obstacles:
            obs_pos = np.array(obs["pose"])
            obs_size = np.array(obs["size"])
            closest_point = self.closest_point_on_box(position, obs_pos, obs_size)
            diff = position - closest_point
            distance = np.linalg.norm(diff)
            if distance < self.R_soi and distance > 1e-6:
                repulsive_force += self.repulsive_scale * ((self.R_soi - distance) / self.R_soi) * (diff / distance)
        return repulsive_force

    def compute_field(self, position):
        return self.f_attract(position) - self.f_repulsive(position)

    def find_path(self, start):
        path = [start]
        position = np.array(start)
        for _ in range(500):  # Iteraciones limitadas para estabilidad
            field = self.compute_field(position)
            norm = np.linalg.norm(field)
            if norm < 0.001:
                break
            position = position + self.step_size * (field / norm)
            path.append(tuple(position))
            if np.linalg.norm(position - self.goal) < 0.05:
                break
        return path

class DroneNavigator:
    def __init__(self):
        rospy.init_node('drone_navigator_pf', anonymous=True)
        self.pose_pub = rospy.Publisher('/command/pose', PoseStamped, queue_size=10)
        self.pose_sub = rospy.Subscriber('/ground_truth/state', Odometry, self.pose_callback)
        self.current_pose = None
        self.obstacles = []
        self.rate = rospy.Rate(10)
        
        with open("/home/alvmorcal/robmov_ws/src/quadrotor_planner/scripts/rrt_input.json", "r") as f:
            self.obstacles = json.load(f)["obstacles"]
        
        rospy.loginfo("Esperando datos de posición...")
        while self.current_pose is None and not rospy.is_shutdown():
            rospy.sleep(0.1)
        rospy.loginfo(f"Posición inicial recibida: ({self.current_pose.x}, {self.current_pose.y}, {self.current_pose.z})")

    def pose_callback(self, msg):
        self.current_pose = msg.pose.pose.position

    def activate_motors(self):
        rospy.loginfo("Activando motores...")
        try:
            subprocess.call(["rosservice", "call", "/enable_motors", "true"])
        except Exception as e:
            rospy.logerr(f"Error al activar los motores: {e}")

    def move_to_waypoints(self, waypoints):
        rospy.loginfo("Moviendo al dron hacia los waypoints...")
        for waypoint in waypoints:
            while not rospy.is_shutdown():
                if self.current_pose is None:
                    rospy.sleep(0.1)
                    continue
                current_position = np.array([self.current_pose.x, self.current_pose.y, self.current_pose.z])
                distance = np.linalg.norm(current_position - np.array(waypoint))
                if distance < 0.1:
                    break
                pose_msg = PoseStamped()
                pose_msg.header.frame_id = "world"
                pose_msg.header.stamp = rospy.Time.now()
                pose_msg.pose.position.x = waypoint[0]
                pose_msg.pose.position.y = waypoint[1]
                pose_msg.pose.position.z = waypoint[2]
                self.pose_pub.publish(pose_msg)
                rospy.sleep(0.1)
        rospy.loginfo("Destino alcanzado.")

    def plan_and_navigate(self):
        start = (self.current_pose.x, self.current_pose.y, self.current_pose.z)
        goal = (float(input("Ingrese X: ")), float(input("Ingrese Y: ")), float(input("Ingrese Z: ")))
        rospy.loginfo(f"Meta establecida: {goal}")
        planner = PotentialFieldPlanner(goal, self.obstacles)
        path = planner.find_path(start)
        self.move_to_waypoints(path)

if __name__ == '__main__':
    try:
        navigator = DroneNavigator()
        navigator.activate_motors()
        navigator.plan_and_navigate()
    except rospy.ROSInterruptException:
        pass



