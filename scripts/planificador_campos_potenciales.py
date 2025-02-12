import rospy
import json
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')  # Backend interactivo para Ubuntu
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import subprocess

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

    def potential_field_navigation(self, goal):
        k_att = 1.0
        k_rep = 2.0
        d0 = 2.0
        dt = 0.1
        epsilon_force = 0.01
        local_min_threshold = 50
        local_min_counter = 0

        trajectory = []
        goal_np = np.array(goal)

        while not rospy.is_shutdown():
            if self.current_pose is None:
                rospy.sleep(0.1)
                continue
            pos = np.array([self.current_pose.x, self.current_pose.y, self.current_pose.z])
            trajectory.append(pos.copy())

            if np.linalg.norm(pos - goal_np) < 0.3:
                rospy.loginfo("Objetivo alcanzado.")
                break

            F_att = -k_att * (pos - goal_np)
            F_rep_total = np.array([0.0, 0.0, 0.0])

            for obs in self.obstacles:
                obs_center = np.array(obs["pose"])
                size = np.array(obs["size"])
                min_corner = obs_center - size / 2.0
                max_corner = obs_center + size / 2.0
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

            if norm_F_total < epsilon_force:
                local_min_counter += 1
            else:
                local_min_counter = 0

            if local_min_counter > local_min_threshold:
                rospy.logwarn("Mínimo local detectado. Generando mapa de fuerzas.")
                self.display_force_field_slice(goal, pos, k_att, k_rep, d0)
                break

            new_pos = pos + F_total * dt
            pose_msg = PoseStamped()
            pose_msg.header.frame_id = "world"
            pose_msg.header.stamp = rospy.Time.now()
            pose_msg.pose.position.x = new_pos[0]
            pose_msg.pose.position.y = new_pos[1]
            pose_msg.pose.position.z = new_pos[2]
            self.pose_pub.publish(pose_msg)
            rospy.sleep(dt)

        return trajectory

        def display_force_field_slice(self, goal, pos_stuck, k_att, k_rep, d0):
        """
        Representa el campo potencial en el plano XY (a la altura donde se detectó
        el mínimo local), usando un mapa de contornos de la magnitud de la fuerza y
        superponiendo el campo vectorial. Además, dibuja los obstáculos (si intersecan
        con el slice) y marca el objetivo.
        """
        from matplotlib.patches import Rectangle

        z_level = pos_stuck[2]
        x_min, x_max = self.bounds["x"]
        y_min, y_max = self.bounds["y"]
        grid_points = 30  # Mayor resolución para la malla
        x_vals = np.linspace(x_min, x_max, grid_points)
        y_vals = np.linspace(y_min, y_max, grid_points)
        grid_x, grid_y = np.meshgrid(x_vals, y_vals)

        F_total_x = np.zeros_like(grid_x)
        F_total_y = np.zeros_like(grid_y)
        M_force = np.zeros_like(grid_x)
        goal_np = np.array(goal)

        # Calcular el campo potencial en cada punto de la malla
        for i in range(grid_x.shape[0]):
            for j in range(grid_x.shape[1]):
                point = np.array([grid_x[i, j], grid_y[i, j], z_level])
                # Fuerza atractiva (potencial cuadrático)
                F_att = -k_att * (point - goal_np)
                # Fuerza repulsiva: se suma la contribución de cada obstáculo
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
                M_force[i, j] = np.linalg.norm(F_total)

        # Crear la figura con un tamaño adecuado
        plt.figure(figsize=(8, 6))
        # Mapa de contornos de la magnitud del campo potencial
        contour = plt.contourf(grid_x, grid_y, M_force, alpha=0.6, cmap='viridis')
        plt.colorbar(contour, label='Magnitud de la fuerza')
        # Superponer el campo vectorial (quiver)
        plt.quiver(grid_x, grid_y, F_total_x, F_total_y, color='white')

        # Dibujar los obstáculos que intersecan con el slice (z = z_level)
        ax = plt.gca()
        for obs in self.obstacles:
            obs_center = np.array(obs["pose"])
            size = np.array(obs["size"])
            z_min_obs = obs_center[2] - size[2] / 2.0
            z_max_obs = obs_center[2] + size[2] / 2.0
            if z_level >= z_min_obs and z_level <= z_max_obs:
                x_obs = obs_center[0] - size[0] / 2.0
                y_obs = obs_center[1] - size[1] / 2.0
                rect = Rectangle((x_obs, y_obs), size[0], size[1],
                                 linewidth=1, edgecolor='r', facecolor='none', alpha=0.8)
                ax.add_patch(rect)

        # Marcar el objetivo
        plt.plot(goal[0], goal[1], 'mo', markersize=8, label='Objetivo')
        plt.title("Campo Potencial en el plano XY a z = {:.2f}".format(z_level))
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.show()

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
            trajectory = self.potential_field_navigation(goal)
            rospy.loginfo("Generando reporte del trayecto...")
            self.display_force_field_slice(goal, start, 1.0, 2.0, 2.0)

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
        navigator.plan_and_navigate()

    except rospy.ROSInterruptException:
        pass



