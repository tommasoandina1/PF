import numpy as np
import yaml

import tf_transformations

import rclpy

from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from landmark_msgs.msg import LandmarkArray


import matplotlib.pyplot as plt
from .plot_utils import plot_initial_particles, plot_particles 




from .utils import residual, state_mean, simple_resample
from .pf import RobotPF
from .probabilistic_models import (
    sample_velocity_motion_model,
    landmark_range_bearing_model,
)

class PF(Node):
    def __init__(self):
        super().__init__('pf')

        # Perform the prediction step at a fixed rate of 20 Hz
        self.declare_parameter('frequency', 20.0)
        self.dt = 1.0 / self.get_parameter('frequency').get_parameter_value().double_value

        dim_x = 3
        dim_u = 2
        eval_gux = sample_velocity_motion_model

        # General noise parameters
        std_lin_vel = 0.01  # [m/s]
        std_ang_vel = np.deg2rad(0.1)  # [rad/s]
        self.sigma_z = np.array([std_lin_vel, std_ang_vel])
        self.sigma_u = np.array([std_lin_vel, std_ang_vel])
        self.vel = np.array([0.01, 0.01])

        # RobotPF initialization
        self.pf = RobotPF(
            dim_x=dim_x,
            dim_u=dim_u,
            eval_gux=eval_gux,
            resampling_fn=simple_resample,
            boundaries=[(-5, 5), (-5, 5), (-np.pi, np.pi)],
            N=1000,
        )
        self.pf.mu = np.array([-2, -0.5, 0])  
        self.pos_gt = np.array([-2, -0.5, 0])
        self.pos_odom = np.array([-2, -0.5, 0])

        self.pf.Sigma = np.diag([0.1, 0.1, 0.1])
        self.pf.initialize_particles()

        # Plot initial particles
        fig, ax = plt.subplots(figsize=(8, 6))
        init_particles_lgnd = plot_initial_particles(self.pf.N, self.pf.particles, ax=ax)
        ax.set_title("Initial Particles Distribution")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        # Assicurati di passare un'etichetta (label) per il grafico
        ax.legend([init_particles_lgnd], ["Initial Particles"])  # Aggiungi un'etichetta qui
        plt.show()  # Shows the plot of initial particles

        # Landmark model setup
        self.eval_hx_landm = landmark_range_bearing_model

        # LANDMARKS LIST
        # Load the YAML file
        with open("/home/tommaso/ros_ws/src/turtlebot3_perception/turtlebot3_perception/config/landmarks.yaml", "r") as file:
            data = yaml.safe_load(file)
        ids = data['landmarks']['id']
        x_coords = data['landmarks']['x']
        y_coords = data['landmarks']['y']
        self.landmarks = np.array([[id_, x, y] for id_, x, y in zip(ids, x_coords, y_coords)])

        # Parametri per il tracciamento delle particelle
        self.track_pf = []  # Lista per tracciare la posizione stimata
        self.particles_plot_step = 10  # Numero di passi tra ogni plot delle particelle


        # Subscription setup
        self.odom_subscriber = self.create_subscription(Odometry, '/odom', self.odom_callback, 10) 
        self.cam_subscriber = self.create_subscription(LandmarkArray, '/landmarks', self.landmarks_callback, 10)
        self.gt_subscriber = self.create_subscription(Odometry, '/ground_truth', self.gt_callback, 10)

        # Publisher setup
        self.state_publisher = self.create_publisher(Odometry, '/pf', 10)

        # Timer to perform prediction operation of the PF
        self.timer = self.create_timer(self.dt, self.prediction)
        
        # Inizializzazione del grafico per le particelle
        self.fig, self.ax = plt.subplots(figsize=(8, 6))


    def odom_callback(self, msg):
        _, _, yaw = tf_transformations.euler_from_quaternion([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])
        self.pos_odom = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, yaw])  
        #self.get_logger().info(f"Odom: {self.pos_odom}")  
        self.vel = np.array([msg.twist.twist.linear.x, msg.twist.twist.angular.z]) + np.array([1e-9, 1e-9])

        
    #def cmd_vel_callback(self, msg):
     #   self.vel = np.array([msg.linear.x, msg.angular.z]) + np.array([1e-9, 1e-9])

    def gt_callback(self, msg):
        _, _, yaw = tf_transformations.euler_from_quaternion([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])
        self.pos_gt = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, yaw])
        #self.get_logger().info(f"GT: {self.pos_gt}")

    def prediction(self):
        self.pf.predict(
            u            = self.vel,
            sigma_u      = self.sigma_u,
            g_extra_args = (self.dt,)
        )
    
        self.update_stmp()
        self.update_tracking()

    def landmarks_callback(self, msg):
        for z in self.landmarks:
            for landmark in msg.landmarks:
                if landmark.id == z[0]:
                    lmark = [z[1], z[2]]
                    z_data = np.array([landmark.range, landmark.bearing])
                    
                    #self.get_logger().info(f"Particle weights before update: {self.pf.weights}")
                    self.pf.update(z=z_data, sigma_z=self.sigma_z, eval_hx=self.eval_hx_landm, hx_args=(lmark, self.sigma_z))
                    #self.get_logger().info(f"Particle weights after update: {self.pf.weights}")
                    
                    
        self.pf.normalize_weights()
        neff = self.pf.neff()
        self.get_logger().info(f"NEFF: {neff}")

        if neff < self.pf.N / 2:
            self.get_logger().info(f"Performing resampling at NEFF: {neff}")
            self.pf.resampling(resampling_fn=self.pf.resampling_fn, resampling_args=(self.pf.weights,))
            self.get_logger().info(f"Particles after resampling: {self.pf.particles}")
            assert np.allclose(self.pf.weights, 1 / self.pf.N)
        self.pf.estimate(mean_fn=state_mean, residual_fn=residual, angle_idx=2)

    def update_tracking(self):
        
        # Traccia la posizione media delle particelle
        self.track_pf.append(self.pf.mu.copy())

        # Plot delle particelle ogni `particles_plot_step` passi
        if len(self.track_pf) % self.particles_plot_step == 0:
            legend_PF1, legend_PF2 = plot_particles(self.pf.particles, self.pos_gt, self.pf.mu, ax=self.ax)
            self.ax.set_title("Particle Filter - Step: " + str(len(self.track_pf)))
            self.ax.set_xlabel("X")
            self.ax.set_ylabel("Y")
            self.ax.legend([legend_PF1, legend_PF2], ["Particles", "Estimated Position"])
            plt.draw()  # Rende visibile il grafico in tempo reale

        # Stampa informazioni sullo stato del filtro
        self.get_logger().info(f"Step: {len(self.track_pf)} - NEFF: {self.pf.neff()}")

    def update_stmp(self):
        state_msg = Odometry()
        state_msg.pose.pose.position.x = self.pf.mu[0]
        state_msg.pose.pose.position.y = self.pf.mu[1]
        state_msg.header.stamp = self.get_clock().now().to_msg()
        self.state_publisher.publish(state_msg)




def main(args=None):
   rclpy.init(args=args)
   
   try:
      seed = 42 
      np.random.seed(seed)


      node = PF()
      rclpy.spin(node)
   except ValueError:
        exit(1)
   except KeyboardInterrupt:
      pass
   finally :
      rclpy.try_shutdown()




if __name__ == '__main__':
    main()