import rospy
import numpy as np
from gazebo_connection import GazeboConnection

class QuadrotorEnv():

    def __init__(self):
        self.gazebo = GazeboConnection()

    def _reset(self):

        # 1st: resets the simulation to initial values
        self.gazebo.resetSim()

        # 2nd: Unpauses simulation
        self.gazebo.unpauseSim()

        # 3rd: resets the robot to initial conditions
        self.check_topic_publishers_connection()
        self.init_desired_pose()
        self.takeoff_sequence()

        # 4th: takes an observation of the initial condition of the robot
        data_pose, data_imu = self.take_observation()
        observation = [data_pose.position.x]
        
        # 5th: pauses simulation
        self.gazebo.pauseSim()

        return observation
        
    def _step(self, action):
