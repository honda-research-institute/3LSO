import casadi as cs
import numpy as np
from scipy.linalg import sqrtm

from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.constants import RENDER_MODE_ONSCREEN, RENDER_MODE_OFFSCREEN
from metadrive.component.vehicle.vehicle_type import DefaultVehicle

from IPython.display import Image
import cv2
import argparse
from metadrive.utils import generate_gif
from panda3d.core import LVector2f, Point2, LVecBase4f
import matplotlib.pyplot as plt
from metadrive.policy.expert_policy import ExpertPolicy
from metadrive.policy.lange_change_policy import LaneChangePolicy
from metadrive.policy.idm_policy import IDMPolicy
import time
import json
import logging
from datetime import datetime
from main.scenarios import ScenarioEnv
from EgoVehicle import EgoVehicle
import torch
import matplotlib.pyplot as plt
import math

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0
        self.alpha_prev = 0
        self.prev_throttle = 0

    def compute_control(self, target, current):
        # Calculate error
        error = (target - current)
        # Proportional term
        proportional = self.kp * error
        # Integral term
        self.integral += error
        integral = self.ki * self.integral * 0.3
        # Derivative term
        derivative = self.kd * (error - self.prev_error)/0.3
        # Save error for next iteration
        self.prev_error = error
        
        # Return control output
        return proportional + integral + derivative
    
    def calculate_throttle(self, v_desired, v, dt=0.3):
        """Calculate throttle using PID control"""
        kp = 0.03
        ki = 0.001
        kd = 0.1

        if dt <= 0:
            return 0 

        v_error = v_desired - v
        self.integral += v_error

        # PID terms
        k_term = kp * v_error
        i_term = ki * self.integral * dt
        d_term = kd * (v_error - self.prev_error) / dt

        # Calculate throttle and clamp to [0, 1]
        throttle = np.clip(self.prev_throttle + k_term + i_term + d_term, 0,1)

        # Update variables
        self.prev_throttle = throttle
        self.prev_error = v_error
        
        return throttle
    
    def calculate_steering(self, ego_pos, yaw, waypoints, speed, dt = 0.3):
        """Calculate steering angle using PurePursuit, Stanley, or MPC"""
        steering = 0
        lookahead_distance = speed
        x,y = ego_pos
        goal_idx = self.get_goal_waypoint_index(x, y, waypoints, lookahead_distance)
        
        
        v1 = [waypoints[goal_idx,0]-x,waypoints[goal_idx,1]-y]
        v2 = [np.cos(yaw), np.sin(yaw)]
        try:
            alpha = self.get_alpha(v1, v2, lookahead_distance)
        except:
            alpha = self.alpha_prev
        self.alpha_prev = alpha
        steering = self.get_steering_direction(v1, v2) * math.atan((2 * 0.3 * math.sin(alpha)) / (2 * speed))
        return steering, goal_idx
    
    def get_goal_waypoint_index(self, x, y, waypoints, lookahead_distance):
        # Placeholder: Return the index of the closest waypoint ahead
        
        for i, waypoint in enumerate(waypoints.tolist()):
            if i == 0: continue
            dist = abs(self.distance_2d(x, y, waypoint[0], waypoint[1])-lookahead_distance)
            if waypoint[0]> x and dist <= 10:
                    return i
        return waypoints.shape[0]-1
    
    def get_alpha(self, v1, v2, lookahead_distance):
        inner_prod = v1[0] * v2[0] + v1[1] * v2[1]
        return math.acos(inner_prod/lookahead_distance)
    
    def get_steering_direction(self, v1, v2):
        # Placeholder: Determine steering direction (left or right)
        return -1 if v1[0] * v2[1] - v1[1] * v2[0] >= 0 else 1
    
    def distance_2d(self, x1, y1, x2, y2):
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def main(args):
    # * Initialize Parameters
    logging.info("---INITIALIZING PARAMETERS---")
    with open("config.json", "r") as f:
        manual_config = json.load(f)

    DT = manual_config["simulation"]["dt"] # dt 
    SIM_TIME = manual_config["simulation"]["SIM_TIME"] # simulation time

    # TODO: Structure the categories of the required parameters. Classify which ones to set as argparse and config.json.
    # * Initialize environment
    env = ScenarioEnv(dict(map="S", log_level=50, traffic_density=0, physics_world_step_size = DT/5), 
                      args, manual_config) ## dt/5 due to decision_repeat = 5.
    env.reset(seed=0)
    vehicles = env.engine.traffic_manager.vehicles
    # vehicles[0].max_speed_m_s = 3
    EGO = EgoVehicle(vehicles[0], vehicles[1:], args, manual_config) 
    pid_steering = PIDController(.3, 0.05, 0.01)
    pid_throttle = PIDController(8, .2, 1)
 
    # * RUN Simulation
    print("--------------Start running simulations--------------")
    comp_time_array = []
    reward_array = []

    movie3d = []
    try:
        frames = []
        for t in range(int(SIM_TIME/DT)):
            vehicles = env.engine.traffic_manager.vehicles
            ego = vehicles[0]
            # EGO.step(ego)

            tic = time.time()
            # waypoints = EGO.get_control(vehicles[1:])
            toc = time.time()
            
            waypoints = np.array([
                    [20,3.5],[30,3.5],[40,3.5],[50,3.5],[60,3.5],[70,3.],[80,3.5],[90,3.5],[100,3.5],[120,3.5]
                ])
            # waypoints = np.array([
            #         [20,0],[30,0],[40,0],[50,0],[60,0],[70,0],[80,0],[90,0],[100,0],[120,0]
            #     ])
            comp_time_array.append(toc - tic)
            
            # Calculate control inputs
            # target_angle = calculate_target_angle(ego.position[:2], [x,y])
            # max_steering = np.deg2rad(ego.config["max_steering"])
            speed = np.linalg.norm(ego.velocity)
            steer,goal_idx = pid_steering.calculate_steering(ego.position, ego.heading_theta, waypoints, speed)
            
            # Set throttle (speed control to maintain constant speed)
            target_speed = 3  # Target speed in m/s
            acc = pid_throttle.calculate_throttle(v_desired = target_speed, v = speed)#/ego.config["max_engine_force"]
            o, r, term, trunc, info = env.step([steer, acc])
            # ego.set_state([x,y])

            reward_array.append(r)

            ######## TOP DOWN RENDER #########
            frame = env.render(mode="topdown",
                                window=False,
                                text={"target": f"{waypoints[goal_idx]}"},
                                screen_size=(400, 400))
            # plt.imshow(frame)
            # plt.show()
            frames.append(frame)
        
        logging.info(f"Average control compute time is {np.mean(comp_time_array)}")
    except:
        env.close()

    if args.save_gif:
        FILENAME = f'{datetime.now().replace(microsecond=0)}_{args.scenario}_{args.object_policy}'
        generate_gif(frames, f"./videos/{FILENAME}.gif",duration = 50)
        Image(open(f"./videos/{FILENAME}.gif", 'rb').read())


if __name__ == "__main__":
    # * Parsing simulation configuration
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="testing", help="[merge, follow]")
    parser.add_argument("--object_policy", type=str, default="IDM", help="[CV, IDM]")
    parser.add_argument("--save_gif", type=bool,
                        default=True, help="Select True to save gif")
    args = parser.parse_args()

    # * Configure logger
    LOG_FILENAME = f'./logs/{datetime.now().replace(microsecond=0)}.log'
    FORMAT = '%(asctime)s %(message)s'
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=LOG_FILENAME, format = FORMAT, level=logging.INFO)

    main(args)
