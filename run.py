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
from tqdm import tqdm
from metadrive.component.pgblock.first_block import FirstPGBlock
import pickle

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0
        self.alpha_prev = 0
        self.prev_throttle = 0
    
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
            if waypoint[0]> x and dist <= 1:
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
    planning_interval = manual_config["simulation"]["interval"]
    # TODO: Structure the categories of the required parameters. Classify which ones to set as argparse and config.json.
    # * Initialize environment
    # vehicle_config = dict(spawn_longitude = manual_config["scenario"][args.scenario]["spawn_longitude"], 
    #                                     spawn_lateral = manual_config["scenario"][args.scenario]["spawn_latitude"],
    #                                     spawn_velocity = manual_config["scenario"][args.scenario]["spawn_velocity"])
    # config = dict(map="S", log_level=50, traffic_density=0.6, physics_world_step_size = DT/5)
    
    # env = MetaDriveEnv(config)
    
    for i in range(20):
        env = ScenarioEnv(dict(map="S", log_level=50, traffic_density=0, physics_world_step_size = DT/5), 
                        args, manual_config, i) ## dt/5 due to decision_repeat = 5.    
        env.reset()
        vehicles = env.engine.traffic_manager.vehicles
        EGO = EgoVehicle(vehicles[0], vehicles[1:], args, manual_config) 
    
        # * RUN Simulation
        print("--------------Start running simulations--------------")
        comp_time_array = []
        reward_array = []

        movie3d = []
        try:
            frames = []
            inputs = []
            infos = []
            min_dists = []
            for t in tqdm(range(int(SIM_TIME/DT)), desc = "Simulation running..."):
                # vehicles = env.engine.traffic_manager.vehicles
                if t % planning_interval == 0:
                
                    EGO.step(vehicles[0])
                    tic = time.time()
                    waypoints, control_input, min_dist = EGO.get_control(vehicles[1:])
                    toc = time.time()
                        # print("----WAYPOINTS----")
                        # print(waypoints)
                        # print("-------------------")
                    
                    comp_time_array.append(toc - tic)
                    inputs += control_input
                    min_dists += min_dist
                
                # Calculate control inputs
                # speed = np.linalg.norm(ego.velocity)
                # steer,goal_idx = pid_steering.calculate_steering(ego.position, ego.heading_theta, waypoints, speed)
                # # Set throttle (speed control to maintain constant speed)
                # target_speed = waypoints[goal_idx, 3]  # Target speed in m/s
                # acc = pid_throttle.calculate_throttle(v_desired = target_speed, v = speed)#/ego.config["max_engine_force"]
                
                k = t%planning_interval
                o, r, term, trunc, info = env.step([0, 0])
                vehicles[0].set_position(waypoints[k,:2])
                vehicles[0].set_heading_theta(waypoints[k,2])
                vehicles[0].set_velocity([waypoints[k,3]*np.cos(waypoints[k,2]),waypoints[k,3]*np.sin(waypoints[k,2])])
                reward_array.append(r)

                ######## TOP DOWN RENDER #########
                text = np.round(waypoints[k],2)
                text2 = np.round(vehicles[0].position,2)
                frame = env.render(mode="topdown",
                                    window=False,
                                    # text={"target": f"{text}",
                                    #       "current": f"{text2}"},
                                    screen_size=(800, 300))
                # if t >= 30:
                #     plt.imshow(frame)
                #     plt.show()
                frames.append(frame)
                try:
                    info["xy"] += waypoints[k,:2].tolist()
                except:
                    info["xy"] = waypoints[k,:2].tolist()
                infos.append(info)
                
            
            logging.info(f"Average control compute time is {np.mean(comp_time_array)}")
            logging.info(f"Reward sum is {np.sum(reward_array)}")
        except:
            env.close()
            FILENAME = f'{datetime.now().replace(microsecond=0)}_{args.scenario}_{args.object_policy}'
            with open(f"./data/{FILENAME}.pkl", "wb") as f:
                pickle.dump(infos,f)
                pickle.dump(inputs,f)
                pickle.dump(min_dists,f)
                pickle.dump(comp_time_array,f)
        finally:
            env.close()
            FILENAME = f'{datetime.now().replace(microsecond=0)}_{args.scenario}_{args.object_policy}'
            with open(f"./data/{FILENAME}.pkl", "wb") as f:
                pickle.dump(infos,f)
                pickle.dump(inputs,f)
                pickle.dump(min_dists,f)
                pickle.dump(comp_time_array,f)


        if args.save_gif:
            FILENAME = f'{datetime.now().replace(microsecond=0)}_{args.scenario}_{args.object_policy}'
            generate_gif(frames, f"./videos/{FILENAME}.gif",duration = 50)
            Image(open(f"./videos/{FILENAME}.gif", 'rb').read())


if __name__ == "__main__":
    # * Parsing simulation configuration
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="merge", help="[merge, follow]")
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
