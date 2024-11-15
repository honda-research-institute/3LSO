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
from metadrive.policy.idm_policy import IDMPolicy
import time
import json
import logging
from datetime import datetime
from main.scenarios import ScenarioEnv
from EgoVehicle import EgoVehicle
import torch
      
def control(ego, vehicles, t):
    # x0 = np.array()
    x_hat = np.array([[vehicle.position[0], vehicle.position[1],
                     vehicle.heading_theta, vehicle.speed] for vehicle in vehicles])  # (Nveh, 4)

    return 10, 0.02


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
    env.reset()
    vehicles = env.engine.traffic_manager.vehicles
    EGO = EgoVehicle(vehicles[0], vehicles[1:], args, manual_config) 
 
    # * RUN Simulation
    comp_time_array = []
    reward_array = []

    movie3d = []
    # try:
    frames = []
    logging.info("---STARTING SIMULATION---")
    for t in range(int(SIM_TIME/DT)):
        vehicles = env.engine.traffic_manager.vehicles
        ego = vehicles[0]
        EGO.step(ego)

        tic = time.time()
        #! ERASE LATER
        #! steer,acc \in [-1,1].
        #! us = S_max * steer; S_max typically 40-50, depending on the vehicle
        #! ua = Fmax * acc; F_max
        steer, acc = EGO.get_control(vehicles[1:])
        toc = time.time()
        comp_time_array.append(toc - tic)
        o, r, term, trunc, info = env.step([steer, acc])

        reward_array.append(r)

        ######## TOP DOWN RENDER #########
        frame = env.render(mode="topdown",
                            window=False,
                            screen_size=(400, 400))
        frames.append(frame)
            
    # except:
    #     env.close()

    # if args.save_gif:
    #     FILENAME = f'{datetime.now().replace(microsecond=0)}_{args.scenario}_{args.object_policy}'
    #     generate_gif(frames, f"./videos/{FILENAME}.gif",duration = 50)
    #     Image(open(f"./videos/{FILENAME}.gif", 'rb').read())


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
