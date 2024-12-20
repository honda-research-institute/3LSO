from metadrive.envs import MetaDriveEnv
from metadrive.manager import BaseManager
from metadrive.component.vehicle.vehicle_type import DefaultVehicle
from IPython.display import clear_output, Image
from metadrive.policy.idm_policy import IDMPolicy
from metadrive.policy.expert_policy import ExpertPolicy

import numpy as np

class ScenarioManager(BaseManager):
    def __init__(self, args,manual_config):
        super(ScenarioManager, self).__init__()
        self.generated_v = []
        self.generate_ts = 0
        self.scenario = args.scenario 
        self.manual_config = manual_config
        self.object_policy = args.object_policy
        self.vehicle_policies = {} 
        
    def before_step(self):
        if self.generated_v:
            if self.object_policy == "CV":
                for i,v in enumerate(self.generated_v):
                    speed = self.manual_config["scenario"][self.scenario]["vehicles"][i][3]
                    phi = v.heading_theta
                    v.set_velocity([speed*np.cos(phi), speed*np.sin(phi)])
                    # v.before_step([0, 0]) # set action
            elif self.object_policy == "IDM":
                for i, v in enumerate(self.generated_v):
                    if v in self.vehicle_policies:
                        # Get the action from IDM policy and apply it
                        policy = self.vehicle_policies[v]
                        action = policy.act(v)  # Calculate IDM action
                        v.before_step([action[0],action[1]])   # Apply IDM action to vehicle
            elif self.object_policy == "custom2":
                for i, v in enumerate(self.generated_v):
                    if i in {0,1,2}:
                        speed = self.manual_config["scenario"][self.scenario]["vehicles"][i][3]
                        phi = v.heading_theta
                        v.set_velocity([speed*np.cos(phi), speed*np.sin(phi)])
                    else:
                        # Get the action from IDM policy and apply it
                        policy = self.vehicle_policies[v]
                        action = policy.act(v)  # Calculate IDM action
                        v.before_step([action[0],action[1]])   # Apply IDM action to vehicle

                
    def after_step(self):
        if self.episode_step == self.generate_ts:
            for x,y,phi,vel in self.manual_config["scenario"][self.scenario]["vehicles"]:
                xr = np.random.uniform(-1,1)
                v = self.spawn_object(DefaultVehicle, vehicle_config = dict(), position = [x+xr,y], heading = phi)
                v.set_velocity([vel*np.cos(phi),vel*np.sin(phi)])
                self.generated_v.append(v)                
                
            # Assign IDMPolicy to each vehicle if needed
            if self.object_policy in {"IDM","custom2"}:
                for v in self.generated_v:
                    self.vehicle_policies[v] = IDMPolicy(v,0)
                    self.vehicle_policies[v].enable_lane_change=True

        elif self.generated_v:
            for v in self.generated_v:
                v.after_step()

class ScenarioEnv(MetaDriveEnv):
    def __init__(self, config, args, manual_config,k):
        np.random.seed(k)
        self.args = args
        self.manual_config = manual_config
        vx, vy = manual_config["scenario"][args.scenario]["spawn_velocity"][0],manual_config["scenario"][args.scenario]["spawn_velocity"][1]
        randx = np.random.uniform(vx-0.5,vx+0.5)
        # randy = np.random.uniform(vy-0.1,vy+0.1)

        xx= np.random.uniform(-0.5,0.5)
        
        config["vehicle_config"] = dict(spawn_longitude = manual_config["scenario"][args.scenario]["spawn_longitude"]+xx, 
                                        spawn_lateral = manual_config["scenario"][args.scenario]["spawn_latitude"],
                                        spawn_velocity = [randx,vy])
        super(ScenarioEnv, self).__init__(config)
    
    def setup_engine(self):
        super(ScenarioEnv, self).setup_engine()
        self.engine.register_manager("exp_mgr", ScenarioManager(self.args,self.manual_config))