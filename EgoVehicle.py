import json
import numpy as np
import sys
import time
import concurrent.futures
from misc.TargetSetters.RTTO import RTTO as Target_Setter
from misc.costs.SGANcost import SGAN_cost_function as cost_function
from misc.Plotter import Plotter
from model.Vehicle import Vehicle
from misc.utils.get_waypoints import *

sys.path.append("../")

class EgoVehicle(Vehicle):
    def __init__(
        self,
        ego,
        vehicles,
        args,
        manual_config

    ) -> None:
        super().__init__(ego, args, manual_config)
        
        scenario = args.scenario
        self.goal = manual_config["scenario"][scenario]["goal"]
            
        
        #* Planner initialization
        self.T = manual_config["planning"]["T"] # [s]
        self.dyn = manual_config["planning"]["dyn"]
        self.S = manual_config["prediction"]["S"]

        if manual_config["planning"]["USE_SPLINE"]:
            self.get_spline_init = get_spline_init
        else:
            self.get_spline_init = get_geodesic_init
        self.waypoints = self.get_spline_init(self.x, self.goal,self.x[3],self.dt)
        self.target_setter = Target_Setter(ego, args, manual_config)

        #* Controller initialization
        self.N_j = manual_config['control']["N_j"]
        self.N_sr = manual_config['control']["N_sr"]
        
        #* Load the cost function (including the SGAN model)
        self.cost_function = cost_function(ego, manual_config)
        
        #* Log data for analysis
        self.log = {
            "x": [self.x],
            "xhat": [],
            "xhat_predictions": [],
            "xhat_gt_predictions" : [],
            "u": [self.u],
            "du": [],
            "target": [],
            "waypoints": [],
            "costs": [],
            "cost_star": [],
            "RCMS" : []
        }

    def step(self, ego):
        self.x = np.array([ego.position[0],ego.position[1],ego.heading_theta, ego.speed]) ## (x,y,phi,v)        
        self.u = np.array(ego.last_action)
        try:
            self.x_history = np.append(self.x_history, self.x[np.newaxis], axis = 0)
        except:
            self.x_history = self.x[np.newaxis] #(1,4)

    
    def update_state(self, ego, u, du):
        x, y, psi, v = ego[0], ego[1], ego[2], ego[3]
        a, dl = (u[0] + du[0] * self.dt, self.u[1] + du[1] * self.dt)
        x += v * self.dt * np.cos(psi + dl)
        y += v * self.dt * np.sin(psi + dl)
        psi += v / self.l_r * self.dt * np.sin(dl)
        v += a * self.dt
        
        return np.array([x, y, psi, v])

        # self.u = [self.u[0] + j * self.dt, self.u[1] + sr * self.dt]
        # super().update_state(x,self.u)
        # self.log["x"].append(self.x)
        # self.log["u"].append(self.u)
        # self.log["du"].append([j, sr])
        # self.log["xhat"].append(np.copy(x_hat).tolist())
    
    def update_target(self, xhat_history, x_history):     

        #* Make multistep prediction for L horizon
        try:
            # The predictions from the previous cost evaluation is available only after 1st step 
            xhat_predictions = self.xhat_predictions 
        except: 
            # Simply use CV prediction for the initial step
            xhat_predictions = self.predict(xhat_history, self.S)
        

        # update target
        target = self.target_setter.get_target(x_history[-1], xhat_predictions, self.goal)
        waypoints = self.target_setter.get_waypoints()

        return target, waypoints
        # self.log["xhat_predictions"].append(xhat_predictions)
        # self.log["target"].append(self.target)
        # self.log["waypoints"].append(self.waypoints)
        


    def eval_cost(self,x_cur, x, x_history, xhat_history, u, target,goal):
        """Return a cost associated with the current state and the predicted states of the obstacles.

        Args:
            x_hat (list): Obstacle states
            u (_type_): Ego action to abtain the ego state
            du (_type_): Delta action, i.e., jerk and steering rate
        """
        cost, spatial_risk = self.cost_function.evaluate_cost(x_cur, x,x_history,xhat_history,u,target,goal,self.cv_predictions)
        
        return cost, spatial_risk

    def get_control(self, vehicles):
        vehicles = np.array([[vehicle.position[0], vehicle.position[1],
                     vehicle.heading_theta, vehicle.speed] for vehicle in vehicles])  # (Nveh, 4)
        try:
            self.xhat_history = np.append(self.xhat_history, vehicles[:,np.newaxis], axis = 1)
        except:
            self.xhat_history = vehicles[:,np.newaxis] # (Nveh, 1, 4)
        xhat_history = self.xhat_history.copy()
        x_history = self.x_history.copy()
        u = self.u.copy()
        u_plan = []
        for k in range(int(self.T / self.dt)):

            #* OPTIMIZATION 1: Target selection
            ## Maneuver vehicles according to the selected behavior model
            vehicles = Vehicle.forward_true(vehicles, x_history[-1], dt=self.dt, dyn = self.dyn)
            xhat_history = np.append(xhat_history, vehicles[:, np.newaxis, :], axis=1)
            target, waypoints = self.update_target(xhat_history, x_history)
            
            #* OPTIMIZATION 2: Control input selection 
            u_candidates = self.actions_arr*self.dt + u
            x_candidates = Vehicle.forward(x_history[-1], u_candidates, self.dt)
            cost, spatial_risk = self.eval_cost(x_history[-1], x_candidates, x_history, xhat_history, u, target, self.goal)

            if np.all(cost==cost[0]):
                istar = int(((self.N_j*self.N_sr)-1)/2)
            else:
                istar = np.argmin(cost)
            ustar = self.actions[istar]

            #*
            self.spatial_risk = spatial_risk[istar]
            self.xhat_predictions = self.cost_function.get_predictions(istar)

            a, dl = a + ustar[0]*self.dt, dl + ustar[1]*self.dt
            u_plan.append([a,dl])

            x = self.update_state(x_history[-1], u_plan[-1], ustar) 
            current_cost = self.evaluate_current_state(ustar)
            trajectory_cost += current_cost
            x_history = np.append(x_history, np.array(x)[np.newaxis, :], axis=0)
            
        

    def predict(self, xhat_history, S):
        """
        TODO: This function will be replaced with NN model.
        """

        """
        Return xhat predictions in a nested list structure: 
        [
        [[x1,y1,psi1,v1]_1,...,[x1,y1,psi1,v1]_L]
        [[x2,y2,psi2,v2]_1,...,[x2,y2,psi2,v2]_L]
        ...
        ]

        Args:
            xhat_history (list(list(list))): history of obstacle states
            H (int): length/timesteps of the history  
            L (int): length of prediction
        """
        # Constant velocity / Constant Steering model: u=[0,0]
        for i in range(S):
            x_hat = self.CVmodel(xhat_history.copy())
            xhat_history = np.append(xhat_history,x_hat,axis=1)
        return xhat_history[:,-S:]

    
    def CVmodel(self,xhat_prediction):
        l_r = 0.5
        x, y, psi, v = xhat_prediction[:,-1, 0], xhat_prediction[:,-1, 1], xhat_prediction[:,-1, 2], xhat_prediction[:,-1, 3]
        a, dl = 0, 0
        
        # Vectorized updates
        x += v * self.dt * np.cos(psi + dl)
        y += v * self.dt * np.sin(psi + dl)
        psi += (v / l_r) * self.dt * np.sin(dl)
        v += a * self.dt

        # Combine the updated values back into a state matrix
        X_updated = np.stack([x, y, psi, v], axis=-1)[:,np.newaxis,:]
        return X_updated

    def evaluate_current_state(self,ego,vehicles,du):
        # Below is to evaluate the current step only, i.e. prediction costs are neglected
        #! unlike track_cost, current_cost is evaluated based on its actual distance to the goal
        #! the safety measure is based on collision checking only
        # TODO: change the cost function that correspond to real cost
        current_cost = (
            10*(self.goal[1] - self.x[1])**2
            + 0.1*self.u[0]**2
            + 0.1*self.u[1]**2  
            + 0.1*(du[0]*self.dt)**2 +0.1*(du[1]*self.dt)**2
        )

        current_cost = np.tanh(current_cost) + self.spatial_risk + self.collision_check(ego,vehicles) 
        self.log["cost_star"].append(current_cost)
        return current_cost


        #+ 100000*(1/(np.clip(self.spatial_tolerance_ub*Nveh-spatial_risk[0],1e-6,None)**2)-1/(self.spatial_tolerance_ub*Nveh)**2)
    def collision_check(self,xhat_history):
        '''
        three circle method for collision checking
        '''
        obstacles = xhat_history[:,-1]
        x, y, theta, _ = self.x
        xi, yi, thetai, _ = obstacles[:, 0], obstacles[:, 1], obstacles[:, 2], obstacles[:, 3]
        h = self.veh_length/2
        w = self.veh_width/2
        # TODO: use different h,w for obs if they are specified.
        hi = h
        wi = w

        # Create offsets for both ego and obstacles in [-1, 0, 1] for both x and y directions
        i_offsets = np.array([-1, 0, 1]) #(3,)
        ego_offsets_x = i_offsets * h * np.cos(theta) #(3,)
        ego_offsets_y = i_offsets * h * np.sin(theta)

        obs_offsets_x = np.outer(i_offsets, hi).T * np.cos(thetai[:, None])  # Shape (Nveh, 3)
        obs_offsets_y = np.outer(i_offsets, hi).T * np.sin(thetai[:, None])  # Shape (Nveh, 3)
        
        # Calculate all pairwise distances between the corners
        ego_x_corners = x + ego_offsets_x  # Shape (3,)
        ego_y_corners = y + ego_offsets_y  # Shape (3,)

        # Expand dimensions for broadcasting
        ego_x_corners = ego_x_corners[None, :]  # Shape (1, 3)
        ego_y_corners = ego_y_corners[None, :]  # Shape (1, 3)

        obs_x_corners = xi[:, None] + obs_offsets_x  # Shape (Nveh, 3)
        obs_y_corners = yi[:, None] + obs_offsets_y  # Shape (Nveh, 3)

        # Calculate distances with broadcasting
        dist_x = ego_x_corners[:, None] - obs_x_corners[:, :, None]  # Shape (Nveh, 3, 3)
        dist_y = ego_y_corners[:, None] - obs_y_corners[:, :, None]  # Shape (Nveh, 3, 3)
        dists = np.sqrt(dist_x**2 + dist_y**2) - (w + wi)

        # Get minimum distance and ensure non-negative
        # min_dist = np.clip(np.min(dists, axis=(1, 2)),1e-3,None) # (Nveh,)
        min_dist = np.min(dists, axis = (1,2))
        if np.any(min_dist<=0) :
            print('Collision!!')
            return 1e2
        return 0 #(1/np.tanh(min_dist**2)-1).item()

def simulate_lane_tendency(lane_tendency, config, scenario, cost_function, Target_Setter):
    x0 = config[scenario]["x0"]  # initial state
    ustar = config[scenario]["u0"]  # initial action
    goal = config[scenario]["goal"]  # (list) [x,y] of a goal state
    x_hat = np.array(config[scenario]["x_hat"])

    T = config["optimization"]["T"]
    dt = config["vehicle"]["dt"]

    xhat_history = np.array([[x_hat[i]] for i in range(len(x_hat))])
    x_history = np.array([x0])
    EGO = EgoVehicle(x0, ustar, goal, veh_config=config["vehicle"], opt_config=config["optimization"],
                     lane_tendency=lane_tendency, target_setter=Target_Setter, cost_function=cost_function)
    trajectory_cost = 0
    u_plan = []
    times = []
    start = time.time()
    for k in range(int(T / dt)):
        if k == 48:
            print('k')
        x_hat = Vehicle.forward_true(x_hat, EGO.x.copy(), [0, 0], dt=dt, dyn = "IDM")
        xhat_history = np.append(xhat_history, x_hat[:, np.newaxis, :], axis=1)
        EGO.update_target(xhat_history, x_history)
        ustar = EGO.get_control(xhat_history)
        u_plan.append(ustar)
        EGO.update_state(ustar[0], ustar[1], x_hat)
        current_cost = EGO.evaluate_current_state(ustar[0], ustar[1], xhat_history)
        trajectory_cost += current_cost
        x_history = np.append(x_history, np.array(EGO.x)[np.newaxis, :], axis=0)

    end = time.time()
    print(f'{end - start} s took to run {T} planning horizon (s) with dt {dt}.')
    print(f'trajectory cost for scenario {scenario} with lane-tendency {lane_tendency} is {trajectory_cost}')

    return trajectory_cost, u_plan, lane_tendency, EGO