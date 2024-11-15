
import numpy as np
import sys, os
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
sgan_source_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), 'src','sgan_source'))
sgan_path = os.path.join(sgan_source_path, 'sgan')
sys.path.insert(0,sgan_path)
sys.path.append('../../')
from src.sgan_source.sgan.predictor import Predictor as SganPredictor
from src.nnmpc_source.utils.vehicle import Vehicle
from src.nnmpc_source.model import Model

class cost_function:
    def __init__(self, ego, manual_config):
        #* Cost function parameters
        self.j_min = manual_config['control']["j_min"]
        self.j_max = manual_config['control']["j_max"]
        self.sr_min = manual_config['control']["sr_min"]
        self.sr_max = manual_config['control']["sr_max"]
        self.N_j = manual_config['control']["N_j"]
        self.N_sr = manual_config['control']["N_sr"]
        self.Nsample = self.N_j*self.N_sr
        
        # Generate action spaces
        self.j_space = np.linspace(self.j_min, self.j_max, self.N_j)
        self.sr_space = np.linspace(self.sr_min, self.sr_max, self.N_sr)
        # Generate a meshgrid
        jv, srv = np.meshgrid(self.j_space, self.sr_space)
        self.actions = [[j, sr] for j, sr in zip(jv.ravel(), srv.ravel())]
        self.du = np.array(self.actions)

        self.min_dist = manual_config["control"]['min_dist'] #? do we need this
        self.alpha_g = manual_config["control"]["alpha_g"]
        self.alpha_s = manual_config["control"]["alpha_s"]
        self.alpha_r = manual_config["control"]["alpha_r"]
        self.gamma = manual_config["control"]["gamma"]
        self.w_v = manual_config["control"]["w_v"] #? do we need this
        self.min_dist = manual_config['control']["min_dist"]
        self.w_v = manual_config['control']["w_v"]
        self.j = self.du[:,0]
        self.sr = self.du[:,1]
        self.l_r = 0.5

        #* Miscellaneous Parameters
        self.vmax = ego.max_speed_m_s
        self.veh_length = ego.LENGTH
        self.veh_width = ego.WIDTH        
        self.dt = manual_config['simulation']["dt"]
        self.S = manual_config["prediction"]["S"]
        self.confidence = manual_config['prediction']["confidence"]

        
        


    def evaluate_cost(self,x_cur, X, x_history, xhat_history,u,target):
        x, y, psi, v = X[:,0], X[:,1], X[:,2], X[:,3]
        x_history = np.array(x_history)[:,np.newaxis,:]

        # Check for collision avoidance
        # Convert the ego position to a numpy array
        rep = X.shape[0]
        # Extract the positions of the obstacles (first two columns of x_hat)
        # Get only the x, y positions
        obstacle_positions = np.repeat(np.array(xhat_history)[:, -1:], rep, axis = 1)
        Nveh = obstacle_positions.shape[0]
        ego_pos = np.repeat(X[np.newaxis, :,:],Nveh, axis = 0)
        # xhat_predictions = np.array(xhat_predictions)[:, :, :2]

        # Compute the Euclidean distances between the ego vehicle and each obstacle
        dists = np.linalg.norm(obstacle_positions[:,:,:2] - ego_pos[:,:,:2], axis=2)
        min_dists = np.clip(np.min(dists,axis=0)-self.min_dist,1e-3,None)
        
        # min_dists = np.min(dists,axis=0)
        preceding_dists = dists[(np.array(xhat_history)[:, -1] - np.array(x_cur))[:,0]>0]
        try:
            preceding_min_dists = np.clip(np.min(preceding_dists,axis=0)-self.min_dist,1e-3,None)
        except:
            preceding_min_dists = np.inf
        # vmax is evaluated based on the minimum distance to obstacles
        # assumed j_min to be the minimum acceleration value. the distance to stop is evaluated based on other vehicles' velocity 1.
        # vmax = np.sqrt(2 * self.j_min *(-min_dists + obstacle_positions[np.argmin(dists,axis=0),np.arange(obstacle_positions.shape[1]),3] / (2 * self.j_min)))
        vmax = np.sqrt(2 * self.j_min *(-preceding_min_dists + 1 / (2 * self.j_min)))
        vlim = np.clip(vmax,0,self.vmax)
        
        vref = self.w_v * vlim
        spatial_risk = self.spatial_risk(ego_pos,obstacle_positions)
        risk_max = np.max(spatial_risk)
        risk_scalar = np.tanh((spatial_risk/self.spatial_tolerance_ub)**2)
        # risk_scalar = np.tanh((min_dists/self.min_dist)**2)
        # temporal_risk = self.temporal_risk(ego_expanded,veh_expanded)
        # Discontinuous cost function previously used
        # cost = w_ref*min((target[0]-x)**2 + (target[1]-y)**2,20) + w_ref* (v-vref)**2 + 1/min(1,(min_dist-0.5)**2) + 10*psi**2 + 30*a**2 + 10*dl**2 + j**2 + 0.2*sr**2
        track_cost = (
            self.w_ref
            * 20
            * np.tanh(0.1*((target[0] - X[:,0]) ** 2 + (target[1] - X[:,1]) ** 2))
            + self.w_ref * (X[:,3] - vref) ** 2
            # + 1 / ((min_dists)**2)
            + self.w_ref * X[:,2]**2
            + 20 * u[:,0]**2
            + 10 * u[:,1]**2
            + self.j**2
            + self.sr**2
        )
        track_max = np.max(track_cost)
        cost = (1-risk_scalar)*track_cost/track_max + risk_scalar* spatial_risk/risk_max
        xhat_history = np.array(xhat_history)[:,:-1]
        xhat_predictions = np.array(xhat_history)[:,-1]
        u = u + self.du*self.dt
        for i in range(1, self.S):
            X = self.forward(X, u, dt=self.dt ,vectorize= True)  
            xhat_predictions = self.predict_obstacles(xhat_predictions)  # forward predict one-step
            ego_pos = np.repeat(X[np.newaxis, :,:],Nveh, axis = 0)
            obstacle_positions = np.repeat(xhat_predictions[:,np.newaxis,:], rep, axis = 1)
            
            dists = np.linalg.norm(obstacle_positions[:,:,:2] - ego_pos[:,:,:2], axis=2)
            # min_dists = np.min(dists,axis=0)
            min_dists = np.clip(np.min(dists,axis=0)-self.min_dist,1e-3,None)
            

            step_track_cost = (
                self.w_ref
                * 20
                * np.tanh(0.1*((target[0] - X[:,0]) ** 2 + (target[1] - X[:,1]) ** 2))
                + self.w_ref * (X[:,3] - vref) ** 2
                # + 1 / ((min_dists)**2)
                + self.w_ref * X[:,2]**2
                + 20 * u[:,0]**2
                + 10 * u[:,1]**2
                + self.j**2
                + self.sr**2
            )
            
            spatial_risk = self.spatial_risk(ego_pos,obstacle_positions)
            # Discontinuous cost function previously used
            # step_cost = w_ref*min((target[0]-x)**2 + (target[1]-y)**2,20) + w_ref* (v-vref)**2 + 1/min(1,(min_dist-0.5)**2) + 10*psi**2 + 30*a**2 + 10*dl**2 + j**2 + 0.2*sr**2
            
            step_cost = (1-risk_scalar)*step_track_cost/track_max + risk_scalar*spatial_risk/risk_max
            cost += self.gamma**i * step_cost
        
        
        # the cost is overrded with infinity if it is not feasible
        cost[v<0] = np.inf
        cost[v>self.vmax] = np.inf

        return cost
    
    
    def spatial_risk(self, candidates_expanded, xhat_predictions):
        '''skewed gaussian. corresponds to the cost in the RCMS paper.'''

        p_ego = candidates_expanded[:, :, :2][:, :, :, np.newaxis]
        p_veh = xhat_predictions[:, :, :2][:, :, :, np.newaxis]
        p_bar = p_veh-p_ego  # (N_veh,N_pred,2,1)
        v_veh_x = xhat_predictions[:, :, 3]*np.cos(xhat_predictions[:,:,2])  # (N_veh,N_pred)
        v_veh_y = xhat_predictions[:, :, 3]*np.sin(xhat_predictions[:,:,2])
        v_veh = np.stack([v_veh_x[:,:,np.newaxis],v_veh_y[:,:,np.newaxis]],axis=2)

        v_ego_x = candidates_expanded[:, :, 3]*np.cos(candidates_expanded[:,:,2])  # (N_veh,N_pred)
        v_ego_y = candidates_expanded[:, :, 3]*np.sin(candidates_expanded[:,:,2])
        v_ego = np.stack([v_ego_x[:,:,np.newaxis],v_ego_y[:,:,np.newaxis]],axis=2)

        v_bar = v_veh-v_ego

        R_veh = np.transpose(self.rotation_matrix(
            xhat_predictions[:, :, 2]), [2, 3, 0, 1])

        cov = np.array([[self.veh_length*self.length_scalar, 0], [0, self.veh_width*self.width_scalar]])
        cov_veh = R_veh @ cov @ np.transpose(R_veh, [0, 1, 3, 2])
        cov_veh_inv = np.linalg.pinv(cov_veh)  # (N_veh,S,2,2)

        dist = 1/(self.alpha_g + np.transpose(p_bar,[0, 1, 3, 2])@cov_veh_inv@p_bar)
        skew = 1 / (1+np.exp(self.alpha_s*np.transpose(p_bar, [0, 1, 3, 2])@v_bar))

        p_road_1 = np.array([[0],[-0.5]])
        p_road_2 = np.array([[0],[3.5]])
        p_bar_lane_1 = (p_ego[0] - p_road_1)[:,1]
        p_bar_lane_2 = (p_road_2 - p_ego[0])[:,1]
        lane_risk = np.exp(-self.alpha_r*p_bar_lane_1*p_bar_lane_1) + np.exp(-self.alpha_r*p_bar_lane_2*p_bar_lane_2)
        risk = np.sum(dist * skew, axis=0)[:, 0, 0] + lane_risk[:,0]

        # risk = np.max(dist * skew, axis=0)[:, 0, 0]
        
        # We don't discount risk here, because cost evaluation already has this factor.
        # risk = np.power(self.gamma, np.arange(0, risk.shape[0], 1))*risk  # discounted risk

        return risk

    @staticmethod
    def rotation_matrix(theta):
        return np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])


    def predict_obstacles(self, Xhat):
        u = np.array([0,0])
        l_r = 0.5
        x, y, psi, v = Xhat[:,0], Xhat[:,1], Xhat[:,2], Xhat[:,3]
        # cv/ca model
        a, dl = u[0], u[1]
        
        # Vectorized updates
        x += v * self.dt * np.cos(psi + dl)
        y += v * self.dt * np.sin(psi + dl)
        psi += (v / l_r) * self.dt * np.sin(dl)
        v += a * self.dt

        # Combine the updated values back into a state matrix
        xhat_predictions = np.stack([x, y, psi, v], axis=-1)
        return xhat_predictions
    
    @staticmethod
    def forward(x, u, dt=0.1, dyn="bicycle", vectorize = False):
        l_r = 0.5
        if vectorize:
            if dyn == "bicycle":
                # x = np.array(x)
                x, y, psi, v = x[:,0], x[:,1], x[:,2], x[:,3]
                a, dl = u[:, 0], u[:, 1]
                
                # Vectorized updates
                x += v * dt * np.cos(psi + dl)
                y += v * dt * np.sin(psi + dl)
                psi += (v / l_r) * dt * np.sin(dl)
                v += a * dt

                # Combine the updated values back into a state matrix
                X_updated = np.stack([x, y, psi, v], axis=-1)
                
            return X_updated
        else:
            if dyn == "bicycle":  # bicycle kinematics
                x, y, psi, v = x[0], x[1], x[2], x[3]
                a, dl = u[0], u[1]
                x += v * dt * np.cos(psi + dl)
                y += v * dt * np.sin(psi + dl)
                psi += v / l_r * dt * np.sin(dl)
                v += a * dt
            return [x, y, psi, v]
        
    
    @staticmethod
    def normalize(arr):
        arr_min = np.min(arr)
        arr_max = np.max(arr)

        return (arr - arr_min)/(arr_max-arr_min) * arr_max