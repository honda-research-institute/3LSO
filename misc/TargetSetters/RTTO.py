import numpy as np
from ..utils.get_waypoints import *


class RTTO:
    def __init__(self, ego, args, manual_config):
        #* Vehicle properties
        self.veh_length = ego.LENGTH
        self.veh_width = ego.WIDTH

        #* Parameters
        self.dt = manual_config["simulation"]["dt"]
        self.S = manual_config["prediction"]["S"]
        self.confidence = manual_config['prediction']["confidence"]
        self.alpha_s = manual_config['control']["alpha_s"]
        self.alpha_g = manual_config['control']["alpha_g"]
        self.alpha_r = manual_config['control']["alpha_r"]        
        self.N_goal_samples = manual_config['planning']["TARGET_SAMPLES"]
        self.goal_dist = manual_config['planning']["TARGET_DIST"]
        self.goal_angle = np.deg2rad(manual_config['planning']["TARGET_ANGLE_RANGE"])
        self.lane_tendency = None
        
    def get_target(self, x_cur, xhat_predictions, goal, lane_tendency):      

        p1 = np.repeat(x_cur[np.newaxis,:3],self.N_goal_samples,axis=0) #(N,3)
        
        #* Sampling waypoings along the biased polar coordinates
        p2 = calc_biased_polar_states_toward_goal_y(goal_y = goal[1],nxy=self.N_goal_samples,d=self.goal_dist,
                                                    a_min=-self.goal_angle,a_max=self.goal_angle,x_cur=x_cur) #(N,3)
        # waypoints = get_spline(p1,p2,x_cur[3],self.dt,xhat_predictions.shape[1], vectorize = True)
        waypoints = get_geodesic(p1,p2,x_cur[3],self.dt)[1:self.S] #(N_pred,N_wp,3)
        waypoints = np.transpose(waypoints,[1,0,2]) #(N_wp,N_pred,3)

        collision = self.collision_check(waypoints, xhat_predictions[:,1:]) 
        spatial_risks = self.spatial_risk(waypoints, xhat_predictions[:,1:], x_cur[3]) #(N_wp, N_pred)
        
        self.waypoints, self.row_idx, self.col_idx = self.get_optimal_sample_path(waypoints,spatial_risks,goal,collision, lane_tendency) #(N_pred,3)
        
        return self.waypoints[self.row_idx, self.col_idx,:2].tolist()

    def get_target_to_obs(self):
        return self.target_to_obs
    
    def get_waypoints(self):
        return self.waypoints
    
    def get_path_cost(self):
        return self.path_cost
    
    def collision_check(self, x_reference, obstacles):
        '''
        x_reference: (Npred,Nsample,3)
        obstacles: (Nveh,Npred,4)
        '''
        Nsample,Npred,_ = x_reference.shape
        Nveh = obstacles.shape[0]
        obstacles = np.repeat(obstacles[:,:,np.newaxis],Nsample,axis=2) # (Nveh,Npred,Nsample,4)
        obstacles = np.transpose(obstacles,[1,2,0,3]) #(Npred,Nsample,Nveh,4)
        
        x, y, theta = x_reference[:,:,0].T, x_reference[:,:,1].T, x_reference[:,:,2].T
        xi, yi, thetai = obstacles[:,:,:,0], obstacles[:,:,:,1], obstacles[:,:,:,2]
        h = np.repeat(np.array(self.veh_length/2),Npred*Nsample,axis=0).reshape(Npred,Nsample,1)
        w = np.repeat(np.array(self.veh_width/2),Npred*Nsample,axis=0).reshape(Npred,Nsample,1)
        
        # TODO: use different h,w for obs if they are specified.
        hi = h
        wi = w

        # Create offsets for both ego and obstacles in [-1, 0, 1] for both x and y directions
        i_offsets = np.repeat(np.array([[-1, 0, 1]]), Npred * Nsample, axis=0).reshape(Npred, Nsample, 3) #(Npred,Nsample,3)
        ego_offsets_x = i_offsets * h * np.cos(theta[:,:,np.newaxis])
        ego_offsets_y = i_offsets * h * np.sin(theta[:,:,np.newaxis])

        obs_offsets_x =  i_offsets[:,:,np.newaxis,:]* hi[:,:,:,np.newaxis] * np.cos(thetai[:,:,:, np.newaxis])  # Shape (Npred,Nsample,Nveh, 3)
        obs_offsets_y =  i_offsets[:,:,np.newaxis,:]* hi[:,:,:,np.newaxis] * np.sin(thetai[:,:,:, np.newaxis])  # Shape (Npred,Nsample,Nveh, 3)
        
        # Calculate all pairwise distances between the corners
        ego_x_corners = x[:,:,np.newaxis] + ego_offsets_x  # Shape (Npred,Nsample,3)
        ego_y_corners = y[:,:,np.newaxis] + ego_offsets_y  # Shape (Npred,Nsample,3)

        # Expand dimensions for broadcasting
        ego_x_corners = ego_x_corners[:,:, np.newaxis, :]  # Shape (Npred,Nsample,1, 3)
        ego_y_corners = ego_y_corners[:,:, np.newaxis, :]  # Shape (Npred,Nsample,1, 3)

        obs_x_corners = xi[:,:,:, np.newaxis] + obs_offsets_x  # Shape (Npred,Nsample,Nveh, 3)
        obs_y_corners = yi[:,:,:, np.newaxis] + obs_offsets_y  # Shape (Npred,Nsample,Nveh, 3)

        # Calculate distances with broadcasting
        dist_x = ego_x_corners[:,:,:,np.newaxis] - obs_x_corners[:, :, :, :, None]  # Shape (Npred,Nsample,Nveh, 3, 3)
        dist_y = ego_y_corners[:,:,:,np.newaxis] - obs_y_corners[:, :, :, :, None] # Shape (Npred,Nsample,Nveh, 3, 3)
        dists = np.sqrt(dist_x**2 + dist_y**2) - (w + wi)[:,:,:,np.newaxis,np.newaxis]  # Shape (Npred,Nsample,Nveh, 3, 3)

        # Get minimum distance and ensure non-negative
        # TODO: use different measure than min to evaluate over Npred.
        min_dist = np.clip(np.min(dists, axis=(3, 4, 2)),0,None) #(Npred, Nsample)
        w = np.power(1/self.confidence, np.arange(0, min_dist.shape[0], 1))
        min_dist = min_dist * np.repeat(w[:,np.newaxis],min_dist.shape[1],axis=1)
        # return np.exp(-min_dist).T#1/(10*min_dist**2)#
        # min_dist = np.clip(np.min(dists, axis=(3, 4, 0, 2)),1e-2,None) #(Npred, Nsample)
        return np.exp(-1*min_dist**2)#1/(10*min_dist**2)
    
    def get_optimal_sample_path(self,waypoints,risk,goal,collision,lane_tendency):
        '''
        waypoints : array(N_wp,N_pred,3)
        risk : array (N_wp, N_pred)
        return: waypoints of array (N_pred,) 
        '''
        lane_deviation = (goal[1]-waypoints[:,:,1])**2
        forward_propagation = waypoints[:,:,0]
        # path_cost = lane_deviation - forward_propagation/np.max(forward_propagation)
        path_cost = 2*self.normalize(lane_deviation) - self.normalize(forward_propagation)
        costs = (1-lane_tendency)* np.tanh((risk/0.4)**2)  + lane_tendency* path_cost
        costs_sum = np.mean(costs,axis=1)
        r = np.argmin(costs_sum)
        # r, c = np.unravel_index(np.argmin(costs), costs.shape)
        return waypoints, r, -1 

    def spatial_risk(self, waypoints, xhat_predictions, v_cur):
        '''skewed gaussian. corresponds to the cost in the RCMS paper.'''
        
        p_ego = np.repeat(waypoints[np.newaxis,:, :, :2],xhat_predictions.shape[0],axis=0)[:, :, :, :, np.newaxis] # (N_veh,N_wp,N_pred,2,1)
        p_veh = np.repeat(xhat_predictions[:,np.newaxis, :, :2],waypoints.shape[0],axis=1)[:, :, :, :, np.newaxis] # (N_veh,N_wp,N_pred,2,1)
        p_bar = p_veh-p_ego  # (N_veh,N_wp,N_pred,2,1)
        v_veh_x = xhat_predictions[:, :, 3]*np.cos(xhat_predictions[:,:,2])  # (N_veh,N_pred)
        v_veh_y = xhat_predictions[:, :, 3]*np.sin(xhat_predictions[:,:,2])
        v_veh = np.stack([v_veh_x[:,:,np.newaxis],v_veh_y[:,:,np.newaxis]],axis=2) #(N_veh,N_pred,2,1)

        v_ego_x = v_cur*np.cos(waypoints[:,:,2]) # (N_wp, N_pred)
        v_ego_y = v_cur*np.sin(waypoints[:,:,2]) # (N_wp, N_pred)
        v_ego = np.stack([v_ego_x[:,:,np.newaxis],v_ego_y[:,:,np.newaxis]],axis=2) # (N_wp, N_pred,2,1)

        v_bar = np.repeat(v_veh[:,np.newaxis,:,:,:],waypoints.shape[0],axis=1) - np.repeat(v_ego[np.newaxis,:,:,:,:],xhat_predictions.shape[0],axis=0)
        #(Nveh,N_wp,N_pred,2,1)

        R_veh = np.transpose(self.rotation_matrix(
            xhat_predictions[:, :, 2]), [2, 3, 0, 1]) # (Nveh,N_pred,2,2)

        cov = np.array([[self.veh_length, 0], [0, self.veh_width]])
        cov_veh = R_veh @ cov @ np.transpose(R_veh, [0, 1, 3, 2])
        # cov_veh_inv = np.power(self.confidence, np.arange(0, v_bar.shape[2], 1))[:,np.newaxis,np.newaxis]*np.linalg.pinv(cov_veh)
        cov_veh_inv = np.linalg.pinv(cov_veh)  # (N_veh,N_pred,2,2)
        cov_veh_inv = np.repeat(cov_veh_inv[:,np.newaxis,:,:,:],waypoints.shape[0],axis=1)  # (N_veh,N_wp,N_pred,2,2)

        dist = 1/(self.alpha_g + np.transpose(p_bar,[0, 1, 2, 4, 3])@cov_veh_inv@p_bar)
        skew = 1 / (1+np.exp(self.alpha_s*np.transpose(p_bar, [0, 1, 2, 4, 3])@v_bar))

        p_road_1 = np.array([[0],[-0.5]])
        p_road_2 = np.array([[0],[9]])
        p_bar_lane_1 = (p_ego[0] - p_road_1)[:,:,1]
        p_bar_lane_2 = (p_road_2 - p_ego[0])[:,:,1]
        np.clip(p_bar_lane_1,0,None,out=p_bar_lane_1)
        np.clip(p_bar_lane_2,0,None,out=p_bar_lane_2)
        lane_risk = np.exp(-self.alpha_r*p_bar_lane_1*p_bar_lane_1) + np.exp(-self.alpha_r*p_bar_lane_2*p_bar_lane_2)
        # risk = np.sum(dist * skew, axis=0)[:, :, 0, 0]  + lane_risk[:,:,0]
        spatial_risk = np.sum(dist * skew, axis=0)[:, :, 0, 0] 
        collision_risk = self.collision_check(waypoints, xhat_predictions) #(Npred, Nsample)
        risk = 0.4*collision_risk + 0.6*spatial_risk.T + 0.2*lane_risk[:,:,0].T #0.5*spatial_risk.T+0.5*collision_risk + 0.2*lane_risk[:,:,0].T
        return risk.T
        # return risk
    
    def normalize(self,arr):
        arr_min = np.min(arr)
        arr_max = np.max(arr)

        return (arr-arr_min)/(arr_max-arr_min)
    @staticmethod
    def rotation_matrix(theta):
        return np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])
    
    


    