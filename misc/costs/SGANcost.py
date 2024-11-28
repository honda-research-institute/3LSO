
import numpy as np
from .cost import cost_function
import sys, os
from concurrent.futures import ThreadPoolExecutor
from scipy.ndimage import convolve
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
sgan_source_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), 'src','sgan_source'))
sgan_path = os.path.join(sgan_source_path, 'sgan')
sys.path.insert(0,sgan_path)
sys.path.append('../../')
from src.sgan_source.sgan.predictor import Predictor as SganPredictor
from src.nnmpc_source.utils.vehicle import Vehicle
from src.nnmpc_source.model import Model
from src.nnmpc_source.utils.IDM import IDM_predictor
np.random.seed(2024)

class SGAN_cost_function(cost_function):
    def __init__(self,ego, manual_config):
        super().__init__(ego, manual_config)
        self.model = Model()
        if manual_config["prediction"]["method"] == "SGAN":
            self.predictor_method = "SGAN"
        elif manual_config["prediction"]["method"] == "IDM":
            self.predictor_method = "IDM"
        elif manual_config["prediction"]["method"] == "CV":
            self.predictor_method = "CV"
        else:
            raise Exception("Choose predictor from [SGAN, IDM, CV]")
        # FILENAME = f'{manual_config['prediction']['prediction_model']}.pt'
        if manual_config['prediction']['prediction_model'] == "SGAN_Student":
            self.model.sgan_model = "SGAN_Student.pt"
        elif manual_config['prediction']['prediction_model'] == "SGAN_Teacher":
            self.model.sgan_model = "SGAN_Teacher.pt"
        else: 
            raise Exception("Only SGAN_Student and SGAN_Teacher is supported.")
        
        # with ThreadPoolExecutor(max_workers=1) as executor:
        #     futures = [executor.submit(SganPredictor, os.path.join(sgan_source_path, 'model', self.model.sgan_model), self.S) for _ in range(1)]
        #     self.predictors = [future.result() for future in futures]
        self.predictor = SganPredictor(os.path.join(sgan_source_path,'model', self.model.sgan_model), self.S) # NN predictor 
    def evaluate_cost(self,x_cur, X, x_history, xhat_history,u,target,goal,cv_predictions):
        '''
        X: Array of (N_j x N_sr, 4): array of all possible states that are induced by applying every action pairs to current state x_cur.
        x_history: list of all previous states; has varying length. 
        xhat_history: array of all previous states of obstacles (N_obstacles, H, 4); H increases every step.
        '''
        x, y, psi, v = X[:,0], X[:,1], X[:,2], X[:,3]
        u_prev = u
        x_cur = np.array(x_cur)
        x_history = np.array(x_history)[:,np.newaxis,:]

        rep = X.shape[0] # N_j x N_sr
        obstacle_positions = np.repeat(xhat_history[:, -1:], rep, axis = 1) #latest history is the current obsrvation.
        Nveh = obstacle_positions.shape[0] # Number of vehicles 
        ego_pos = np.repeat(X[np.newaxis, :,:],Nveh, axis = 0) # for vectorization
        dists = np.linalg.norm(obstacle_positions[:,:,:2] - ego_pos[:,:,:2], axis=2)
        # min_dists = np.clip(np.min(dists-self.min_dist,axis=0),1e-6,None)
        preceding_idxs = (xhat_history[:, -1] - x_cur)[:,0]>0
        same_land_idxs = np.abs((xhat_history[:, -1] - x_cur)[:,1]) <= 2.5
        
        preceding_dists = dists[np.logical_and(preceding_idxs, same_land_idxs)] # required to compute dynamic speed limit 
        

        try:
            preceding_min_dists = np.clip(np.min(preceding_dists,axis=0)-self.min_dist,1e-3,None)
        except:
            preceding_min_dists = np.inf
        # vmax is evaluated based on the minimum distance to preceding obstacles
        # ! assumed j_min to be the minimum acceleration value. the distance to stop is evaluated based on other vehicles' velocity 1.
        # TODO: use a_min if such information of an obstacle vehicle is given.
        vmax = np.sqrt(2 * self.j_min *(-preceding_min_dists + 1 / (2 * self.j_min)))
        vlim = np.clip(vmax,0,self.vmax) 
        
        try:
            reference_vehicle_idx = np.argmin(preceding_dists,axis=0)
            vref = xhat_history[np.logical_and(preceding_idxs, same_land_idxs)][reference_vehicle_idx,-1,3]
        except:
            vref = self.w_v * vlim # ? Should we obtain desired velocity profile
 
        Nveh = np.sum(~preceding_idxs) # only consider vehicles that are not preceding the ego.
        x_history = np.repeat(x_history,self.Nsample,axis=1)[:,:,:2] # (H,Nsample,2)
        xhat_history_temp = np.repeat(xhat_history[~preceding_idxs,:-1],self.Nsample,axis=0)
        xhat_history_temp_2 = np.transpose(xhat_history_temp,[1,0,2]) # (H,Nsample*Nveh,4)
        u = u+self.du*self.dt
        du = self.du
        x_reference = X[np.newaxis,:,:]
        u_reference = u[np.newaxis,:,:]
        du_reference = du[np.newaxis,:,:]
        # Propagate reference ego states using u(t+s) = u(t+1) for all 1<=s<=S
        for _ in range(1,self.S):
            X = self.forward(X.copy(), u.copy(), dt=self.dt ,vectorize= True)
            x_reference = np.append(x_reference,X[np.newaxis,:,:],axis=0)
            u_reference = np.append(u_reference,u[np.newaxis,:,:],axis=0)
            du_reference = np.append(du_reference,du[np.newaxis],axis=0)
            
            # du = np.clip(np.random.normal(du,0.1,du.shape),-0.5,0.5)
            # du = np.random.uniform(-0.4,0.4,du.shape)
            # u = u+du*self.dt
            # u = u*0

        if self.predictor_method == "SGAN":
            xhat_predictions = self.predictor.predict_batch(x_history,xhat_history_temp_2[:,:,:2],Nveh, self.Nsample ,self.dt, self.model.dt, x_reference)
            #* Masking preceding vehicles with CV predictions
            obstacle_positions = xhat_predictions[:,1:] # 0th vehicle is the ego vehicle
            cv_predictions = np.repeat(cv_predictions[:,:,np.newaxis],self.Nsample,axis=2)
            cv_predictions[~preceding_idxs] = np.transpose(obstacle_positions,[1,0,2,3])
            cv_predictions = np.transpose(cv_predictions,[1,0,2,3])
            obstacle_positions = cv_predictions
        elif self.predictor_method == "IDM":
            obstacle_positions = IDM_predictor(x_reference, xhat_history[:,-1], self.dt)
        elif self.predictor_method == "CV":
            cv_predictions = np.repeat(cv_predictions[:,:,np.newaxis],self.Nsample,axis=2)
            cv_predictions = np.transpose(cv_predictions,[1,0,2,3])
            obstacle_positions = cv_predictions
            
                

        spatial_risk,min_dist,lane_dist = self.spatial_risk(x_reference,obstacle_positions) #(Npred,Nsample)

        # ! THIS WORKS
        track_cost = (
            # 2000*np.tanh(0.1*((target[0] - x_reference[:,:,0]) ** 2 + (target[1] - x_reference[:,:,1]) ** 2)) # target position tracking
            150*((target[0] - x_reference[:,:,0]) ** 2 + (target[1] - x_reference[:,:,1]) ** 2)
            + 50*(x_reference[:,:,3] - vref) ** 2 # target velocity tracking 
            + 500*(x_reference[:,:,2])**2 # target heading angle tracking
            + 50*(u_reference[:,:,0])**2 + 200*(u_reference[:,:,1])**2  # control effort
            # + 40*j**2 + 80*sr**2
            + 50*(du_reference[:,:,0])**2 + 2000*(du_reference[:,:,1])**2 # driving comfort 
            + 3000*spatial_risk + 30*np.log(1+np.exp(-10*(min_dist-2))) + 5*np.log(1+np.exp(-5*(lane_dist-1)))#+100*np.exp(-10*(lane_dist)) #10*(1/min_dist)**2 + 0.1*(1/lane_dist)**2 #+ 2*(1/min_dist)**2 + 1*(1/lane_dist)**2# 20*np.exp(-(min_dist-0.5)) #+10*np.exp(-10*(lane_dist-0.2))
        ) #(N_pred,Nsample)

        track_max = np.max(track_cost,axis=1) #for normalization 
        risk_max = np.max(spatial_risk,axis=1) # for normalization
        # cost = (1-risk_scalar.T)*track_cost.T/track_max + risk_scalar.T* min_dist.T
        cost = track_cost.T #+ 5*np.exp(-3*min_dist).T + 3*np.exp(-lane_dist).T # + spatial_risk.T
        # cost = (1-risk_scalar.T)*track_cost.T/track_max + risk_scalar.T* spatial_risk.T/risk_max #+ 2*np.exp(-3*min_dist).T + np.exp(-lane_dist).T
        cost = np.power(self.gamma, np.arange(0, cost.shape[1], 1))*cost  # discounted cost over the prediction
        cost = np.mean(cost,axis=1) #+ collision_risk # summation over the prediction horizon
        
        # the cost is overrded with infinity if it is not feasible
        # TODO: add other constraints here if needed
        # cost[v<0] = np.inf
        # cost[v>self.vmax] = np.inf
        # cost[min_dist[0] < 1] = np.inf
        # cost[np.any(lane_dist< 0.1,axis=0)] = 1e6
        cost = self.convolve(cost)
        
        self.xhat_history = obstacle_positions #self.xhat_history[-self.S:].reshape(self.S,Nveh,-1,4)
        return cost, spatial_risk[0]
    def convolve(self,arr):
        N_sr = self.N_sr
        N_j = self.N_j

        # Reshape the array to (M, M)
        array_2d = arr.reshape((N_sr, N_j))

        # Define the averaging kernel
        kernel = np.array([[1, 1, 1],
                        [1, 1, 1],
                        [1, 1, 1]]) / 9

        # Pad the array to handle edges and corners
        padded_array = np.pad(array_2d, pad_width=1, mode='constant', constant_values=0)

        # Perform the convolution
        convolved_array = convolve(padded_array, kernel, mode='constant', cval=0.0)

        # Remove padding to get the final result
        result = convolved_array[1:-1, 1:-1]
        return result.flatten()

    def collision_check(self, x_reference, obstacles):
        '''
        x_reference: (Npred,Nsample,4)
        obstacles: (Npred,Nveh,Nsample,4)
        '''
        obstacles = np.transpose(obstacles,[0,2,1,3])
        Npred,Nsample,Nveh,_ = obstacles.shape
        x, y, theta = x_reference[:,:,0], x_reference[:,:,1], x_reference[:,:,2]
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
        min_dist = np.min(dists, axis=(3, 4, 2)) #(Npred, Nsample)
        w = np.power(1/self.confidence, np.arange(0, min_dist.shape[0], 1))
        min_dist = min_dist * np.repeat(w[:,np.newaxis],min_dist.shape[1],axis=1)
        # min_dist = np.mean(min_dist, axis =2 )
        return min_dist #np.clip(min_dist,1e-3,None) #np.exp(-1*min_dist**2) #1/(min_dist**2) #np.exp(-min_dist**2) #1/(10*min_dist**2)#
    
    def get_predictions(self,arg):
        return np.transpose(self.xhat_history[:,:,arg],[1,0,2])
    
    def spatial_risk(self, candidates_expanded, xhat_predictions):
        '''skewed gaussian. corresponds to the cost in the RCMS paper.'''

        p_ego = np.repeat(candidates_expanded[:, np.newaxis, :, :2], xhat_predictions.shape[1],axis=1)
        p_veh = xhat_predictions[:, :, :, :2] #(Npred,Nveh,Nsample,2)
        p_bar = p_veh-p_ego 
        v_veh_x = xhat_predictions[:, :, :, 3]*np.cos(xhat_predictions[:,:,:,2])
        v_veh_y = xhat_predictions[:, :,:, 3]*np.sin(xhat_predictions[:,:,:,2])
        v_veh = np.stack([v_veh_x,v_veh_y],axis=-1) #(Npred,Nveh,Nsample,2)

        v_ego_x = candidates_expanded[:, :, 3]*np.cos(candidates_expanded[:,:,2])  # (N_veh,N_pred)
        v_ego_y = candidates_expanded[:, :, 3]*np.sin(candidates_expanded[:,:,2])
        v_ego = np.stack([v_ego_x,v_ego_y],axis=-1)[:,np.newaxis,:,:]#(Npred,1,Nsample,2)

        v_bar = v_veh-v_ego #(Npred,Nveh,Nsample,2)

        R_veh = np.transpose(self.rotation_matrix(
            xhat_predictions[:, :, :, 2]), [2, 3, 4, 0, 1])

        cov =  np.array([[self.veh_length, 0], [0, self.veh_width]])
        cov_veh = R_veh @ cov @ np.transpose(R_veh, [0, 1, 2, 4, 3])
        # cov_veh_inv = np.power(self.confidence, np.arange(0, v_bar.shape[2], 1))[:,np.newaxis,np.newaxis]*np.linalg.pinv(cov_veh)  # (Npred,N_veh,Nsample,2,2)
        cov_veh_inv = np.linalg.pinv(cov_veh)
        dist = 1/(self.alpha_g + p_bar[:,:,:,np.newaxis,:]@cov_veh_inv@p_bar[:,:,:,:,np.newaxis]) #(Npred,Nveh,Nsample,1)
        skew = 1 / (1+np.exp(self.alpha_s*p_bar[:,:,:,np.newaxis,:]@v_bar[:,:,:,:,np.newaxis]))

        p_road_1 = np.array([[0],[-0.5]])
        p_road_2 = np.array([[0],[9]])
        p_bar_lane_1 = (p_ego[:,0,:,:,np.newaxis] - p_road_1)[:,:,1]
        p_bar_lane_2 = (p_road_2 - p_ego[:,0,:,:,np.newaxis])[:,:,1]

        np.clip(p_bar_lane_1,0,None,out=p_bar_lane_1)
        np.clip(p_bar_lane_2,0,None,out=p_bar_lane_2)
        lane_dist = np.minimum(p_bar_lane_1,p_bar_lane_2)

        lane_risk = np.exp(-self.alpha_r*p_bar_lane_1*p_bar_lane_1) + np.exp(-self.alpha_r*p_bar_lane_2*p_bar_lane_2) #(Nveh,Nsample,1)
        # risk = np.sum(dist * skew, axis=1)[:, :, 0, 0]  + lane_risk[:,:,0]
        # spatial_risk = np.mean(dist * skew, axis=1)[:, :, 0, 0]  #+ lane_risk[:,:,0]
        min_dist = self.collision_check(candidates_expanded, xhat_predictions) #(Npred, Nsample)
        spatial_risk = 5*np.mean(dist * skew, axis=1)[:, :, 0, 0]  + 0.25*lane_risk[:,:,0]

        # risk = 0.5*collision_risk + 0.5*spatial_risk+0.2*lane_risk[:,:,0] #0.5*spatial_risk+0.5*collision_risk + 0.2*lane_risk[:,:,0]
        return spatial_risk, min_dist, lane_dist[:,:,0] #np.clip(lane_dist[:,:,0],1e-3,None)
