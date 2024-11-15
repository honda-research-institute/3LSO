#!/usr/bin/env python

import math, copy
# from typing import NamedTuple
import numpy as np
from scipy.interpolate import interp1d
import sys, os
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from utils import dist_betw_vehicles, euclidean_dist, get_nearest_index, get_closest_dist_to_traj, plan_speed_qp
from vehicle import State
# from lane import Lane
from frenet_utils import get_frenet,next_point_index, distance
import rospy
from config import nnmpc_params
from collections import namedtuple

sgan_source_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'sgan_source'))
sgan_path = os.path.join(sgan_source_path, 'sgan')

sys.path.insert(0,sgan_path)
from sgan_source.sgan.predictor import Predictor
from lane_keep_opt_planner import *

MERGING_LANE_ID = 100

class nnmpc(object):
    """docstring for nnmpc."""
    def __init__(self, config_dict):
        super(nnmpc, self).__init__()
        self.config = config_dict
        self.predictor = Predictor(os.path.join(sgan_source_path,'model',self.config["sgan_model"]))
        # self.dead_end = self.config["dead_end"]
        # self.a_prev = 0.0
        # self.delta_prev = 0.0
        self.debug = self.config["debug"]
        self.safety_bound_front = self.config["safety_bound_front"]
        self.safety_bound_rear = self.config["safety_bound_rear"]
        self.qp_speed_planning = False
        self.speed_limit = 0
        self.a_max = 0
        self.close_ind_to_target_lane = 0
        self.close_ind_to_source_lane = 0
        self.Configs = namedtuple("Configs", ['speed_limit','a_max','a_min', 'dead_end_s'])
        

    def get_waypoints(self, ego, others, source_lane, target_lane, configs = False):
        """
        Get the waypoints in time.
        """
        if len(target_lane.list) == 0 or len(source_lane.list) == 0:
            rospy.logwarn("[ NNMPC] Empty lane info.")
            return [] 
        
        self.close_ind_to_source_lane = get_nearest_index(ego.pos, source_lane.list)
        self.close_ind_to_target_lane = get_nearest_index(ego.pos, target_lane.list)

        # print("============================= GET WAYPOINT ================================")
        # print("in_get_waypoints")
        # print("slane id:{} width:{} list len:{}".format(source_lane.id, source_lane.width, len(source_lane.list)))
        # print("tlane id:{} width:{} list len:{}".format(target_lane.id, target_lane.width, len(target_lane.list)))
        # if configs: 
        #     self.Configs.speed_limit = v_max
        #     self.Configs.speed_limit = v_max
        # else:
        #     self.Configs.speed_limit = self.Configs.speed_limit 

        if configs:
            self.Configs.speed_limit = configs.speed_limit
            self.Configs.a_min = configs.a_min
            self.Configs.a_max = configs.a_max
            self.Configs.dead_end_s = configs.dead_end_s
            # print("configs.dead_end_s: {}".format(configs.dead_end_s))
        else:
            self.Configs.speed_limit = self.config["v_des"]
            self.Configs.a_min = self.config["a_min"]
            self.Configs.a_max = self.config["a_max"]
            self.Configs.dead_end_s = 0
        # print("NNMPC: speed_limit: {}".format(self.Configs.speed_limit))

        # ego = copy.copy(ego_)
        # others = copy.copy(others_)
        self.source_lane = source_lane
        self.target_lane = target_lane

        # print("NNMPC: merging_point_ahead: {}".format(self.merging_point_ahead(ego)))



        # self.shouldRefineActionSpace = self.config["intention_based_trajectory"] and ego.s <= self.dead_end - 20
        N_receding = int(round(self.config["T"]/self.config["dt"])) # MPC control horz
        findOpt = len(ego.records) >= int(round(self.config["T_obs"]/self.config["dt"]))
        isEnoughObservation = False
        # print("targetlane width: {} ego.d: {}".format(target_lane.width, ego.d))
        self.onLaneChange = abs(ego.d) > 1/2*target_lane.width/2+0.3
        self.onLaneKeeping = abs(ego.d) <= 1/2*target_lane.width/2+0.3
        # self.onLaneChange  = source_lane.id != target_lane.id
        # self.onLaneKeeping = source_lane.id == target_lane.id
        self.onRight = ego.d >= 0
        # self.onLaneChange = True
        # self.onLaneKeeping = False
        # if self.onRight: dir = "left"
        # else: dir = "right"
        # if self.onLaneChange: rospy.loginfo("[ planner] Lane Changing to {}".format(dir))
        # else: rospy.loginfo("[ planner] Lane keeping")


        # =================== prediction accruacy analysis =====================
        #                       (Sangjae, March 29, 2021)
        # if self.compute_error:
        #     # for id in others.keys():
        #     for id in self.vehs_to_compute_error:
        #         if not id in self.pred_errors.keys(): self.pred_errors[id] = []
        #         if not id in self.pred_prev.keys(): self.pred_prev[id] = [others[id].x,others[id].y]
        #         err = np.sqrt((others[id].x-self.pred_prev[id][0])**2+(others[id].y-self.pred_prev[id][1])**2)
        #         err_x = err*abs(np.cos(others[id].theta))
        #         err_y = err*abs(np.sin(others[id].theta))
        #         print("prediction error x: {:.2f} y: {:.2f} (id: {}, pred: {:.2f}, actual: {:.2f})".format(err_x,err_y, id, self.pred_prev[id][0], others[id].x))
        #         self.max_pred_error = min(max(self.max_pred_error, err), self.config["thred_safety"])
        #         self.compute_error = False
        #         t = time.time()
        #         self.err_x_record.append([t, err_x])
        #         self.err_y_record.append([t, err_y])
        #         np.savetxt("{}_error_x.csv".format(self.begin_time), self.err_x_record, delimiter=",")
        #         np.savetxt("{}_error_y.csv".format(self.begin_time), self.err_y_record, delimiter=",")
        # ======================================================================
        time_gap = 3.0 # sec
        front_ind_target = self.get_front_ind(ego, others, target_lane, range_m = max(int(time_gap * ego.v),20))
        front_ind_source = self.get_front_ind(ego, others, source_lane, range_m = max(int(time_gap * ego.v),20))
    
        if self.onLaneKeeping:
            print("lane keeping")
            
            if front_ind_target:
                rospy.loginfo("Lane following: front_ind_target: {}, v: {}".format(front_ind_target, others[front_ind_target].v))
            # print("close_ind_to_target_lane: ",self.close_ind_to_target_lane)
            # print("source_lane.id: ",source_lane.id)
            # print("target_lane.id: ",target_lane.id)
            # print("target_lane len: ", len(target_lane.list))
            traj = self.lane_change_traj(ego, others, target_lane, front_ind_target, target_ind_ahead = max(int(time_gap * ego.v),20), close_ind = self.close_ind_to_target_lane)
            return traj

        print("lane changing")
        # run IDM if data is not enough to run SGAN
        if front_ind_target:
            isEnoughObservation = len(others[front_ind_target].records) > \
                            int(self.config["T_obs"]/self.config["dt"])*(int(self.config["dt"]/self.config["timestep"]))+5.0 # Best 20
        if front_ind_target and not isEnoughObservation:
            a = self.idm_acc(ego, others, target_lane)
            a = min(a,self.Configs.a_max)/5 # keep it slow until having enough observations
            waypoints = self.control_to_waypoint(ego,np.zeros(N_receding),a*np.ones(N_receding), target_lane)
            rospy.loginfo("Not enough observation. IDM running.")
            return waypoints

        # get indices of the near vehicles
        ind_near = self.get_indices_near_vehs(ego, others, target_lane, range_m = max(ego.v*3,10))

        # rospy.loginfo("[ planner] all vehicles (total {}) in range: {}.".format(len(others),others.keys()))
        if self.debug and len(ind_near) >= 1:
            rospy.loginfo("[ planner] vehicles (total {}) in range: {}.".format(len(ind_near),ind_near))
            glob_min_dist, id = self.get_global_min_dist(ego, others, ind_near)
            for i in ind_near:
                if self.is_front(ego, others[i]):
                    rospy.loginfo("Distance to front {} veh: {} (safety bound: {})".format(
                        i, self.inter_veh_gap(ego,others[i]), self.safety_bound_front))
                else:
                    rospy.loginfo("Distance to rear {} veh: {} (safety bound: {})".format(
                        i, self.inter_veh_gap(ego,others[i]), self.safety_bound_rear))
            if self.is_front(ego, others[id]):
                rospy.loginfo("minimum (front) distance: {} to veh {} (safety bound: {})".format(glob_min_dist, id, self.safety_bound_front))
            else:
                rospy.loginfo("minimum (rear) distance: {} to veh {} (safety bound: {})".format(glob_min_dist, id, self.safety_bound_rear))


        if len(ind_near) >= 1:
            cands = []
            cands_name = []
            front_gap = 100 # arbitrarily large
            
            if front_ind_source:
                front_gap = self.inter_veh_gap(ego,others[front_ind_source])
            
            d_short = min(max(int(ego.v*1),10), front_gap)
            d_mid = min(max(int(ego.v*2),15), front_gap)
            d_long = min(max(int(ego.v*3),20), front_gap)
            
            if front_gap <= d_short:
                [CHECK_SHORT, CHECK_MID, CHECK_LONG] = [True, False, False]
                d_short = max(d_short, 5) # lower bound the minimum lane changing distance
            elif front_gap <= d_mid:
                [CHECK_SHORT, CHECK_MID, CHECK_LONG] = [True, True, False]
            else: # front_gap <= d_long
                [CHECK_SHORT, CHECK_MID, CHECK_LONG] = [True, True, True]
            # print("front gap: {} flags (short, mid, long) = {}".format(front_gap, [CHECK_SHORT, CHECK_MID, CHECK_LONG]))
            
            # ==================================
            # Generate the trajectory candidates
            # ==================================
            # traj_to_source_in_d = self.lane_change_traj(ego, others, source_lane, False)
            if CHECK_SHORT:
                traj_to_target_in_d_short = self.lane_change_traj(ego, others, target_lane, front_ind_target, target_ind_ahead = d_short, close_ind = self.close_ind_to_target_lane)
                if len(traj_to_target_in_d_short) >= N_receding:
                    wp = self.convert_waypoints_in_time(traj_to_target_in_d_short)
                    if len(list(wp)) >= N_receding:
                        cands.append(wp)
                        cands_name.append("change to TARGET -- SHORT") 
                    
            if CHECK_MID:
                traj_to_target_in_d_mid = self.lane_change_traj(ego, others, target_lane, front_ind_target, target_ind_ahead = d_mid, close_ind = self.close_ind_to_target_lane)
                if len(traj_to_target_in_d_mid) >= N_receding:
                    wp = self.convert_waypoints_in_time(traj_to_target_in_d_mid)
                    if len(list(wp)) >= N_receding:
                        cands.append(wp)
                        cands_name.append("change to TARGET -- MED")
                
            if CHECK_LONG:                
                traj_to_target_in_d_long = self.lane_change_traj(ego, others, target_lane, front_ind_target, target_ind_ahead = d_long, close_ind = self.close_ind_to_target_lane)
                if len(traj_to_target_in_d_long) >= N_receding:
                    wp = self.convert_waypoints_in_time(traj_to_target_in_d_long)
                    if len(list(wp)) >= N_receding:
                        cands.append(wp)
                        cands_name.append("change to TARGET -- LONG")

            

            # print("traj_to_target_in_d_mid len: {}".format(len(traj_to_target_in_d_mid)))
            

            # print("traj_to_target_in_d_long len: {}".format(len(traj_to_target_in_d_long)))
            

            
            # if len(traj_to_source_in_d) >= N_receding:
            #     waypoints_source = self.convert_waypoints_in_time(traj_to_source_in_d)
            #     if len(waypoints_source) >= N_receding:
            #         cands.append(waypoints_source)
            #         cands_name.append("return to SOURCE")

            # waypoints_acc    = self.fixed_acc_traj(ego, np.random.rand())
            # waypoints_dcc    = self.fixed_acc_traj(ego, -np.random.rand())
            # cands.append(waypoints_acc)
            # cands.append(waypoints_dcc)
            # cands_name.append("constant ACC")
            # cands_name.append("constant DCC")

            rospy.loginfo("length of candidates: {}, {}".format(len(cands),cands_name))
            rospy.loginfo("vehs to consider: {}, {}".format(len(ind_near),ind_near))
            waypoints = False
            if len(cands) > 0:
                # print("line 204: evaluating trajectories")
                waypoints = self.eval_cands(ego, others, ind_near,
                                            target_lane,
                                            cands,
                                            cands_name,
                                            lane_width = target_lane.width)
                # rospy.loginfo("waypoint after eval: {}".format(waypoints))


        else: # if no other vehicles around
            rospy.loginfo("[ NNMPC] No vehicle is around. Free-flow lane changing.")
            front_ind_source = self.get_front_ind(ego, others, source_lane, range_m = max(ego.v * 4,30))
            if front_ind_source:
                dist_to_lane_change = dist_betw_vehicles(ego, others[front_ind_source])-1
            else:
                dist_to_lane_change = max(int(ego.v*5),20)
            # waypoints = self.lane_change_traj(ego, others, target_lane, front_ind_target, target_ind_ahead = min(int(dist_to_lane_change),35), close_ind = self.close_ind_to_target_lane)
            waypoints = self.lane_change_traj(ego, others, target_lane, front_ind_target, 
                            target_ind_ahead = max(dist_to_lane_change,7), 
                            close_ind = self.close_ind_to_target_lane)
            # waypoints = self.lane_change_traj(ego, others, target_lane, False)

        # feasible solution found
        
        if waypoints:
            if self.qp_speed_planning:
                try:
                    vf = False; df = False
                    if front_ind_target:
                        veh = others[front_ind_target]
                        df = dist_betw_vehicles(ego, veh)
                        vf = veh.v
                    xy_profile = [[point[0],point[1]] for point in waypoints]
                    v,d,_,_ = plan_speed_qp(ego.v,vf,df,a_min=self.Configs.a_min,a_max=self.Configs.a_max,v_max = self.Configs.speed_limit)
                    ds_arr = [euclidean_dist(point, xy_profile[i+1]) for i, point in enumerate(xy_profile[:-1])]
                    ds_arr.insert(0, 0)
                    d_arr = np.cumsum(ds_arr)
                    v_profile = np.interp(d_arr, d, v).tolist()
                    # rebuild waypoints
                    waypoints = [[xy[0], xy[1], v] for xy, v in zip(xy_profile, v_profile)]
                except:
                    rospy.logwarn("NNMPC: quadratic programming failed. Linear mapping applied.")
            return waypoints

        # infeasible solution
        else:
            # print("infeasible waypoint: {}".format(waypoints))
            rospy.logwarn("[ NNMPC] solution infeasible")
            front_ind_source = self.get_front_ind(ego, others, source_lane, range_m = max(ego.v * 3,20))

            # Merging or dead end scenario
            if (target_lane.id == MERGING_LANE_ID and self.merging_point_ahead(ego)) or \
                    self.is_dead_lock_ahead(ego, others, source_lane):
                rospy.logwarn("NNMPC: Dead end ahead.")
                if front_ind_source:
                    dist_to_front = dist_betw_vehicles(ego, others[front_ind_source])
                    if dist_to_front <= self.config["dead_zone"]:
                        return -1 # we are in the dead zone. stop and wait
                    else:
                        return self.lane_change_traj(ego, others, source_lane, front_ind_source, close_ind = self.close_ind_to_source_lane)
                elif self.get_dead_lock_dist(ego, others, source_lane) <= self.config["dead_zone"]:
                    return -1 # we are in the dead zone. stop and wait
                else:
                    # return to source lane
                    print("Returning to source")
                    return self.lane_change_traj(ego, others, source_lane, False, close_ind = self.close_ind_to_source_lane)
            
            # Same direction scenario
            else:
                if ego.v >= 5 and abs(ego.d) > target_lane.width/2+0.3:
                    print("Returning to source")
                    return self.lane_change_traj(ego, others, source_lane, front_ind_source, close_ind = self.close_ind_to_source_lane)
                elif ego.v >= 5 and abs(ego.d) <= target_lane.width/2+0.3:
                    print("Already in target. Push forward to the target lane")
                    # front_ind_target  = self.get_front_ind(ego, others, target_lane, range_m = max(ego.v * 3,20))
                    return self.lane_change_traj(ego, others, target_lane, front_ind_target, close_ind = self.close_ind_to_target_lane)
                else:
                    return -1



    def eval_cands(self, ego, others, ind_near, target_lane,
                                        cands, cands_name, lane_width = 3.0, gamma = 0.9):
        """
        Return the best waypoints from the candidate,
        """
           
        # Filter out unimportant vehicles
        others_ = {}
        ind_near_ = []
        for id in ind_near:
            if self.inter_veh_gap(ego, others[id]) <= self.config["range_m"]:
                if id not in others_:
                    others_[id] = others[id]
                    ind_near_.append(id)

        # Define local vars
        N_sim = len(cands)
        N_obs = int(round(self.config["T_obs"]/self.config["dt"])) # observation horz
        N_receding = int(round(self.config["T"]/self.config["dt"])) # MPC control horz
        N_near_vehs = len(ind_near_)
        N_vehs = len(ind_near_) + 1 # to include ego
        Ns = int(self.config["dt"]/self.config["timestep"])
        sub_opt_ind = -1


        # Return it if no evaluation is needed
        if len(ind_near_) == 0 or len(list(others_)) == 0:
            print("no evaulation needed after checking near vehicles")
            return cands[-1]

        # initialize metrics for control candidates
        # print("ego at the beginning: ({},{})".format(ego.x,ego.y))
        ego_vec = np.array([copy.deepcopy(ego) for i in range(N_sim)])
        others_vec = np.array([copy.deepcopy(others_) for i in range(N_sim)]) # list of dict
        # ego_vec = np.array([copy.deepcopy(ego) for i in range(N_sim)])
        # others_vec = np.array([others.copy() for i in range(N_sim)]) # list of dict
        min_dist_vec = float('inf')*np.ones(N_sim)
        feas_inds = range(N_sim)
        cost_vec = np.zeros(N_sim)
        lane_offset_vec = np.array([ego.d for i in range(N_sim)]) # list
        # dist_to_end_vec = dist_to_end * np.ones(N_sim) # vector for distance to end
        # print("-- evaluating cands")
        for ell in range(N_receding-1):
            # safety check
            # print("ell: {} safety checking".format(ell))
            inds_to_remove = []
            for (ni,n) in enumerate(feas_inds): # feasible simulation inds
                for id in ind_near_: # nearby vehicle inds
                    _, d_ego, _ = get_frenet(ego_vec[n].x, ego_vec[n].y, target_lane.list, target_lane.s_map)
                    _, d_veh, _ = get_frenet(others_vec[n][id].x, others_vec[n][id].y, target_lane.list, target_lane.s_map)
                    if abs(d_ego) <= target_lane.width/2+0.5 and abs(d_veh) >= target_lane.width/2+0.3:
                        # rospy.logwarn("ignore veh: {} as in different lane".format(id))
                        continue # ignore vehicle in different lane
                    
                    dist = self.inter_veh_gap(ego_vec[n], others_vec[n][id])
                    min_dist_vec[n] = min(min_dist_vec[n], dist)
                    if self.is_rear(ego_vec[n],others_vec[n][id]):
                        if min_dist_vec[n] <= self.safety_bound_rear:
                            inds_to_remove.append(ni)
                    else:
                        if min_dist_vec[n] <= self.safety_bound_front:
                            inds_to_remove.append(ni)

            # remove the infeasible candidate
            if len(inds_to_remove) >= 1:
                # print("ell: {} removing cands: {}".format(ell, inds_to_remove))
                feas_inds = np.delete(feas_inds, inds_to_remove)
                if len(feas_inds) == 0: 
                    max_ell=ell-1
                    rospy.loginfo("No feasible candidates remaining. Evaluation terminated.")
                    break
                rospy.logwarn("AFTER REMOVE, feas_inds: {}".format(feas_inds))

            # initialize sgan input
            # print("ell: {} initializing sgan".format(ell))
            obs_traj = []
            for t in range(N_obs):
                obs_traj_t = []
                for n in feas_inds:
                    for i in range(N_vehs):
                        if i == 0: # ego
                            veh = ego_vec[n]
                        else: # other
                            veh = others_vec[n][ind_near_[i-1]]
                        try:
                            pos = [veh.records[(N_obs-1)*Ns-t*Ns].x, veh.records[(N_obs-1)*Ns-t*Ns].y]
                        except:
                            pos = [veh.records[-1].x, veh.records[-1].y]
                        obs_traj_t.append(pos)
                obs_traj.append(obs_traj_t)

            # relative observation trajectory
            obs_traj_rel = []
            for t in range(N_obs):
                obs_traj_rel_t = []
                for j in range(len(obs_traj[0])):
                    if t == 0:
                        pos = [0.0, 0.0]
                    else:
                        pos = [obs_traj[t][j][0]-obs_traj[t-1][j][0],   obs_traj[t][j][1]-obs_traj[t-1][j][1]]
                    obs_traj_rel_t.append(pos)
                obs_traj_rel.append(obs_traj_rel_t)

            # start end sequence
            seq_start_end = []
            for (ni,n) in enumerate(feas_inds): seq_start_end.append([ni*N_vehs, (ni+1)*N_vehs])

            # Predict next positions
            next_pred_traj = self.predictor.predict_batch(obs_traj, obs_traj_rel, seq_start_end)

            for (ni,n) in enumerate(feas_inds):
                for i in range(N_near_vehs):
                    id = ind_near_[i]
                    # isStopped = abs(obs_traj[-1][N_vehs*ni+i+1][0])-abs(obs_traj[-2][N_vehs*ni+i+1][0]) <= 0.0001 \
                    #         and abs(obs_traj[-1][N_vehs*ni+i+1][1])-abs(obs_traj[-2][N_vehs*ni+i+1][1])  <= 0.0001
                    isStopped = others[id].v <= 0.1
                    if isStopped:
                        next_pred_traj[N_vehs*ni+i+1][0] = obs_traj[-1][N_vehs*ni+i+1][0]
                        next_pred_traj[N_vehs*ni+i+1][1] = obs_traj[-1][N_vehs*ni+i+1][1]

            # ============== for prediction errors ====================
            #                 (March 31, 2021, Sangjae)
            # if ell == 0 and self.compute_error:
            #     # print("obs_traj: {}".format(obs_traj))
            #     # print("next_pred_traj: {}".format(next_pred_traj))
            #     ni = 0 # first candidate
            #     self.compute_error = True
            #     self.vehs_to_compute_error = ind_near_
            #     for i, id in enumerate(ind_near_):
            #         self.pred_prev[id] = [next_pred_traj[N_vehs * ni + i+1][0], next_pred_traj[N_vehs * ni + i+1][1]]
            # =========================================================


            # propagate other vehicles
            # print("ell: {} propagating vehs".format(ell))
            for (ni,n) in enumerate(feas_inds):
                # update the scene with the predicted predictions
                for i in range(N_near_vehs):
                    id = ind_near_[i]
                    veh = others_vec[n][id]
                    x_ = next_pred_traj[N_vehs * ni + i+1][0]
                    y_ = next_pred_traj[N_vehs * ni + i+1][1]
                    x_diff = x_ - veh.x
                    y_diff = y_ - veh.y

                    if x_diff == 0:
                        theta_ = veh.theta
                    else:
                        theta_ = np.arctan(y_diff / x_diff)
                    v_ = np.sqrt(x_diff**2 + y_diff**2)/self.config["dt"]
                    veh.set_state(x_,y_,theta_,v_)
                    others_vec[n][id] = veh

            # propagate ego
            for n in feas_inds:
                ego_ = ego_vec[n]
                x_=cands[n][ell+1][0]
                y_=cands[n][ell+1][1]
                v_=cands[n][ell+1][2]
                x_diff = x_ - ego_.x
                y_diff = y_ - ego_.y
                if x_diff == 0: theta_ = ego_.theta
                else: theta_ = np.arctan(y_diff / x_diff)
                distance_, lane_offset_, _ = get_frenet(x_, y_, target_lane.list, target_lane.s_map) # XXX sangjae, Nov 15, 2021
                ego_.set_state(x_,y_,theta_,v_,d=lane_offset_,s=distance_)
                ego_vec[n] = ego_
                # print("ell:{}, cand:{}, ego_: ({},{},{},{})".format(ell, cands_name[n], ego_.x,ego_.y,ego_.theta,ego_.v))
                # print("ell:{}, cand:{}, ego: ({},{},{},{})".format(ell, cands_name[n], ego.x,ego.y,ego.theta,ego.v))


            # compute the cumulative cost
            for n in feas_inds:
                ego_ = ego_vec[n]
                # _, lane_offset_vec[n] = get_frenet(ego_.x, ego_.y, ego_.theta, target_lane_line_list)
                _, lane_offset_vec[n], _ = get_frenet(ego_.x, ego_.y, target_lane.list, target_lane.s_map)
                cost_vec[n] += gamma**(ell)*((self.config["lambda_v"])*(ego_.v - self.Configs.speed_limit)**2
                                                  + (self.config["lambda_div"])*(lane_offset_vec[n])**2)
            sub_opt_ind = np.argmin(cost_vec) # -- keep the best up to this iteration



        # Check if the last position is in the dead_zone
        if self.is_dead_lock_ahead(ego, others, self.source_lane):
            rospy.loginfo("Deadlock detected")
            inds_to_remove = []
            for (ni,n) in enumerate(feas_inds): # feasible simulation inds
                ego_ = ego_vec[n]
                # if (ego_.s >= self.config["dead_end"]-self.config["dead_zone"] and abs(ego_.d) > lane_width/2*1.7):
                if (self.get_dead_lock_dist(ego, others, self.source_lane) <= self.config["dead_zone"]
                    and abs(ego_.d) > lane_width/2*1.7):
                    inds_to_remove.append(ni)
            feas_inds = np.delete(feas_inds, inds_to_remove)


        # return if feasible solution exists
        if len(feas_inds) >= 1:
            opt_ind = np.argmin(cost_vec[feas_inds])
            rospy.logwarn("Best intention: {}".format(cands_name[feas_inds[opt_ind]]))
            rospy.logwarn("Minimum distance over control horizon: {}".format(min_dist_vec))
            return cands[feas_inds[opt_ind]]#, cands_name[feas_inds[opt_ind]]

        # if self.config["is_dead_lock_scenario"]:
        #     rospy.logwarn("Best intention: STOP and WAIT")
        return False


    def fixed_acc_traj(self, ego, acc):
        """
        Generate waypoints with fixed acceleration/decelerations
        """
        N_receding = int(round(self.config["T"]/self.config["dt"]))
        dt = self.config["dt"]
        x=ego.x
        y=ego.y
        v=ego.v
        theta=ego.theta
        waypoints = [[x,y,v]]
        for l in range(N_receding):
            x = x + (v*dt + 1/2*acc*dt**2) * np.cos(theta)
            y = y + (v*dt + 1/2*acc*dt**2) * np.sin(theta)
            v = v + acc*dt
            waypoints.append([x,y,v])
        return waypoints


    def lane_change_traj(self, ego, others, lane, front_ind, target_ind_ahead = 10, close_ind = -1, space_step = False, N_path = 50):
        """
        Optimization for lane-change maneuver
        """
        
        try:
            if distance(ego.x,ego.y,lane.list[0].x,lane.list[0].y) >= 500: # handle initialization error
                ego.x,ego.y = lane.list[0].x,lane.list[0].y 
            if close_ind == -1:
                close_ind = get_nearest_index(ego.pos, lane.list)
            # print("close_ind: {}, target_ind_ahead: {}, len(lane.list): {}".format(close_ind, target_ind_ahead, len(lane.list)))
            # print("close point: {} ego state: {}".format(lane.list[close_ind],[ego.x,ego.y]))
            goal_state = lane.list[min(close_ind + int(target_ind_ahead), len(lane.list)-1)]
            goal_state_local = get_local_goal_state(goal_state, [ego.x,ego.y,ego.theta])
            path = plan_path(goal_state_local)
            path = list(map(list, zip(*path)))  ## swap the axes of list of lists
            path = transform_path(path, [ego.x,ego.y,ego.theta])
            if front_ind: # car following
                # if dist_betw_vehicles(ego, others[front_ind]) <= 5: # too close 
                #     profile = follow_profile(path, ego.v, 0, self.Configs.a_max,
                #                 [others[front_ind].x, others[front_ind].y, max(others[front_ind].v,0)])
                # else:
                profile = follow_profile(path, ego.v, self.Configs.speed_limit, self.Configs.a_max,
                            [others[front_ind].x, others[front_ind].y, max(others[front_ind].v,0)])
            else: # nominal
                profile = nominal_profile(path, ego.v, self.Configs.speed_limit, self.Configs.a_max, self.Configs.a_min)

            # If it should be space step, interporate 
            if space_step and len(profile) > 1: 
                s_list = [np.sqrt((point[0]-profile[0][0])**2 + (point[1]-profile[0][1])**2) for point in profile]
                
                # interpolate with the space step
                # print("s_list : {}",s_list)
                xs = np.array(profile)[:,0]#[point[0] for point in profile]
                ys = np.array(profile)[:,1]#[point[1] for point in profile]
                vs = np.array(profile)[:,2]#[point[2] for point in profile]
                fx = interp1d(s_list,xs,fill_value="extrapolate")
                fy = interp1d(s_list,ys,fill_value="extrapolate")
                fv = interp1d(s_list,vs,fill_value="extrapolate")

                s_list_des = [i*space_step for i in range(max(len(s_list),51))]
                x_interp = fx(s_list_des)
                y_interp = fy(s_list_des)
                v_interp = fv(s_list_des)

                profile = [[x,y,v] for x,y,v in zip(x_interp, y_interp, v_interp)]
            return profile
        except:
            rospy.logwarn("Check if the lane info is being published.")
            return []



    def convert_waypoints_in_time(self, waypoints_dis, dd= 0.3):
        """
        Convert waypoints in distance domain to time domain
        """
        tot_dis = dd * (len(waypoints_dis)-1)

        ds = np.transpose(waypoints_dis)[2][:-1] * self.config["dt"]
        ds = np.insert(ds,0,0.0)
        ds = np.cumsum(ds)

        a=np.arange(0,tot_dis,dd)
        b=np.transpose(waypoints_dis)
        ll=min(len(a),len(b[0]))
        ds = ds[ds<=tot_dis]
        ds = ds[ds<=max(a[:ll])]

        # try:
        xs = interp1d(a[:ll], b[0][:ll])(ds)
        ys = interp1d(a[:ll], b[1][:ll])(ds)
        vs = interp1d(a[:ll], b[2][:ll])(ds)
        path = [xs.tolist(), ys.tolist(), vs.tolist()]
        path = list(map(list, zip(*path)))  ## swap the axes of list of lists
        return path
        # except:
        #     rospy.logwarn("ERROR IN ")
        #     # print("tot_dis: {}".format(tot_dis))
        #     # print("ds: {}, xold: {}, yold: {}".format(ds,a,b))
        #     # print("ll: {}, ds: {}, xold[:ll]: {}, yold[:ll]: {}".format(ll,ds,a[:ll],b[:ll]))
        #     return []
        # xs = np.insert(xs, 0, waypoints_dis[0][0])
        # ys = np.insert(ys, 0, waypoints_dis[0][1])
        # vs = np.insert(vs, 0, waypoints_dis[0][2])


    def is_body_in(self, ego, lane):
        return abs(ego.d) - np.sqrt((ego.length/2)**2+(ego.width/2)**2)*abs(np.cos(np.pi/2-(ego.rel_angle+np.arctan(ego.width/ego.length)))) <= lane.width/2
                # > veh.width/2 #


    def get_indices_near_vehs(self, ego, others, lane, range_m = 10):
        if self.onLaneChange:
            ind_near = np.array([id for id in others
                                    if euclidean_dist((ego.x,ego.y),(others[id].x, others[id].y)) <= range_m
                                    and ( self.is_front(ego,others[id]) or
                                          ( not self.is_front(ego,others[id]) 
                                            and self.get_projection(ego,others[id]) <= ego.length/2+others[id].length/2# -1.0
                                          )
                                        )# XXX BEST: -1.0 XXX
                                    and len(others[id].records) > int(self.config["T_obs"]/self.config["dt"])#])
                                    and abs(others[id].d) <= lane.width/2]) # this only considers target lane
                                    #and not self.is_body_in(ego,others[id],lane)])
        else: # lane keeping
            ind_near = np.array([id for id in others
                                    if euclidean_dist((ego.x,ego.y),(others[id].x, others[id].y)) <= range_m
                                    and self.is_front(ego, others[id])
                                    and abs(others[id].d) <= lane.width/2 # only in the target lane
                                    and len(others[id].records) > int(self.config["T_obs"]/self.config["dt"])])
        return ind_near



    def get_projection(self, a, b):
        """
        Get projection of a onto b
        a : vehiche
        b : vehicle
        """
        x,y,theta=a.x,a.y,a.theta
        xi,yi,thetai=b.x,b.y,b.theta

        avec = [x-xi,y-yi]
        bvec = [np.cos(thetai),np.sin(thetai)]
        return np.dot(avec,bvec)/np.linalg.norm(bvec)


    def get_ind_front_veh(self, ego, others, lane, type="current", range_m = 10):
        min_dist = float("inf")
        ind_front = None
        for id in others:
            dist =  euclidean_dist((ego.x,ego.y),(others[id].x, others[id].y))
            # front vehicle on the target lane
            if  type == "target" and (dist <= min_dist
                                and dist <= range_m
                                and self.is_front(ego, others[id])
                                and abs(others[id].d) <= lane.width/2): # on the target lane
                min_dist = dist
                ind_front = id
            # front vehicle on the source lane
            if type == "source" and (dist <= min_dist
                and dist <= range_m
                and self.is_front(ego, others[id])
                and abs(others[id].d) > lane.width/2): # on the source lane
                min_dist = dist
                ind_front = id
            # front vehicle on the current lane (either target or source)
            if type == "current" and (dist <= min_dist
                and dist <= range_m
                and self.is_front(ego, others[id])
                and abs(ego.d-others[id].d) <= lane.width/2): # on the current lane (either target, source, or any other)
                min_dist = dist
                ind_front = id
        return ind_front

    def is_front(self, ego, veh):
        return np.dot([veh.x-ego.x, veh.y-ego.y],[np.cos(ego.theta),np.sin(ego.theta)]) >= 0

    def is_rear(self, ego, veh):
        return np.dot([veh.x-ego.x, veh.y-ego.y],[np.cos(ego.theta),np.sin(ego.theta)]) < 0

    def inter_veh_gap(self, ego, other):
        """
        Using three circle model
        """
        x,y,theta = ego.x,ego.y,ego.theta
        xi,yi,thetai = other.x,other.y,other.theta
        h = ego.length/2
        w = ego.width/2
        hi = other.length/2
        wi = other.width/2

        min_dist = float("inf")
        for i in [-1,0,1]:
            for j in [-1,0,1]:
                dist = np.sqrt(( (x + i*(h)*np.cos(theta)) - (xi+ j*(hi)*np.cos(thetai)) )**2
                         +( (y + i*(h)*np.sin(theta)) - (yi+ j*(hi)*np.sin(thetai)) )**2
                         ) - (w + wi)
                min_dist = min(dist, min_dist)
        return max(0, min_dist)


    def merging_point_ahead(self, ego):
        # check if ego passed the merging point
        if isinstance(self.Configs.dead_end_s, float):
            return ego.s < self.Configs.dead_end_s
        else: 
            return False


    def is_dead_lock_ahead(self, ego, others, source_lane):
        for id in others.keys():
            veh = others[id]
            if self.is_dead_lock(ego, veh, source_lane):
                return True
        return False


    def is_dead_lock(self, ego, veh, source_lane, N=100, VN=20):
        # check if the veh is in the source lane
        if self.is_in_lane(veh, source_lane):
            # check if they are within N meters
            # if euclidean_dist(ego.pos,veh.pos) <= N:
                # check if the veh has been stopped for the past VN steps
            if abs(veh.records[0].x - veh.records[-1].x) <= 0.01 and abs(veh.records[0].y - veh.records[-1].y) <= 0.01:
                # if all([s.v <= 0.1 for s in veh.records[:min(VN,len(veh.records-1))]]):
                return True


    def is_in_lane(self, veh, lane):
        return get_closest_dist_to_traj(veh.pos, lane.list) <= lane.width/2


    def get_dead_lock_dist(self, ego, others, source_lane, N=100, VN=20):
        min_dist = float('inf')
        for id in others.keys():
            veh = others[id]
            dist = euclidean_dist(ego.pos,veh.pos)
            if (dist <= N
                and self.is_in_lane(veh, source_lane)
                and all([s.v <= 0.1 for s in veh.records[:VN]])):
                min_dist = min(min_dist, dist)
        return min_dist


    def get_global_min_dist(self, ego, others, ind_near):
        min_dist = float("inf")
        for id in ind_near:
            dist = self.inter_veh_gap(ego, others[id])
            if dist < min_dist:
                min_dist = dist
                min_id = id
        return min_dist, min_id


    def get_front_ind(self, ego, others, lane, range_m = False, time_to_consider = 3):
        if range_m == False:
            range_m = ego.v * time_to_consider
        
        min_dist = float("inf")
        ind_front = False
        for id in list(others.keys()):
            veh = others[id]
            dist =  euclidean_dist((ego.x,ego.y),(veh.x, veh.y))
            _, d, _ = get_frenet(veh.x, veh.y, lane.list, lane.s_map)
            if dist <= min_dist and dist <= range_m and self.is_front(ego, veh) and abs(d) <= lane.width/2:
                min_dist = dist
                ind_front = id

        return ind_front


    def idm_acc(self, ego, others, target_lane):
        """
        Compute acceleration using IDM
        """
        front_ind_current = self.get_front_ind(ego, others, target_lane)
        if front_ind_current:
            rospy.logwarn("not enough data for SGAN (data length: {}). Running IDM.".format(others[front_ind_current]))
        k_spd = 1.0     # proportional constant for speed tracking when in freeflow [s^-1]
        dlta = 4.0         # acceleration exponent [-]
        T = 1.5         # desired time headway [s]
        s_min = 2.0     # minimum acceptable gap [m]
        a_max = 1.0     # maximum acceleration ability [m/s^2]
        d_cmf = 2.0     # comfortable deceleration [m/s^2] (positive)
        d_max = 9.0     # maximum deceleration [m/s^2] (positive)
        dt = self.config["dt"]        # timestep to simulate [s]
        v_offset = 1.0  # offset from the desired speed [m/s]
        if front_ind_current:
            veh = others[front_ind_current]
            headway = self.inter_veh_gap(ego, veh)
            Delta_v = veh.v - ego.v
            s_des = s_min + ego.v*T - ego.v*Delta_v / (2*np.sqrt(self.Configs.a_max*abs(self.Configs.a_min)))
            v_ratio = ego.v/self.Configs.speed_limit
            a = self.Configs.a_max * (1.0 - v_ratio**dlta - (s_des/headway))
        else:
            # no lead vehicle, just drive to match desired speed
            Delta_v = self.Configs.speed_limit - ego.v
            a = Delta_v*k_spd # predicted accel to match target speed
            a = max(a, self.Configs.a_max)
        return a

    def propagate(self, veh, a, delta):
        # aa = veh.length/2; bb = veh.length/2
        aa = 1.5*veh.length/2/2; bb = 1.5*veh.length/2/2
        L = aa+bb
        l = -bb

        x,y,theta,v,dt = veh.x,veh.y,veh.theta,veh.v,self.config["dt"]
        s = max(0,v*dt + a*dt**2/2)
        v_ = max(0,v + a *dt)

        if abs(delta) < 0.01:
            return State(x+s*np.cos(theta),y+s*np.sin(theta),theta, v_)
        else:
            R = L/np.tan(delta)
            beta = s/R
            xc = x - R*np.sin(theta) + l*np.cos(theta)
            yc = y + R*np.cos(theta) + l*np.sin(theta)

            theta_ = np.mod(theta + beta, 2*np.pi)
            x_ = xc + R*np.sin(theta + beta) - l*np.cos(theta_)
            y_ = yc - R*np.cos(theta + beta) - l*np.sin(theta_)

            return State(x_,y_,theta_,v_)


    def control_to_waypoint(self, veh, delta_seq, a_seq, target_lane):
        # N_receding = int(round(self.config["T"]/self.config["dt"])) # MPC control horz
        waypoints = []
        waypoints.append([veh.x,veh.y,veh.v])
        # if self.onLaneChange:
        for ell in range(len(delta_seq)-1):
            if ell == 0:
                self.a_prev = a_seq[ell]
                self.delta_prev = delta_seq[ell]

            state_ = self.propagate(veh, a_seq[ell], delta_seq[ell])
            veh.set_state(state_.x, state_.y, state_.theta, state_.v)
            waypoints.append([veh.x,veh.y,veh.v])

        return waypoints
