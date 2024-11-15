#!/usr/bin/env python
from distutils.log import warn
import math, copy
import numpy as np
import sys, os, time, warnings
from utils_refactored import closest_point_ind
from scipy.interpolate import interp1d

sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))

import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Pose, Point

sgan_source_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'sgan_source'))
sgan_path = os.path.join(sgan_source_path, 'sgan')

sys.path.insert(0,sgan_path)
from sgan_source.sgan.predictor import Predictor
from lane_keep_opt_planner import *

from collections import namedtuple
Point2D = namedtuple('Point2D', ['x', 'y'])


# =========================
# REFACTORING FROM HERE
# ===========================
# from utils_refactored import LocalConfig, euclidean_dist, follow_profile_idm, follow_profile_qp, get_front_ind, get_min_dist_to_lane_from_point, get_nearest_index, dist_betw_agents,\
#                         get_projection, inter_veh_gap, interp_waypoint_with_space_step, is_curvy_road, is_emergency_ahead,\
#                         is_front, is_rear, merging_point_ahead, speed_profile_with_acc_limit, waypoints_with_fixed_dt, get_frenet,\
#                         distance, dist_betw_vehs_bump, get_parallel_translated_traj, plan_speed_qp, get_parallel_translation, angle_between, from_list_to_Point2D_vec, point_ind_to_distance
from utils_refactored import *
from lane import Lane


SOURCE_LANE_ID = 0 # ego-centric lane numbering
MERGING_LANE_ID = 100 # merging lane has fixed id

LANE_KEEPING = 0
LC_FREE_FLOW = 1
LC_CONSERVATIVE = 2
LC_INTERACTIVE = 3
SCENARIO_NAMES = ["LANE KEEPING", 
                  "FREE FLOW LANE CHANGING", 
                  "CONSERVATIVE LANE CHANGING", 
                  "INTERACTIVE LANE CHANGING"]

SPEED_IDM = 0
SPEED_CONST_ACC = 1
SPEED_QP = 2

SMOOTH_ACC = 0.73
SMOOTH_DCC = -0.43

InDeviation = 0
InReturn = 1

# TODO: add a safety layer for lane changing trajectory towards the front vehicle
# TODO: consider a front vehicle when generating a lane changing trajectory

class nnmpc(object):
    """docstring for nnmpc."""
    def __init__(self, config_dict):
        super(nnmpc, self).__init__()
        self.config_global = config_dict
        self.config_local = LocalConfig()
        
        # Global variables
        self.predictor = Predictor(os.path.join(sgan_source_path,'model',
                                                self.config_global["sgan_model"])) # NN predictor 
        
        # Local variables
        self.prev_change = False # previous change command
        self.slane_ind_ego = 0 # Int
        self.tlane_ind_ego = 0 # Int
        self.source_lane = 0 # Lane()
        self.target_lane = 0 # Lane()
        self.front_ind_source = 0 # Agent()
        self.front_ind_target = 0 # Agent()
        self.deviation_target = False
        self.ego = 0 # Vehicle()
        self.others = 0 # List()
        self.merging_scenario = False
        self.dist_to_merge = 0
        self.scenario = LANE_KEEPING
        self.others_to_consider = [] 
        self.others_to_interact = []
        self.lane_change_trigger_start_time = -1
        self.time_taken_lane_change = 0 
        self.agents_s_in_target = []
        self.is_curvy_road = False
        self.safety_bound_rear = 1 # meter
        self.safety_bound_front = 1 # meter
        self.safety_bound_lat = 1 # meter
        self.N_path = 51
        self.dev_from_left = 0 # deviation distance FROM Wleft
        self.dev_from_right = 0 # deviation distance FROM right
        self.speed_planning_option = SPEED_CONST_ACC # 
        self.is_in_merging_area = False
        self.spiral = self.config_global["spiral"]
        # self.gapInfo = GapInfo()
        self.prev_opt_v = False
        self.prev_stopping_a = False
        self.min_plan_horizon = 10
        self.gap_selection_buffer = []
        self.gap_selection_update_thred = 3
        self.prev_gap_selection = -1
        
        # ------------------------------
        # Deviation params -- TO BE MOVED TO A CONFIG FILE
        # ------------------------------
        # self.deviation_mode = InReturn
        # self.deviation_min_offset = 11 
        # self.deviation_first_piece_type = "interp" # {interp, straight}
        # self.deviation_first_piece_min_len = 3 # meters
        
        
        # self.deviation_inDev_p2_lon_shift = 4 # meters -- to be deprecated
        # self.deviation_inDev_p2_lat_shift = 1.2 # meter -- to be deprecated
        # self.deviation_inRet_p1_lon_shift = 5 # meters -- to be deprecated
        # self.deviation_inRet_p1_lat_shift = 1.5 # meter -- to be deprecated
                
        # self.deviation_call_buffer = [] 
        self.deviation_lane_change_min_len = self.config_global["deviation_lane_change_min_len"]
        self.deviation_lane_change_dur     = self.config_global["deviation_lane_change_dur"]
        self.deviation_p2_lon_shift        = self.config_global["deviation_p2_lon_shift"]
        self.deviation_p3_lon_shift        = self.config_global["deviation_p3_lon_shift"]
        self.deviation_p2_lat_shift        = self.config_global["deviation_p2_lat_shift"]
        self.deviation_p3_lat_shift        = self.config_global["deviation_p3_lat_shift"]
        
        self.deviation_inDev_min_d = False
        self.smooth_deviation = False
        self.prev_dev_lane = False
        self.dev_lane_used = False
        
        self.point_markers_pub = rospy.Publisher("/region/pw_points", MarkerArray, queue_size = 1)
        self.invalid_point_markers_pub = rospy.Publisher("/region/pw_invalid_points", MarkerArray, queue_size = 1)
        self.traj_markers_pub = rospy.Publisher("/region/pw_traj", MarkerArray, queue_size = 1)

    
    def update_local_vars(self, ego, others, source_lane, target_lane, config_local:LocalConfig, reinit):
        self.ego = ego
        self.others = others
        # print("ego s {:.2} s_target: {:.2} v {:.2} d {:.2} d_target {:.2}".format(ego.s, ego.s_target, ego.v, ego.d, ego.d_target))
        
        # Update the local configurations
        # self.config_local = copy.copy(config_local)
        self.config_local = config_local
        # self.config_local.ref_v = self.config_local.speed_limit # reset the reference speed
        self.slane_ind_ego = self.config_local.slane_ind_ego
        self.tlane_ind_ego = self.config_local.tlane_ind_ego
    
        # Update merging scenario
        self.merging_scenario = target_lane.id == MERGING_LANE_ID
        if self.merging_scenario:
            # If the ego vehicel is within the merging area, the curvature following will be inactive
            self.is_in_merging_area = ego.s >= config_local.merge_start_s and ego.s <= config_local.merge_end_s + 5
            if isinstance(ego.s, float) and isinstance(config_local.merge_start_s,float) and isinstance(config_local.merge_end_s,float):
                rospy.logdebug("ego.s: {:.2} merge start: {:.2} merge end: {:.2}".format(ego.s, config_local.merge_start_s, config_local.merge_end_s))
        
        # Update lane info
        self.source_lane = source_lane
        self.target_lane = target_lane
        if source_lane.id == target_lane.id:
            self.time_taken_lane_change = 0
            self.lane_change_trigger_start_time = -1
        else:
            if self.lane_change_trigger_start_time == -1: # this is the first call for lane changing
                self.lane_change_trigger_start_time = rospy.Time.now()
            self.time_taken_lane_change = (rospy.Time.now() - self.lane_change_trigger_start_time).to_sec()
        
        # Update localization & perception
        self.others_to_consider = {}
        self.others_to_interact = {}
        agents_id_in_target = []
        agents_s_in_target = [] 
        self.agents_id_in_target = [] # sorted
        self.agents_s_in_target = [] # sorted
        for id in list(others):
            veh = others[id]
            dist = dist_betw_agents(ego,veh)
            # print("veh {} lane num {} d {} target lane id {}".format(id, veh.lane_num, veh.d, target_lane.id))
        
            # Update agents in perception range 
            if dist <= ego.v * self.config_global["time_range_to_consider"] or dist <= 40\
                and abs(veh.d_target) <= target_lane.width/2: # only within the target lane
                self.others_to_consider[id] = veh

                # Update agents in interaction range
                if dist <= ego.v * self.config_global["time_range_to_interact"] or dist <= 15:
                    self.others_to_interact[id] = veh

                # Update gaps
                if veh.lane_num == target_lane.id or veh.d_target <= target_lane.width/2:
                    agents_id_in_target.append(veh.id)
                    agents_s_in_target.append(ego.s_target + (1-2*is_rear(ego,veh))*dist_betw_agents(ego,veh) - ego.width)

        # Update gaps
        sorted_inds = np.argsort(agents_s_in_target)
        for ind in sorted_inds:
            self.agents_id_in_target.append(agents_id_in_target[ind])
            self.agents_s_in_target.append(agents_s_in_target[ind])
            # print("veh {} s: {}".format(agents_id_in_target[ind], agents_s_in_target[ind]))
        # print("self.agents_id_in_target: {}".format(self.agents_id_in_target))
            
        # gaps = np.diff(self.agents_s_in_target)
        
        # Update deviations
        if self.config_global["deviation"]:
            self.dev_from_left = 0
            self.dev_from_right = 0
            for id in list(self.others_to_interact):
                veh = self.others_to_interact[id]
                # check if:
                # (1) close longitudinal position
                # (2) in the next lane
                if abs(self.ego.s_target-veh.s_target) <= self.ego.length or\
                    dist_betw_agents(self.ego, veh) < self.source_lane.width: # only if the veh gets close
                    # Terminate if not enough data
                    if len(self.ego.records) < 2:
                        break  
                    # Check if veh is on left or right
                    ref_point = np.array([self.ego.records[1].x,self.ego.records[1].y])
                    vec_ego = np.array([ego.x,ego.y])-ref_point
                    vec_veh = np.array([veh.x,veh.y])-ref_point
                    dot = vec_ego[0]*(-vec_veh[1]) + vec_ego[1]*vec_veh[0]
                    on_right = dot > 0 # left otherwise
                    if on_right:
                        self.dev_from_right = abs(veh.d)
                    else: # veh on left
                        self.dev_from_left = abs(veh.d)
        
        # Check if the scenario reinitialized
        if reinit == True:
            # print("reinitialized")
            self.prev_change = False
        

        # # Update closets index to lanes
        # self.slane_ind_ego = get_nearest_index(ego.pos, source_lane.list[self.slane_ind_ego:self.slane_ind_ego+min(100, len(source_lane.list)-1)]) \
        #                     + self.slane_ind_ego
        # if source_lane.id == target_lane.id:
        #     self.tlane_ind_ego = self.slane_ind_ego
        # else:
        #     self.tlane_ind_ego = get_nearest_index(ego.pos, target_lane.list[self.tlane_ind_ego:self.tlane_ind_ego+min(100, len(target_lane.list)-1)])\
        #                         + self.tlane_ind_ego
        rospy.logdebug("slane ind ego {} tlane ind ego {}".format(self.slane_ind_ego, self.tlane_ind_ego))
        # print("slane ind ego {} tlane ind ego {}".format(self.slane_ind_ego, self.tlane_ind_ego))

        # Check curvy road
        # TEMP:: Fix the indexing error, 20230503
        # if len(source_lane.list) > self.slane_ind_ego + 5: # prevent index error
        #     self.is_curvy_road = is_curvy_road(source_lane, self.slane_ind_ego)
        #     # if self.is_curvy_road:
        #     #     rospy.logdebug("This is curvy road. Follow the lane center.")
        # else:
        self.is_curvy_road = False
        
        # Update front agents -- can be optimized using perception data directly
        self.front_ind_source = self.config_local.front_ind_source
        self.front_ind_target = self.config_local.front_ind_target
        # self.front_ind_source = get_front_ind(ego, others, source_lane, 
        #                                             range_m = max(int(self.config_global["time_range_to_consider"] * ego.v),50),merging_scenario=self.merging_scenario)
        # if self.front_ind_source:
        #     self.config_local.ref_v = others[self.front_ind_source].v # ADDED 
        # # if source_lane.id == target_lane.id or abs(ego.d_target) <= target_lane.width/2:
        # #     self.front_ind_target = self.front_ind_source
        # # else:
        # self.front_ind_target = get_front_ind(ego, others, target_lane, 
        #                                             range_m = max(int(self.config_global["time_range_to_consider"] * ego.v),50), merging_scenario = self.merging_scenario)
        # if self.front_ind_target and abs(ego.d_target) <= target_lane.width/2:
        #     self.config_local.ref_v = others[self.front_ind_target].v # ADDED 
            

        rospy.logdebug("front ind source: {} front ind target: {} ref_v: {}".format(self.front_ind_source, self.front_ind_target, self.config_local.ref_v))
        # With restricted merging area 
        self.dist_to_merge = 0

        # Update scenario
        if source_lane.id == target_lane.id or\
            (self.merging_scenario and distance(self.ego.x,self.ego.y,
                     target_lane.list[min(self.tlane_ind_ego, len(target_lane.list)-1)].x,
                     target_lane.list[min(self.tlane_ind_ego, len(target_lane.list)-1)].y) >= target_lane.width*2 ):
            self.scenario = LANE_KEEPING
        elif len(self.others_to_consider) == 0:
            self.scenario = LC_FREE_FLOW
        else:
            if (self.config_local.with_dg and self.config_local.dg_negotiation_called) \
                or self.config_global["enforce_interactive_planning"]:
                    if self.config_global["enable_interactive_planning"] == True:
                        self.scenario = LC_INTERACTIVE
                    else:
                        rospy.logdebug("[ NNMPC] enable_interactive_planning = False. Run conservative.")
                        self.scenario = LC_CONSERVATIVE         
            elif self.config_local.with_dg and not self.config_local.dg_negotiation_called:
                self.scenario = LC_CONSERVATIVE
            else: # decision is needed
                # if (any(gaps >= 20) or\
                if self.time_taken_lane_change <= self.config_global["time_to_trigger_interaction"]: # if a gap exists
                    self.scenario = LC_CONSERVATIVE 
                else:
                    if self.config_global["enable_interactive_planning"] == True:
                        self.scenario = LC_INTERACTIVE
                    else:
                        rospy.logdebug("[ NNMPC] enable_interactive_planning = False. Run conservative.")
                        self.scenario = LC_CONSERVATIVE         
            
            # if not self.config_global["enforce_interactive_planning"] \
            #     and (any(gaps >= 20) \
            #     or self.time_taken_lane_change <= self.config_global["time_to_trigger_interaction"]): # if a gap exists
            #     self.scenario = LC_CONSERVATIVE
            # else:
            #     if self.config_local.with_dg and not self.config_local.dg_negotiation_called:
            #         self.scenario = LC_CONSERVATIVE
            #     else:
            #         self.scenario = LC_INTERACTIVE

        rospy.logdebug("[ NNMPC] {}".format(SCENARIO_NAMES[self.scenario]))


    def get_waypoints(self, ego, others, source_lane, target_lane, config_local, reinit = False):
        """
        Run NNMPC to generate waypoints (array of [x,y,speed]).

        args:
            ego: Vehicle()
            others: array of Agent(), including Vehicle(), Pedestrian()
            source_lane: Lane()
            target_lane: Lane()
            config_local: namedtuple("config", ['speed_limit','a_max','a_min', 'dead_end_s', 'ref_v', 'merge_end_point_x', 'merge_end_point_y'])

        returns:
            waypoints: 2d Array() of [x,y,speed]
        """
        if len(source_lane.list) == 0 or len(target_lane.list) == 0:
            rospy.logwarn("[ NNMPC] No lane information. Motion planning terminated.")
            return []

        print("\n======================================")

        # Update local variables
        t = time.time()
        self.update_local_vars(ego, others, source_lane, target_lane, config_local, reinit)
        elapsed = time.time() - t
        # rospy.logdebug("update local vars time: {:.3f}".format(elapsed))
        # rospy.logdebug("initial reference v: {}".format(self.config_local.ref_v))

        # Run motion planner
        t = time.time()
        traj = self.plan_traj()
        elapsed = time.time() - t
        # rospy.logdebug("plan_traj time: {:.3f}".format(elapsed))
        
        # Spatial interporations
        if self.config_global["space_interpolation"] and len(traj) > 2:
            # print("Interpolation: Before, len {} dist {}".format(len(traj),euclidean_dist(traj[0],traj[-1])))
            t = time.time()
            traj = self.traj_with_fixed_space_step(traj)
            elapsed = time.time() - t
            # rospy.logdebug("interpolation time: {:.3f}".format(elapsed))
            # print("Interpolation: After, len {} dist {}".format(len(traj),euclidean_dist(traj[0],traj[-1])))
            
        # Deviation if needed
        if self.config_global["deviation"]:
            # TODO: resume here to add deviation for conflicting point -- Nov 1, 2022
            dev = self.dev_from_right - self.dev_from_left # positive: deviate to left, negative: deviate to right
            # dev = 1 # TEMP: for testing
            if abs(dev) >= 0.1:
                if dev > 0:
                    dir = "LEFT"
                else:
                    dir = "RIGHT"
                rospy.logdebug("[ NNMPC] Deviation to {} needed: d={}".format(dir,dev))
                # NOTE: if deviation is needed, just shifting the goal point might be sufficient
                if len(traj) > 2:
                    t = time.time()
                    traj_shifted = get_parallel_translated_traj(traj, min(abs(dev),self.source_lane.width/2), dev > 0)
                    traj = [[point[0],point[1],v] for point,v in zip(traj_shifted, np.array(traj)[:,2])]
                    elapsed = time.time() - t
                    # rospy.logdebug("deviation traj time: {:.3f}".format(elapsed))
                else:
                    rospy.logwarn("[ NNMPC] Empty trajectory.")
                    
        # Extend the trajectory along with the lane, if the trajectory is shorter than x meter
        if len(traj)>=5 \
            and self.config_local.prevent_off_steering\
            and self.slane_ind_ego <= len(self.source_lane.list)-1-20\
            and not self.is_curvy_road\
            and distance(traj[0][0], traj[0][1], traj[-1][0], traj[-1][1]) <= 20: # meter
            try:
                rospy.logdebug("Extends the trajectory")
                traj = self.extend_traj(traj)
            except:
                rospy.logwarn("Trajectory extension failed. Check the reference path.")
                
        # Address empty traj
        if len(traj) <= 2 and self.config_local.prevent_off_steering:
            rospy.logwarn("Fixing empty traj to be aligned with the lane center.")
            traj = self.traj_aligned_lane(plan_speed=False)
            
        # check the reference speed for debugging 
        # rospy.logdebug("updated reference v: {}".format(self.config_local.ref_v))
        # rospy.logdebug("returned traj length: {}".format(len(traj)))
        return traj
   
   
    def plan_traj(self):
        """Get planned trajectory based on the scenario.

        Returns:
            2d-array: in the form of [[x,y,v],...]
        """
        if self.config_global["collision_warning"] and self.collision_warning():
            traj = self.max_brake_traj()
        elif self.scenario == LANE_KEEPING:
            traj = self.lane_follow_traj()
        elif self.scenario == LC_FREE_FLOW:
            traj = self.free_lane_change_traj()
        elif self.scenario == LC_CONSERVATIVE:
            traj = self.conservative_lane_change_traj()
        elif self.scenario == LC_INTERACTIVE:
            traj = self.interactive_lane_change_traj()
            
        return traj
    
    
    def collision_warning(self):
        ego = self.ego
        for id in list(self.others):
            veh = self.others[id]
            dist = dist_betw_vehs_bump(self.ego, veh)
            vec_ego = [np.cos(ego.theta),np.sin(ego.theta)]
            vec_ego_to_veh = [veh.x-ego.x, veh.y-ego.y]
            angle = angle_between(vec_ego, vec_ego_to_veh)
            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)
            lon_dir = dist*abs(cos_angle)
            lat_dir = dist*abs(sin_angle)
            same_dir = abs(ego.theta-veh.theta) <= 0.11
            # print("veh {} angle diff: {} same dir: {}".format(id, abs(ego.theta-veh.theta), same_dir))
            # print("veh: {} dist {} angle {} cos {} sin {}".format(id, dist, angle*180/np.pi, cos_angle, sin_angle))
            if dist <= 0.5 or \
                (not same_dir and dist <= 1): # This is the red flag. Must stop. 
                rospy.logwarn("[ NNMPC] Collision warning: veh {} dist {} lon_dir {} lat_dir {}".format(id, dist, lon_dir, lat_dir))
                return True 
        return False 
    
    
    
    def max_brake_traj(self):
        self.config_local.ref_v = 0.0
        self.config_local.a_min = -3.0
        traj = self.traj_aligned_lane(plan_speed=True)    
        return traj


    def lane_follow_traj(self):
        """Generate lane following trajectory

        Returns:
            profile: lane follow trajectory
        """
        # If the road is curvy (e.g., on ramp), spiral curve will lead the ego off-road
        if self.is_curvy_road or\
            (self.merging_scenario and 
             self.ego.s <= self.config_local.merge_start_s and 
             not self.is_in_merging_area):
            rospy.logdebug("This is curvy road. Follow the lane center.")
            traj = self.traj_aligned_lane()
        else:
            # Otherwise, spiral curve opt
            traj = self.lane_change_traj(self.target_lane, self.front_ind_target, self.tlane_ind_ego)
        
        # print("a_max: ", self.config_local.a_max)
        # print("ref v: ", self.config_local.ref_v)
        # print("traj len: ",len(traj))
        return traj


    def traj_aligned_lane(self, lane = False, plan_speed= True, point_offset=3):
        """Generate trajectory that follows a source lane

        Returns:
            _type_: _description_
        """
        N_path = self.N_path
        if lane is False:
            lane = self.source_lane
            i_nearest = self.slane_ind_ego
        else:
            i_nearest = closest_point_ind(lane.list, self.ego.x, self.ego.y)
        
        if lane.id < 100 and lane.id >10: # this is the opposite direction
            if i_nearest - point_offset <= 0:
                point_offset -= i_nearest
            i_nearest -= point_offset
            # print("i nearest: {} point offset {}".format(i_nearest, point_offset))
            remaining_len = len(lane.list[i_nearest::-1])
            if remaining_len < N_path:
                path = np.array(lane.list[i_nearest::-1])
            else:
                path = np.array(lane.list[i_nearest:i_nearest-N_path:-1])
        else:
            if i_nearest + point_offset >= len(lane.list)-1:
                point_offset = len(lane.list)-1 -point_offset
            i_nearest += point_offset
            path = np.array(lane.list[i_nearest:min(i_nearest+N_path,len(lane.list)-1)])
            
        
        if plan_speed:
            # Plan speed
            # print("in traj_aligned_lane ref v: {}".format(self.config_local.ref_v))
            
            if self.front_ind_source:
                traj = follow_profile(path, self.ego.v, self.config_local.ref_v, self.config_local.a_max, self.config_local.a_min,
                                    [path[-1][0], path[-1][1], max(self.others[self.front_ind_source].v,0)])
            # Just max out accel/decel if no traffic
            else: 
                # # TODO: check if any vehicle is too close on the path
                nominal = True
                if self.front_ind_target:
                    dist = dist_betw_vehs_bump(self.ego, self.others[self.front_ind_target])
                    # print("in traj_aligned_lane, ego d: {} dist to front ind target {}".format(self.ego.d, dist))
                    if abs(self.ego.d_target) <= self.target_lane.width*2/3:
                        if dist <= 1:
                            rospy.logwarn("[ NNMPC] too close to the front car {} in the target lane".format(self.front_ind_target))
                            nominal = False
                if nominal:
                    traj = nominal_profile(path, self.ego.v, self.config_local.ref_v, self.config_local.a_max, self.config_local.a_min)
                else:
                    traj = follow_profile(path, self.ego.v, self.config_local.ref_v, self.config_local.a_max, self.config_local.a_min,
                                    [path[-1][0], path[-1][1], max(self.others[self.front_ind_target].v,0)])
        else:
            # print("len(path): {}, path[0]: {}".format(len(path), path[0]))
            traj = [[point[0], point[1], 0] for point in path]
        
        return traj


    def lane_change_to_target(self):
        return self.lane_change_traj(self.target_lane, self.front_ind_target, self.tlane_ind_ego)


    def lane_change_to_source(self):
        return self.lane_change_traj(self.source_lane, self.front_ind_source, self.slane_ind_ego)


    def free_lane_change_traj(self):
        return self.lane_change_traj(self.target_lane, False, self.tlane_ind_ego)
    
    
    def lane_change_traj(self, lane, front_ind, lane_ind_ego):
        """Generate lane changing trajectory with spiral curves

        Args:
            lane (Lane): target lane instance
            front_ind (int): front agent id within the target lane
            lane_ind_ego (int): closese index of lane path to the ego vehicle

        Returns:
            2d-array: motion planning trajectory
        """
        front_dist = max(int(self.config_global["lane_change_duration"] * self.ego.v),self.min_plan_horizon)
        
        # Check the frontal distance
        # if not self.merging_scenario:
        if self.front_ind_source:
            front_dist_source = self.config_local.front_dist_source
            front_dist = min(front_dist_source-self.ego.length/2, front_dist)
            rospy.logdebug(f'Front ind source: {self.front_ind_source}, front dist: {front_dist_source}, lane changing dist: {front_dist}')
        
        if not self.config_local.prevent_off_steering:
            rospy.logdebug(f'Front ind target: {self.front_ind_target}')
            if self.front_ind_target:
                if self.front_ind_source != self.front_ind_target:
                    front_dist_target = dist_betw_agents(self.ego, self.others[self.front_ind_target])
                    front_dist = max(min(front_dist_target, front_dist),20)
                else:
                    front_dist_target = front_dist

        # Update the target index
        lane_ind_target = max(int(front_dist),10)
        lane_ind_target = min(lane_ind_target, 60)
            
        if self.spiral:
            if abs(self.ego.d) <= self.target_lane.width/3:
                traj= self.traj_aligned_lane(lane=self.target_lane)
            else:
                traj= self.spiral_traj(lane, front_ind, lane_ind_ego, lane_ind_target = lane_ind_target)
        else:
            traj= self.traj_aligned_lane()
        
        # print("in lane change traj, lane id: {} lane ind ego {} lane ind target {}".format(lane.id, lane_ind_ego, lane_ind_target))
        # rospy.logdebug("lane change traj len {}".format(len(traj)))
        
        return traj
    
    
    def conservative_lane_change_traj(self):
        """Generate a **defensive** lane changing trajectory 

        Args:
            lane (_type_): _description_
            front_ind (_type_): _description_
            lane_ind_ego (_type_): _description_
            lane_ind_target (int, optional): _description_. Defaults to 10.

        Returns:
            2d-array: motion planning trajectory
        """
        #! NOTE: Depending on the specific lane changing scenario (e.g., merging or nominal lane changing),
        #!      the *waiting* strategy may differ. In merging (within merging area) scenario, the ego vehicle
        #!      should slow down. In the nominal lane changing scenario, we continue on the source lane and
        #!      try it in the next time step if needed.
        
        # Lane change decision
        # if self.merging_scenario:
        t = time.time()
        lane_change_ok, veh_headway = self.should_change_lane_now()
        elapsed = time.time() - t
        rospy.logdebug("should lane change now elapsed in: {:.3f} sec".format(elapsed))
        
        # ----------------------------------------------------------------------------------
        # Generate trajectory
        #
        # 1) Lane changing is fine right now. Generate lane changing trajectory
        if lane_change_ok:
            traj = self.lane_change_to_target()
        
        # 2) We should wait in the source lane. 
        else:
            # -- If it is a merging scenario, we should slow down
            if self.merging_scenario and self.is_in_merging_area:
                rospy.logdebug("[ NNMPC] No merging now. Cruising in the merging area.")
                # Option 1: Naive, slow down gradually until the hard stop point
                traj = self.traj_in_merging_area(veh_headway)
                
                # Option 2: Optimal trajectory, targeting the optimal gap
                # self.gapInfo = self.get_gap_info() # gets the gap info
                # # TODO: check the target vehicle
            
            # -- Otherwise, we just continue on the source lane and see a chance in the next time step
            else:
                find_gap = False
                if find_gap:
                    traj = self.traj_for_best_gap()
                else:
                    traj = self.traj_when_lane_change_fails()
        
        return traj


    def traj_for_best_gap(self):        
        # Get the best gap
        ref_v, goal_s, _ = self.get_the_best_gap()
        
        # Get the path for source lane following
        # print("refv: {} goal s: {}".format(ref_v, goal_s))
        if goal_s >= 1: # target a gap ahead
            self.config_local.ref_v = (ref_v <= 5)*ref_v*1.8 +\
                                      (ref_v > 5 and ref_v <=10)*ref_v*1.5 +\
                                      (ref_v > 10)*ref_v*1.3 # speed up
        else:
            self.config_local.ref_v = (ref_v <= 5)*ref_v*0.5 +\
                                      (ref_v > 5 and ref_v <=10)*ref_v*0.7 +\
                                      (ref_v > 10)*ref_v*0.8 # speed down
                                      
        self.config_local.ref_v = min(self.config_local.ref_v, self.config_local.speed_limit)
        
        # Use comfortable acc dcc limit
        self.config_local.a_max = SMOOTH_ACC # TEMP
        self.config_local.a_min = SMOOTH_DCC # TEMP
        # print("updated ref v in config: {}".format(self.config_local.ref_v))
            
        # Generate the trajectory 
        traj = self.traj_aligned_lane(plan_speed=True)
        return traj
    
    
    def get_the_best_gap(self, min_range = 40, target_point_shift = 4):
        """Computes the best gap in the target lane within a range 

        Args:
            min_range (int, optional): _description_. Defaults to 40.
        """
        def get_cost():
            """Inner function for gap cost

            Returns:
                _type_: _description_
            """
            cost = (abs(del_v) + # 
                (del_d < 0) * (-20) * min(del_d,-1) + (del_d >= 0) *0.5* del_d   # better not to wait
                + (d_gap < self.ego.length+4) * 10 * d_gap - (d_gap >= self.ego.length+4) * 6 * min(d_gap,15)   # larger gap is definitely good
                    - 4*v_gap # speed difference, 
                    + 20*(veh.v<1))
            return cost
        
        # -------------
        # Get the vehicles on the target lane 
        # -------------
        ids = self.agents_id_in_target        
        ss = self.agents_s_in_target
        costs = [] # storing cost for each gap
        vs = [] # storing speed reference for each gap
        d_gaps = []
        best_gap_ind = -1
        # print("ids in target: {}".format(ids))
        # print("ids in others: {}".format(self.others.keys()))
        
        if len(self.target_lane.merge_end_point) != 0:
            merge_scenario = self.target_lane.merge_end_point[0] != 0 and self.target_lane.merge_end_point[1] != 0
            if merge_scenario:
                dist_to_merge_end = distance(self.ego.x, self.ego.y, self.target_lane.merge_end_point[0], self.target_lane.merge_end_point[1])
                # print("dist to merge end:", dist_to_merge_end)
        else:
            rospy.logwarn("[ NNMPC] Merge point not received.")
            merge_scenario = False

        for i, id in enumerate(ids):
            # target vehicle
            veh = self.others[id] 
            s_target = ss[i]-target_point_shift
            vs.append(veh.v)

            # -------------
            # Compute del_v, del_d, d_gap, v_gap
            #   del_v: speed difference to the target vehicle from the ego
            #   del_d: distance to the target vehicle (+ for ahead, - for behind)
            #   d_gap: gap distance with the rear vehicle (+ is good)
            #   v_gap: speed difference with the rear vehicle (+ is good, gap being bigger, - is not good, gap being narrower)
            # -------------
            # Compute speed difference
            del_v = veh.v - self.ego.v
            
            # Compute distance to the target vehicle
            # del_d = veh.s_target - self.ego.s_target # + for ahead, - for behind
            # print("veh {} s_target {} ego_s {}".format(id, s_target, self.ego.s_target))
            del_d = s_target - self.ego.s_target # + for ahead, - for behind

            # Compute the d_gap and v_gap 
            if i == 0: # this is the last (most behind) vehicle 
                rear_end = self.ego.s_target - max(self.ego.v * self.config_global["time_range_to_consider"], min_range)
                d_gap = s_target - rear_end/2
                v_gap = 0 
            else: # this is the vehicle in between 
                d_gap = s_target - ss[i-1]
                v_gap = veh.v - self.others[ids[i-1]].v
            
            d_gaps.append(d_gap)
            # -------------
            # Compute the cost of the gap 
            # -------------
            cost = get_cost()
            if merge_scenario:
                if del_d >= dist_to_merge_end:
                    cost += 100 # this gap is not feasible.
            # print("veh id: {} del_v: {:.2f} del_d: {:.2f} d_gap: {:.2f} v_gap: {:.2f} cost: {:.2f} ".format(veh.id, del_v, del_d, d_gap, v_gap, cost))
            costs.append(cost)
            
        # Also consider the gap ahead the first vehicle
        if len(ids) > 0:
            veh = self.others[ids[-1]]
            front_end = self.ego.s_target + max(self.ego.v * self.config_global["time_range_to_consider"], min_range)
            del_v = veh.v - self.ego.v 
            del_d = (ss[-1] + 15) - self.ego.s_target
            d_gap = front_end - ss[-1]
            v_gap = 0
            ss.append(ss[-1]+max(self.ego.v*5,20))
            vs.append(veh.v)
            d_gaps.append(d_gap)
            cost = get_cost()
            if merge_scenario:
                if del_d >= dist_to_merge_end:
                    cost += 100 # this gap is not feasible.
            
            # print("first gap, del_v: {:.2f} del_d: {:.2f} d_gap: {:.2f} v_gap: {:.2f} cost: {:.2f} ".format(del_v, del_d, d_gap, v_gap, cost))
            costs.append(cost)
        
            # print("number of gaps: {}".format(len(costs)))
            # print("costs: {} ".format(costs))
            
            # Get the best gap
            best_gap_ind = np.argmin(costs)
            # print("best gap index: {} from behind".format(best_gap_ind))
            if best_gap_ind == len(costs)-1:
                print("choose gap in front of car {}".format(ids[-1]))
            else:
                print("choose gap behind car {}".format(ids[best_gap_ind]))
        else:
            rospy.logwarn("no gaps. Check the perception info. len(others): {}".format(len(self.others)))
            return self.ego.v, 0, 0
            
        # ----------------------------------
        # Update the gap choice with tolerence
        # ----------------------------------
        self.gap_selection_buffer.append(best_gap_ind)
        if len(self.gap_selection_buffer) > self.gap_selection_update_thred:
            self.gap_selection_buffer.pop(0)
        
        # Check the same choice has been made over the buffer 
        if all(np.array(self.gap_selection_buffer) == best_gap_ind):
            print("update the gap")
            self.prev_gap_selection = best_gap_ind
            return_ind = best_gap_ind
        else:
            print("in buffer. Keep the previous choice.")
            return_ind = self.prev_gap_selection
            
        # TODO:: Currently assume the number of gaps is consistent, which might not be the case.
        if return_ind < len(vs):
            return vs[return_ind], ss[return_ind]-self.ego.s_target, d_gaps[return_ind] # return v_ref and goal_s (it could be negative)
        else:
            rospy.logwarn("Gap length changed. Use the new gap ind.")
            return vs[best_gap_ind], ss[best_gap_ind]-self.ego.s_target - target_point_shift, d_gaps[best_gap_ind] # return v_ref and goal_s (it could be negative)

        
    def interactive_lane_change_traj(self):
        """
        Return interactive lane changing trajectory. Safety evaluation will involve interative SGAN predictions

        Returns:
            2d-array: motion planning trajectory
        """
        # Define function variables for convenience
        ego = self.ego
        others = self.others
        target_lane = self.target_lane
        N_receding = int(round(self.config_global["T"]/self.config_global["dt"])) # NNMPC evaluation horizon

        # Filter out target vehicles for interaction
        target_agent_inds = self.find_target_agents_to_interact(ego, self.others_to_interact, target_lane, range_m = 20)

        # 1) No interaction is needed 
        if len(target_agent_inds) == 0:
            # Decision making is not needed. Do nominal lane changing
            print("No interaction is needed.")
            traj = self.conservative_lane_change_traj()
        
        # 2) Interaction is needed
        else:
            # Decision making is needed. Run NNMPC
            if self.front_ind_target:
                self.config_local.ref_v = others[self.front_ind_target].v+2
            traj_cands , traj_cands_names = self.generate_traj_cands(ego, others, target_lane, N_receding)        
            print("traj cands: {}".format(traj_cands_names))
            best_cand_ind, is_sol_found = self.eval_cands_single_thread(target_agent_inds, traj_cands, traj_cands_names)

            if is_sol_found: # if solution is found
                traj = traj_cands[best_cand_ind] 
            else:
                # Returning to the source lane
                traj = self.traj_when_lane_change_fails()
        return traj

    
    def traj_in_merging_area(self, veh_headway):
        """Generate trajectory within the merging area

        Args:
            veh_headway (_type_): _description_

        Returns:
            _type_: _description_
        """
        i_nearest = self.slane_ind_ego
        N_path = self.N_path
        i_nearest += self.config_global["stay_lane_lookahead"]
        xy_profile = np.array(self.source_lane.list[i_nearest:i_nearest+N_path])
        
        # Plan speed
        v_profile = [self.ego.v for i in range(N_path)]
        d2merge = max(self.config_local.dead_end_s - self.ego.s, 0)
        
        # Compute the minimum distance for full brake
        d2_stop = -1/2 * self.ego.v * (self.ego.v/self.config_local.a_min) + 2 # adding a buffer 
        rospy.logdebug("d2stop: {} d2merge: {}".format(d2_stop, d2merge))
        if d2merge <= d2_stop: # just immediate stop
            rospy.logwarn("[ NNMPC] emmergency stop due to the infeasible merging")
            v_profile = [self.ego.v + i*self.config_local.a_min if self.ego.v + i*self.config_local.a_min > 0 else 0 for i in range(len(xy_profile))]
            traj = [[xy_profile[i][0], xy_profile[i][1], v_profile[i]] for i in range(len(xy_profile))] # stopping behavior with full acceleration
        else:
            d_buffer = 6 # distance buffer toward the front car
            ref_T = max(veh_headway+self.config_global["headway_bound"],0) # time to merge point of the reference vehicle
            v_f = 2*max(d2merge - d_buffer,0)/ref_T - self.ego.v # targeted speed 
            a = 2*(v_f-self.ego.v)/ref_T # desired acceleration
            if a > 0: 
                a = -0.3 # limit acc to be non positive
            if a < -1.3: a = -1.3 # comfortable limit, but this may not guarantee safety
            if self.prev_stopping_a != False:
                a = (a*2 + self.prev_stopping_a)/3
            self.prev_stopping_a = a
            v_profile = speed_profile_with_acc_limit(self, v_profile, dt=0.1, alim = a)
            rospy.logdebug("time-to-merge: {} target v: {} gradually slow down with selected a: {}".format(veh_headway, v_f, a))
            
            # Construct traj
            traj = [[xy_profile[i][0], xy_profile[i][1], v_profile[i]] for i in range(len(xy_profile))]
        rospy.logdebug("traj in merging area, traj len: {} \
        #     total distance: {}".format(len(traj),distance(xy_profile[0,0],xy_profile[0,1],xy_profile[-1,0],xy_profile[-1,1])))
        
        return traj
        
    
    def traj_when_lane_change_fails(self):
        """Generate trajectory when NNMPC finds no solution

        Returns:
            2d-array: motion planning trajectory
        """
        others = self.others_to_consider
        # -----------------------------------------------------------------------------------------------------
        # Merging or dead end scenario
        # 1) Keep enough space to the front agent for steering
        # 2) Keep enough space to the emergency agent
        # 3) Return to the source otherwise
        emergency_ahead = is_emergency_ahead(self.ego, others, self.source_lane)
        if (self.merging_scenario and self.is_in_merging_area) or emergency_ahead:
            rospy.logwarn("NNMPC: Dead end ahead.")
            # 1) Keep enough space to the front vehicle
            if self.front_ind_source:
                dist_to_front = dist_betw_agents(self.ego, others[self.front_ind_source])
                if dist_to_front <= self.config_global["dead_zone"]:
                    traj = []
                else:
                    traj = self.lane_change_to_source()
            elif self.front_ind_target: # for one point merging
                dist_to_front = dist_betw_agents(self.ego, others[self.front_ind_target])
                if dist_to_front <= self.config_global["dead_zone"]:
                    traj = []
                else:
                    traj = self.lane_change_to_source()
            
            # 2) Keep enough space to the emergency vehicle/agent
            elif (emergency_ahead and 
                dist_betw_agents(self.ego, others[emergency_ahead]) <= self.config_global["dead_zone"]):
                rospy.logdebug("Full brake for the stalling vehicle")
                traj = []
            
            # 3) Return to the source lane
            else:
                rospy.logdebug("Returning to source")
                traj = self.lane_change_to_source()
        
        # -----------------------------------------------------------------------------------------------------
        # Same direction scenario (e.g., highway)
        # 1) Return to the source lane if ego vehicle is cruising 
        # 2) Continue on to the target lane if ego vehicle is already intervening the target lane
        # 3) Fully brake otherwise
        else:
            if self.ego.v >= 5 and abs(self.ego.d) > self.target_lane.width/2+0.3:
                rospy.logdebug("merging scenario: {} in_merging_are: {}, emergency ahead: {}".format(
                    self.merging_scenario, self.is_in_merging_area, emergency_ahead))
                warn("Same direction. Returning to source")
                traj = self.lane_change_to_source()
            elif self.ego.v >= 5 and abs(self.ego.d) <= self.target_lane.width/2+0.3:
                rospy.logdebug("Already in target. Push forward to the target lane")
                traj = self.lane_change_to_target()
            else:
                return_to_source = False
                if return_to_source:
                    rospy.logdebug("No lane change solution, cruise in the source lane")
                    traj = self.lane_change_to_source()
                else:
                    rospy.logdebug("No lane change solution, stop and wait")
                    self.config_local.ref_v = 0.0
                    traj = self.traj_aligned_lane(plan_speed= True) # stopping brake.
        
        return traj


    def should_change_lane_now(self):
        """Check if **defensive** lane changing is available now
        
        Returns:
            Boolean: TRUE if lane changing now is fine
        """
        ego = self.ego
        others = self.others_to_consider
        target_lane = self.target_lane
        lane_change_ok = True
        front_min = float('inf')
        rear_min = float('inf')
        veh_headway = float('inf')
        merging_scenario = self.target_lane.id == MERGING_LANE_ID
        # merging_scenario = self.config_local.merging_scenario
        
        if merging_scenario:
            d2merge = self.config_local.merge_end_s - ego.s
            # d2merge = distance(self.config_local.merge_end_point_x, self.config_local.merge_end_point_y, ego.x, ego.y)
        else:
            d2merge = 30
        
        # -------------------------------------------------------------------------------------------
        # Case 1: free flow -- no vehicles on the merging lane
        # 
        if len(others) == 0 or abs(ego.d) <= target_lane.width/2 or d2merge < 0: # No decision needed.
            rospy.logdebug(f'ego.d: {ego.d}, d2merge: {d2merge}')
            if len(others) == 0:
                rospy.logdebug("[ NNMPC] in should_change_lane_now, no vehicle to consider")  
            if abs(ego.d) <= target_lane.width/2:
                rospy.logdebug("[ NNMPC] already in the target lane")
            if d2merge <0:
                rospy.logdebug("[ NNMPC] merging point passed")
            lane_change_ok = True
            if self.front_ind_target and self.front_ind_target in others:
                front_veh = others[self.front_ind_target]
                front_min = front_veh.s - ego.s 
                rospy.logdebug("front min: {}".format(front_min))
                if front_min <= 30:
                    self.config_local.ref_v = min(front_veh.v, self.config_local.speed_limit)
                    # print("ref_v updated: ", self.config_local.ref_v)
        
        # -------------------------------------------------------------------------------------------
        # Case 2: oncoming vehicles exist
        # 
        else: # Decision needed
            # print("in should_change_lane_now, need decision")
            # Get ego headway
            ego_headway = d2merge/max(min(ego.v+self.config_local.a_max/2, 
                                          (ego.v+self.config_local.speed_limit)/2),1)
            
            # Sort vehicles from behind
            others_ids = [id for id in list(others) if self.others[id].lane_num == target_lane.id]
            # print("others_ids: {}".format(others_ids))
            others_s = [self.others[id].s_target for id in others_ids]
            inds_order = np.argsort(others_s, kind="mergesort")
            for ind in inds_order:
                id = others_ids[ind]
                veh = self.others[id]
                
                # VEH IS FRONT
                if veh.s_target >= ego.s_target: 
                    dist = dist_betw_vehs_bump(ego, veh)
                    front_headway = (dist + max(veh.v-3,0)+max(veh.v-6,0)+max(veh.v-9,0))/max(1, ego.v)
                    if dist < front_min and dist <= 40:
                        front_min = dist
                        self.config_local.ref_v = min(veh.v, self.config_local.speed_limit)
                    rospy.logdebug("ego hw: {:.2f}, front veh {} v: {:.2f} dist: {:.2f} front hw: {:.2f}".format(ego_headway, id, veh.v, dist, front_headway))
                    
                    # We have enough distance at front
                    if  dist >= self.config_global["distance_front"] \
                        and front_headway >= self.config_global["front_headway_bound"]:
                        lane_change_ok = True
                    
                    # We are in the process of lane changing. Continue lane changing to prevent swerving.
                    elif self.prev_change == True and dist >= self.config_global["distance_front"]/2\
                        and front_headway >= 1/2*self.config_global["front_headway_bound"]: # prevent swerving
                        lane_change_ok = True
                        rospy.logdebug("prevent swerving")
                    
                    # Otherwise, we should slow down
                    else:
                        lane_change_ok = False
                        slow_for_front = True
                        # veh_headway = front_headway
                        veh_d2merge = distance(veh.x,veh.y,self.config_local.merge_end_point_x, 
                                            self.config_local.merge_end_point_y)
                        veh_headway = veh_d2merge/max(veh.v,1)
                        break
                
                # VEH IS REAR
                else:
                    # dist = dist_betw_vehs_bump(ego, veh)
                    dist = ego.s_target - veh.s_target
                    if merging_scenario:
                        veh_d2merge = distance(veh.x,veh.y,self.config_local.merge_end_point_x, 
                                            self.config_local.merge_end_point_y)
                    else:
                        veh_d2merge = 30+dist
                    veh_headway = veh_d2merge/max(veh.v, 1)

                    if dist < rear_min:
                        rear_min = dist
                    rospy.logdebug("ego hw: {:.2f}, rear veh {} v: {:.2f} dist: {:.2f} hw: {:.2f}".format(ego_headway, id, veh.v, dist, veh_headway))
                    
                    # Ego will arrive at the merging point sooner than the veh. Lane changing is fine.
                    if (ego_headway + self.config_global["headway_bound"] < veh_headway and dist >= 3)\
                        or dist >= 40:
                        lane_change_ok = True
                        
                    # Otherwise, 
                    else:
                        # Ego is in the process of lane changing. Continue lane changing to prevent swerving.
                        if self.prev_change == True and ego_headway + self.config_global["headway_bound"]/2 <= veh_headway:
                            rospy.logdebug("prevent swerving back to the source lane")
                            lane_change_ok = True
                        
                        # Otherwise, we should wait in the source lane
                        else:
                            lane_change_ok = False
                            break
        
        # Store the previous decision on lane changing
        self.prev_change = lane_change_ok
        rospy.logdebug("lane change allowed: {}".format(lane_change_ok))
        
        return lane_change_ok, veh_headway
            

    def spiral_traj(self, lane, front_ind, lane_ind_ego, lane_ind_target = 10, ref_v = False):
        """Generate a motion planning trajectory based on a spiral curve optimization

        Args:
            lane (Lane): target lane
            front_ind (Int): front agent id in the target lane
            lane_ind_ego (Int): index of the lane path closest to the ego vehicle
            lane_ind_target (Int, optional): look ahead for the target point in the lane path. Defaults to 10.

        Returns:
            2d-Array: in the form of [[x0,y0,v0],[x1,y1,v1],...]
        """
        # Define function variables
        ego = self.ego
        others = self.others
        if ref_v == False:
            ref_v = self.config_local.ref_v
            
        if distance(ego.x,ego.y,lane.list[0].x,lane.list[0].y) >= 2000: # handle initialization error
            ego.x,ego.y = lane.list[0].x,lane.list[0].y 

        # Compute speed to keep distance to the front agent
        # if str(front_ind) != 'False':
        if front_ind is not False:
            ref_v = others[front_ind].v
            # traj = follow_profile(path, ego.v, self.config_local.ref_v, self.config_local.a_max,
            #             [others[front_ind].x, others[front_ind].y, max(others[front_ind].v,0)])
            if self.speed_planning_option == SPEED_IDM:
                # Option 1: using IDM
                traj = follow_profile_idm(path, ego, others[front_ind])
            elif self.speed_planning_option == SPEED_QP:
                traj = follow_profile_qp(path, ego, others[front_ind], 
                                         a_max=self.config_local.a_max, 
                                         v_max=ref_v)
            else: #self.speed_planning_option == SPEED_CONST_ACC:
                # Option 2: using lane keep opt
                
                # -- TEMP: for HPCC testing
                if front_ind == self.config_local.front_ind_source:
                    dist_to_front = self.config_local.front_dist_source
                elif front_ind == self.config_local.front_ind_target:
                    dist_to_front = self.config_local.front_dist_target
                else:
                    dist_to_front = dist_betw_agents(ego, others[front_ind])
                    
                # rospy.logdebug("In sprial traj, front ind: {} dist to front: {}".format(front_ind, dist_to_front))
                # if self.config_local.prevent_off_steering and\
                #      dist_to_front <= max(ego.v*self.config_global["lane_change_duration"], 60): # if the distance is too short
                if True:
                    
                    # Compute the optimal speed reference using QP
                    # opt_ref_v = follow_profile_qp(path, ego, others[front_ind], a_max=self.config_local.a_max, v_max=ref_v)[2][-1]
                    t = time.time()
                    v,d,_,_ = plan_speed_qp(ego.v,others[front_ind].v, dist_to_front, 
                                            a_max = self.config_local.a_max,
                                            d_min = max(2*ego.v, 2),
                                            v_max = ref_v+1.5, 
                                            soft=True)
                    elapsed = time.time() - t
                    rospy.logdebug("QP time: {:.4}".format(elapsed))
                    
                    # Choose the optimal speed considering the control delay
                    if ego.v <= 5:
                        opt_ref_v = v[2]
                    else:
                        opt_ref_v = v[1] # consider the control delay
                    rospy.logdebug("ego v: {} computed opt ref v: {} ".format(ego.v, opt_ref_v))
                    if self.prev_opt_v != False:
                        opt_ref_v = (opt_ref_v*2/3 + self.prev_opt_v/3)
                    self.prev_opt_v = opt_ref_v
                    rospy.logdebug("[ NNMPC] Front distance {} v: {:.2f} smoothed optimal ref v: {:.3f}".format(dist_to_front, ego.v, opt_ref_v))
                    
                    goal_dist = max(ego.v*self.config_global["lane_change_duration"], self.min_plan_horizon)
                    # goal_dist = 65
                    traj = self.spiral_traj(lane, False, lane_ind_ego, lane_ind_target, ref_v = opt_ref_v)
                else:
                    traj = follow_profile(path, ego.v, ref_v, self.config_local.a_max,
                                [path[-1][0], path[-1][1], max(others[front_ind].v,0)])
                    
        # Just max out accel/decel if no traffic
        else:
            # Set a goal state
            goal_ind = min(lane_ind_ego + min(3,int(np.ceil(ego.v*0.1))) + int(lane_ind_target), len(lane.list)-1)
            goal_state = lane.list[goal_ind]
            print("lane ind ego: {} lane ind target {} dist to goal state {}".format(lane_ind_ego, lane_ind_target, euclidean_dist(ego.pos, goal_state)))
            
            # For sharper lane change
            if self.config_local.sharp_turn:
                t = time.time()
                p1 = [lane.list[goal_ind-1].x, lane.list[goal_ind-1].y]
                p2 = [goal_state.x, goal_state.y]
                shifted_p_rear, shifted_p_front = get_parallel_translation(p1,p2,min(lane.width/4,abs(self.ego.d)),self.ego.d<0)
                goal_state = Point2D(shifted_p_front[0],shifted_p_front[1])
                elapsed = time.time() - t
                # rospy.logdebug("parallel translation elapsed in: {:.3f} sec".format(elapsed))
                
            
            # rospy.logdebug(f'lane_ind_ego: {lane_ind_ego}, lane_ind_target: {lane_ind_target} goal ind:{goal_ind}')
            # rospy.logdebug(f'ego state: {[ego.x,ego.y,ego.theta]} goal state {[goal_state]} distance {np.sqrt((ego.x-goal_state.x)**2 + (ego.y-goal_state.y)**2)}')
            t = time.time()
            goal_state_local = get_local_goal_state(goal_state, [ego.x,ego.y,ego.theta])
            elapsed = time.time() - t
            # rospy.logdebug("get local goal state elapsed in: {:.3f} sec".format(elapsed))
            
            # Generate path with spiral curve optimization
            t = time.time()
            path = plan_path(goal_state_local)
            elapsed = time.time() - t
            # rospy.logdebug("plan_path elapsed in: {:.3f} sec".format(elapsed))
            
            path = list(map(list, zip(*path)))  ## swap the axes of list of lists
            
            t = time.time()
            path = transform_path(path, [ego.x,ego.y,ego.theta])
            elapsed = time.time() - t
            # rospy.logdebug("transform path elapsed in: {:.3f} sec".format(elapsed))
            
            rospy.logdebug("Nominal profile is generated")
            t = time.time()
            traj = nominal_profile(path, ego.v, ref_v, self.config_local.a_max, self.config_local.a_min)
            elapsed = time.time() - t
            # rospy.logdebug("nominal_profile elapsed in: {:.4f} sec".format(elapsed))
            

        
        
        traj = traj[2:] # consider the control delay 
        # print("===================================")
        # print("traj length: {}".format(len(traj)))
        # print("lane ind target: {}".format(lane_ind_target))
        # print("goal ind: {}".format(goal_ind - lane_ind_ego))
        # print("expected traj distance: {}".format(int(self.config_global["lane_change_duration"] * self.ego.v)))
        # print("actual traj distance:{}".format(distance(traj[0][0],traj[0][1],traj[-1][0],traj[-1][1])))

        return traj

    
    def generate_traj_cands(self, ego, others, target_lane, N_receding):
        """
        Return trajectory candidates and their name

        returns:
            traj_cands: List of [[traj],[traj],...], trajectory candidates
            traj_cands_names: List of [name, name, ...], trajectory candidate names
        """

        # Initialize trajectory candidates
        traj_cands = []
        traj_cands_name = []
        
        # -----------------------------------------------------------------------------------------------------
        # Calcualte frontal gap
        #
        if self.front_ind_source:
            front_gap = inter_veh_gap(ego,others[self.front_ind_source])
        else:
            front_gap = float('inf')
        
        d_short = min(max(int(ego.v*1),10), front_gap)
        d_mid = min(max(int(ego.v*2),15), front_gap)
        d_long = min(max(int(ego.v*3),20), front_gap)
        
        if front_gap <= d_short:
            [CHECK_SHORT, CHECK_MID, CHECK_LONG] = [True, False, False]
            d_short = max(d_short, 5) # lower bound the minimum lane changing distance
        elif front_gap <= d_mid:
            [CHECK_SHORT, CHECK_MID, CHECK_LONG] = [True, True, False]
        else:
            [CHECK_SHORT, CHECK_MID, CHECK_LONG] = [True, True, True]
        rospy.logdebug("traj length [short, mid, long] : {}".format([d_short, d_mid, d_long]))
        rospy.logdebug("traj availability [short, mid, long] : {}".format([CHECK_SHORT, CHECK_MID, CHECK_LONG]))

        # -----------------------------------------------------------------------------------------------------
        # Generate candidates
        # 
        front = self.front_ind_target
        if CHECK_SHORT:
            traj_to_target_in_d_short = self.spiral_traj(target_lane, front, 
                                                    lane_ind_target = d_short, lane_ind_ego = self.tlane_ind_ego)
            if len(traj_to_target_in_d_short) >= N_receding:
                # Convert the waypoints in fixed time step to match with SGAN timestep
                wp = waypoints_with_fixed_dt(traj_to_target_in_d_short,self.config_global["dt"],lane=target_lane) 
                if len(wp) >= N_receding:
                    traj_cands.append(wp)
                    traj_cands_name.append("change to TARGET -- SHORT") 
                
        if CHECK_MID:
            traj_to_target_in_d_mid = self.spiral_traj(target_lane, False, 
                                                    lane_ind_target = d_mid, lane_ind_ego = self.tlane_ind_ego)
            if len(traj_to_target_in_d_mid) >= N_receding:
                wp = waypoints_with_fixed_dt(traj_to_target_in_d_mid,self.config_global["dt"],lane=target_lane)
                if len(wp) >= N_receding:
                    traj_cands.append(wp)
                    traj_cands_name.append("change to TARGET -- MED")
            
        # if CHECK_LONG:                
        #     traj_to_target_in_d_long = self.spiral_traj(target_lane, False, 
        #                                             lane_ind_target = d_long, lane_ind_ego = self.tlane_ind_ego)
        #     if len(traj_to_target_in_d_long) >= N_receding:
        #         wp = waypoints_with_fixed_dt(traj_to_target_in_d_long,self.config_global["dt"],lane=target_lane)
        #         if len(wp) >= N_receding:
        #             traj_cands.append(wp)
        #             traj_cands_name.append("change to TARGET -- LONG")
        
        return traj_cands, traj_cands_name


    def find_target_agents_to_interact(self, ego, others, lane, range_m):
        """Find agents to interact 

        Args:
            ego (_type_): _description_
            others (_type_): _description_
            lane (_type_): _description_
            range_m (_type_): _description_

        Returns:
            _type_: _description_
        """
        inds_target = np.array([id for id in others
                                    if euclidean_dist((ego.x,ego.y),(others[id].x, others[id].y)) <= range_m
                                    and ( is_front(ego,others[id]) or
                                          ( not is_front(ego,others[id]) 
                                            and get_projection(ego,others[id]) 
                                                    <= ego.length/2+others[id].length/2
                                          )
                                        )
                                    and len(others[id].records) > int(self.config_global["T_obs"]/self.config_global["dt"])
                                    and abs(others[id].d) <= lane.width/2]) # only within the target lane 
        return inds_target
     
     
    def eval_cands_single_thread(self, ind_near, cands, cands_name, gamma = 0.9):
        """
        Return the best waypoints from the candidates
        """
        
        # Define function variables
        ego = copy.deepcopy(self.ego)
        others = copy.deepcopy(self.others)
        target_lane = self.target_lane
        N_sim = len(cands)
        N_obs = int(round(self.config_global["T_obs"]/self.config_global["dt"])) # observation horz
        N_receding = int(round(self.config_global["T"]/self.config_global["dt"])) # evaluation horz
        N_near_vehs = len(ind_near)
        N_vehs = len(ind_near) + 1 # to include ego
        Ns = int(self.config_global["dt"]/self.config_global["timestep"])
        lane_width = target_lane.width
        sub_opt_ind = -1

        # initialize metrics for control candidates
        ego_vec = np.array([copy.deepcopy(ego) for i in range(N_sim)])
        others_vec = np.array([copy.deepcopy(others) for i in range(N_sim)]) # list of dict
        min_dist_vec = float('inf')*np.ones(N_sim)
        feas_inds = [i for i in range(N_sim)]
        cost_vec = np.zeros(N_sim)
        lane_offset_vec = np.array([ego.d for i in range(N_sim)]) # list

        ####################################################
        ####################################################
        # Evaluate candidates
        ####################################################
        ####################################################
        for ell in range(N_receding-1):
            # ------------------------------------------------------------------------------
            # Check safety
            #
            inds_to_remove = []
            if ell <= 3:
                for (ni,n) in enumerate(feas_inds): # feasible simulation inds
                    # print("ell {} n {} ego[n].x {} ego[n].y {}".format(ell, ni, ego_vec[n].x, ego_vec[n].y))
                    for id in ind_near: # nearby vehicle inds
                        s_ego, d_ego, _ = get_frenet(ego_vec[n].x, ego_vec[n].y, target_lane.list, target_lane.s_map)
                        s_veh, d_veh, _ = get_frenet(others_vec[n][id].x, others_vec[n][id].y, target_lane.list, target_lane.s_map)
                        if abs(d_ego) <= target_lane.width/2+0.5 and abs(d_veh) >= target_lane.width/2+0.3:
                            continue # ignore vehicle in different lane
                        
                        if s_ego >= s_veh: # ego at front
                            if (s_ego-s_veh <= (ego.length/2 + others[id].length/2) + self.safety_bound_rear 
                                and abs(d_ego-d_veh) <= others[id].width/2+ego.width/2+ self.safety_bound_lat):
                                inds_to_remove.append(ni)
                        else: # ego behind 
                            if (s_veh-s_ego <= (ego.length/2 + others[id].length/2) + self.safety_bound_front 
                                and abs(d_ego-d_veh) <= others[id].width/2+ego.width/2+ self.safety_bound_lat):
                                inds_to_remove.append(ni)

                        # dist = inter_veh_gap(ego_vec[n], others_vec[n][id])
                        # min_dist_vec[n] = min(min_dist_vec[n], dist)
                        # if is_rear(ego_vec[n],others_vec[n][id]):
                        #     if min_dist_vec[n] <= self.safety_bound_rear:
                        #         inds_to_remove.append(ni)
                        # else:
                        #     if min_dist_vec[n] <= self.safety_bound_front:
                        #         inds_to_remove.append(ni)

            # Remove the infeasible candidate
            if len(inds_to_remove) >= 1:
                # print("ell: {} removing cands: {}".format(ell, inds_to_remove))
                feas_inds = np.delete(feas_inds, inds_to_remove)
                rospy.logwarn("At ell: {} AFTER REMOVE, feas_inds: {}".format(ell, feas_inds))
            
            if len(feas_inds) == 0: 
                max_ell=ell-1
                rospy.logdebug("No feasible candidates remaining. Evaluation terminated.")
                break

            # ------------------------------------------------------------------------------
            # SGAN
            # 
            # Initialize sgan input
            obs_traj = []
            for t in range(N_obs):
                obs_traj_t = []
                for n in feas_inds:
                    for i in range(N_vehs):
                        if i == 0: # ego
                            veh = ego_vec[n]
                        else: # other
                            veh = others_vec[n][ind_near[i-1]]
                        try:
                            pos = [veh.records[(N_obs-1)*Ns-t*Ns].x, veh.records[(N_obs-1)*Ns-t*Ns].y]
                        except:
                            pos = [veh.records[0].x, veh.records[0].y]
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
            for (ni,n) in enumerate(feas_inds):
                seq_start_end.append([ni*N_vehs, (ni+1)*N_vehs])
            # print("feas_inds: {}".format(feas_inds))
            # print("N_vehs: {}".format(N_vehs))
            # print("obs traj: {}".format(obs_traj))
            # print("obs traj rel: {}".format(obs_traj_rel))            
            # print("seq start end: {}".format(seq_start_end))


            # Predict next positions
            next_pred_traj = self.predictor.predict_batch(obs_traj, obs_traj_rel, seq_start_end)

            # Predict stopped vehicles remaining stopped
            for (ni,n) in enumerate(feas_inds):
                for i in range(N_near_vehs):
                    id = ind_near[i]
                    isStopped = others[id].v <= 0.1
                    if isStopped:
                        next_pred_traj[N_vehs*ni+i+1][0] = obs_traj[-1][N_vehs*ni+i+1][0]
                        next_pred_traj[N_vehs*ni+i+1][1] = obs_traj[-1][N_vehs*ni+i+1][1]
            
            # ---------------------------------------------------------------------------------
            # Propagate
            # 
            # Propagate other vehicles
            for (ni,n) in enumerate(feas_inds):
                # update the scene with the predicted predictions
                for i in range(N_near_vehs):
                    id = ind_near[i]
                    veh = others_vec[n][id]
                    x_ = next_pred_traj[N_vehs * ni + i+1][0]
                    y_ = next_pred_traj[N_vehs * ni + i+1][1]
                    x_diff = x_ - veh.x
                    y_diff = y_ - veh.y

                    if x_diff == 0:
                        theta_ = veh.theta
                    else:
                        theta_ = np.arctan(y_diff / x_diff)
                    v_ = np.sqrt(x_diff**2 + y_diff**2)/self.config_global["dt"]
                    veh.set_state(x_,y_,theta_,v_)
                    others_vec[n][id] = veh

            # Propagate ego
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

            # ------------------------------------------------------------------------------
            # Cost evaluation
            #
            for n in feas_inds:
                ego_ = ego_vec[n]
                _, lane_offset_vec[n], _ = get_frenet(ego_.x, ego_.y, target_lane.list, target_lane.s_map)
                cost_vec[n] += gamma**(ell)*((self.config_global["lambda_v"])*(ego_.v - self.config_local.ref_v)**2
                                            + (self.config_global["lambda_div"])*(lane_offset_vec[n])**2)
            sub_opt_ind = np.argmin(cost_vec) # -- keep the best up to this iteration


        # Check if the last position is in the dead_zone
        dead_lock_ind = is_emergency_ahead(ego, others, self.source_lane)
        if dead_lock_ind:
            rospy.logdebug("Deadlock detected")
            inds_to_remove = []
            for (ni,n) in enumerate(feas_inds): # feasible simulation inds
                ego_ = ego_vec[n]
                # if (ego_.s >= self.config_global["dead_end"]-self.config_global["dead_zone"] and abs(ego_.d) > lane_width/2*1.7):
                if (dist_betw_agents(ego_, others[dead_lock_ind]) <= self.config_global["dead_zone"]
                    and abs(ego_.d) > lane_width/2*1.7):
                    inds_to_remove.append(ni)
            feas_inds = np.delete(feas_inds, inds_to_remove)


        # return if feasible solution exists
        if len(feas_inds) >= 1:
            opt_ind = np.argmin(cost_vec[feas_inds])
            rospy.logwarn("Best intention: {}".format(cands_name[feas_inds[opt_ind]]))
            rospy.logwarn("Minimum distance over control horizon: {}".format(min_dist_vec))
            return feas_inds[opt_ind], True
        else:
            return False, False
        
        
    def traj_with_fixed_space_step(self, traj):
        # Use different space step for differenet speed
        # if self.ego.v >= 10: 
        #     space_step = 1.0
        if self.ego.v <= 6: 
            space_step = 0.6
        else: 
            space_step = self.ego.v * 0.1
        
        # Extract Path
        xy_profile = np.array(traj)[:,:-1]
        
        # Extract Speed
        v_profile = np.array(traj)[:,-1]
        
        # Set desired distance list
        len_fixed = min(len(xy_profile),len(v_profile))
        ds_arr = [euclidean_dist(point, xy_profile[i+1]) for i, point in enumerate(xy_profile[:-1])]
        ds_arr.insert(0, 0)
        s_list = np.cumsum(ds_arr)
        
        # Interpolate
        traj_interp = interp_waypoint_with_space_step(self, 
                                            xy_profile[:len_fixed,0], # x
                                            xy_profile[:len_fixed,1], # y
                                            v_profile[:len_fixed],  # v
                                            space_step, # ds
                                            s_list[:len_fixed], # desired distance steps
                                            N_path=self.N_path, dt=0.1, v0=self.ego.v)
        
        return traj_interp


    def extend_traj(self, traj, length = 50):
        """Extend the trajectory following the lane center. 

        Args:
            traj (_type_): _description_

        Returns:
            _type_: _description_
        """
        rospy.logdebug("[ NNMPC] Extend the trajectory.")
        
        # Find the closest lane
        length = self.N_path
        traj = np.array(traj)
        traj_end_point = np.array([[traj[-1][0],traj[-1][1]]])
        try:
            dist_to_source = get_min_dist_to_lane_from_point(traj_end_point, self.source_lane.list, self.slane_ind_ego)
            dist_to_target = get_min_dist_to_lane_from_point(traj_end_point, self.target_lane.list, self.tlane_ind_ego)
        except:
            dist_to_source = 0
            dist_to_target = 0
            rospy.logdebug("[ NNMPC] End road")                # print(traj_end_point, self.source_lane.list, self.slane_ind_ego)
        if dist_to_source < dist_to_target:
            lane = self.source_lane
            ego_ind_lane = self.slane_ind_ego
        else:
            lane = self.target_lane
            ego_ind_lane = self.tlane_ind_ego
        
        # Extend the trajectory with the final speed of the trajectory
        xs = traj[:,0]    
        ys = traj[:,1]
        vs = traj[:,2]
        len_lane = len(lane.list)
        near_ind = get_nearest_index(traj_end_point, lane.list[ego_ind_lane:min(len_lane-1,ego_ind_lane+length+20)])
        if ego_ind_lane+near_ind+length-1 <= len_lane:
            xs = np.array(xs.tolist()+[lane.list[ego_ind_lane+near_ind+i].x for i in range(length)])
            ys = np.array(ys.tolist()+[lane.list[ego_ind_lane+near_ind+i].y for i in range(length)])
            vs = np.array(vs.tolist()+[traj[-1,2] for i in range(length)])
            path = [xs.tolist(), ys.tolist(), vs.tolist()]
            traj = list(map(list, zip(*path)))  ## swap the axes of list of lists
        else:
            rospy.logwarn("[ NNMPC] Reference path ended.")
            traj = []
        return traj
        
