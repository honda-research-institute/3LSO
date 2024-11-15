#!/usr/bin/env python

from collections import namedtuple
from nnmpc_source.utils_refactored import LocalConfig, get_min_dist_to_lane_from_point, get_front_ind, get_nearest_index
import rospy
import time, math, copy
import numpy as np
import sys, os
from scipy.interpolate import interp1d
from std_msgs.msg import Float32, Int32
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped
# from carla_msgs.msg import CarlaWorldInfo, CarlaEgoVehicleStatus, CarlaCollisionEvent ,CarlaEgoVehicleControl
from traffic_msgs.msg import PerceptionLanes, VehicleState, VehicleStateArray, Waypoint, WaypointArray, CenterLanes, Decision, DecisionTrajectory
from tf.transformations import euler_from_quaternion, quaternion_from_euler

sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nnmpc_source'))
from nnmpc_source.frenet_utils import *
# from nnmpc_source.utils import *
from nnmpc_source.config_refactored import nnmpc_params
from nnmpc_source.nnmpc_refactored import nnmpc, MERGING_LANE_ID
from nnmpc_source.vehicle import Vehicle
from nnmpc_source.lane import Lane

import matplotlib.pyplot as plt

class NNMPC_NODE(object):
    def __init__(self, *args):
        super(NNMPC_NODE, self).__init__(*args)
        
        # Get ROS args
        self.with_decision_governor = rospy.get_param("/with_dg")
        self.visualize = rospy.get_param("/visualize")
        self.hz = rospy.get_param("/hz")
        self.prevent_off_steering = rospy.get_param("/prevent_off_steering")
        self.debug = rospy.get_param("/debug")
        self.dg_deviation = rospy.get_param("/dg_deviation")

        # Other class variables
        self.ego = Vehicle(-1,-1,-1)
        self.others = {}
        self.lanes = {}
        self.source_lane = Lane(id = 0)
        self.target_lane = Lane(id = 0)
        self.global_route = Lane(id = 0)
        self.planner = nnmpc(nnmpc_params)
        self.LANE_CHANGE_DECISION = Decision.LANE_CHANGE
        self.NEGOTIATE_DECISION = Decision.NEGOTIATION
        self.decision = 0
        self.dead_end_s = 0.0
        self.reinit = True
        self.config_local = LocalConfig()
        self.config_local.with_dg = self.with_decision_governor
        self.config_local.prevent_off_steering = self.prevent_off_steering
        self.config_local.sharp_turn = rospy.get_param("/sharp_turn","true")
        self.N_path = self.planner.N_path
        self.ego_state_ready = False
        
        # Subscribers
        rospy.Subscriber('/region/ego_state', VehicleState, self.callback_ego_state)
        rospy.Subscriber("/region/target_lane_id", Int32, self.target_lane_id_callback)
        rospy.Subscriber("/region/lanes_center", CenterLanes, self.lanes_center_callback)
        rospy.Subscriber("/region/lanes_perception", PerceptionLanes, self.lanes_perception_callback)
        rospy.Subscriber('/region/decision', Decision, self.decision_callback)
        rospy.Subscriber('/region/global_route', Path, self.global_route_callback)
        
        # Publishers
        self.pub_lane_change_traj = rospy.Publisher('/region/lane_change_traj', DecisionTrajectory, queue_size=10)
        self.pub_neogitation_traj = rospy.Publisher('/region/negotiation_traj', DecisionTrajectory, queue_size=10)
        self.pub_lane_follow_dev_traj = rospy.Publisher('/region/lane_follow_dev_traj', DecisionTrajectory, queue_size=10)
        self.pub_spiral_traj = rospy.Publisher('/region/spiral_traj', Path, queue_size=1)
        self.pub_final_waypoints = rospy.Publisher('/region/final_waypoints', WaypointArray, queue_size=1)
        self.pub_final_path = rospy.Publisher('/region/final_path', Path, queue_size=1)
        
        
        print("with_dg: {}".format(self.with_decision_governor))
        print("visualize: {}".format(self.visualize))
        print("hz: {}".format(self.hz))
        print("prevent_off_steering: {}".format(self.prevent_off_steering))
        print("debug: {}".format(self.debug))
        print("sharp turn: {}".format(self.config_local.sharp_turn))
        
    def lanes_perception_callback(self, msg):
        """Callback for lanes perception

        Args:
            msg (PerceptionLanes): ros msg
        """
        others = self.others
        source_lane = self.source_lane
        target_lane = self.target_lane
        
        # put the data into objects
        objects = []
        objects_ids = []
        objects_lane_num = []
        for lane_id, vehs in zip(msg.ids, msg.vehicles):
            for veh in vehs.vehicles:
                if not veh.lifetime_id in objects_ids:
                    objects.append(veh)
                    objects_ids.append(veh.lifetime_id)
                    objects_lane_num.append(lane_id)

        # remove vehicles
        existing_veh_ids = list(others.keys()) # Make a copy such that iterable does not change size during iteration
        new_veh_ids = [veh.lifetime_id for veh in objects]
        if len(existing_veh_ids) >= 1:
            for id in existing_veh_ids:
                if id not in new_veh_ids:
                    if others[id].lost_count >= 5:
                        try: del others[id]
                        except: pass
                        rospy.logdebug("[ NNMPC] veh {} removed (outside the range).".format(id))
                    else: others[id].lost_count += 1
                else: others[id].lost_count = 0

        # update vehicles
        for (i,veh) in enumerate(list(objects)):
            # update positions
            lane = Lane()
            lane_num = objects_lane_num[i]
            lane = Lane()
            if lane_num in list(self.lanes):
                lane = self.lanes[lane_num]
            else:
                rospy.logwarn("No lane num {} in lanes".format(lane_num))
            # if lane_num == target_lane.id or lane_num == source_lane.id: # only consider vehicles on the target lane
            if veh.lifetime_id not in others: # not existing if not others.has_key(veh.lifetime_id):
                others[veh.lifetime_id] = Vehicle(veh.lifetime_id,
                                                veh.width, # width
                                                veh.length) # length
                rospy.logdebug("[ NNMPC] veh {} added.".format(veh.lifetime_id))
            position = veh.pose.pose.position
            orientation_q = veh.pose.pose.orientation
            orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
            (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
            s_target, d_target, _ = get_frenet(position.x, position.y, target_lane.list, target_lane.s_map)
            if lane_num in list(self.lanes):
                s, d, _ = get_frenet(position.x, position.y, lane.list, lane.s_map)
            else:
                s, d = s_target, d_target
            v = np.sqrt(veh.twist.twist.linear.x**2 + veh.twist.twist.linear.y**2)
            others[veh.lifetime_id].set_state(position.x, position.y, yaw, v, d=d, s=s)
            others[veh.lifetime_id].s_target_prev = others[veh.lifetime_id].s_target
            others[veh.lifetime_id].s_target = s_target
            others[veh.lifetime_id].d_target = d_target
            others[veh.lifetime_id].lane_num = lane_num # lane number
            

    def callback_ego_state(self, msg):
        """Localization

        Args:
            msg (_type_): _description_
        """
        
        # target_lane = self.target_lane
        
        if self.ego.id == -1:
            self.ego = Vehicle(msg.lifetime_id, msg.width, msg.length)
            print("ego id: {}".format(self.ego.id))

        if len(self.target_lane.list) < 1:
            rospy.logwarn("[ NNMPC] Check if target lane is published.")
            return 
        position = msg.pose.pose.position
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, yaw) = euler_from_quaternion (orientation_list)
        s_target, d, _ = get_frenet(position.x, position.y, self.target_lane.list, self.target_lane.s_map)
        v = np.sqrt(msg.twist.twist.linear.x**2 + msg.twist.twist.linear.y**2)
        

        # ego.set_state(position.x, position.y, yaw, msg.twist.twist.linear.x, d=msg.d, s=msg.s) # msg.d erroneous
        self.ego.set_state(position.x, position.y, yaw, v, d=d, s=msg.s) # d<0 : deviation to right
        self.ego.s_target = s_target    
        self.ego.d_target = d
        
        # dist_ind = np.argmin([get_min_dist_to_lane_from_point([self.ego.x, self.ego.y], 
        # [id].list) for id in list(self.lanes)])
        # lane_ids = [id for id in list(self.lanes)]
        # self.ego.lane_num = lane_ids[dist_ind]
        # print("ego lane id: {}".format(self.ego.lane_num))
        # rospy.logdebug("[ ego state] x:{:.2f} y:{:.2f} theta:{:.2f} v:{:.2f} d:{:.2f} s:{:.2f}".format(self.ego.x, self.ego.y, self.ego.theta, self.ego.v, self.ego.d, self.ego.s))
        # print("ego target d : {}".format(d))
        # nnmpc_node.NNMPC_control()
        if msg.s <= 20:
            # At the starting position. Reinitialize.
            self.reinit = True 
            self.config_local.slane_ind_ego = False
            self.config_local.tlane_ind_ego = False
        
        
        # -----------------------------------
        # Local variables
        # ------------------------------------
        if self.ego_state_ready == False:
            self.config_local.slane_ind_ego = False
            self.config_local.tlane_ind_ego = False
            self.ego_state_ready = True    
            
        if self.reinit:
            self.config_local.slane_ind_ego = False
            self.config_local.tlane_ind_ego = False
        
        
        # Update the local vars
        t = time.time()
        self.update_local_vars()
        # print("compt. time to update local vars: {}".format(time.time() - t))
        

    def update_local_vars(self):
        """Update the local variables 
        """
        
        # Update closets index to lanes
        if self.config_local.slane_ind_ego == False and self.config_local.tlane_ind_ego == False:
            self.config_local.slane_ind_ego = get_nearest_index(self.ego.pos, self.source_lane.list)
            self.config_local.tlane_ind_ego = get_nearest_index(self.ego.pos, self.target_lane.list)
        else:
            if len(self.source_lane.list) > 0 and len(self.target_lane.list) > 0:
                self.config_local.slane_ind_ego = get_nearest_index(self.ego.pos, 
                                                                    self.source_lane.list[self.config_local.slane_ind_ego:self.config_local.slane_ind_ego
                                                                                        +min(100, len(self.source_lane.list)-1)]) + self.config_local.slane_ind_ego
                self.config_local.tlane_ind_ego = get_nearest_index(self.ego.pos, self.target_lane.list[self.config_local.tlane_ind_ego:self.config_local.tlane_ind_ego+min(100, len(self.target_lane.list)-1)]) + self.config_local.tlane_ind_ego
            else:
                self.config_local.slane_ind_ego = 0
                self.config_local.tlane_ind_ego = 0
        
        # Get the lane angle
        
                                    
        # Get the front ind in source and target lane
        # print("target lane id: {}".format(self.target_lane.id))
        self.config_local.front_ind_source = get_front_ind(self.ego, self.others, self.source_lane, 
                                                    range_m = max(int(5 * self.ego.v),50),merging_scenario=self.target_lane.id == MERGING_LANE_ID)
        self.config_local.ref_v = self.config_local.speed_limit
        if self.config_local.front_ind_source:
            self.config_local.ref_v = self.others[self.config_local.front_ind_source].v # ADDED 
        
        self.config_local.front_ind_target = get_front_ind(self.ego, self.others, self.target_lane, 
                                                    range_m = max(int(5 * self.ego.v),50), 
                                                    merging_scenario = self.target_lane.id == MERGING_LANE_ID)
        if self.config_local.front_ind_target and abs(self.ego.d_target) <= self.target_lane.width/2:
            self.config_local.ref_v = self.others[self.config_local.front_ind_target].v # ADDED 
        
        # Get the front dist
        if self.config_local.front_ind_source:
            veh = self.others[self.config_local.front_ind_source]
            self.config_local.front_dist_source = max(0,distance(self.ego.x, self.ego.y, veh.x, veh.y)-(self.ego.length+veh.length)/2)
        if self.config_local.front_ind_target:
            veh = self.others[self.config_local.front_ind_target]
            self.config_local.front_dist_target = max(0,distance(self.ego.x, self.ego.y, veh.x, veh.y)-(self.ego.length+veh.length)/2)
            
        # Extend the lane info if needed
        if self.planner.config_global["extend_lane_info"]:
            if self.config_local.slane_ind_ego is not False and self.config_local.slane_ind_ego < len(self.source_lane.list): 
                if self.config_local.slane_ind_ego >= len(self.source_lane.list)-1 -self.planner.N_path:
                    # Do not have enough lane info for the planning horizon. Extend the lane.
                    ext_list, ext_s_map = self.extended_lane_info(self.config_local.slane_ind_ego, self.source_lane)
                    self.source_lane.list += ext_list
                    self.source_lane.s_map += ext_s_map
            if self.config_local.tlane_ind_ego is not False and self.config_local.tlane_ind_ego < len(self.target_lane.list): 
                if self.config_local.tlane_ind_ego >= len(self.target_lane.list)-1 -self.planner.N_path:
                    # Do not have enough lane info for the planning horizon. Extend the lane.
                    ext_list, ext_s_map = self.extended_lane_info(self.config_local.tlane_ind_ego, self.target_lane)
                    self.target_lane.list += ext_list
                    self.target_lane.s_map += ext_s_map
                    
    
    def extended_lane_info(self, ind_ego:int, lane:Lane):
        """Extend the lane info with the remaining points.

        Args:
            ind_ego (int): _description_
            lane (Lane): _description_

        Returns:
            _type_: _description_
        """
        base_point = lane.list[ind_ego] # Point2D
        ss = [distance(point.x,point.y,base_point.x,base_point.y) for point in lane.list[ind_ego:]]
        s_step = np.mean(np.diff(ss))
        ext_ss = [ss[-1]+s_step*(i+1) for i in range(self.N_path)]
        ext_xs = interp1d(ss, [point.x for point in lane.list[ind_ego:]],fill_value="extrapolate")(ext_ss)
        ext_ys = interp1d(ss, [point.y for point in lane.list[ind_ego:]],fill_value="extrapolate")(ext_ss)
        
        return [Point2D(x,y) for x,y in zip(ext_xs,ext_ys)], ext_ss
        

    def target_lane_id_callback(self, msg):
        """
        Update target lane object
        """
        self.target_lane.id = msg.data

        if self.target_lane.id in self.lanes.keys():
            self.target_lane.width = self.lanes[self.target_lane.id].width
            self.target_lane.list = self.lanes[self.target_lane.id].list
            self.target_lane.s_map = self.lanes[self.target_lane.id].s_map
            self.target_lane.merge_end_point = self.lanes[self.target_lane.id].merge_end_point


    def global_route_callback(self, msg):
        path_list,s_map,angle_list = path_to_list(msg,return_angle_list=True)
        self.global_route.id = 0 # same as the source lane
        self.global_route.list = path_list
        self.global_route.angle_list = angle_list
        self.global_route.s_map = s_map
        self.global_route.width = self.source_lane.width
        self.config_local.global_lane = self.global_route
        

    def lanes_center_callback(self, msg):
        """Callback function for subscribing /region/lanes_center

        Args:
            msg (_type_): _description_
        """
        print("lane center callback")
        merging_scenario = False
        for id, path, width, speed_limit, acc_min, acc_max, merge_start_points, merge_end_points\
            in zip(msg.ids, msg.center_lines, msg.lanes_width, msg.speed_limits, msg.acc_min_limits, msg.acc_max_limits, msg.merge_start_points, msg.merge_end_points):

            path_list,s_map,angle_list = path_to_list(path.path,return_angle_list=True)
            self.lanes[id] = Lane(id=id, width=width, list=path_list, s_map=s_map, merge_end_point=[merge_end_points.x, merge_end_points.y], angle_list=angle_list)

            if id == self.source_lane.id:
                if speed_limit < 1:
                    speed_limit = 5
                self.config_local.speed_limit = speed_limit
                self.config_local.a_min = acc_min
                self.config_local.a_max = acc_max
                self.config_local.ref_v = speed_limit
                self.source_lane.list = path_list
                self.source_lane.s_map = s_map
                self.source_lane.width = width
                self.source_lane.angle_list = angle_list

            if id == self.target_lane.id:
                self.target_lane.list = path_list
                self.target_lane.s_map = s_map
                self.target_lane.width = width
                self.target_lane.angle_list = angle_list
                if self.target_lane.id == MERGING_LANE_ID or (merge_end_points.x != 0 and merge_end_points.y != 0):
                    merging_scenario = True
                    # self.config_local.merging_scenario = merging_scenario

            if merging_scenario:
                start_s, _, conv_s = get_frenet(merge_start_points.x, merge_start_points.y, self.source_lane.list, self.source_lane.s_map)
                end_s, _, conv_e = get_frenet(merge_end_points.x, merge_end_points.y, self.source_lane.list, self.source_lane.s_map)
                self.config_local.merge_end_point_x = merge_end_points.x
                self.config_local.merge_end_point_y = merge_end_points.y
                if conv_s:
                    self.config_local.merge_start_s = start_s                    
                if conv_e:
                    self.config_local.merge_end_s = end_s
                    self.config_local.dead_end_s = end_s
                else:
                    rospy.logwarn("Cannot convert dead end point")
        
        self.config_local.lanes = self.lanes


    def decision_callback(self, msg):
        """Decision from Decision Governor

        Args:
            msg (_type_): _description_
        """
        self.decision = msg.action


    def NNMPC_control(self):
        """Run NNMPC
        """
        # print("in nnmpc_Control")
        t = time.time()
        # if self.config_local.ref_v < 1: ## TEMP for some numerical error
        #     self.config_local.ref_v = 5
        ###############################################
        # *** Run NNMPC ***
        # 
        waypoints = self.planner.get_waypoints(self.ego,
                                        self.others,
                                        self.source_lane,
                                        self.target_lane, 
                                        self.config_local,
                                        reinit = self.reinit)
        ###############################################
        elapsed = time.time() - t
        
        # For debugging
        # rospy.loginfo("[ NNMPC] computation. time: {:.3f} sec".format(elapsed))
        if elapsed >= 0.07:
            rospy.loginfo("[ NNMPC] computation. time: {:.3f} sec".format(elapsed))
        if waypoints == -1 or waypoints == None or len(waypoints) == 0:
            rospy.logwarn("[ NNMPC] NO FEASIBLE SOLUTION. DO NOT MOVE")
            waypoints = [[self.ego.x, self.ego.y, 0.0] for i in range(self.N_path)]

        
        # With decision governor, publish to /lane_change_traj
        if self.with_decision_governor and self.decision == Decision.LANE_CHANGE:
            self.publish_lane_change_traj(waypoints)
            
        elif self.with_decision_governor and self.decision == Decision.NEGOTIATION:
            self.publish_negotiation_traj(waypoints)
            
        # Otherwise, publish to /final_path
        else:
            self.publish_final_path(waypoints)
        

        self.reinit = False

        # Visualize
        if len(waypoints) > 1 and self.visualize:
            dt = self.hz
        
            plt.clf()
            plt.subplot(211)
            plt.plot(np.arange(0, len(waypoints),1)*dt, np.array(waypoints)[:,2], color='b', label='Speed')
            plt.grid(True)
            plt.ylabel("[m/s]")
            plt.title("NNMPC planned trajectory")
            plt.legend()

            plt.subplot(212)
            plt.plot(np.arange(0, len(waypoints)-1,1)*dt, np.diff(np.array(waypoints)[:,2]), color='r', label='Accel')
            plt.ylabel("[m/s2]")
            plt.grid(True)
            plt.legend()
            plt.pause(0.0001)


    def publish_lane_change_traj(self,final_path):
        """Publish lane changing trajectory (with Decision Governor)

        Args:
            final_path (2d-array): waypoints to be published
        """
        waypointArray = WaypointArray()
        waypointArray.header.frame_id = "map"
        waypointArray.header.stamp = rospy.Time.now()
        if len(final_path) >= 1 and len(final_path[0]) == 3:
            for (i,p) in enumerate(final_path):
                wp = Waypoint()
                wp.pose.pose.position.x = p[0]
                wp.pose.pose.position.y = p[1]
                wp.twist.twist.linear.x = p[2] # mps
                waypointArray.waypoints.append(wp)

            # publising for decision governor
            lane_change_traj = DecisionTrajectory()
            lane_change_traj.header.stamp = rospy.Time.now()
            lane_change_traj.decision.action = self.LANE_CHANGE_DECISION
            lane_change_traj.trajectory_planned = waypointArray
            lane_change_traj.is_valid = True
            self.pub_lane_change_traj.publish(lane_change_traj)
            # print("lane change traj published")
            
    
    def publish_spiral_traj(self,final_path):
        """Publish lane changing trajectory (with Decision Governor)

        Args:
            final_path (2d-array): waypoints to be published
        """
        path_ = Path()
        path_.header.frame_id = "map";
        path_.header.stamp = rospy.Time.now()
        for p in final_path:
            pose = PoseStamped()
            pose.pose.position.x = p[0]
            pose.pose.position.y = p[1]
            pose.pose.position.z = 0.0
            path_.poses.append(pose)
            
        self.pub_spiral_traj.publish(path_)
        # print("spiral traj published")
            
            
    def publish_negotiation_traj(self,final_path):
        """Publish negotiation trajectory (with Decision Governor)

        Args:
            final_path (2d-array): waypoints to be published
        """
        
        waypointArray = WaypointArray()
        waypointArray.header.frame_id = "map"
        waypointArray.header.stamp = rospy.Time.now()
        if len(final_path) >= 1 and len(final_path[0]) == 3:
            for (i,p) in enumerate(final_path):
                wp = Waypoint()
                wp.pose.pose.position.x = p[0]
                wp.pose.pose.position.y = p[1]
                wp.twist.twist.linear.x = p[2] # mps
                waypointArray.waypoints.append(wp)

            # publising for decision governor
            negotiation_traj = DecisionTrajectory()
            negotiation_traj.header.stamp = rospy.Time.now()
            negotiation_traj.decision.action = self.NEGOTIATE_DECISION
            negotiation_traj.trajectory_planned = waypointArray
            negotiation_traj.is_valid = True
            self.pub_neogitation_traj.publish(negotiation_traj)
            # print("negotiation traj published")


    def publish_final_path(self, final_path):
        """Publish final path (this is without Decision Governor)

        Args:
            final_path (2d-array): waypoints to be published
        """
        waypoints = WaypointArray()
        waypoints.header.frame_id = "map"
        if len(final_path) >= 1:
            for (i,p) in enumerate(final_path):
                wp = Waypoint()
                wp.pose.pose.position.x = p[0]
                wp.pose.pose.position.y = p[1]
                if len(p) == 3:
                    wp.twist.twist.linear.x = p[2]
                else:
                    wp.twist.twist.linear.x = 0.0
                waypoints.waypoints.append(wp)

            self.pub_final_waypoints.publish(waypoints)

            # publishing final path to visualize waypoints
            path_ = Path()
            path_.header.frame_id = "map";
            for p in final_path:
                pose = PoseStamped()
                pose.pose.position.x = p[0]
                pose.pose.position.y = p[1]
                pose.pose.position.z = 0.0
                path_.poses.append(pose)
            self.pub_final_path.publish(path_)


    def lane_change_control(self):
        """Run NNMPC
        """
        
        t = time.time()
        waypoints = self.planner.piecewise_lanechange_waypoints(
                                        self.ego,
                                        self.others,
                                        self.source_lane,
                                        self.target_lane,
                                        self.config_local,
                                        reinit = self.reinit)
        elapsed = time.time() - t
        if elapsed >= 0.07:
            rospy.loginfo("[ NNMPC-LC] computation time: {:.2}".format(elapsed))
        
        self.publish_spiral_traj(waypoints)
        print("spiral traj published of len: {}".format(len(waypoints)))
        self.reinit = False
        

    def deviation_control(self):
        """Run NNMPC
        """
        
        t = time.time()
        waypoints = self.planner.piecewise_deviation_waypoints(
                                        self.ego,
                                        self.others,
                                        self.source_lane,
                                        self.target_lane,
                                        self.config_local,
                                        reinit = self.reinit)
        elapsed = time.time() - t
        if elapsed >= 0.07:
            rospy.loginfo("[ NNMPC-dev] computation time: {:.2}".format(elapsed))
        
        self.publish_spiral_traj(waypoints)
        

        self.reinit = False

        
if __name__ == '__main__':
    nnmpc_node = NNMPC_NODE()
    if nnmpc_node.debug:
        rospy.init_node('nnmpc', anonymous=False, log_level=rospy.DEBUG)
    else:
        rospy.init_node('nnmpc', anonymous=False)
        
    rate = rospy.Rate(nnmpc_node.hz) # 10hz
    while not rospy.is_shutdown():
        ###############################
        # RUN NNMPC
        ###############################
        if nnmpc_node.with_decision_governor:
                if nnmpc_node.decision == Decision.LANE_CHANGE:
                    nnmpc_node.config_local.dg_negotiation_called = False
                    nnmpc_node.NNMPC_control()
                    # nnmpc_node.lane_change_control()
                elif nnmpc_node.decision == Decision.NEGOTIATION:
                    nnmpc_node.config_local.dg_negotiation_called = True
                    nnmpc_node.NNMPC_control()
                elif nnmpc_node.decision == Decision.LANE_FOLLOW_DEV and nnmpc_node.dg_deviation:
                    nnmpc_node.config_local.dg_deviation_called = True
                    # nnmpc_node.lane_change_control()
                    nnmpc_node.deviation_control()
                else:
                    # Do nothing otherwise
                    pass
        else:
            nnmpc_node.NNMPC_control()
        
        try:
            rate.sleep()
        except rospy.exceptions.ROSTimeMovedBackwardsException:
            rospy.logwarn("ROSTimeMovedBackwards -- scenario reinitialized")
            nnmpc_node.reinit = True
            time.sleep(0.01)