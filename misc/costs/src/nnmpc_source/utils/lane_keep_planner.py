#!/usr/bin/env python3
import random
from math import sqrt
import numpy as np

import carla

# from agents.navigation.global_route_planner import GlobalRoutePlanner
# from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO

lookahead_distance = 10.0

# Using d = (v_f^2 - v_i^2) / (2 * a), compute the distance
# required for a given acceleration/deceleration.
def calc_distance(v_i, v_f, a):
    """Computes the distance given an initial and final speed, with a constant
    acceleration.

    args:
        v_i: initial speed (m/s)
        v_f: final speed (m/s)
        a: acceleration (m/s^2)
    returns:
        d: the final distance (m)
    """
    return (v_f*v_f-v_i*v_i)/2/a

# Using v_f = sqrt(v_i^2 + 2ad), compute the final speed for a given
# acceleration across a given distance, with initial speed v_i.
# Make sure to check the discriminant of the radical. If it is negative,
# return zero as the final speed.
def calc_final_speed(v_i, a, d):
    """Computes the final speed given an initial speed, distance travelled,
    and a constant acceleration.

    args:
        v_i: initial speed (m/s)
        a: acceleration (m/s^2)
        d: distance to be travelled (m)
    returns:
        v_f: the final speed (m/s)
    """
    temp = v_i*v_i+2*d*a
    if temp < 0: return 0.0000001
    else: return sqrt(temp)

# Computes a profile for following a lead vehicle..
def follow_profile(path, start_speed, desired_speed, acc_max, lead_car_state, time_gap=3.0):
    """Computes the velocity profile for following a lead vehicle.

    args:
        path: Path (global frame) that the vehicle will follow.
            Format: [x_points, y_points, t_points]
                    x_points: List of x values (m)
                    y_points: List of y values (m)
                    t_points: List of yaw values (rad)
                Example of accessing the ith point's y value:
                    paths[1][i]
            It is assumed that the stop line is at the end of the path.
        start_speed: speed which the vehicle starts with (m/s)
        desired_speed: speed which the vehicle should reach (m/s)
        lead_car_state: the lead vehicle current state.
            Format: [lead_car_x, lead_car_y, lead_car_speed]
                lead_car_x and lead_car_y   : position (m)
                lead_car_speed              : lead car speed (m/s)
    internal parameters of interest:
        acc_max: maximum acceleration/deceleration of the vehicle (m/s^2)
        time_gap: Amount of time taken to reach the lead vehicle from
            the current position
    returns:
        profile: Updated follow vehicle profile which contains the local
            path as well as the speed to be tracked by the controller
            (global frame).
            Length and speed in m and m/s.
            Format: [[x0, y0, v0],
                      [x1, y1, v1],
                      ...,
                      [xm, ym, vm]]
            example:
                profile[2][1]:
                returns the 3rd point's y position in the local path
                profile[5]:
                returns [x5, y5, v5] (6th point in the local path)
    """
    print("FOLLOW profile with v_desired:{}".format, desired_speed)
    profile = []
    # Find the closest point to the lead vehicle on our planned path.
    min_index = len(path[0]) - 1
    min_dist = float('Inf')
    for i in range(len(path)):
        dist = np.linalg.norm([path[0][i] - lead_car_state[0],
                                path[1][i] - lead_car_state[1]])
        if dist < min_dist:
            min_dist = dist
            min_index = i

    # Compute the time gap point, assuming our velocity is held constant at
    # the minimum of the desired speed and the ego vehicle's velocity, from
    # the closest point to the lead vehicle on our planned path.
    desired_speed = min(lead_car_state[2], desired_speed)
    ramp_end_index = min_index
    distance = min_dist
    distance_gap = desired_speed * time_gap
    while (ramp_end_index > 0) and (distance > distance_gap):
        distance += np.linalg.norm([path[0][ramp_end_index] - path[0][ramp_end_index-1],
                                    path[1][ramp_end_index] - path[1][ramp_end_index-1]])
        ramp_end_index -= 1

    # We now need to reach the ego vehicle's speed by the time we reach the
    # time gap point, ramp_end_index, which therefore is the end of our ramp
    # velocity profile.
    if desired_speed < start_speed:
        decel_distance = calc_distance(start_speed, desired_speed, -acc_max)
    else:
        decel_distance = calc_distance(start_speed, desired_speed, acc_max)

    # Here we will compute the speed profile from our initial speed to the
    # end of the ramp.
    vi = start_speed
    for i in range(ramp_end_index + 1):
        dist = np.linalg.norm([path[0][i+1] - path[0][i],
                                path[1][i+1] - path[1][i]])
        if desired_speed < start_speed:
            vf = calc_final_speed(vi, -acc_max, dist)
        else:
            vf = calc_final_speed(vi, acc_max, dist)

        profile.append([path[0][i], path[1][i], vi])
        vi = vf

    # Once we hit the time gap point, we need to be at the desired speed.
    # If we can't get there using a_max, do an abrupt change in the profile
    # to use the controller to decelerate more quickly.
    for i in range(ramp_end_index + 1, len(path[0])):
        profile.append([path[0][i], path[1][i], desired_speed])

    return profile

# Computes a profile for nominal speed tracking.
def nominal_profile(path, start_speed, desired_speed, acc_max, time_gap=2.0):
    print("nominal profile with v_desired:{}".format, desired_speed)
    """Computes the velocity profile for the local planner path in a normal
    speed tracking case.

    args:
        path: Path (global frame) that the vehicle will follow.
            Format: [x_points, y_points, t_points]
                    x_points: List of x values (m)
                    y_points: List of y values (m)
                    t_points: List of yaw values (rad)
                Example of accessing the ith point's y value:
                    paths[1][i]
            It is assumed that the stop line is at the end of the path.
        desired_speed: speed which the vehicle should reach (m/s)
    internal parameters of interest:
        acc_max: maximum acceleration/deceleration of the vehicle (m/s^2)
    returns:
        profile: Updated nominal speed profile which contains the local path
            as well as the speed to be tracked by the controller (global frame).
            Length and speed in m and m/s.
            Format: [[x0, y0, v0],
                      [x1, y1, v1],
                      ...,
                      [xm, ym, vm]]
            example:
                profile[2][1]:
                returns the 3rd point's y position in the local path
                profile[5]:
                returns [x5, y5, v5] (6th point in the local path)
    """
    profile = []
    # Compute distance travelled from start speed to desired speed using
    # a constant acceleration.
    if desired_speed < start_speed:
        accel_distance = calc_distance(start_speed, desired_speed, -acc_max)
    else:
        accel_distance = calc_distance(start_speed, desired_speed, acc_max)

    # Here we will compute the end of the ramp for our velocity profile.
    # At the end of the ramp, we will maintain our final speed.
    ramp_end_index = 0
    distance = 0.0
    while (ramp_end_index < len(path[0])-1) and (distance < accel_distance):
        distance += np.linalg.norm([path[0][ramp_end_index+1] - path[0][ramp_end_index],
                                    path[1][ramp_end_index+1] - path[1][ramp_end_index]])
        ramp_end_index += 1

    # Here we will actually compute the velocities along the ramp.
    vi = start_speed
    for i in range(ramp_end_index):
        dist = np.linalg.norm([path[0][i+1] - path[0][i],
                                path[1][i+1] - path[1][i]])
        if desired_speed < start_speed:
            vf = calc_final_speed(vi, -acc_max, dist)
            # clamp speed to desired speed
            if vf < desired_speed:
                vf = desired_speed
        else:
            vf = calc_final_speed(vi, acc_max, dist)
            # clamp speed to desired speed
            if vf > desired_speed:
                vf = desired_speed

        profile.append([path[0][i], path[1][i], vi])
        vi = vf

    # If the ramp is over, then for the rest of the profile we should
    # track the desired speed.
    for i in range(ramp_end_index+1, len(path[0])):
        profile.append([path[0][i], path[1][i], desired_speed])

    return profile

# def calculate_route(world, start): # start = [x, y, z]
#     """
#     Calculate the source lane line
#     """
#     start_location = carla.Location(start[0],
#                                     -start[1],
#                                     start[2])
#
#     start_waypoint = world.get_map().get_waypoint(start_location)
#     goal_waypoint = random.choice(start_waypoint.next(lookahead_distance))
#     goal = goal_waypoint.transform
#
#     goal_location = carla.Location(goal.location.x,
#                                     goal.location.y,
#                                     goal.location.z)
#
#     # dao = GlobalRoutePlannerDAO(world.get_map())
#     # grp = GlobalRoutePlanner(dao)
#     grp.setup()
#     lane_keep_route = grp.trace_route(start_location, goal_location)
#
#     # print(lane_keep_route)
#     path = []
#     for wp in lane_keep_route:
#         p = [0, 0, 0]
#         # print(type(wp))
#         p[0] = wp[0].transform.location.x
#         p[1] = -wp[0].transform.location.y
#         p[2] = wp[0].transform.rotation.yaw
#         path.append(p)
#
#     return path
#
#
# def main():
#     """
#     main function
#     """
#     carla_client = carla.Client(host="127.0.0.1", port=2000)
#     carla_client.set_timeout(2)
#
#     carla_world = carla_client.get_world()
#
#
# if __name__ == "__main__":
#     main()
