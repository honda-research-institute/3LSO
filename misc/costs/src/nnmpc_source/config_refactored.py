nnmpc_params = {
    # NNMPC params
    "timestep": 0.1, # simulation time step
    "is_dead_lock_scenario": False,
    "dead_end": 50.0, # deadend location
    "dead_zone": 30.0, # deadend zone -- do not want be within {} meters zone of the dead-end
    "delta_max": 0.5, # maximum steering angle
    "delta_min": -0.5, # minimum steering angle
    "a_max": 3, # maximum acceleration
    "a_min": -5.0, # minimum acceleration
    "dc_max": -9.0, # maximum deceleration while accelerating
    "T": 2.4, # evaluation horizon
    "dt": 0.3, # evaluation time step
    "T_obs": 2.4, # observation horizon of sgan -- must bt 8 x dt
    "v_des": 5, # desired speed, m/s
    "range_m": 12.0, # range to negotiate
    "Delta_a_max": 2.0, # maximum jerk
    "Delta_delta_max": 0.4, # maximum steering rate
    "lambda_v": 3000.0, # penalty on reference speed
    "lambda_div": 15000.0, # penalty on lane tracking
    "lambda_delta": 3000.0, # penalty on steering
    "lambda_a": 1000.0, # penalty on acceleration
    "lambda_Delta_delta": 3000.0, # penalty on steering rate
    "lambda_Delta_a": 1000.0, # penalty on jerk
    "safety_bound_front": 1, # for fore vehicle, make sure ot set less than the half of the lane width
    "safety_bound_rear": 1, # for rear vehicle, make sure ot set less than the half of the lane width
    "safety_bound_lane_keeping": 1.0, # when the ego vehicle is keeping the lane
    "pred_model_type": "sgan", # {sgan, const_speed}
    "sgan_model": "model_v7_2.pt", # sgan model
    "veh_shape_type": "circle", # vehicle shape model {circle, ellipsoid}
    "eta_coop": 0.8, # cooperation parameter
    "eta_percept": 0.3, # perception parameter
    "debug": False, # set True to see outputs
    
    # Hitachinaka highway merging scneario
    "headway_bound": 1.5, # [s] -- greater, more conservative for merging behavior (default: 1)
    "front_headway_bound": 1, # [s] -- greater, more conservative for merging behavior (default: 1)
    "distance_front": 2, # [m] -- greater, more conservative for following behavior (default: 3)
    "change_lane_lookahead": 45, #[m] -- greater, smoother (default: 45)
    "stay_lane_lookahead": 3, #[m] -- greater, smoother (default: 15)
    "lane_change_duration": 5,
    
    # Piecewise linear deviation planner
    "deviation_lane_change_min_len": 6, # [m] min lane change path length
    "deviation_lane_change_dur": 3, # [s] desired lane change duration
    "deviation_lane_change_padding_by_lat_deviation": True, # 
    "deviation_p2_lon_shift": 5, # [m] longitudinal shift from the center point (+ is further from the stopped car)
    "deviation_p3_lon_shift": 5, # [m] longitudinal shift from the center point (+ is further from the stopped car) 
    "deviation_p2_lat_shift": 2.5, # [m] lateral shift from the stop car (+ is further from the stopped car)
    "deviation_p3_lat_shift": 2.5, # [m] lateral shift from the stop car (+ is further from the stopped car)
    
    # Other params
    "deviation": False,
    "spiral": True, 
    "space_interpolation": True,
    "time_range_to_consider": 5,
    "time_range_to_interact": 1,
    "time_to_trigger_interaction": 10,
    "enforce_interactive_planning": False,
    "enable_interactive_planning": False,
    "extend_lane_info": True,
    "collision_warning": False
}
