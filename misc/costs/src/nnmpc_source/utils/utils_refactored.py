#!/usr/bin/env python


from cmath import isnan
import numpy as np
from scipy.interpolate import interp1d
from collections import namedtuple
from scipy.spatial.distance import cdist

Point2D = namedtuple('Point2D', ['x', 'y'])


def is_curvy_road(lane,lane_ind,deg=0.5, look_ahead = 3):
    """
    Return TRUE if the lane is curvy
    """
    if lane_ind + look_ahead > len(lane.list)-1:
        look_ahead = len(lane.list)-1-lane_ind
    v1 = [lane.list[lane_ind+look_ahead].x-lane.list[lane_ind].x,lane.list[lane_ind+look_ahead].y-lane.list[lane_ind].y]
    v2 = [lane.list[lane_ind+1].x-lane.list[lane_ind].x,lane.list[lane_ind+1].y-lane.list[lane_ind].y]
    angle = get_angle_betw_vectors(v1,v2)
    is_curvy =  angle >= deg * np.pi/180
    return is_curvy

def get_angle_betw_vectors(v1, v2):
    unit_v1 = v1 / max(np.linalg.norm(v1),0.1)
    unit_v2 = v2 / max(np.linalg.norm(v2),0.1)
    dot_product = np.dot(unit_v1, unit_v2)
    dot_product = min(dot_product, 1)
    dot_product = max(dot_product, -1)
    angle = np.arccos(dot_product)
    return angle

def from_list_to_Point2D_vec(l):
    return [Point2D(p[0],p[1]) for p in l]
    

def get_nearest_index(point, traj):
    """
    Return the nearest index within traj from point
    """
    dists = [np.linalg.norm(np.array([p.x, p.y])-np.array(point)) for p in traj]
    if len(dists) > 0:
        return dists.index(min(dists))
    else:
        # rospy.logwarn("[ NNMPC] Empty dists. Check the traj length in get_nearest_index()")
        return 0
        


def get_front_ind(ego, others, lane, range_m, merging_scenario = False):
    """
    Return the index of the front agent within the lane
    """
    min_dist = float("inf")
    ind_front = False
    for id in list(others.keys()):
        veh = others[id]
        dist =  euclidean_dist((ego.x,ego.y),(veh.x, veh.y))
        if dist <= range_m and is_front(ego, veh):
            # print("veh {} lane num {} veh.d_target: {} ego.d: {}".format(veh.id, veh.lane_num, veh.d_target, ego.d))
            if veh.lane_num == lane.id:
                # or (merging_scenario and abs(veh.d_target - ego.d) < ego.width):
                if dist <= min_dist:
                    min_dist = dist
                    ind_front = id
    # print("ind front: {}".format(ind_front))
    return ind_front


def is_front(ego, veh):
    """
    Return TRUE if ego is in front of veh
    """
    return np.dot([veh.x-ego.x, veh.y-ego.y],[np.cos(ego.theta),np.sin(ego.theta)]) >= 0


def dist_betw_agents(agent1,agent2):
    return euclidean_dist([agent1.x, agent1.y],[agent2.x, agent2.y])


def euclidean_dist(pos1, pos2):
    return np.sqrt((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2)


def get_frenet_concise(obj, veh, lane):
    path = lane.list
    s_map = lane.s_map
    x,y = veh.x, veh.y
    if lane.id == 0: # source lane:
        ind_closest = obj.close_ind_to_source_lane
    elif lane.id == 100: # for target lane for merging scenario
        ind_closest = obj.close_ind_to_target_lane
    else:
        ind_closest = False
    s,d,success = get_frenet(x,y,path,s_map,ind_closest=ind_closest)
    return s,d,success


# Transform from Cartesian x,y coordinates to Frenet s,d coordinates
def get_frenet(x, y, path, s_map, ind_closest = False):
    if path == None:
        print("Empty map. Cannot return Frenet coordinates")
        return 0.0, 0.0, False

    if ind_closest == False:
        ind_closest = closest_point_ind(path, x, y)

    # Determine the indices of the 2 closest points
    if ind_closest < len(path):
        # Check if we are at the end of the segment
        if ind_closest == len(path) - 1:
            use_previous = True
        elif ind_closest == 0:
            use_previous = False
        else:
            dist_prev = distance(path[ind_closest-1].x, path[ind_closest-1].y, x, y)
            dist_next = distance(path[ind_closest+1].x, path[ind_closest+1].y, x, y)

            if dist_prev <= dist_next:
                use_previous = True
            else:
                use_previous = False

        # Get the 2 points
        if use_previous:
            p1 = Point2D(path[ind_closest - 1].x, path[ind_closest - 1].y)
            p2 = Point2D(path[ind_closest].x, path[ind_closest].y)
            prev_idx = ind_closest - 1
        else:
            p1 = Point2D(path[ind_closest].x, path[ind_closest].y)
            p2 = Point2D(path[ind_closest + 1].x, path[ind_closest + 1].y)
            prev_idx = ind_closest

        # Get the point in the local coordinate with center p1
        theta = np.arctan2(p2.y - p1.y, p2.x - p1.x)
        local_p = global_to_local(p1, theta, Point2D(x,y))

        # Get the coordinates in the Frenet frame
        p_s = s_map[prev_idx] + local_p.x
        p_d = local_p.y # p_d>0: left, p_d<0: right

    else:
        print("Incorrect index")
        return 0.0, 0.0, False

    return p_s, p_d, True


def distance(x1, y1, x2, y2):
    return np.sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2))


def closest_point_ind(path, x, y):
    index = 0
    closest_index = 0
    min_dist = 10000.0
    for p in path:
        dist = distance(p.x, p.y, x, y)
        if dist < min_dist:
            min_dist = dist
            closest_index = index
        index += 1
    return closest_index

def point_ind_to_distance(path, x, y, dist_thred=0):
    dists = np.array([np.linalg.norm(np.array([p.x, p.y])-np.array([x,y])) for p in path])
    for i,dist in enumerate(dists):
        if dist <= dist_thred:
            return i 
    return 0


def global_to_local(ref_orig, orientation, p):
    delta = Point2D(p.x - ref_orig.x, p.y - ref_orig.y)

    s = np.sin(-orientation)
    c = np.cos(-orientation)

    out = Point2D(delta.x * c - delta.y * s,
    delta.x * s + delta.y * c)

    return out


def waypoints_with_fixed_dt(waypoints, dt, lane=False):
    """Convert the waypoints from with fixed distance step to with fixed time step

    Args:
        waypoints (_type_): waypoints in fixed time step
        dt (_type_): timestep
        lane (bool, optional): the assigned lane to extrapolate the trajectory over

    Returns:
        _type_: _description_
    """
    dists = []
    times = [0]
    waypoints = np.array(waypoints)
    for i,p in enumerate(waypoints[:-1]):
        p1 = waypoints[i+1]
        dist = distance(p1[0],p1[1],p[0],p[1])
        dists.append(dist)
        times.append(dist/max((p[2]+p1[2])/2,0.1))
    dists_accum = np.cumsum(dists)
    times_accum = np.cumsum(times)
    
    # print("time accum {} dt* 10 {}".format(times_accum[-1],dt*10))
    # print("len waypoints: {}, len times_accum: {}".format(len(waypoints[:,0]), len(times_accum)))
    des_times = np.arange(dt,max(times_accum[-1],dt*10),dt)
    
    
    # If lane information is not given, just extrapolate
    if lane == False:
        xs = interp1d(times_accum, waypoints[:,0],fill_value="extrapolate")(des_times)
        ys = interp1d(times_accum, waypoints[:,1],fill_value="extrapolate")(des_times)
        vs = interp1d(times_accum, waypoints[:,2],fill_value="extrapolate")(des_times)
    else: # if lane info is given, extrapolate over the lane
        xs = interp1d(times_accum, waypoints[:,0])(des_times[des_times<=times_accum[-1]])
        ys = interp1d(times_accum, waypoints[:,1])(des_times[des_times<=times_accum[-1]])
        vs = interp1d(times_accum, waypoints[:,2])(des_times[des_times<=times_accum[-1]])
        ind = get_nearest_index([waypoints[-1,0],waypoints[-1,1]], lane.list)
        xs = np.array(xs.tolist()+[lane.list[ind+i].x for i in range(20) if ind+i <= len(lane.list)-1])
        ys = np.array(ys.tolist()+[lane.list[ind+i].y for i in range(20) if ind+i <= len(lane.list)-1])
        vs = np.array(vs.tolist()+[waypoints[-1,2] for i in range(min(20,len(ys)-1)) ])
    path = [xs.tolist(), ys.tolist(), vs.tolist()]
    path = list(map(list, zip(*path)))  ## swap the axes of list of lists

    # tot_dis = dd * (len(waypoints_dis)-1)

    # ds = np.transpose(waypoints_dis)[2][:-1] * dt
    # ds = np.insert(ds,0,0.0)
    # ds = np.cumsum(ds)

    # a=np.arange(0,tot_dis,dd)
    # b=np.transpose(waypoints_dis)
    # ll=min(len(a),len(b[0]))
    # ds = ds[ds<=tot_dis]
    # ds = ds[ds<=max(a[:ll])]

    # xs = interp1d(a[:ll], b[0][:ll])(ds)
    # ys = interp1d(a[:ll], b[1][:ll])(ds)
    # vs = interp1d(a[:ll], b[2][:ll])(ds)
    # path = [xs.tolist(), ys.tolist(), vs.tolist()]
    # path = list(map(list, zip(*path)))  ## swap the axes of list of lists
    return path


def is_front(ego, veh, theta = False):
    """
    Return TRUE if veh is ahead of ego
    """
    if theta is False:
        return np.dot([veh.x-ego.x, veh.y-ego.y],[np.cos(ego.theta),np.sin(ego.theta)]) > 0
    else:
        return np.dot([veh.x-ego.x, veh.y-ego.y],[np.cos(theta),np.sin(theta)]) > 0

def is_rear(ego, veh):
    """
    Return TRUE if veh is behind of ego
    """
    return np.dot([veh.x-ego.x, veh.y-ego.y],[np.cos(ego.theta),np.sin(ego.theta)]) < 0


def inter_veh_gap(ego, veh):
    """
    Return bump-to-bump distance between ego and veh using three circle models
    """
    x,y,theta = ego.x,ego.y,ego.theta
    xi,yi,thetai = veh.x,veh.y,veh.theta
    h = ego.length/2
    w = ego.width/2
    hi = veh.length/2
    wi = veh.width/2

    min_dist = float("inf")
    for i in [-1,0,1]:
        for j in [-1,0,1]:
            dist = np.sqrt(( (x + i*(h)*np.cos(theta)) - (xi+ j*(hi)*np.cos(thetai)) )**2
                        +( (y + i*(h)*np.sin(theta)) - (yi+ j*(hi)*np.sin(thetai)) )**2
                        ) - (w + wi)
            min_dist = min(dist, min_dist)
    return max(0, min_dist)


def get_projection(a, b):
    """
    Get projection of a onto b

    args:
        a: Vehicle()
        b: Vehicle()
    """
    x,y,theta=a.x,a.y,a.theta
    xi,yi,thetai=b.x,b.y,b.theta

    avec = [x-xi,y-yi]
    bvec = [np.cos(thetai),np.sin(thetai)]
    return np.dot(avec,bvec)/np.linalg.norm(bvec)


def is_emergency_ahead(ego, others, lane):
    for id in list(others.keys()):
        veh = others[id]
        if is_dead_lock(veh, lane):
            return id
    return False


def is_dead_lock(veh, lane, N=100, VN=20):
        # check if the veh is in the source lane
        if is_in_lane(veh, lane):
            # check if they are within N meters
            # if euclidean_dist(ego.pos,veh.pos) <= N:
                # check if the veh has been stopped for the past VN steps
            if abs(veh.records[0].x - veh.records[-1].x) <= 0.01 and abs(veh.records[0].y - veh.records[-1].y) <= 0.01:
                # if all([s.v <= 0.1 for s in veh.records[:min(VN,len(veh.records-1))]]):
                return True


def is_in_lane(veh, lane):
    return get_closest_dist_to_traj(veh.pos, lane.list) <= lane.width/2


def get_closest_dist_to_traj(point, traj):
    dists = [np.linalg.norm(np.array(p)-np.array(point)) for p in traj]
    return min(dists)


def get_dead_lock_dist(ego, others, source_lane, N=100, VN=20):
        min_dist = float('inf')
        for id in others.keys():
            veh = others[id]
            dist = euclidean_dist(ego.pos,veh.pos)
            if (dist <= N
                and is_in_lane(veh, source_lane)
                and all([s.v <= 0.1 for s in veh.records[:VN]])):
                min_dist = min(min_dist, dist)
        return min_dist


def merging_point_ahead(obj,buffer:float=0):
    """Checks if it is a merging scenario

    Args:
        obj (NNMPC): NNMPC instance

    Returns:
        bool: True if merging point is ahead
    """
    # check if ego passed the merging point
    if isinstance(obj.config_local.dead_end_s, float):
        return obj.ego.s < obj.config_local.dead_end_s+buffer
    else: 
        return False
    
    
def dist_betw_vehs_bump(veh1,veh2):
    return dist_betw_three_circles(veh1.x,veh1.y,veh1.theta,
                                   veh2.x,veh2.y,veh2.theta)
    
    
def dist_betw_three_circles(x1,y1,theta1,x2,y2,theta2,h1=5,w1=2,h2=5,w2=2):
    min_dist = float("inf")
    cos_theta1 = np.cos(theta1)
    sin_theta1 = np.sin(theta1)
    cos_theta2 = np.cos(theta2)
    sin_theta2 = np.sin(theta2)

    for i in [-1,0,1]:
        for j in [-1,0,1]:
            dist = np.sqrt(( (x1 + i*(h1/2)*cos_theta1) - (x2+ j*(h2/2)*cos_theta2) )**2
                     +( (y1 + i*(h1/2)*sin_theta1) - (y2+ j*(h2/2)*sin_theta2) )**2
                     ) - (w1 + w2)/2
            min_dist = min(dist, min_dist)

    return max(0, min_dist)
    
    
def interp_waypoint_with_space_step(obj, xs, ys, vs, space_step, s_list, N_path=51, dt=0.1, v0=0):
    """
    Manipulate the waypoint with a fixed space step
    """
    s_list_des = [i*space_step for i in range(min(max(len(s_list),N_path),100))]
    x_interp = np.interp(s_list_des, s_list, xs)
    y_interp = np.interp(s_list_des, s_list, ys)
    v_interp = np.interp(s_list_des, s_list, vs)
    
    # shift speed 
    np.insert(v_interp, 0, v0)
    v_interp = v_interp[:-1]
    
    # restrict accelerations
    v_interp_new = speed_profile_with_acc_limit(obj, v_interp, dt)
    return [[x,y,v] for x,y,v in zip(x_interp, y_interp, v_interp_new)]
    
    
def speed_profile_with_acc_limit(obj, vs, dt, alim=False, boost=False):
    if alim == False:
        alim = obj.config_local.a_max
    vs_new = []
    accs = np.diff(vs) # accelerations
    vr = obj.config_local.ref_v
    
    if boost:
        # If boost, cramp up
        accs_cramped = [max(a, alim*dt) for a in accs]
    else:
        # If not, cramp down
        accs_cramped = [min(a, alim*dt) for a in accs]
    # print("accs_cramped: ", accs_cramped)
    v0 = vs[0]
    vs_new.append(v0)

    [vs_new.append(min(vs_new[i]+a, obj.config_local.ref_v)) if v0 <= vr # if current speed is slower than reference
    else vs_new.append(max(vs_new[i]+a, obj.config_local.ref_v)) # if current speed is faster than reference
    for i,a in enumerate(accs_cramped)]
    
    return vs_new


def get_parallel_translated_traj(traj,d = 1, counter_clock = True, to_left = -1, with_Point2D = False):
    """
    Args:
        traj: trajectory from which the parallel trajectory is generated
        d: distance of the parallel trajectory from traj
        couter_clock: True (left bound), False (right bound)
    
    Returns:
        trans: Shifted trajectory
    """
    if to_left != -1: # this is just for backward compatibility
        counter_clock = to_left
    trans = [get_parallel_translation(traj[i],traj[i+1],d,counter_clock)[0] for i in range(len(traj)-1)]
    if len(trans) > 1:
        trans.append(get_parallel_translation(traj[-2],traj[-1],d,counter_clock)[1])
    if with_Point2D:
        trans = [Point2D(p[0],p[1]) for p in trans]
    return trans


def ccw(A,B,C):
    return (C.y-A.y) * (B.x-A.x) > (B.y-A.y) * (C.x-A.x)

# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)


def get_parallel_translation(p1, p2, d, counter_clock = True, as_Point2D = False):
    """
    Args:
        p1: point 1 [x1,y1] -- rear point
        p2: point 2 [x2,y2] -- front point
        d: distance from vector (p2-p1)

    Returns
        p3: point 3 -- rear point
        p4: point 4 -- front point
    """
    x1,y1,x2,y2 = p1[0],p1[1],p2[0],p2[1]
    r = max(np.sqrt((x2-x1)**2 + (y2-y1)**2),1)
    if counter_clock: # to the left
        dx = d/r*(y1-y2)
        dy = d/r*(x2-x1)
    else: # to the right
        dx = d/r*(y2-y1)
        dy = d/r*(x1-x2)
    
    if as_Point2D:
        p3 = Point2D(x1 + dx, y1 + dy)
        p4 = Point2D(x2 + dx, y2 + dy)
    else:
        p3 = [x1 + dx, y1 + dy]
        p4 = [x2 + dx, y2 + dy]
    
    return p3, p4


def calc_idm_acc(ego, front_veh, s0=1, a=0.73, b=1.67, delta = 4, T=1.5):
    """_summary_

    Args:
        ego (_type_): _description_
        front_veh (_type_): _description_

    Returns:
        _type_: _description_
    """
    delta_v = ego.v - front_veh.v
    s = inter_veh_gap(ego, front_veh)
    vel = ego.v
    s_star_raw = s0 + vel * T + (vel * delta_v) / (2 * np.sqrt(a*b))
    s_star = max(s_star_raw, s0)
    acc = a * (1 - np.power(vel / ego.v, delta) - (s_star **2) / (s**2))
    acc = max(acc, -3)
    if isnan(acc):
        acc = 0
    return acc


def follow_profile_idm(path, ego, front_veh, dt = 0.1):
    acc = calc_idm_acc(ego, front_veh)
    profile = [[path[i][0], path[i][1], ego.v+dt*acc*i] for i in range(len(path))]
    return profile
    
    
    
def follow_profile_qp(path, ego, front_veh, dt = 0.1, a_max=0.78, v_max=10):
    xy_profile = [[path[i][0], path[i][1]] for i in range(len(path))]
    ds_arr = [euclidean_dist(point, xy_profile[i+1]) for i, point in enumerate(xy_profile[:-1])]
    ds_arr.insert(0, 0)
    d_arr = np.cumsum(ds_arr)
    v,d,_,_ = plan_speed_qp(ego.v,front_veh.v,front_veh.dist_to(ego), a_max=a_max, v_max = v_max, soft=True)
    v_profile = np.interp(d_arr, d, v).tolist()
    profile = [[path[i][0], path[i][1], v_profile[i]] for i in range(len(path))]
    return profile
    

def diag_block_mat_boolindex(L):
    """
    Build a diagonal matrix with matrices
    """
    shp = L[0].shape
    mask = np.kron(np.eye(len(L)), np.ones(shp))==1
    out = np.zeros(np.asarray(shp)*len(L),dtype=int)
    out[mask] = np.concatenate(L).ravel()
    return out


def plan_speed_qp(v0,vf,df0,d0=0,a_prev=0,T=3.2,dt=0.4,
                  lambda_v=1,lambda_d=0,lambda_df=4,lambda_a=5,lambda_Da=0,
                  d_min=False,a_min=-1.36,a_max=0.78, v_max = 3, soft = False):
    """
    Speed planning through quadratic programming.
    
    state = [v, d, df] # speed (m/s), distance (m), distance to front (m)
    input = [a]
    """
    if vf == False: vf = v_max
    if df0 == False: df0 = 100
    if d_min == False: d_min = max(1, 1.5*v0)

    # t = time.time()
    N = int(T/dt)
#     print("N: {}, var_len: {}".format(N, 4*N+3))

    # Useful matrieces
    zero_mat_N = np.zeros((N,N))
    zero_mat_Np1 = np.zeros((N+1,N+1))
    zero_mat_NbyNp1 = np.zeros((N,N+1))
    zero_mat_Np1byN = np.zeros((N+1,N))
    ones_N = np.ones(N)
    ones_Np1 = np.ones(N+1)
    zeros_N = np.zeros(N)
    zeros_Np1 = np.zeros(N+1)
    eye_N = np.eye(N)
    eye_Np1 = np.eye(N+1)
    low_tri_mat = np.eye(N+1,k=-1)
    upper_tri_mat = np.eye(N+1,k=1)
    dyn_mat = eye_Np1 - low_tri_mat;

    # Objective
    Q_v = lambda_v * eye_Np1
    Q_d = lambda_d * eye_Np1
    Q_df = lambda_df * eye_Np1
    Q_a = ((lambda_a + 2*lambda_Da) * eye_Np1 - lambda_Da * low_tri_mat - lambda_Da * upper_tri_mat); 
    Q_a[-2,-2] = lambda_a + lambda_Da
    Q = diag_block_mat_boolindex((Q_v,Q_d,Q_df,Q_a))[:-1,:-1]
    Q = Q.astype('float')
    Q = 2*matrix(Q)

    p_a = zeros_N; 
    p_a[0] = -2 * a_prev * lambda_Da
    if soft:
        p = np.hstack((-2*vf*lambda_v*ones_Np1, zeros_Np1, -2*d_min*lambda_df*ones_Np1, p_a))
    else:
        p = np.hstack((-2*vf*lambda_v*ones_Np1, zeros_Np1, zeros_Np1, p_a))
    p = matrix(p)

    # Inequality
    G_vmax = eye_Np1
    G_vmin = -eye_Np1
    G_df = -eye_Np1; G_df[0,0] = 0
    G_amax = eye_N;
    G_amin = -eye_N;
    if soft:
        G_row1 = np.hstack((zero_mat_Np1, zero_mat_Np1, zero_mat_Np1, zero_mat_Np1byN))
    else:
        G_row1 = np.hstack((zero_mat_Np1, zero_mat_Np1, G_df, zero_mat_Np1byN))
    G_row2 = np.hstack((zero_mat_NbyNp1, zero_mat_NbyNp1, zero_mat_NbyNp1, G_amax)) # a_max
    G_row3 = np.hstack((zero_mat_NbyNp1, zero_mat_NbyNp1, zero_mat_NbyNp1, G_amin)) # a_min
    G_row4 = np.hstack((G_vmax, zero_mat_Np1, zero_mat_Np1, zero_mat_Np1byN)) # v_max
    G_row5 = np.hstack((G_vmin, zero_mat_Np1, zero_mat_Np1, zero_mat_Np1byN)) # v_max
    G = np.vstack((G_row1, G_row2, G_row3, G_row4, G_row5))
    G = matrix(G)

    if soft:
        h_df = zeros_Np1
    else:
        h_df = -d_min * ones_Np1; h_df[0] = 0 # for hard constrained minimum distance
    h_vmax = v_max*ones_Np1
    h_vmin = zeros_Np1
    h = np.hstack((h_df, a_max*ones_N, -a_min*ones_N, h_vmax, h_vmin))
    h = matrix(h)

    # Equality
    A_row1 = np.hstack((dyn_mat, zero_mat_Np1, zero_mat_Np1, np.vstack((zeros_N, -dt*eye_N))))
    A_row2 = np.hstack((-dt*low_tri_mat, dyn_mat, zero_mat_Np1, zero_mat_Np1byN))
    A_row3 = np.hstack((dt*low_tri_mat, zero_mat_Np1, dyn_mat, zero_mat_Np1byN))
    A = np.vstack((A_row1, A_row2, A_row3))
    A = matrix(A)

    b = np.zeros(np.size(A,0))
    b[0], b[N+1], b[2*(N+1)] = v0, d0, df0
    b[2*(N+1)+1:] = vf*dt
    b = matrix(b)

    solvers.options['show_progress'] = False
    solvers.options['maxiters'] = 8
    solvers.options['feastol'] = 1e-3
    # solvers.options['tm_lim'] = 100
    sol=solvers.qp(Q, p, G, h, A, b)
    v = np.array(sol['x'][:N+1])[:,0]
    d = np.array(sol['x'][N+1:2*(N+1)])[:,0]
    df = np.array(sol['x'][2*(N+1):3*(N+1)])[:,0]
    a = np.array(sol['x'][3*(N+1):])[:,0]

    # elapsed = time.time() - t
    # print("[ NNMPC] QP time: {:.3f} sec".format(elapsed))

    return v,d,df,a

def get_min_dist_to_lane_from_point(p, lane_list_2dPoint, search_start_ind=0):
    """Returns minimum distance to the lane from the point

    Args:
        p (_type_): _description_
        lane_list_2dPoint (_type_): _description_
        search_start_ind (_type_): _description_

    Returns:
        _type_: _description_
    """
    lane_list = np.array([[point.x, point.y] for point in lane_list_2dPoint[search_start_ind:min(len(lane_list_2dPoint)-1,search_start_ind+60)]])
    dist_to_lane = np.argmin(cdist(p, lane_list))
    return dist_to_lane


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

class LocalConfig():
    def __init__(self):
        self.speed_limit=0 # from perception
        self.a_max=0 # from perception
        self.a_min=0 # from perception
        self.dead_end_s=0 # internal use from NNMPC
        self.ref_v = 0 # internal use from NNMPC
        self.merge_start_s=0 # internal use from NNMPC
        self.merge_end_s=0 # internal use from NNMPC
        self.merge_end_point_x=0 # from perception
        self.merge_end_point_y=0 # from perception
        self.with_dg=False # with decision governor
        self.dg_negotiation_called=False # flag for calling negotiation intention
        self.dg_deviation_called=False # flag for calling negotiation intention
        self.prevent_off_steering=True
        self.sharp_turn= True # for sharp turning, shift the goal point
        self.slane_ind_ego=False
        self.tlane_ind_ego=False
        self.front_ind_source=False
        self.front_ind_target=False
        self.front_dist_source=False
        self.front_dist_target=False
        self.lanes={}
        self.merging_scenario = False
        self.global_lane = False