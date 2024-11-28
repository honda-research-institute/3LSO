import numpy as np

def IDM_predictor(x_reference, xhat, dt):
    '''
    x_reference (S, Nsample, 4)
    xhat (Nveh,4)
    dt int
    return xhat_predictions (S,Nveh,Nsample,4)
    '''
    S, Nsample = x_reference.shape[0], x_reference.shape[1]
    Nveh = xhat.shape[0]

    obstacles = np.repeat(xhat[:,np.newaxis],Nsample,axis=1) #(Nveh,Nsample,4) 
    xhat_predictions = np.zeros((S,Nveh,Nsample,4))
    for i in range(S):    
        obstacles = IDM_one_step(obstacles,x_reference[i], dt)
        xhat_predictions[i] = obstacles

    return xhat_predictions



def IDM_one_step(x,ego_x,dt):
    '''
    ego_x (Nsample,4)
    x (Nveh, Nsample, 4)
    return (Nveh, Nsample, 4)
    '''
    s_0 = 8 #minimum safe distance for IDM
    a_max = 1.5 # maximum acceleration for IDM
    b = 1 #comfortable deceleration for IDM
    delta = 4 #-- acceleration exponent for IDM (typically 4)
    
    T=0.5
    v_des = 3#-- desired speed for each vehicle
    Nveh = x.shape[0]
    # v_des = np.random.uniform(0.5,2,(x.shape[0]+1,))
    # T = np.random.uniform(0.5,2,(x.shape[0]+1,)) #1.8 # desired time headway for IDM
    
    ego_x = ego_x[np.newaxis,:] # (1, Nsample, 4)
    x = np.append(ego_x,x, axis = 0) #(Nveh+1, Nsample,4)
    x_pos, y_pos, psi, v = x[:, :, 0], x[:,:, 1], x[:,:, 2], x[:,:, 3] # (Nveh+1,Nsample) each

    # Compute pairwise distances between vehicles
    x_diff = x_pos[:, np.newaxis] - x_pos[np.newaxis] #(Nveh+1,1,Nsample)-(1,Nveh+1,Nsample) = (Nveh+1,Nveh+1,Nsample)
    y_diff = y_pos[:, np.newaxis] - y_pos[np.newaxis]
    distances = np.sqrt(x_diff**2 + y_diff**2)

    # Mask for vehicles that are in front and in the near lane
    front_mask = (x_diff < 0) & (np.abs(y_diff) <= 2) # 1m is the threshold to consider as near lane

    # Set distance to infinity where vehicles are not in front or not in the same lane
    distances = np.where(front_mask, distances, np.inf)

    # Find the nearest lead vehicle for each vehicle
    min_distances = np.min(distances, axis=1) # (Nveh+1,Nsample)
    lead_vehicle_indices = np.argmin(distances, axis=1) 
    indices = np.arange(v.shape[1])  # Sample indices (0 to Nsample-1)
    v_lead = v[lead_vehicle_indices, indices]  # Extract lead vehicle velocities
    # Get the velocities of the lead vehicles
    # v_lead = v[lead_vehicle_indices]
    # v_des = v_des[lead_vehicle_indices]
    # T = T[lead_vehicle_indices]
    # Calculate desired gap for IDM
    s_star = s_0 + v * T + (v * (v - v_lead)) / (2 * np.sqrt(a_max * b))

    # IDM acceleration calculation for each vehicle
    a_idm = a_max * (1 - (v / v_des)**delta - (s_star / min_distances)**2)
    a_idm = np.where(min_distances == np.inf, a_max * (1 - (v / v_des)**delta), a_idm)  # If no lead, accelerate to v_des

    # Update each vehicle's state based on IDM dynamics
    x_pos += v * dt * np.cos(psi)
    y_pos += v * dt * np.sin(psi)
    v = np.maximum(0, v + a_idm * dt)  # Ensure velocity doesn't go negative

    # Combine the updated values back into a state matrix
    X_updated = np.stack([x_pos, y_pos, psi, v], axis=-1)
    return X_updated[1:] # exclude ego
