import numpy as np

class Vehicle:
    def __init__(self, ego, args, manual_config) -> None:
        self.x = np.array([ego.position[0],ego.position[1],ego.heading_theta, ego.speed]) ## (x,y,phi,v)        
        self.u = np.array([0,0])
        self.vmax = ego.max_speed_m_s
        self.veh_length = ego.LENGTH
        self.veh_width = ego.WIDTH
        self.dt = manual_config['simulation']["dt"]
        self.l_r = 0.5

    # def update_state(self, u):
    #     """Return an updated state from the current state x0 with the action u
    #     Args:
    #         x0 (_type_): Current state [x,y,psi,v]
    #         u (_type_): Action [acc,steer]
    #         dt (_type_, Optional): Timestep, Defaults to 0.1 [s]
    #         dyn (_type_, Optional): System dynamics, Defaults to bicycle kinematics
    #     """
    #     x, y, psi, v = self.x[0], self.x[1], self.x[2], self.x[3]
    #     a, dl = u[0], u[1]
    #     x += v * self.dt * np.cos(psi + dl)
    #     y += v * self.dt * np.sin(psi + dl)
    #     psi += v / self.l_r * self.dt * np.sin(dl)
    #     v += a * self.dt
    #     self.x = [x, y, psi, v]

    #     return [x, y, psi, v]

    @staticmethod
    def forward(x, u, dt=0.1, dyn="bicycle"):
        l_r = 0.5
        if dyn == "bicycle":
            # x = np.array(x)
            x, y, psi, v = x[0], x[1], x[2], x[3]
            a, dl = u[:, 0], u[:, 1]
            
            # Vectorized updates
            x += v * dt * np.cos(psi + dl)
            y += v * dt * np.sin(psi + dl)
            psi += (v / l_r) * dt * np.sin(dl)
            v += a * dt

            # Combine the updated values back into a state matrix
            X_updated = np.stack([x, y, psi, v], axis=-1)
            
            return X_updated
    
    @staticmethod
    def forward_true(x, ego_x, dt=0.1, dyn="CV"):
        l_r = 0.5
        if dyn == "IDM":
            s_0 = 8 #minimum safe distance for IDM
            a_max = 0.5 # maximum acceleration for IDM
            b = 1 #comfortable deceleration for IDM
            delta = 4 #-- acceleration exponent for IDM (typically 4)
            
            T=1.2
            v_des = 2#-- desired speed for each vehicle
            Nveh = x.shape[0]
            # v_des = np.random.uniform(0.5,2,(x.shape[0]+1,))
            # T = np.random.uniform(0.5,2,(x.shape[0]+1,)) #1.8 # desired time headway for IDM
            
            ego_x = np.array(ego_x)[np.newaxis,:] # (1,4)
            x = np.append(ego_x,x, axis = 0)
            x_pos, y_pos, psi, v = x[:, 0], x[:, 1], x[:, 2], x[:, 3]

            # Compute pairwise distances between vehicles
            x_diff = x_pos[:, np.newaxis] - x_pos[np.newaxis, :]
            y_diff = y_pos[:, np.newaxis] - y_pos[np.newaxis, :]
            distances = np.sqrt(x_diff**2 + y_diff**2)

            # Mask for vehicles that are in front and in the near lane
            front_mask = (x_diff < 0) & (np.abs(y_diff) <= 1) # 1m is the threshold to consider as near lane

            # Set distance to infinity where vehicles are not in front or not in the same lane
            distances = np.where(front_mask, distances, np.inf)

            # Find the nearest lead vehicle for each vehicle
            min_distances = np.min(distances, axis=1)
            
            lead_vehicle_indices = np.argmin(distances, axis=1)

            # Get the velocities of the lead vehicles
            v_lead = v[lead_vehicle_indices]
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

        elif dyn == "CV":
            # Use the existing bicycle model code if needed
            l_r = 0.5
            x, y, psi, v = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
            a, dl = 0,0

            # Vectorized updates
            x += v * dt * np.cos(psi + dl)
            y += v * dt * np.sin(psi + dl)
            psi += (v / l_r) * dt * np.sin(dl)
            v += a * dt

            # Combine the updated values back into a state matrix
            X_updated = np.stack([x, y, psi, v], axis=-1)
        return X_updated