import numpy as np

def get_spline_init(p1, p2, v, dt):
    """
    Get spline from start point p1 to goal point p2

    input:
        p1: start point [x,y,heading]
        p1: goal point [x,y,heading]
    return:
        list: in the form of [[x,y,0],...]
    """

    x1, y1, theta1 = p1[0], p1[1], p1[2]
    x2, y2, theta2 = p2[0], p2[1], 0

    dx1 = np.cos(theta1)
    dy1 = np.sin(theta1)
    dx2 = np.cos(theta2)
    dy2 = np.sin(theta2)

    dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    t = np.linspace(0, dist * 1.2, int(1.2*dist / (max(v,1)*dt)))

    t0 = t[0]
    t1 = t[-1]

    # Matrix form to be inversed
    A = np.asarray(
        [
            [1, t0, t0**2, t0**3],  # x  @ 0
            [0, 1, 2 * t0, 3 * t0**2],  # x' @ 0
            [1, t1, t1**2, t1**3],  # x  @ 1
            [0, 1, 2 * t1, 3 * t1**2],  # x' @ 1
        ]
    )

    # Compute for X
    X = np.asarray([x1, dx1, x2, dx2]).transpose()
    bx = np.linalg.solve(A, X)

    # Compute for Y
    Y = np.asarray([y1, dy1, y2, dy2]).transpose()
    by = np.linalg.solve(A, Y)

    x = np.dot(np.vstack([np.ones_like(t), t, t**2, t**3]).transpose(), bx)
    y = np.dot(np.vstack([np.ones_like(t), t, t**2, t**3]).transpose(), by)
    psi = np.append(theta1, np.arctan2(y[1:]-y[:-1], x[1:]-x[:-1]))
    traj = [[xx, yy, ps] for xx, yy, ps in zip(x, y, psi)]

    return traj

def get_spline(p1, p2, v, dt, S, vectorize = False):
    if vectorize:
        x1, y1, theta1 = p1[:,0], p1[:,1], p1[:,2]
        x2, y2, theta2 = p2[:,0], p2[:,1], p2[:,2]

        dx1 = np.cos(theta1)
        dy1 = np.sin(theta1)
        dx2 = np.cos(theta2)
        dy2 = np.sin(theta2)

        dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        count = int(1.2*dist[0] / (max(v,1)*dt))
        t = np.linspace(0, dist * 1.2, count)

        t0 = t[0]
        t1 = t[-1]

        # Matrix form to be inversed
        A = np.stack([
            np.stack([np.ones_like(t0), t0, t0**2, t0**3], axis=-1),  # x @ t0
            np.stack([np.zeros_like(t0), np.ones_like(t0), 2 * t0, 3 * t0**2], axis=-1),  # x' @ t0
            np.stack([np.ones_like(t1), t1, t1**2, t1**3], axis=-1),  # x @ t1
            np.stack([np.zeros_like(t1), np.ones_like(t1), 2 * t1, 3 * t1**2], axis=-1)   # x' @ t1
        ], axis=1)  # Shape will be (d, 4, 4) for batch processing

        # Compute for X and Y
        # Ensure X and Y are in the correct shape for batch solving, (d, 4) for bx and by
        X = np.stack([x1, dx1, x2, dx2], axis=-1)
        Y = np.stack([y1, dy1, y2, dy2], axis=-1)

        # Now solve for bx and by for all trajectories in parallel
        bx = np.linalg.solve(A, X)
        by = np.linalg.solve(A, Y)

        # Compute x and y positions for all time samples `t`
        # First, create the time basis matrix for t, which will be (d, len(t), 4)
        t_basis = np.stack([np.ones_like(t), t, t**2, t**3], axis=-1)

        # Now calculate x and y using the time basis and the solved coefficients
        x = np.einsum('dij,ij->di', t_basis, bx)  # Shape: (d, len(t))
        y = np.einsum('dij,ij->di', t_basis, by)  # Shape: (d, len(t))

        # Calculate psi (heading) using the arctangent of differences in y and x positions
        psi = np.append(theta1[np.newaxis,:], np.arctan2(y[1:] - y[:-1], x[1:] - x[:-1]), axis=0)

        # Create trajectories and store them
        trajs = np.stack((x,y,psi),axis=-1)
    else:
        trajs = []
        for i in range(p1.shape[0]):
            x1, y1, theta1 = p1[i,0], p1[i,1], p1[i,2]
            x2, y2, theta2 = p2[i,0], p2[i,1], p2[i,2]

            dx1 = np.cos(theta1)
            dy1 = np.sin(theta1)
            dx2 = np.cos(theta2)
            dy2 = np.sin(theta2)

            dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            t = np.linspace(0, dist * 1.2, int(1.2*dist / (max(v,1)*dt)))

            t0 = t[0]
            t1 = t[-1]

            # Matrix form to be inversed
            A = np.asarray(
                [
                    [1, t0, t0**2, t0**3],  # x  @ 0
                    [0, 1, 2 * t0, 3 * t0**2],  # x' @ 0
                    [1, t1, t1**2, t1**3],  # x  @ 1
                    [0, 1, 2 * t1, 3 * t1**2],  # x' @ 1
                ]
            )

            # Compute for X
            X = np.asarray([x1, dx1, x2, dx2]).transpose()
            bx = np.linalg.solve(A, X)

            # Compute for Y
            Y = np.asarray([y1, dy1, y2, dy2]).transpose()
            by = np.linalg.solve(A, Y)

            x = np.dot(np.vstack([np.ones_like(t), t, t**2, t**3]).transpose(), bx)
            y = np.dot(np.vstack([np.ones_like(t), t, t**2, t**3]).transpose(), by)
            psi = np.append(theta1, np.arctan2(y[1:]-y[:-1], x[1:]-x[:-1]))
            traj = [[xx, yy, ps] for xx, yy, ps in zip(x, y, psi)]
            trajs.append(traj[:S])
    return trajs

def get_geodesic(p1,p2,v, dt):
    x1, y1, theta1 = p1[:,0], p1[:,1], p1[:,2] # (Nsample,) for each
    x2, y2, theta2 = p2[:,0], p2[:,1], p2[:,2]

    dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    count = int(1.2*dist[0] / (max(v,1)*dt))
    
    x = np.linspace(x1,x2,count)
    y = np.linspace(y1,y2,count)
    theta = np.linspace(theta1,theta2,count)
    trajs = np.stack((x,y,theta),axis=-1)
    return  trajs

def get_geodesic_init(p1,p2,v,dt):
    x1, y1, theta1 = p1[0], p1[1], p1[2]
    x2, y2, theta2 = p2[0], p2[1], 0

    dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    count = int(1.2*dist / (max(v,1)*dt))
    
    x = np.linspace(x1,x2,count)
    y = np.linspace(y1,y2,count)
    theta = np.linspace(theta1,theta2,count)
    trajs = np.stack((x,y,theta),axis=-1)
    return  trajs.tolist()


def sample_states_vectorized(angle_samples, a_min, a_max, d, x_cur):
    angle_samples = np.asarray(angle_samples)
    a = a_min + (a_max - a_min) * angle_samples[:, None]

    xf = x_cur[0] + d * np.cos(a)
    yf = x_cur[1] + d * np.sin(a)

    yawf = x_cur[2] / 2 + a
    
    states = np.stack([xf, yf, yawf], axis=-1).reshape(-1, 3)
    return states

def calc_biased_polar_states_toward_goal_y(goal_y, nxy, d, a_min, a_max, x_cur):
    ns=100

    # Sample angles uniformly
    asi = np.linspace(a_min, a_max, ns - 1)
    
    # Compute the corresponding y-values for each angle
    y_values = x_cur[1] + d * np.sin(asi)
    
    # Bias toward points closer to goal_y (use inverse distance for biasing)
    cnav = 1 / (1 + np.abs(y_values - goal_y))

    # Normalize bias values
    cnav_sum = np.sum(cnav)
    cnav_max = np.max(cnav)
    cnav = (cnav_max - cnav) / (cnav_max * ns - cnav_sum)

    # Cumulative sum for normalized bias
    csumnav = np.cumsum(cnav)

    # Interpolation to get biased angle samples
    i_vals = np.linspace(0, 1, nxy)
    di_indices = np.searchsorted(np.linspace(0, 1, ns - 1), i_vals)

    di = csumnav[di_indices]
    states = sample_states_vectorized(di, a_min, a_max, d, x_cur)

    return states