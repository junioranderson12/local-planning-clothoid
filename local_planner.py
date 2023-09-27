import numpy as np
import math
import matplotlib.pyplot as plt
from path_planning_utils import generate_maneuver
from path_planning_utils import GetClothoidPath
import matplotlib.patches as patches
# from path_optimizer import optimize_speed
import time

class Vehicle(object):
    def __init__(self, x, y, theta, kappa, v, dt, lane=0):
        self.x = x
        self.y = y
        self.theta = theta
        self.kappa = kappa
        self.v = v
        self.dt = dt
        self.lane = lane

    def make_step(self, curv_rate, accel):
        self.x = self.x + self.v*np.cos(self.theta)*self.dt
        self.y = self.y + self.v*np.sin(self.theta)*self.dt
        self.theta = self.theta + self.kappa*self.v*self.dt
        self.kappa = self.kappa + curv_rate*self.dt
        self.v = self.v + accel*self.dt

class Lane(object):
    def __init__(self, y, ds, length):
        self.x = np.linspace(0, length, num=int(length/ds))
        self.y = y*np.ones(int(length/ds))
        self.theta = np.zeros(len(self.x))
        self.theta = np.arctan2( np.diff(self.y),np.diff(self.x) )
        path = np.zeros((len(self.x), 2))
        path[:, 0] = self.x
        path[:, 1] = self.y
        self.s, self.kappa = self.initial_guess( path )

    def initial_guess( self, path ):
        # Get vector between adjacent points
        vector = np.diff(path[:, 0:2], axis=0)

        # Get heading and magnitude of path vectors
        theta = np.arctan2(vector[:,1], vector[:,0])
        magnitude = np.sqrt(((vector[:,0]**2 + vector[:,1]**2)))

        # Get heading variation
        dtheta = np.diff(theta);

        # Clip between -pi and pi
        dtheta = np.mod(dtheta + math.pi, 2 * math.pi) - math.pi

        # Calculate curvature
        kappa_mag = np.sqrt(magnitude[0:len(magnitude)-1] * magnitude[1:len(magnitude)])
        kappa = 2 * np.sin(dtheta / 2) / kappa_mag

        # Calculate arc length
        arclength = np.concatenate(( [0], np.cumsum(magnitude) ))

        # Initial and end curvature calculation
        #     Initial: Solve for kappa and dkappa using 2nd and 3rd points
        A = ([1, 0],\
             [1, magnitude[1]])
        b = kappa[0:2]
        kappa_1 = np.array([1, -magnitude[0]]).dot(np.linalg.lstsq(A,b)[0])

        #     Final: Solve for kappa and dkappa using the two last available points
        A = ([1, -magnitude[len(magnitude)-2]],\
             [1, 0])
        b = kappa[len(kappa)-2:len(kappa)]
        kappa_end = np.array([1, magnitude[len(magnitude)-1]]).dot( np.linalg.lstsq(A,b)[0])

        #     Concatenate them into one vector
        kappa = np.concatenate(( ([kappa_1]), kappa, ([kappa_end]) ))

        return arclength, kappa

def draw_circ(radius, obs):

    n_points = 360
    X = np.zeros(n_points)
    Y = np.zeros(n_points)
    for n in range(n_points):
        X[n] = radius*np.cos(np.deg2rad(n)) + obs[0]
        Y[n] = radius*np.sin(np.deg2rad(n)) + obs[1]
    
    return (X, Y)

def navigate():
    ds = 0.1
    dt = 0.1
    
    # traj_h = [15, 30, 45, 60, 75]
    # traj_v = [-1.25, -.75, -.25,
    #                 0., .25, .50, .75]
    #Sampling parameters
    traj_h = [15, 50]
    traj_v = [-1.25, -1., -.75, -.50, -.25,
                    0., .25, .50, .75]
    
    ax = plt.figure(num=None, figsize=(10, 5)).gca()
    plt.axis("equal")

    #Obstacle definition
    obs = [100, -0.5]
    r_obs = 1.0
    x_circ, y_circ = draw_circ(r_obs, obs)
    plt.plot(x_circ, y_circ, "b")

    cost = []
    traj_sharp = []
    traj_init = []

    #Create lanes
    lane_len = 500
    lane_width = 3
    lane1 = Lane(0, ds, lane_len)
    lane2 = Lane(lane_width, ds, lane_len)
    plt.plot(np.arange(0, lane_len), (-lane_width/2)*np.ones(lane_len), "k")
    plt.plot(np.arange(0, lane_len), (lane_width + lane_width/2)*np.ones(lane_len), "k")
    plt.plot(np.arange(0, lane_len), (lane_width/2)*np.ones(lane_len), "k--")

    #Create vehicle
    v0 = 8
    ego = Vehicle(0, 0, 0, 0, v0, dt)

    #Time step
    k = 0

    #Terminal condition
    max_ego_x = 200.

    #Patches list
    rect_list = []

    while(ego.x < max_ego_x):
        
        error = False

        #Find closest point on the lane
        allDistances = np.sqrt((ego.x-lane1.x)**2+(ego.y-lane1.y)**2)
        n = np.argmin(allDistances)
        s_curr = lane1.s[n]

        #Compute final vertical samples
        s_final = s_curr + traj_h[len(traj_h)-1]
        allDistances = np.sqrt((s_final-lane1.s)**2)
        n_final = np.argmin(allDistances)

        #Vector point to sample points at s_final
        vec_final = np.array([np.cos(lane1.theta[n_final]-math.pi/2),
                        np.sin(lane1.theta[n_final]-math.pi/2)])
        theta_final = lane1.theta[n_final]
        x_hor_final = lane1.x[n_final]
        y_hor_final = lane1.y[n_final]

        start_time = time.time()
        #Sample trajectories
        for i in range(len(traj_h)):
            s_hor = s_curr + traj_h[i]
            allDistances = np.sqrt((s_hor-lane1.s)**2)
            n = np.argmin(allDistances)

            #Vector point to sample points at s_horizon
            vec = np.array([np.cos(lane1.theta[n]-math.pi/2),
                            np.sin(lane1.theta[n]-math.pi/2)])
            theta_sample = lane1.theta[n]

            for j in range(len(traj_v)):

                x_hor = lane1.x[n]
                y_hor = lane1.y[n]

                x_sample = traj_v[j]*vec[0] + x_hor
                y_sample = traj_v[j]*vec[1] + y_hor

                #Consider the curvature at the sample point
                r = 1/(abs(lane1.kappa[n]) + 1e-8)
                if traj_v[j] < 0:
                    kappa_sample = lane1.kappa[n] - 1/r
                else:
                    kappa_sample = lane1.kappa[n] + 1/r

                init1, sharp1 = generate_maneuver(ego.x, ego.y, ego.theta, ego.kappa,
                                    x_sample, y_sample, theta_sample, kappa_sample)

                if sum(init1) == -4:
                    error = True

                if traj_h[i] != traj_h[len(traj_h)-1]:
                    x_final = traj_v[j]*vec_final[0] + x_hor_final
                    y_final = traj_v[j]*vec_final[1] + y_hor_final

                    #Consider the curvature at the sample point
                    r = 1/(abs(lane1.kappa[n_final]) + 1e-8)
                    if traj_v[j] < 0:
                        kappa_final = lane1.kappa[n_final] - 1/r
                    else:
                        kappa_final = lane1.kappa[n_final] + 1/r

                    init2, sharp2 = generate_maneuver(x_sample, y_sample, theta_sample, kappa_sample,
                                        x_final, y_final, theta_final, kappa_final)

                    if sum(init2) == -4:
                        error = True

                    init = np.concatenate((init1, init2))
                    sharp = np.concatenate((sharp1, sharp2))
                else:
                    init = init1
                    sharp = sharp1


                if not error:

                    path = GetClothoidPath( init, sharp, 2 )
                    # print(path)
                    if not k % 50: 
                        plt.plot(path[0], path[1], "r")
                        plt.pause(0.001)
                        # plt.show()

                    #Compute trajectory cost
                    Kref = 1
                    Kcomf = 80

                    allDistances_obs = np.sqrt((path[0]-obs[0])**2+(path[1]-obs[1])**2)
                    if (allDistances_obs < (r_obs+1.)).any():
                        Cost = np.inf
                    else:
                        #Compute distance to reference cost
                        #Compute lateral error
                        C_ref = 0
                        # start_time = time.time()
                        for p in range(len(path)):
                            id = np.argmin(np.sqrt((path[0][p]-lane1.x)**2+(path[1][p]-lane1.y)**2))
                            coef_b = 1
                            coef_a = -np.tan(lane1.theta[id])
                            coef_c = np.tan(lane1.theta[id])*lane1.x[id]-lane1.y[id]
                            lat_error = (coef_a*path[0][p] + coef_b*path[1][p] + coef_c)/np.sqrt(coef_a**2+coef_b**2)
                            C_ref = abs(lat_error) + C_ref

                        #Compute comfort cost
                        C_comf = sum(abs(sharp[:,0]))

                        #Compute cost for right most trajectories
                        if traj_v[j] > 0:
                            C_right = 1
                        else:
                            C_right = 0

                        Cost = Kref*C_ref + Kcomf*C_comf + C_right


                else:
                    Cost = np.inf

                cost.append(Cost)
                traj_init.append(init)
                traj_sharp.append(sharp)


        min_cost_traj = np.argmin(cost)
        init = traj_init[min_cost_traj]
        sharp = traj_sharp[min_cost_traj]

        cost = []
        traj_sharp = []
        traj_init = []
        print(time.time() - start_time)
        # path = GetClothoidPath( init, sharp, 1 )
        # plt.plot(path[0], path[1], 'b.')
        # draw_circ(r_obs, obs)
        # plt.plot(ego.x, ego.y, 'bo')
        vehicle_length = 2.2
        vehicle_width = 1.0
        psi = ego.theta
        rect = patches.Rectangle((ego.x - (vehicle_length)/2.0, 
                                        ego.y - (vehicle_width)/2.0),
                                        vehicle_length,
                                        vehicle_width,
                                        psi*180/np.pi, 
                                        facecolor="blue",
                                        edgecolor="black")

        # Add the patch to the Axes
        ax.add_patch(rect)
        rect_list.append(rect)

        x_range = 30
        x_min = ego.x - x_range/2
        x_max = ego.x + x_range/2
        y_min = -2
        y_max = lane_width + 2
        
        plt.xlim([x_min, x_max])
        plt.ylim([y_min, y_max])
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        
        plt.pause(0.001)
        curv_der = sharp[0, 0]*ego.v
        ego.make_step(curv_der, 0)
        
        k += 1


navigate()
