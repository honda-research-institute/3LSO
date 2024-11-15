import warnings
import numpy as np

class Vehicle:
	def __init__(self, id, width, length, buffer_size = 30, is_oncoming = False):
		self.buffer_size = buffer_size
		self.id = id # unique id (lifetime id)
		self.width = width
		self.length = length
		self.x = 0.0 # x coordinate (cartesian)
		self.y = 0.0 # y coordinate (cartesian)
		self.theta = 0.0 # heading (yaw angle) [rad]
		self.v = 0.0 # speed [m/s]
		self.d = 0.0 # lane offset [m] (Frenet)
		self.s = 0.0 # driving distance [m] from the current lane
		self.s_target = 0.0 # driving distance [m] w.r.t. the target lane
		self.s_target_prev = 0.0 # this is to check if the new s_target jumps from the previous one
		self.d_target = 0.0
		self.isFirstCall = True
		self.records = []
		self.rel_angle = 0.0
		self.lost_count = 0.0
		self.a = 0.0 # acceleration
		self.type = "vehicle"
		self.is_oncoming = is_oncoming
		self.lane_num = False
		self.pos = [self.x,self.y]
		self.t = 0

	def set_state(self, x,y,theta,v,d=1.0,s=1.0,a=0.0, lane_num = 0):
		self.x = x
		self.y = y
		self.theta = theta
		self.v = v
		self.d = d
		self.s = s
		self.a = a
		self.lane_num = lane_num
		self.isFirstCall = False
		if len(self.records) >= self.buffer_size:
			del self.records[-1]
		self.records.insert(0,State(x,y,theta,v,s)) # push the recent state
		self.pos = [x,y]

	def get_state(self):
		return (self.x, self.y, self.theta, self.v)

	def get_lane_offset(self):
		return self.d

	def get_state_records(self):
		return [(state.x,state.y,state.theta,state.v) for state in self.records]

	def is_approaching_to_junction(self):
		if len(self.records) < 2:
			warnings.warn("self.record length must be larger than 2")
			return None
		return self.records[0].s-self.records[1].s < 0

	def predict(self, duration = 15): # duratin in number of seconds
		"""
		predict future positions in the next time duration, based on the current velocity
		"""
		cos_theta = np.cos(self.theta)
		sin_theta = np.sin(self.theta)
		predicted = [[self.x + self.v * cos_theta * t, self.y + self.v * sin_theta * t] for t in range(duration)]
		return predicted

	def dist_to(self, veh):
		"""bump to bump distance to {veh} in longitudinal direction

		Args:
			veh (_type_): _description_
		"""
		return np.sqrt((self.x-veh.x)**2 + (self.y-veh.y)**2)-(self.width+veh.width)/2
    		

class State:
	def __init__(self, x,y,theta,v, s=0):
		self.x = x
		self.y = y
		self.theta = theta
		self.v = v
		self.s = s
