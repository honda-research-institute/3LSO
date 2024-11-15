import warnings
from utils import euclidean_dist
import numpy as np

class Pedestrian:
	def __init__(self, id, width=1.0, length=1.0, buffer_size = 1000):
		self.buffer_size = buffer_size
		self.id = id # unique id (lifetime id)
		self.width = width
		self.length = length
		self.x = 0.0 # x coordinate (cartesian)
		self.y = 0.0 # y coordinate (cartesian)
		self.theta = 0.0 # heading (yaw angle) [rad]
		self.v = 0.0 # speed [m/s]
		self.d = 0.0 # lane offset [m] (Frenet)
		self.s = 0.0 # driving distance [m]
		self.isFirstCall = True
		self.records = []
		self.rel_angle = 0.0
		self.lost_count = 0.0
		self.type = "pedestrian"

	def set_state(self, x,y,theta,v,d=1.0,s=1.0):
		self.x = x
		self.y = y
		self.theta = theta
		self.v = v
		self.d = d
		self.s = s
		self.isFirstCall = False
		if len(self.records) >= self.buffer_size:
			del self.records[-1]
		self.records.insert(0,State(x,y,theta,v,s)) # push the recent state

	def get_state(self):
		return (self.x, self.y, self.theta, self.v)

	def get_lane_offset(self):
		return self.d

	def get_state_records(self):
		return [(state.x,state.y,state.theta,state.v) for state in self.records]

	def is_approaching_to(self, veh, prev_ind = 4):
		if len(self.records) < prev_ind:
			warnings.warn("self.record length must be larger than {}".format(prev_ind))
			return False
		prev_pos = [self.records[prev_ind-1].x, self.records[prev_ind-1].y]
		curr_pos = [self.records[0].x, self.records[0].y]
		if isinstance(veh, list):
			return euclidean_dist(curr_pos, veh) < euclidean_dist(prev_pos, veh)
		else:
			return euclidean_dist(curr_pos, [veh.x, veh.y]) < euclidean_dist(prev_pos, [veh.x, veh.y])

	def is_approaching_to_junction(self):
		if len(self.records) < 2:
			warnings.warn("self.record length must be larger than 2")
			return None
		return self.records[0]-self.records[1] < 0

	def predict(self, duration = 15):
		"""
		predict future positions in the next time duration, based on the current velocity
		"""
		cos_theta = np.cos(self.theta)
		sin_theta = np.sin(self.theta)
		predicted = [[self.x + self.v * cos_theta * t, self.y + self.v * sin_theta * t] for t in range(duration)]
		# print("cos_theta: {}, sin_theta: {}".format(cos_theta, sin_theta))
		# print("predicted: {}".format(predicted))
		return predicted


class State:
	def __init__(self, x,y,theta,v,s=0):
		self.x = x
		self.y = y
		self.theta = theta
		self.v = v
		self.s = s
