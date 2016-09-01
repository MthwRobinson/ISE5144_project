"""
This module defines classes and methods for an energy
grid simulation. See README.txt for documentation.
"""

from __future__ import division
import numpy as np



class SimError(Exception):
	"""
	Sim Error Error Class
	Defines a custom error messages for exceptions relating
	to the simulation package
	"""
	def __init__(self,value):
		self.value=value
	def __str__(self):
		return repr(self.value)


class Climate():
	"""
	The climate class simulates random weather
	events for the simulation
	"""
	def  __init__(self, P, init_states = None, 
		state_names = None, random_seed = None):
		"""
		Initializes an instance of the Climate class
		Input:  1. P: the transition probabilty matrix
					(an nxn numpy array)
				2. init_states: the initial weather conditions 
					(an nx1 numpy array) -- if no value is specified
					then all states are equally likely
				3. state_names: an optional dictionary
					that names the states in the system (dict)
				4. random_seed: specifies which random seed to 
					use (integer)
		Output: 1. Initialized climate instance
		"""
		if type(P) == np.ndarray:
			self.P = P
		elif type(P) == list:
			self.P = np.array(P)
		else:
			err_string = """
				The transition probability matrix must be
				entered as either a list or a numpy array
			"""
			raise SimError(err_string)
		if self.P.shape[0] != self.P.shape[1]:
			err_string = """
				The transition probability matrix must
				be a square matrix
			"""

		if state_names != None:
			if len(state_names) != self.P.shape[0]:
				err_string = """
					The number of states must be equal to the
					number of the dimensions in the matrix
				"""
				raise SimError(err_string)
			else:
				state_names = [str(name) for name in state_names]
				self.state_names = state_names
		else:
			state_names = range(self.P.shape[0])
			state_names = [str(name) for name in state_names]
			self.state_names = state_names
		self.index_dict = {}
		for i, state in enumerate(self.state_names):
			self.index_dict[state] = i
		self.state_dict = {v: k for k, v in self.index_dict.items()}

		if type(init_states) == np.ndarray:
			if init_states.shape[0] != self.P.shape[0]:
				err_string = """
					The number of states in init_states must
					be the same as the number of states in
					the transition probability matrix
				"""
				raise SimError(err_string)
			else:
				self.init_states = init_states
		elif type(init_states) == list:
			if len(init_states) != self.P.shape[0]:
				err_string = """
					The number of states in init_states must
					be the same as the number of states in
					the transition probability matrix
				"""
				raise SimError(err_string)
			else:
				self.init_states = np.array(init_states)
		else:
			n = len(self.state_names)
			init_states = [1/n for x in state_names]
			self.init_states = np.array(init_states)

		if random_seed != None:
			self.random_seed = random_seed

	def transition(self, state, probs):
		"""
		Transition from the specified state to the next
			step according to the probability specified
			by the markov chain
		Input:  1. state: the current state
		Output: 1. The next state
		"""
		if probs == 'init':
			probs = list(self.init_states)
		else:
			probs = list(self.P[state,:])
		totals = [0]
		for i,prob in enumerate(probs):
			totals.append(totals[i] + prob)
		variate = np.random.uniform(0,1)
		for i in range(len(probs)):
			if variate >= totals[i] and variate < totals[i+1]:
				return i

	def simulate(self, n = 10, output = 'formatted'):
		"""
		Produces a list of n random weather events based
			on the markov chain
		Input:  1. n: the number of events to be simulated
		Output: 1. A list of n weather events
		"""
		# Generate the initial weather
		last_event = self.transition(state = 'init', probs = 'init')
		events = [last_event]
		if n > 1:
			for i in range(n-1):
				last_event = self.transition(state = last_event, probs = 'P')
				events.append(last_event)
		if output == 'formatted':
			return [self.state_dict[x] for x in events]
		else:
			return events








