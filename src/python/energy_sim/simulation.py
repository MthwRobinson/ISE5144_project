"""
This module defines classes and methods for an energy
grid simulation. See README.txt for documentation.
"""

from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from train_model import (
	train_demand_model,
	train_weather_model
)

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

def initialize_demand_sim(state):
	df, model = train_demand_model()
	df['month'] = [x.month for x in df.date]
	state_df = df.groupby('state').get_group(state)
	months = state_df.groupby('month')
	seasons = state_df.groupby('season')
	mean_dict = dict(months.mean()['total_energy'])
	var_dict = dict(months.std()['total_energy'])
	return state_df, mean_dict, var_dict

def simulate_demand(mean_dict, var_dict, start_month = 4, 
	n = 12,	level = 'state', interval = 'monthly', pop = None):
	if level not in ['state','city']:
		err_string = """
			The level must be either state or city
		"""
		raise SimError(err_string)
	if interval not in ['monthly', 'daily']:
		err_string = """
			The interval must be either monthly or daily
		"""
		raise SimError(err_string)
	if interval == 'daily' and pop == None:
		err_string = """
			Population is required to simulate at the
			city level
		"""
		raise SimError(err_string)
	if (pop < 0 or pop > 1) and pop != None:
		err_string = """
			Population is a percentage and must be
			expressed as a float in 0 <= x <= 1
		"""
		raise SimError(err_string)
	months = []
	demand = []
	for i in range(start_month, n+start_month):
		month = i%12
		if month == 0:
			month = 12
		mu = mean_dict[month]
		sigma = var_dict[month]
		if interval == 'monthly':
			month_demand = np.random.normal(mu,sigma)
			demand.append(month_demand)
			months.append(month)
		elif interval == 'daily':
			month_demand  = np.random.normal(mu,sigma,30)
			for dem in month_demand:
				demand.appen(dem/30)
				months.append(month)
	if level == 'city':
		demand = [pop*x for x in demand]
	return demand

def simulate_wind_level(n = 360):
	modeldir = '/home/matt/ISE5144_project/src'
	modeldir += '/python/energy_sim/models/'
	mean_list = pickle.load(open(modeldir+'wind_mean.p', 'rb'))
	std_list = pickle.load(open(modeldir+'wind_std.p', 'rb'))
	std_list = list(std_list)
	month = 0
	winds = []
	for i in range(n):
		mu = mean_list[i%360]
		sigma = std_list[month%12]
		winds.append(max(0,np.random.normal(mu,sigma)))
	return winds


def gen_wind_sim_plots():
	plt.clf()
	plt.subplot(2,1,1)
	datadir = '/home/matt/ISE5144_project/data/raw/'
	datafile = 'noaa_phl_weather.csv'
	df = pd.read_csv(datadir+datafile)
	df_wind = df[['WSF2','DATE']]
	df_wind['WSF2'][:360].plot()
	plt.ylim([0,60])
	plt.xlim([0,360])
	plt.title('Observed Wind Values, PHL')
	plt.xlabel('Day of Year')
	plt.ylabel('Max Wind Speed')

	plt.subplot(2,1,2)
	winds = simulate_wind_level()
	plt.plot(winds)
	plt.ylim([0,60])
	plt.xlim([0,360])
	plt.title('Simulated Wind Values, PHL')
	plt.xlabel('Day of Year')
	plt.ylabel('Max Wind Speed')

def gen_demand_sim_plots():
	plt.clf()
	plt.subplot(2,2,1)
	df, mean_dict, var_dict = initialize_demand_sim('NY')
	demand = simulate_demand(mean_dict, var_dict, n = 109)
	df['sim'] = demand
	df['total_energy'].plot()
	df['sim'].plot()
	plt.xlabel('Date')
	plt.ylabel('Energy Demand')
	plt.legend()
	plt.title('New York')

	plt.subplot(2,2,2)
	df, mean_dict, var_dict = initialize_demand_sim('CA')
	demand = simulate_demand(mean_dict, var_dict, n = 109)
	df['sim'] = demand
	df['total_energy'].plot()
	df['sim'].plot()
	plt.xlabel('Date')
	plt.ylabel('Energy Demand')
	plt.legend()
	plt.title('California')

	plt.subplot(2,2,3)
	df, mean_dict, var_dict = initialize_demand_sim('TX')
	demand = simulate_demand(mean_dict, var_dict, n = 109)
	df['sim'] = demand
	df['total_energy'].plot()
	df['sim'].plot()
	plt.xlabel('Date')
	plt.ylabel('Energy Demand')
	plt.legend()
	plt.title('Texas')

	plt.subplot(2,2,4)
	df, mean_dict, var_dict = initialize_demand_sim('CO')
	demand = simulate_demand(mean_dict, var_dict, n = 109)
	df['sim'] = demand
	df['total_energy'].plot()
	df['sim'].plot()
	plt.xlabel('Date')
	plt.ylabel('Energy Demand')
	plt.legend()
	plt.title('Colorado')

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
