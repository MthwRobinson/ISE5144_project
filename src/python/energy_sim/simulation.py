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
	train_weather_model,
	train_sun_model
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
	n = 12,	level = 'state', interval = 'monthly', pop = None,
	drift = 0, varc = 1):
	"""
	energy demand simulation returns energy in kWh
	"""
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
		mu = mean_dict[month]*(1+drift*(i/n))
		sigma = var_dict[month]*varc
		if interval == 'monthly':
			month_demand = np.random.normal(mu,sigma)
			demand.append(month_demand)
			months.append(month)
		elif interval == 'daily':
			month_demand  = np.random.normal(mu,sigma,30)
			for dem in month_demand:
				demand.append(dem/30)
				months.append(month)
	if level == 'city':
		demand = [(pop*x) for x in demand]
	return demand

def simulate_wind_level(city, n = 360, varc = 1, mean_list = None,
		std_list = None):
	city_name = city.lower().replace(' ','')
	modeldir = '/home/matt/ISE5144_project/src'
	modeldir += '/python/energy_sim/models/'
	mean_list = pickle.load(
		open(modeldir+'wind_mean_' + city_name + '.p', 'rb'))
	std_list = pickle.load(
		open(modeldir+'wind_std_' + city_name + '.p', 'rb'))
	std_list = list(std_list)
	month = 0
	winds = []
	for i in range(n):
		mu = mean_list[i%360]
		sigma = std_list[month%12]*varc
		if city_name in ['newyork']:
			winds.append(max(1,np.random.normal(mu,sqrt(sigma))))
		else:
			winds.append(max(1,np.random.normal(mu,sigma)))
	return winds

def simulate_solar_panels(city, n = 360, max_list = None,
		med_list = None, min_list = None, panels = None):
	city_name = city.lower().replace(' ','')
	modeldir = '/home/matt/ISE5144_project/src'
	modeldir += '/python/energy_sim/models/'
	if max_list == None:
		max_list = pickle.load(
			open(modeldir+'sun_max_' + city_name + '.p', 'rb'))
		max_list = [max_list[x] for x in max_list]
	if med_list == None:
		med_list = pickle.load(
			open(modeldir+'sun_med_' + city_name + '.p', 'rb'))
		med_list = [med_list[x] for x in med_list]
	if min_list == None:
		min_list = pickle.load(
			open(modeldir+'sun_min_' + city_name + '.p', 'rb'))
		min_list = [min_list[x] for x in min_list]
	if panels == None:
		panels = pickle.load(
			open(modeldir+'solar_panels.p', 'rb'))
	city_panels = panels[city_name]
	month = 0
	sunshine = []
	for i in range(n):
		a = min_list[i%12]
		b = med_list[i%12]
		c = max_list[i%12]
		sun = np.random.triangular(a,b,c)
		pct_diff = 1+((sun-b)/b)
		month_avg = city_panels[str(i%12)]
		sunshine.append(max(0,month_avg*pct_diff))
	return sunshine

def convert_wind(C, wind_speed):
	"""
	C = Rated capacity in Kw
	wind_speed = wind speed for the day
	"""
	pct_prod = min((wind_speed)**3/(30)**3,1)
	E = pct_prod*C*24
	return E

def convert_solar(A, S):
	"""
	A = area 
	S = amount produced for a day per sq meter
	"""
	return A*S

def gen_wind_sim_plots(city):
	plt.clf()
	plt.subplot(2,1,1)
	city_name = city.lower().replace(' ', '')
	datadir = '/home/matt/ISE5144_project/data/cleaned/'
	datafile = 'noaa_weather_' + city_name + '.csv'
	df = pd.read_csv(datadir+datafile)
	df_wind = df[['AWND','DATE']]
	df_wind['AWND'][:360].plot()
	plt.ylim([0,25])
	plt.xlim([0,360])
	title_name = 'Observed Wind Values, ' + city
	plt.title(title_name)
	plt.xlabel('Day of Year')
	plt.ylabel('Average Wind Speed (mph)')
	plt.subplot(2,1,2)
	winds = simulate_wind_level(city, varc = 1)
	plt.plot(winds)
	plt.ylim([0,25])
	plt.xlim([0,360])
	title_name = 'Simulated Wind Values, ' + city
	plt.title(title_name)
	plt.xlabel('Day of Year')
	plt.ylabel('Average Wind Speed (mph)')

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
	df, mean_dict, var_dict = initialize_demand_sim('FL')
	demand = simulate_demand(mean_dict, var_dict, n = 109,
		drift = .3, varc = .55)
	df['Simulation'] = demand
	df['Observed'] = df['total_energy']
	df['Observed'].plot()
	df['Simulation'].plot()
	plt.xlabel('Date')
	plt.ylabel('Energy Demand')
	plt.legend(loc = 2)
	plt.title('Florida')

	plt.subplot(2,2,3)
	df, mean_dict, var_dict = initialize_demand_sim('TX')
	demand = simulate_demand(mean_dict, var_dict, n = 109)
	df['sim'] = demand
	df['total_energy'].plot()
	df['sim'].plot()
	plt.xlabel('Date')
	plt.ylabel('Energy Demand (kWh)')
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










