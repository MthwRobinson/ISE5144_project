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

def simulate_wind_level(city, n = 360):
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
		sigma = std_list[month%12]
		if city_name in ['newyork']:
			winds.append(max(1,np.random.normal(mu,sqrt(sigma))))
		else:
			winds.append(max(1,np.random.normal(mu,sigma)))
	return winds

def simulate_sun_level(city, n = 360):
	city_name = city.lower().replace(' ','')
	modeldir = '/home/matt/ISE5144_project/src'
	modeldir += '/python/energy_sim/models/'
	max_list = pickle.load(
		open(modeldir+'sun_max_' + city_name + '.p', 'rb'))
	max_list = [max_list[x] for x in max_list]
	med_list = pickle.load(
		open(modeldir+'sun_med_' + city_name + '.p', 'rb'))
	med_list = [med_list[x] for x in med_list]
	min_list = pickle.load(
		open(modeldir+'sun_min_' + city_name + '.p', 'rb'))
	min_list = [min_list[x] for x in min_list]
	month = 0
	sunshine = []
	for i in range(n):
		a = min_list[i%12]
		b = med_list[i%12]
		c = max_list[i%12]
		sunshine.append(np.random.triangular(a,b,c))
	return sunshine


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
	plt.ylabel('Max Wind Speed')
	plt.subplot(2,1,2)
	winds = simulate_wind_level(city)
	plt.plot(winds)
	plt.ylim([0,25])
	plt.xlim([0,360])
	title_name = 'Observed Wind Values, ' + city
	plt.title(title_name)
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

