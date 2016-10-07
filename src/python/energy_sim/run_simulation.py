from __future__ import division
from simulation import (
	initialize_demand_sim,
	simulate_demand,
	simulate_wind_level,
	simulate_solar_panels,
	convert_wind,
	convert_solar
)
import pandas as pd
import matplotlib.pyplot as plt

def simulate_florida(wind_mills, panel_area, pop, runs = 1):
	e_df, e_mean, e_var = initialize_demand_sim('FL')
	demand = simulate_demand(mean_dict = e_mean,
		var_dict = e_var, start_month = 1, level = 'city',
		interval = 'daily', pop = pop/18000000)
	wind = simulate_wind_level('Miami', n = 360)
	wind_energy = [wind_mills*convert_wind(wind_speed = x, C = 1500)
		for x in wind]
	sunshine = simulate_solar_panels(city = 'Miami', n = 360)
	solar_energy = [convert_solar(A = panel_area, S = x) 
		for x in sunshine]
	dem_met = 0
	outside = 0
	for i, dem in enumerate(demand):
		prod = solar_energy[i] + wind_energy[i]
		if dem >= prod:
			outside += (dem-prod)
		else:
			dem_met += 1
	print 'Pct of days demand was met: ', dem_met/360
	print 'Energy Purchased from Grid: ', outside

	return {'demand': demand, 'wind' : wind_energy, 'solar' : solar_energy}


# plt.plot(dict['demand'])
# plt.plot(dict['wind'])
# plt.plot(dict['solar'])