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
import numpy as np
import applpy
import random
import pickle

def simulate_florida(wind_mills, panel_area, storage, price,
		pop, runs = 1, verbose = False, model_dict = None):
	total_demand = []
	mean_prod = []
	demand_met = []
	reward_risk = []
	energy_sold = []
	energy_purchased = []
	if model_dict == None:
		modeldir = '/home/matt/ISE5144_project/src'
		modeldir += '/python/energy_sim/models/'
		max_list = pickle.load(
			open(modeldir+'sun_max_miami.p', 'rb'))
		max_list = [max_list[x] for x in max_list]
		med_list = pickle.load(
			open(modeldir+'sun_med_miami.p', 'rb'))
		med_list = [med_list[x] for x in med_list]
		min_list = pickle.load(
			open(modeldir+'sun_min_miami.p', 'rb'))
		min_list = [min_list[x] for x in min_list]
		panels = pickle.load(
			open(modeldir+'solar_panels.p', 'rb'))
		mean_list = pickle.load(
			open(modeldir+'wind_mean_miami.p', 'rb'))
		std_list = pickle.load(
			open(modeldir+'wind_std_miami.p', 'rb'))
		e_df, e_mean, e_var = initialize_demand_sim('FL')
	else:
		max_list = model_dict['max_list']
		med_list = model_dict['med_list']
		min_list = model_dict['min_list']
		panels = model_dict['panels']
		mean_list = model_dict['mean_list']
		std_list = model_dict['std_list']
		e_df = model_dict['e_df']
		e_mean = model_dict['e_mean']
		e_var = model_dict['e_var']
	for i in range(runs):
		demand = simulate_demand(mean_dict = e_mean,
			var_dict = e_var, start_month = 1, level = 'city',
			interval = 'daily', pop = pop/18000000)
		wind = simulate_wind_level('Miami', n = 360,
			mean_list = mean_list, std_list = std_list)
		wind_energy = [wind_mills*convert_wind(wind_speed = x, C = 1500)
			for x in wind]
		sunshine = simulate_solar_panels(city = 'Miami', n = 360,
			max_list = max_list, med_list = med_list,
			min_list = min_list)
		solar_energy = [convert_solar(A = panel_area, S = x) 
			for x in sunshine]
		dem_met = 0
		outside = 0
		sold = 0
		stored = 0
		total_prod = []
		for i, dem in enumerate(demand):
			prod = solar_energy[i] + wind_energy[i]
			total_prod.append(prod)
			available = prod + stored
			if dem >= available:
				outside += dem - available
			else:
				dem_met += 1
				sold += max(available - dem - storage, 0)
				stored = min(available - dem, storage)
		prod_array = np.array(total_prod)
		avg_prod = prod_array.mean()
		inv_cov = prod_array.mean()/prod_array.std()
		mean_prod.append(avg_prod)
		demand_met.append(dem_met/360)
		reward_risk.append(inv_cov)
		energy_sold.append(price*sold)
		energy_purchased.append(price*outside)
	prod = np.mean(mean_prod)
	prod_025 = np.percentile(mean_prod, 2.5)
	prod_975 = np.percentile(mean_prod, 97.5)
	dem = np.mean(demand_met)
	dem_025 = np.percentile(demand_met, 2.5)
	dem_975 = np.percentile(demand_met, 97.5)	
	risk = np.mean(reward_risk)
	risk_025 = np.percentile(reward_risk, 2.5)
	risk_975 = np.percentile(reward_risk, 97.5)
	sold = np.mean(energy_sold)
	sold_025 = np.percentile(energy_sold, 2.5)
	sold_975 = np.percentile(energy_sold, 97.5)
	pur = np.mean(energy_purchased)
	pur_025 = np.percentile(energy_purchased, 2.5)
	pur_975 = np.percentile(energy_purchased, 97.5)
	cost = (150*panel_area + 1000000 * wind_mills 
			+ 1100*storage)
	wm_dim = np.floor(np.sqrt(wind_mills))+1
	area = (panel_area + ((wm_dim*10)*(wm_dim*100)))
	if verbose == True:
		print 'Mean Energy Production: ', prod
		print 'Pct of days demand was met: ', dem
		print 'Reward-Risk Coefficient: ', risk
		print 'Energy Purchased from Grid: ', pur
		print 'Energy Sold to Grid: ', sold
		print 'Area for Renewables: ', area
		print 'Installation Cost: ', cost

	data = {
		'population' : [pop],
		'wind_mills' : [wind_mills],
		'panel_area' : [panel_area],
		'storage' : [storage],
		'price' : [price],
		'area' : [area],
		'cost' : [cost],
		'total_prod' : [prod],
		'demand_met' : [dem],
		'reward_risk' : [risk],
		'energy_sold' : [sold],
		'energy_purchased' : [pur],
		'total_prod_025' : [prod_025],
		'demand_met_025' : [dem_025],
		'reward_risk_025' : [risk_025],
		'energy_sold_025' : [sold_025],
		'energy_purchased_025' : [pur_025],
		'total_prod_975' : [prod_975],
		'demand_met_975' : [dem_975],
		'reward_risk_975' : [risk_975],
		'energy_sold_975' : [sold_975],
		'energy_purchased_975' : [pur_975]
	}
	return data


# plt.plot(dict['demand'])
# plt.plot(dict['wind'])
# plt.plot(dict['solar'])

if __name__ == '__main__':
	modeldir = '/home/matt/ISE5144_project/src'
	modeldir += '/python/energy_sim/models/'
	max_list = pickle.load(
		open(modeldir+'sun_max_miami.p', 'rb'))
	max_list = [max_list[x] for x in max_list]
	med_list = pickle.load(
		open(modeldir+'sun_med_miami.p', 'rb'))
	med_list = [med_list[x] for x in med_list]
	min_list = pickle.load(
		open(modeldir+'sun_min_miami.p', 'rb'))
	min_list = [min_list[x] for x in min_list]
	panels = pickle.load(
		open(modeldir+'solar_panels.p', 'rb'))
	mean_list = pickle.load(
		open(modeldir+'wind_mean_miami.p', 'rb'))
	std_list = pickle.load(
		open(modeldir+'wind_std_miami.p', 'rb'))
	e_df, e_mean, e_var = initialize_demand_sim('FL')
	model_dict = {
		'max_list' : max_list,
		'med_list' : med_list,
		'min_list' : min_list,
		'panels' : panels,
		'mean_list' : mean_list,
		'std_list' : std_list,
		'e_df' : e_df,
		'e_mean' : e_mean,
		'e_var' : e_var
	}
	data = {
		'population' : [],
		'wind_mills' : [],
		'panel_area' : [],
		'storage' : [],
		'price' : [],
		'area' : [],
		'cost' : [],
		'total_prod' : [],
		'demand_met' : [],
		'reward_risk' : [],
		'energy_sold' : [],
		'energy_purchased' : [],
		'total_prod_025' : [],
		'demand_met_025' : [],
		'reward_risk_025' : [],
		'energy_sold_025' : [],
		'energy_purchased_025' : [],
		'total_prod_975' : [],
		'demand_met_975' : [],
		'reward_risk_975' : [],
		'energy_sold_975' : [],
		'energy_purchased_975' : []
	}
	df = pd.DataFrame(data)
	mills = range(0,20)
	solar = range(0,20)
	solar = [100*x for x in solar]
	store = range(20)
	store = [100*x for x in store]
	for wm in mills:
		for slr in solar:
			for sto in store:
				random.seed(8675309)
				data = simulate_florida(
					wind_mills = wm,
					panel_area = slr,
					storage = sto,
					price = .12,
					pop = 50000,
					runs = 1000,
					verbose = False,
					model_dict = model_dict
				)
				df_t = pd.DataFrame(data)
				df = pd.concat([df,df_t])
				print wm, slr, sto, 'done'
	df.dropna()
	output_dir = '/home/matt/ISE5144_project/output/'
	df.to_csv(output_dir + 'sim_out.csv')
	print 'done ... yay!'
