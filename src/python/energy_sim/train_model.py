"""
This module trains the times series models that
are used to generate random weather and energy
demand events
"""

from __future__ import division
import pandas as pd 
import numpy as np
import statsmodels.formula.api as sm
import matplotlib.pyplot as plt
import seaborn
import pickle
from scipy.interpolate import UnivariateSpline
from scipy.signal import cubic

def train_demand_model():
	datadir = '/home/matt/ISE5144_project/data/cleaned/'
	datafile = 'nrel_energy_consumption.csv'

	df = pd.read_csv(datadir+datafile)
	df['year'] = [str(x) for x in df['year']]
	df['month'] = [str(x) for x in df['month']]
	df['date'] = df['year']+'-'+df['month']
	df['date'] = pd.to_datetime(df['date'])
	season = []
	for month in df['month']:
		if month in ['12','1','2']:
			season.append('winter')
		elif month in ['3','4','5']:
			season.append('spring')
		elif month in ['6','7','8']:
			season.append('summer')
		else:
			season.append('fall')
	df['season'] = season

	panel = pd.DataFrame()
	groups = df.groupby('state_code')
	for name, group in groups:
		group_total = group[group['producer_type'] == 
			'Total Electric Power Industry']
		group_sum = group.groupby(['date','season']).sum().sort_index()
		group_sum['season'] = group_sum.index.get_level_values('season')
		group_sum['date'] = group_sum.index.get_level_values('date')
		group_sum['total_energy_1'] = group_sum['total_energy'].shift(1)
		group_sum['total_energy_2'] = group_sum['total_energy'].shift(2)
		group_sum['total_energy_3'] = group_sum['total_energy'].shift(3)
		group_sum['state'] = name
		group_sum.index = group_sum['date']
		panel = pd.concat([panel,group_sum])
	panel = panel.dropna()
	del panel['Unnamed: 0']
	panel = panel[panel['state'] != 'US-TOTAL']

	formula = """
		total_energy ~ total_energy_1 + total_energy_2
			+ total_energy_3 + season
	"""
	model = sm.ols(formula = formula, data = panel)
	res = model.fit()
	#res.summary()

	modeldir = '/home/matt/ISE5144_project/src'
	modeldir += '/python/energy_sim/models/'
	pickle.dump(res, open(modeldir+'energy_demand.p', 'wb'))
	return panel, res

def train_weather_model():
	datadir = '/home/matt/ISE5144_project/data/raw/'
	datafile = 'noaa_phl_weather.csv'
	df = pd.read_csv(datadir+datafile)
	df_wind = df[['WSF2','DATE']]
	df_wind['DATE'] = [pd.to_datetime(str(x)) for x in df_wind['DATE']]
	df_wind.index = df_wind['DATE']
	df_wind['day'] = [x.day for x in df_wind['DATE']]
	df_wind['month'] = [x.month for x in df_wind['DATE']]
	df_wind['year'] = [x.year for x in df_wind['DATE']]
	df_avgwind = df_wind.groupby('month').mean()
	df_stdwind = df_wind.groupby('month').std()
	df_avgwind = df_avgwind[:360]
	df_avgwind.index = range(len(df_avgwind))
	x = np.array(range(15,360,30))
	y = df_avgwind['WSF2']
	spl = UnivariateSpline(x,y,k=5)
	mean_list = spl(range(360))
	std_list = df_stdwind['WSF2']
	modeldir = '/home/matt/ISE5144_project/src'
	modeldir += '/python/energy_sim/models/'
	pickle.dump(mean_list, open(modeldir+'wind_mean.p', 'wb'))
	pickle.dump(std_list, open(modeldir+'wind_std.p', 'wb'))
	return mean_list, std_list

def gen_weather_plots(df):
	datadir = '/home/matt/ISE5144_project/data/raw/'
	datafile = 'noaa_phl_weather.csv'
	df = pd.read_csv(datadir+datafile)
	df_wind = df[['WSF2','DATE']]
	df_wind['DATE'] = [pd.to_datetime(str(x)) for x in df_wind['DATE']]
	df_wind.index = df_wind['DATE']
	df_wind['day'] = [x.day for x in df_wind['DATE']]
	df_wind['month'] = [x.month for x in df_wind['DATE']]
	df_wind['year'] = [x.year for x in df_wind['DATE']]
	df_avgwind = df_wind.groupby('month').mean()
	x = np.array(range(15,360,30))
	y = df_avgwind['WSF2']
	spl = UnivariateSpline(x,y,k=5)
	signal = spl(range(360))
	df_daily = df_wind.groupby(['month','day']).mean()
	df_daily.index = range(len(df_daily))
	df_daily['WSF2'].plot()
	plt.plot(signal, color = 'red')
	plt.title('Daily Expected Wind Level: PHL')
	plt.ylabel('Day of Years')
	plt.xlabel('Winds Speed')


def gen_plots(panel, res):
	plt.clf()
	plt.subplot(2,2,1)
	xnew = panel[panel['state'] == 'NY']
	pred = res.predict(xnew)
	pred = pd.DataFrame(pred)
	pred.columns = ['Predicted']
	pred.index = xnew.index
	obs = [x for x in xnew['total_energy']]
	xnew['Observed'] = obs
	xnew['Observed'].plot()
	pred['Predicted'].plot()
	plt.xlabel('Date')
	plt.ylabel('Energy Demand')
	plt.legend()
	plt.title('New York')

	plt.subplot(2,2,2)
	xnew = panel[panel['state'] == 'CA']
	pred = res.predict(xnew)
	pred = pd.DataFrame(pred)
	pred.columns = ['Predicted']
	pred.index = xnew.index
	obs = [x for x in xnew['total_energy']]
	xnew['Observed'] = obs
	xnew['Observed'].plot()
	pred['Predicted'].plot()
	plt.xlabel('Date')
	plt.ylabel('Energy Demand')
	plt.legend()
	plt.title('California')

	plt.subplot(2,2,3)
	xnew = panel[panel['state'] == 'TX']
	pred = res.predict(xnew)
	pred = pd.DataFrame(pred)
	pred.columns = ['Predicted']
	pred.index = xnew.index
	obs = [x for x in xnew['total_energy']]
	xnew['Observed'] = obs
	xnew['Observed'].plot()
	pred['Predicted'].plot()
	plt.xlabel('Date')
	plt.ylabel('Energy Demand')
	plt.legend()
	plt.title('Texas')

	plt.subplot(2,2,4)
	xnew = panel[panel['state'] == 'CO']
	pred = res.predict(xnew)
	pred = pd.DataFrame(pred)
	pred.columns = ['Predicted']
	pred.index = xnew.index
	obs = [x for x in xnew['total_energy']]
	xnew['Observed'] = obs
	xnew['Observed'].plot()
	pred['Predicted'].plot()
	plt.xlabel('Date')
	plt.ylabel('Energy Demand')
	plt.legend()
	plt.title('Colorado')

if __name__ == '__main__':
	panel, res = train_demand_model()