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