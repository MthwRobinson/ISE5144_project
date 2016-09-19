import sys, os
import pandas as pd 

def parse_weather_data(file, destdir):
	station_dict = {
		'GHCND:USW00003017' : 'Denver',
		'GHCND:USW00012839' : 'Miami',
		'GHCND:USW00094789' : 'New York',
		'GHCND:USW00024233' : 'Seattle',
		'GHCND:USW00012960' : 'Houston'
	}

	df = pd.read_csv(file)
	df['city_name'] = [station_dict[x] for x in df['STATION']]
	groups = df.groupby('city_name')
	for name, group in groups:
		savefile = 'noaa_weather_' + name.lower().replace(' ','') + '.csv'
		group.to_csv(destdir + savefile)


if __name__ == '__main__':
	filedir = '/home/matt/ISE5144_project/data/raw/'
	destdir = '/home/matt/ISE5144_project/data/cleaned/'
	file = 'noaa_weather.csv'
	parse_weather_data(filedir + file, destdir)