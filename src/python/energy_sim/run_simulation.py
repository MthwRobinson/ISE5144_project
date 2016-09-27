from simulation import (
	initialize_demand_sim,
	simulate_demand,
	simulate_wind_level,
	simulate_sun_level,
	convert_wind,
	convert_solar
)

def simulate_florida(runs = 1):
	e_df, e_mean, e_var = initialize_demand_sim('FL')
	demand = simulate_demand(mean_dict = e_mean,
		var_dict = e_var, start_month = 1, level = 'city',
		interval = 'daily', pop = 0.0016)
	wind = simulate_wind_level('Miami', n = 360)
	wind_energy = [convert_wind(wind_speed = x,
		rho = 0.076474252, radius = 10, Cp = 0.35, n = 20)
		for x in wind]
	sunshine = simulate_sun_level(city = 'Miami', n = 360)
	solar_energy = [x*convert_solar(n = 1618, rating = 4,
		loss = 0.16, hours = 8) for x in sunshine]
	plt.plot(demand)
	plt.plot(wind_energy)
	plt.plot(solar_energy)