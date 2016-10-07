import json
import pickle


if __name__ == '__main__':
	data_dir = '/home/matt/ISE5144_project/data/raw/'
	file = 'nrel_solar_panel.json'
	data = json.load(open(data_dir+file, 'r'))
	target_dir = '/home/matt/ISE5144_project'
	target_dir += '/src/python/energy_sim/models/'
	pick_file = 'solar_panels.p'
	pickle.dump(data, open(target_dir+pick_file, 'wb'))
