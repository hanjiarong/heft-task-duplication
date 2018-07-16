import argparse, json

from utils.utils import mkdir
from utils.utils import get_stats

import numpy as np
import scipy as sp
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


from matplotlib.ticker import AutoMinorLocator

def plot_bars(data, ccrs, algorithms):

	# plot
	fig, ax = plt.subplots(figsize=(13, 6))

	# variables
	ylabel = 'y_label'
	title = 'title'
	number_of_bars = len(algorithms)
	minorLocator = AutoMinorLocator()
	
	# data
	bar_means = list()
	bar_cis = list()
	bar_positions = list()
	bar_width = 0.80	
	position = 0

	#################
	# creating bars #
	#################

	#preparing data
	for ccr in ccrs:
		for algorithm in algorithms:

			mean = data[ccr][algorithm]['mean']
			ci = data[ccr][algorithm]['ci']
			
			bar_means.append(mean)
			bar_cis.append(ci)

			position += bar_width
			bar_positions.append(position)

		position += bar_width

	# bar configs
	rects = list()
	colors = ['red', 'lightsalmon',  'lightgreen', 'royalblue', 'yellowgreen', 'deepskyblue', 'silver', 'silver']
	opacity = 0.9
	error_config = {'ecolor': '0.0'}
	#patterns = ('..',  'x', 'xx',  '--')

	# creating rects
	for i in range(number_of_bars):
		rect = ax.bar(bar_positions[i::number_of_bars], bar_means[i::number_of_bars], bar_width,
			#hatch=patterns[i],
			alpha=1.0,
			color=colors[i],
			yerr=bar_cis[i::number_of_bars],
			error_kw=error_config,
			label='%s' % (algorithms[i]))
		rects.append(rect)

	#################
	# x-axis config #
	#################
	
	# ensembles tick labels
	ensemble_xticklabels = [r'$%s$' % (i)  for i in ccrs]
	ensemble_xticks = map(lambda x: x  + ((number_of_bars * bar_width)/2), bar_positions[::number_of_bars])

	
	# divisor line between ensenbles sizes
	divisor_xticks = map(lambda x: x - bar_width/2 , bar_positions[number_of_bars::number_of_bars])


	# set ticks and labels
	ax.set_xticks( ensemble_xticks)
	ax.set_xticks( divisor_xticks, minor=True )
	ax.set_xticklabels(ensemble_xticklabels, fontsize=20)
	v_y = [-0.03] * len(ensemble_xticklabels)
	for t, y in zip( ax.get_xticklabels(), v_y ):
		t.set_y(y)

	ax.xaxis.labelpad = 15
	ax.set_xlabel(r'CCR', fontsize=20)
	ax.grid('off', axis='x', which='minor')
	ax.tick_params( axis='x', which='minor', direction='out', length=30, top='off')
	ax.tick_params( axis='x', which='major', bottom='on', top='off', length=2 )
	ax.set_xlim(right=bar_positions[-1] + bar_width )


	#################
	# general config#
	#################
	
	# Hide these grid behind plot objects
	ax.set_axisbelow(True)




	plt.show()

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, h


def create_data_field(ccrs, algorithms):
	return dict((ccr, dict((algorithm, dict({'mean': 0.0, 'ci': 0.0, 'values':list()})) for algorithm in algorithms)) for ccr in ccrs)


def calculate_stats(data, ccrs, algorithms):

	for ccr in ccrs:
		for algorithm in algorithms:

			values = data[ccr][algorithm]['values']

			#(mean , CI) = get_stats(values)
			(mean , CI) = mean_confidence_interval(values)

			data[ccr][algorithm]['mean'] = mean
			data[ccr][algorithm]['ci'] = CI

	return data

def parse_json(json_filename, ccrs, algorithms):

	makespans =  create_data_field(ccrs, algorithms)
	communications =  create_data_field(ccrs, algorithms)
	runtimes =  create_data_field(ccrs, algorithms)

	try:
		with open(json_filename) as file:
			try:
				json_data = json.load(file)

				for ccr in ccrs:
					
					for key in json_data[ccr][0].keys():

						raw_data = json_data[ccr][0][key]

						for result in raw_data:
							
							algorithm = result['flavour']
							
							makespan = result['makespan']
							runtime = result['runtime']
							communication = result['totalBytesSent']

							makespans[ccr][algorithm]['values'].append(makespan)
							runtimes[ccr][algorithm]['values'].append(runtime)
							communications[ccr][algorithm]['values'].append(communication)


			except (ValueError) as e:
				print ("Json parser error on file:  %s") % (e)

	except (IOError, OSError) as e:
		print ("file not found:  %s") % (e)


	## calculating averages and CIs
	makespans = calculate_stats(makespans, ccrs, algorithms)
	communications = calculate_stats(communications, ccrs, algorithms)
	runtimes = calculate_stats(runtimes, ccrs, algorithms)

	plot_bars(makespans, ccrs, algorithms)
	plot_bars(communications, ccrs, algorithms)


if __name__ == '__main__':


	algorithms = ['HEFT','HEFT-TaskDuplication','HEFT-LookAhead-TaskDuplication','HEFT-Ilia-W-0.05', 'HEFT-Ilia-W-0.10', 'HEFT-Ilia-W-0.50', 'HEFT-Ilia-W-0.90']
	app_names =  ['MONTAGE', 'CYBERSHAKE', 'GENOME', 'LIGO', 'SIPHT']
	app_sizes = ['50', '100', '300', '500', '1000']
	vm_files = ['heft.2.yaml', 'heft.10.yaml']
	ccrs = ['0.1', '0.5', '1.0', '1.5', '2.0']

	data_dir = '/local1/thiagogenez/mulitple-workflow-simulation'
	plots_dir = '/local1/thiagogenez/mulitple-workflow-simulation'


	parser = argparse.ArgumentParser(description='Simulator runner', add_help=True, prog='run.py', usage='python %(prog)s [options]', epilog='Mail me (thiagogenez@ic.unicamp.br) for more details', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--app_names', nargs='+', type=str, help='App names for simulations', action='store', default=app_names)
	parser.add_argument('--app_sizes', nargs='+', help='Size of the application', type=int, action='store', default=app_sizes)
	parser.add_argument('--vm_files', nargs='+', help='VM files', type=int, action='store', default=vm_files)
	parser.add_argument('--algorithms', nargs='+', type=str, help='Scheduling policies', action='store', default=algorithms)
	parser.add_argument('--version', action='version', version='%(prog)s 1.0')
	parser.add_argument('--ccrs', nargs='+', help='communication to computation ratios (CCR)', type=float, action='store', default=ccrs)
	parser.add_argument('--data_dir', nargs='?', help='Simulation data', type=str, action='store', default=data_dir)
	parser.add_argument('--plots_dir', nargs='?', help='Simulation data', type=str, action='store', default=plots_dir)

	# parsing arguments
	try:
		args = parser.parse_args()
	except IOError as ioerr:
		parser.print_usage()


	# prepare dir to receive the plot files
	output_dir = '%s/plots' % (args.plots_dir)
	mkdir(output_dir)


	for app_name in args.app_names:
		for app_size in args.app_sizes:
			for vm_file in args.vm_files:
				json_filename= '%s/%s/%s/%s/results/simulation.json' % (args.data_dir, app_name, app_size, vm_file)
				parse_json(json_filename, args.ccrs, args.algorithms)
