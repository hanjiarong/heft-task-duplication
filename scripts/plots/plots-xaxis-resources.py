import argparse, json

from utils.utils import mkdir
from utils.utils import get_stats

import numpy as np
import scipy as sp
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


from matplotlib.ticker import AutoMinorLocator
from matplotlib.ticker import FuncFormatter

def y_fmt(y, pos):
	decades = [1e9, 1e6, 1e3, 1e0, 1e-3, 1e-6, 1e-9 ]
	suffix  = ["G", "M", "k", "" , "m" , "u", "n"  ]
	if y == 0:
		return str(0)
	for i, d in enumerate(decades):
		if np.abs(y) >=d:
			val = y/float(d)
			signf = len(str(val).split(".")[1])
			if signf == 0:
				return '{val:d} {suffix}'.format(val=int(val), suffix=suffix[i])
			else:
				if signf == 1:
					if str(val).split(".")[1] == "0":
						return '{val:d} {suffix}'.format(val=int(round(val)), suffix=suffix[i]) 
				tx = "{"+"val:.{signf}f".format(signf = signf) +"} {suffix}"
				return tx.format(val=val, suffix=suffix[i])
	return y


def get_correct_legend(labels):

	new_labels = list()


	for label in labels:

		if 'HEFT' == label:
			new_labels.append(r'HEFT')

		elif 'HEFT-TaskDuplication' == label:
			new_labels.append(r'TaskDuplication')			

		elif 'HEFT-LookAhead-TaskDuplication' == label:
			new_labels.append(r'LookAhead-TaskDuplication')

		elif 'HEFT-Ilia-W-0.05' == label:
			new_labels.append(r'W-$0.05$')

		elif 'HEFT-Ilia-W-0.10' == label:
			new_labels.append(r'W-$0.10$')

		elif 'HEFT-Ilia-W-0.50' == label:
			new_labels.append(r'W-$0.50$')

		elif 'HEFT-Ilia-W-0.90' == label:
			new_labels.append(r'W-$0.90$')

		else:
			new_labels.append('Unknown ylabel')


	return new_labels

def plot_errorbars(data, resources, algorithms, title='Title', ylabel='y_label', plot_filename='/tmp/plot', format='pdf'):

	#fig, ax = plt.subplots()
	fig, ax = plt.subplots(figsize=(10, 5))

	minorLocator = AutoMinorLocator()

	x = [float(resource) for resource in resources]
	errorbars = list()

	#confg
	linestyle = ['solid', 'dashed', 'dashdot', 'dotted', 'solid']
	markers = ['v', 'o', 'x', 'D', '*']
	colours = ['black', 'red', 'blue', 'green', 'purple']

	#preparing data
	
	for i in range(len(algorithms)):	
		y = list()
		yerr = list()

		algorithm = algorithms[i]

		for resource in resources:	

			mean = data[resource][algorithm]['mean']
			ci = data[resource][algorithm]['ci']

			y.append(mean)
			yerr.append(ci)

		errorbar = ax.errorbar(x, y, yerr=yerr, ls=linestyle[i], marker=markers[i], color=colours[i], markerfacecolor="None", capsize=5, markersize=6, linewidth=1)
		errorbars.append(errorbar[0])

	#################
	# x-axis config #
	#################
	ax.xaxis.labelpad = 15
	ax.set_xticks([float(resource) for resource in resources])
	ax.set_xticklabels([int(resource) for resource in resources])
	ax.set_xlabel(r'Number of Resources', fontsize=14)

	#################
	# y-axis config #
	#################
	ax.set_ylabel(ylabel, fontsize=14)
	ax.yaxis.labelpad = 9
	ax.yaxis.set_minor_locator(minorLocator)
	#ax.yaxis.set_major_formatter(FuncFormatter(y_fmt))
	

	#################
	# general config#
	#################
	
	ax.set_title(title, fontsize=17)

	# Hide these grid behind plot objects
	ax.set_axisbelow(True)

	# Legend
	plt.legend(errorbars, get_correct_legend(algorithms), loc='best', ncol=1, numpoints=2, fontsize=10)


	
	# tight layout
	plt.tight_layout(True)
	
	#grid
	ax.grid('on', axis='both', ls='dashed')

	#################
	#     saving    #
	#################
	plt.savefig( '%s-errobar.%s' % (plot_filename, format), format=format, bbox_inches='tight') #dpi=300,  rasterized=True)		
	plt.clf()
	plt.close('all')



def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, h


def create_data_field(resources, algorithms):
	return dict((resource, dict((algorithm, dict({'mean': 0.0, 'ci': 0.0, 'values':list()})) for algorithm in algorithms)) for resource in resources)


def calculate_stats(data, resources, algorithms):

	for resource in resources:
		for algorithm in algorithms:

			values = data[resource][algorithm]['values']

			#(mean , CI) = get_stats(values)
			(mean , CI) = mean_confidence_interval(values)

			data[resource][algorithm]['mean'] = mean
			data[resource][algorithm]['ci'] = CI

	return data

def parse_json(resources, algorithms, data_dir, app_name, app_size, ccr):

	makespans =  create_data_field(resources, algorithms)
	communications =  create_data_field(resources, algorithms)
	runtimes =  create_data_field(resources, algorithms)
	duplicatas = create_data_field(resources, algorithms)

	for resource in resources:

		json_filename = '%s/%s/%s/%s/results/simulation.json' % (args.data_dir, app_name, app_size, resource)

		try:
			with open(json_filename) as file:
				try:
					json_data = json.load(file)

					
					for simulation_number in json_data[ccr][0].keys():

						raw_data = json_data[ccr][0][simulation_number]

						for result in raw_data:
							
							algorithm = result['flavour']

							if algorithm in algorithms:
							
								makespan = result['makespan']
								runtime = result['runtime']
								communication = result['totalBytesSent']
								duplicata = result['duplicatas']

								makespans[resource][algorithm]['values'].append(makespan)
								runtimes[resource][algorithm]['values'].append(runtime)
								communications[resource][algorithm]['values'].append(communication)
								duplicatas[resource][algorithm]['values'].append(duplicata)


				except (ValueError) as e:
					print ("Json parser error on file:  %s") % (e)

		except (IOError, OSError) as e:
			print ("file not found:  %s") % (e)

	## calculating averages and CIs
	makespans = calculate_stats(makespans, resources, algorithms)
	communications = calculate_stats(communications, resources, algorithms)
	runtimes = calculate_stats(runtimes, resources, algorithms)
	duplicatas = calculate_stats(duplicatas, resources, algorithms)

	return (makespans, communications, runtimes, duplicatas)


def calculate_relative(data, resources, algorithm, relative='HEFT'):

	import operator
	import functools

	# function to div element by element of 2 lists
	multi_div = functools.partial(map, operator.div)

	new_data = create_data_field(resources, algorithms)

	for resource in resources:

		comparator_values = data[resource][relative]['values']

		for algorithm in algorithms:

			original_values = data[resource][algorithm]['values']

			new_values = multi_div(original_values, comparator_values)

			new_data[resource][algorithm]['values'] = new_values


	new_data = calculate_stats(new_data, resources, algorithms)

	return new_data




def call_plots(data, resources, algorithms, title='title', ylabel='y_label', plot_filename='plot'):


	plot_errorbars(data, resources, algorithms, ylabel=ylabel, title=title, plot_filename=plot_filename)
	#plot_bars(data, ccrs, algorithms, ylabel=ylabel, title=title, plot_filename=plot_filename)
	


if __name__ == '__main__':


	algorithms = ['HEFT','HEFT-Ilia-W-0.05', 'HEFT-TaskDuplication','HEFT-LookAhead-TaskDuplication'] #,'HEFT-Ilia-W-0.10', 'HEFT-Ilia-W-0.50', 'HEFT-Ilia-W-0.90']
	app_names =  ['MONTAGE', 'CYBERSHAKE', 'GENOME', 'LIGO', 'SIPHT']
	app_sizes = ['50', '100', '300', '500', '1000']
	resources = ['5', '10', '15', '20', '25', '30', '35']
	ccrs = ['0.1', '0.5', '1.0', '2.0', '5.0', '10.0']

	data_dir = '/local1/thiagogenez/mulitple-workflow-simulation'


	parser = argparse.ArgumentParser(description='Simulator runner', add_help=True, prog='run.py', usage='python %(prog)s [options]', epilog='Mail me (thiagogenez@ic.unicamp.br) for more details', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--app_names', nargs='+', type=str, help='App names for simulations', action='store', default=app_names)
	parser.add_argument('--app_sizes', nargs='+', help='Size of the application', type=str, action='store', default=app_sizes)
	parser.add_argument('--resources', nargs='+', help='Size of the application', type=str, action='store', default=resources)
	parser.add_argument('--algorithms', nargs='+', type=str, help='Scheduling policies', action='store', default=algorithms)
	parser.add_argument('--version', action='version', version='%(prog)s 1.0')
	parser.add_argument('--ccrs', nargs='+', help='communication to computation ratios (CCR)', type=str, action='store', default=ccrs)
	parser.add_argument('--data_dir', nargs='?', help='Simulation data', type=str, action='store', default=data_dir)
	

	# parsing arguments
	try:
		args = parser.parse_args()
	except IOError as ioerr:
		parser.print_usage()


	

	for app_name in args.app_names:
		for app_size in args.app_sizes:
			for ccr in args.ccrs:

				# prepare dir to receive the plot files
				output_dir = '%s/plots/xaxis-resources/%s' % (args.data_dir, app_name)
				mkdir(output_dir)
				
				title = '%s with %s tasks - CCR %s' % (app_name, app_size, ccr)
				plot_filename = '%s-%s-%s' % (app_name, app_size, ccr)


				

				(makespans, communications, runtimes, duplicatas) = parse_json(args.resources, args.algorithms, args.data_dir, app_name, app_size, ccr)
				

				call_plots(makespans, args.resources, algorithms, title=title, ylabel='Average makespan', plot_filename='%s/makespan-%s' % (output_dir, plot_filename))
				call_plots(communications, args.resources, algorithms, title=title, ylabel='Average makespan', plot_filename='%s/communication-%s' % (output_dir, plot_filename))

				call_plots(duplicatas, args.resources, algorithms, title=title, ylabel='Average of Clones', plot_filename='%s/duplicatas-%s' % (output_dir, plot_filename))

				

				
				makespans_relative = calculate_relative(makespans, args.resources, algorithms)
				call_plots(makespans_relative, args.resources, algorithms, title=title, ylabel='Normalised makespan', plot_filename='%s/relative-makespan-%s' % (output_dir, plot_filename))

				
				communications_relative = calculate_relative(communications, args.resources, algorithms)
				call_plots(communications_relative, args.resources, algorithms, title=title, ylabel='Normalised communication cost', plot_filename='%s/relative-communication-%s' % (output_dir, plot_filename))

				#duplicatas_relative = calculate_relative(duplicatas, ccrs, ['HEFT-TaskDuplication','HEFT-LookAhead-TaskDuplication'])


