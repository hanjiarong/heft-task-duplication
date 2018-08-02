

import argparse, json

from utils.utils import mkdir
from utils.utils import get_stats

import numpy as np
import scipy as sp
import pandas as pd
import scipy.stats


import seaborn as sns
sns.set(style="white")


def plot_errorbars(data, ccrs, algorithm, title='Title', ylabel='y_label', plot_filename='/tmp/plot', format='pdf'):

	#fig, ax = plt.subplots()
	fig, ax = plt.subplots(figsize=(10, 5))

	minorLocator = AutoMinorLocator()

	x = [float(ccr) for ccr in ccrs]
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

		for ccr in ccrs:	

			mean = data[ccr][algorithm]['mean']
			ci = data[ccr][algorithm]['ci']

			y.append(mean)
			yerr.append(ci)

		errorbar = ax.errorbar(x, y, yerr=yerr, ls=linestyle[i], marker=markers[i], color=colours[i], markerfacecolor="None", capsize=5, markersize=6, linewidth=1)
		errorbars.append(errorbar[0])

	#################
	# x-axis config #
	#################
	ax.xaxis.labelpad = 15
	ax.set_xticks([float(ccr) for ccr in ccrs])
	ax.set_xticklabels([float(ccr) for ccr in ccrs])
	ax.set_xlabel(r'CCR', fontsize=14)

	#################
	# y-axis config #
	#################
	ax.set_ylabel(r'%s' % (ylabel), fontsize=14)
	ax.yaxis.labelpad = 9
	ax.yaxis.set_minor_locator(minorLocator)
	#ax.yaxis.set_major_formatter(FuncFormatter(y_fmt))
	

	#################
	# general config#
	#################
	
	ax.set_title(r'%s' % (title), fontsize=17)

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
	cost = create_data_field(ccrs, algorithms)

	try:
		with open(json_filename) as file:
			try:
				json_data = json.load(file)

				for ccr in ccrs:
					
					for simulation_number in json_data[ccr][0].keys():

						raw_data = json_data[ccr][0][simulation_number]

						for result in raw_data:
							
							algorithm = result['flavour']

							if algorithm in algorithms:
							
								makespan_scheduled = result['makespan-scheduled']
								makespan_simulated = result['makespan-simulated']

								
								if int(makespan_scheduled) != int(makespan_simulated):
									print 'diferent makespan, simulated %s\t scheduled %s\t' % (makespan_scheduled, makespan_simulated)


							
								cost_simulated = result['cost-simulated']

								makespans[ccr][algorithm]['values'].append(makespan_scheduled)
								cost[ccr][algorithm]['values'].append(cost_simulated)

						
								


			except (ValueError) as e:
				print ("Json parser error on file:  %s") % (e)

	except (IOError, OSError) as e:
		print ("file not found:  %s") % (e)

	## calculating averages and CIs
	makespans = calculate_stats(makespans, ccrs, algorithms)
	cost = calculate_stats(cost, ccrs, algorithms)

	return (makespans, cost)



if __name__ == '__main__':


	algorithms = ['HEFT','HEFT-Ilia-W-0.05', 'HEFT-TaskDuplication','HEFT-LookAhead','HEFT-LookAhead-TaskDuplication'] #,'HEFT-Ilia-W-0.10', 'HEFT-Ilia-W-0.50', 'HEFT-Ilia-W-0.90']
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

	

	for app_size in args.app_sizes:
		for app_name in args.app_names:

			# prepare dir to receive the plot files
			output_dir = '%s/plots/pareto/%s' % (args.data_dir, app_name)
			mkdir(output_dir)

			pareto = dict((resource, dict({'cost': None, 'makespan': None})) for resource in args.resources)

			

			for resource in args.resources:
				
				json_filename = '%s/%s/%s/%s/results/simulation.json' % (args.data_dir, app_name, app_size, resource)
				
				(pareto[resource]['makespan'], pareto[resource]['cost']) = parse_json(json_filename, args.ccrs, args.algorithms)


			for ccr in args.ccrs:

				df = pd.DataFrame(columns=['cost', 'makespan', 'resource', 'algorithm'])
				
				i = 0
				for algorithm in args.algorithms:

			
					for resource in args.resources:


						df2 = pd.DataFrame({'cost':pareto[resource]['cost'][ccr][algorithm]['values'],
							'makespan':pareto[resource]['makespan'][ccr][algorithm]['values'],
							'resource': float(resource), 
							'algorithm': algorithm})


						df = df.append(df2, ignore_index=True)

						#sns.pairplot(df2, vars=["cost", "makespan"])
						#import matplotlib.pyplot as plt
						#plt.show()
					i = i + 1
				print df.dtypes

				print df.dtypes['algorithm']

				#sns.pairplot(df, hue="resource")
				df['resource'].astype('int64')
				
				

				#sns.relplot(x="makespan", y="cost", hue="algorithm", size="resource", sizes=(30,300),  alpha=.4, height=6, data=df)
				sns.lineplot(x="makespan", y="cost", hue="algorithm", truncate=True, height=5, data=df)
				
				import matplotlib.pyplot as plt
				plt.show()
				
				#exit(0)

