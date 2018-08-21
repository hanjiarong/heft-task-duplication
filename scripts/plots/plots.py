import argparse, json, os

from utils.utils import mkdir
from utils.utils import get_stats

import numpy as np
import scipy as sp
import scipy.stats
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
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

		elif 'HEFT-TaskDuplication2' == label:
			new_labels.append(r'HEFT-TD')	

		elif 'Flexible-Scheduler' == label:
			new_labels.append(r'Cost-aware')	

		elif 'HEFT-LookAhead-TaskDuplication' == label:
			new_labels.append(r'Lookahead-TD')

		elif 'HEFT-Ilia-W-0.05' == label:
			new_labels.append(r'HEFT with W=$0.05$')

		elif 'HEFT-Ilia-W-0.10' == label:
			new_labels.append(r'HEFT with W=$0.10$')

		elif 'HEFT-Ilia-W-0.50' == label:
			new_labels.append(r'HEFT with W=$0.50$')

		elif 'HEFT-Ilia-W-0.90' == label:
			new_labels.append(r'HEFT with W=$0.90$')

		elif 'HEFT-LookAhead' == label:
			new_labels.append(r'Lookahead')

		else:
			new_labels.append('Unknown ylabel')


	return new_labels




def plot_cdf(data, ccrs, algorithms, title='Title', ylabel='y_label', plot_filename='/tmp/plot', format='pdf', type='type'):

	import numpy as np
	#fig, ax = plt.subplots()
	

	linestyle = ['solid', 'dashed', 'dashdot', 'dotted', 'solid', 'solid']
	markers = ['v', 'o', 'x', 'D', '*', 'v']
	colours = ['black', 'red', 'blue', 'green', 'purple', 'orange', 'magenta']

	
	for ccr in ccrs:	
		
		i = 0
		#fig, ax = plt.subplots(figsize=(10, 5))
		fig, ax = plt.subplots()
		minorLocator = AutoMinorLocator()

		for i in range(len(algorithms)):	

	
			algorithm = algorithms[i]

			X = data[ccr][algorithm]['values'] 
			Xs = np.sort(X)  # Or data.sort(), if data can be modified

			#method 1
			#n_bins=len(Xs)
			#counts, bin_edges, patches = ax[0].hist(X, bins=n_bins,  histtype='step', cumulative=True, normed=True)

			#method 2
			y = np.arange(1, len(X) + 1) / np.float(len(X))
			ax.plot(Xs, y, color=colours[i], label=get_correct_legend([algorithm])[0], ls=linestyle[i], lw=1)


			#method 3
			#cnt, edges = np.histogram(X, bins=n_bins, normed=1, density=False)
			#ax[2].step(edges[:-1], cnt.cumsum())

			i = i + 1

		#################
		# x-axis config #
		#################
		#ax.xaxis.labelpad = 7
		ax.set_xlabel('%s - CCR %s' % (ylabel, ccr), fontsize=16)

		#################
		# y-axis config #
		#################
		ax.set_ylabel(r'CDF', fontsize=16)
		#ax.yaxis.labelpad = 0
		ax.yaxis.set_minor_locator(minorLocator)
		#ax.yaxis.set_major_formatter(FuncFormatter(y_fmt))
		if 'communication' in type:
			ax.xaxis.set_major_formatter(FuncFormatter(y_fmt))
		

		ax.tick_params(axis='x', labelsize=13, pad=7)
		ax.tick_params(axis='y', labelsize=14, pad=0)

		#################
		# general config#
		#################
		ax.set_title(r'%s' % (title), fontsize=16)
		plt.margins(0.02) # keeps data off plot
		ax.set_axisbelow(True) # Hide these grid behind plot objects
		plt.legend(loc='best', ncol=1, numpoints=2, fontsize=14)		
		plt.tight_layout(True) # tight layout
		ax.grid('on', axis='both', ls='dashed') #grid
			
		#################
		#     saving    #
		#################
		#plt.subplots_adjust(left=0.0, right=0.9, top=0.9, bottom=0.0)
		plt.savefig( '%s-%s-CDF.%s' % (plot_filename, ccr, format), format=format, bbox_inches='tight', pad_inches=0.05) #dpi=300,  rasterized=True)		
		plt.clf()
		plt.close('all')
		


def plot_errorbars(data, ccrs, algorithms, title='Title', ylabel='y_label', plot_filename='/tmp/plot', format='pdf', type='type'):

	#fig, ax = plt.subplots()
	#fig, ax = plt.subplots(figsize=(10, 5))
	fig, ax = plt.subplots()
	minorLocator = AutoMinorLocator()

	x = [float(ccr) for ccr in ccrs]
	errorbars = list()

	#confg
	linestyle = ['solid', 'dashed', 'dashdot', 'dotted', 'solid', 'dashed', 'dashdot']
	markers = ['v', 'o', 'x', 'D', '*', '>', ',', '^', 'H']
	colours = ['black', 'red', 'blue', 'green', 'purple', 'orange', 'magenta']

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
	#ax.xaxis.labelpad = 0
	ax.set_xticks([float(ccr) for ccr in ccrs])
	ax.set_xticklabels([float(ccr) for ccr in ccrs])
	ax.set_xlabel(r'CCR', fontsize=16)

	#################
	# y-axis config #
	#################
	ax.set_ylabel(r'%s' % (ylabel), fontsize=16)
	#ax.yaxis.labelpad = 5
	ax.yaxis.set_minor_locator(minorLocator)
	if 'communication' in type:
		ax.yaxis.set_major_formatter(FuncFormatter(y_fmt))

	
	ax.tick_params(axis='x', labelsize=13, pad=0)
	plt.xticks(rotation=40)
	ax.tick_params(axis='y', labelsize=14, pad=0)

	#################
	# general config#
	#################
	
	ax.set_title(r'%s' % (title), fontsize=16)

	# Hide these grid behind plot objects
	ax.set_axisbelow(True)



	# Legend
	plt.legend(errorbars, get_correct_legend(algorithms), loc='best', ncol=1, numpoints=2, fontsize=14)


	
	# tight layout
	plt.tight_layout(True)
	
	#grid
	ax.grid('on', axis='both', ls='dashed')

	#################
	#     saving    #
	#################
	#plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.0)
	plt.savefig( '%s-errobar.%s' % (plot_filename, format), format=format, bbox_inches='tight', pad_inches=0.01) #dpi=300,  rasterized=True)		
	plt.clf()
	plt.close('all')

def plot_bars(data, app_names, algorithms, title='Title', xlabel='x_label', ylabel='y_label', plot_filename='/tmp/plot', format='pdf'):

	# plot
	#fig, ax = plt.subplots(figsize=(13, 6))
	fig, ax = plt.subplots()

	# variables
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
	for app_name in app_names:
		for algorithm in algorithms:

			mean = data[app_name][algorithm]['mean']
			ci = data[app_name][algorithm]['ci']
			
			bar_means.append(mean)
			bar_cis.append(ci)

			position += bar_width
			bar_positions.append(position)

		position += bar_width

	# bar configs
	rects = list()
	#colors = ['white', 'white',  'white', 'black', 'white', 'deepskyblue', 'silver', 'silver']
	colours = ['black', 'red', 'blue', 'green', 'purple', 'orange', 'magenta']
	error_config = {'ecolor': '0.0'}
	patterns = (' ', '..',  'x',  '--', 'xxx', '\\', '+', '.')

	# creating rects
	for i in range(number_of_bars):
		rect = ax.bar(bar_positions[i::number_of_bars], bar_means[i::number_of_bars], bar_width,
			hatch=patterns[i],
			alpha=0.7,
			color=colours[i],
			yerr=bar_cis[i::number_of_bars],
			error_kw=error_config,
			capsize=5,
			edgecolor='black',
			#bottom=min(bar_means) - 0.20 *min(bar_means),
			label='%s' % (algorithms[i]))
		rects.append(rect)

	#################
	# x-axis config #
	#################
	
	# ensembles tick labels
	xticklabels = [r'$%s$' % (i)  for i in app_names]
	xticks = map(lambda x: x  + ((number_of_bars * bar_width)/2), bar_positions[::number_of_bars])

	
	# divisor line between ensenbles sizes
	divisor_xticks = map(lambda x: x - bar_width/2 , bar_positions[number_of_bars::number_of_bars])


	# set ticks and labels
	ax.set_xticks( xticks)
	ax.set_xticks( divisor_xticks, minor=True )
	ax.set_xticklabels(xticklabels, fontsize=14)
	v_y = [-0.03] * len(xticklabels)
	for t, y in zip( ax.get_xticklabels(), v_y ):
		t.set_y(y)

	ax.xaxis.labelpad = 10
	ax.set_xlabel(xlabel, fontsize=15)
	ax.grid('off', axis='x', which='minor')
	ax.tick_params( axis='x', which='minor', direction='out', length=1, top='off')
	ax.tick_params( axis='x', which='major', bottom='on', top='off', length=0 )
	ax.set_xlim(right=bar_positions[-1] + bar_width )

	#################
	# y-axis config #
	#################

	ax.set_ylabel(ylabel, fontsize=14)
	ax.yaxis.labelpad = 9
	ax.yaxis.grid(color='black',linestyle='dotted', linewidth=1)
	ax.yaxis.set_minor_locator(minorLocator)
	#ax.yaxis.set_major_formatter(FuncFormatter(y_fmt))
	ax.tick_params( axis='both', labelsize=14)
	ax.set_ybound(upper=2)

	#################
	# general config#
	#################
	
	ax.set_title(title, fontsize=17)

	# Hide these grid behind plot objects
	ax.set_axisbelow(True)

	# log scale
	ax.set_yscale('log')

	# Legend
	plt.legend(rects, get_correct_legend(algorithms), loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=5, fontsize=12)

	
	# tight layout
	plt.tight_layout(True)


	#################
	#     saving    #
	#################
	plt.savefig( '%s-bar.%s' % (plot_filename, format), format=format, bbox_inches='tight') #dpi=300,  rasterized=True)		
	plt.clf()
	plt.close('all')

def mean_confidence_interval(data, confidence=0.95):
	a = 1.0*np.array(data)
	n = len(a)
	m, se = np.mean(a), scipy.stats.sem(a)
	h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
	return m, h


def create_data_field(field1, field2):
	return dict((f1, dict((f2, dict({'mean': 0.0, 'ci': 0.0, 'values':list()})) for f2 in field2)) for f1 in field1)


def calculate_stats(data, filed1, field2):

	for f1 in filed1:
		for f2 in field2:

			values = data[f1][f2]['values']

			#(mean , CI) = get_stats(values)
			(mean , CI) = mean_confidence_interval(values)

			data[f1][f2]['mean'] = mean
			data[f1][f2]['ci'] = CI

	return data

def parse_json(json_filename, ccrs, algorithms, runtimes, app_name):

	makespans =  create_data_field(ccrs, algorithms)
	communications =  create_data_field(ccrs, algorithms)
	duplicatas = create_data_field(ccrs, algorithms)
	cost = create_data_field(ccrs, algorithms)
	reschedules = create_data_field(ccrs, algorithms)

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

								
								#if int(makespan_scheduled) != int(makespan_simulated):
								#	print 'diferent makespan, simulated %s\t scheduled %s\t' % (makespan_scheduled, makespan_simulated)

								
								communication_scheduled = result['totalBytesSent-scheduled']
								communication_simulated = result['totalBytesSent-simulated']

								cost_simulated = result['cost-simulated']


								#if communication_scheduled != communication_simulated:
								#	print 'diferent communication, simulated %s\t scheduled %s\t' % (communication_scheduled, communication_simulated)
								

								runtime = result['runtime']
								duplicata = result['duplicatas']
								rescheduled = result['rescheduled']

	


								makespans[ccr][algorithm]['values'].append(makespan_simulated)
								communications[ccr][algorithm]['values'].append(communication_simulated)
								cost[ccr][algorithm]['values'].append(cost_simulated)

								
								duplicatas[ccr][algorithm]['values'].append(duplicata)
								reschedules[ccr][algorithm]['values'].append(rescheduled)

								runtimes[app_name][algorithm]['values'].append(runtime)
								


			except (ValueError) as e:
				print ("Json parser error on file:  %s") % (e)

	except (IOError, OSError) as e:
		print ("file not found:  %s") % (e)

	## calculating averages and CIs
	makespans = calculate_stats(makespans, ccrs, algorithms)
	communications = calculate_stats(communications, ccrs, algorithms)
	duplicatas = calculate_stats(duplicatas, ccrs, algorithms)
	cost = calculate_stats(cost, ccrs, algorithms)
	reschedules = calculate_stats(reschedules, ccrs, algorithms)

	return (makespans, communications, duplicatas, reschedules, cost, runtimes)


def calculate_relative(data, ccrs, algorithm, relative='HEFT'):

	import operator
	import functools

	# function to div element by element of 2 lists
	multi_div = functools.partial(map, operator.div)

	new_data = create_data_field(ccrs, algorithms)

	for ccr in ccrs:

		comparator_values = data[ccr][relative]['values']

		mean = data[ccr][relative]['mean']

		comparator_values = [mean if x <= 0.0 else x for x in comparator_values]

		for algorithm in algorithms:

			original_values = data[ccr][algorithm]['values']


			new_values = multi_div(original_values, comparator_values)

			new_data[ccr][algorithm]['values'] = new_values


	new_data = calculate_stats(new_data, ccrs, algorithms)

	return new_data



def calculate_ratio(data1, data2, ccrs, algorithms):

	import operator
	import functools

	# function to div element by element of 2 lists
	multi_div = functools.partial(map, operator.div)

	new_data = create_data_field(ccrs, algorithms)

	for ccr in ccrs:

		for algorithm in algorithms:

			values1 = data1[ccr][algorithm]['values']

			values2 = data2[ccr][algorithm]['values']

			new_values = multi_div(values1, values2)

			new_data[ccr][algorithm]['values'] = new_values

	new_data = calculate_stats(new_data, ccrs, algorithms)

	return new_data



if __name__ == '__main__':


	algorithms = ['HEFT','HEFT-TaskDuplication2','HEFT-LookAhead', 'HEFT-LookAhead-TaskDuplication'] #,'HEFT-Ilia-W-0.50', ]# 'Flexible-Scheduler' 'HEFT-TaskDuplication',  'HEFT-Ilia-W-0.10', 'HEFT-LookAhead']#, 'HEFT-Ilia-W-0.50', 'HEFT-Ilia-W-0.90']
	app_names =  ['MONTAGE', 'CYBERSHAKE', 'GENOME', 'LIGO', 'SIPHT', 'FORKJOIN.A.1'] #, 'FORKJOIN.A.2' ]
	app_sizes = ['50', '100', '500', '1000']
	resources = ['5', '10', '15', '20', '25', '30', '35'] #'2', 
	ccrs = ['0.5', '1.0', '2.0', '5.0', '10.0'] #'0.1', 

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

	

	
	runtimes =  create_data_field(args.app_names, args.algorithms)

	for app_size in args.app_sizes:
		for app_name in args.app_names:
			for resource in args.resources:

				# prepare dir to receive the plot files
				output_dir = '%s/plots/xaxis-ccr/%s/%s' % (args.data_dir, app_name, app_size)
				mkdir(output_dir)

				
				## check err
				
				err_file  = '%s/%s/%s/%s/runners/simulation.err' % (args.data_dir, app_name, app_size, resource)
				if os.path.getsize(err_file) != 0:
					print 'ERR: %s' % (err_file)

				json_filename = '%s/%s/%s/%s/results/simulation.json' % (args.data_dir, app_name, app_size, resource)
				
				title = '%s with %s tasks - %s resources' % (app_name, app_size, resource)
				plot_filename = '%s-%s-%s' % (app_name, app_size, resource)


				
				(makespans, communications, duplicatas, rescheduled, cost, runtimes) = parse_json(json_filename, args.ccrs, args.algorithms, runtimes, app_name)


				# call cdf plots (separatelly)
				mkdir('%s/cdf' % output_dir)	
				plot_cdf(makespans, ccrs, algorithms, ylabel='Makespan', title=title, plot_filename='%s/cdf/makespan-%s' % (output_dir, plot_filename))
				plot_cdf(communications, ccrs, algorithms, ylabel='Amount of data transfer', title=title, plot_filename='%s/cdf/communication-%s' % (output_dir, plot_filename), type='communication')
				plot_cdf(cost, ccrs, algorithms, ylabel='Execution Cost', title=title, plot_filename='%s/cdf/cost-%s' % (output_dir, plot_filename))
				
				#plot_cdf(duplicatas, ccrs, algorithms, ylabel='Number of duplicate tasks', title=title, plot_filename='%s/cdf/duplicatas-%s' % (output_dir, plot_filename))
				#plot_cdf(rescheduled, ccrs, algorithms, ylabel='Number of tasks rescheduled', title=title, plot_filename='%s/cdf/rescheduled-%s' % (output_dir, plot_filename))
				

				
				# normal plots
				plot_errorbars(makespans, args.ccrs, args.algorithms, title=title, ylabel='Average makespan', plot_filename='%s/makespan-%s' % (output_dir, plot_filename))
				plot_errorbars(communications, args.ccrs, args.algorithms, title=title, ylabel='Average amount of data transfer', plot_filename='%s/communication-%s' % (output_dir, plot_filename), type='communication')
				plot_errorbars(cost, args.ccrs, args.algorithms, title=title, ylabel='Average Execution Cost (\\$)', plot_filename='%s/cost-%s' % (output_dir, plot_filename))

				plot_errorbars(duplicatas, args.ccrs, args.algorithms, title=title, ylabel='Average of duplicates', plot_filename='%s/duplicatas-%s' % (output_dir, plot_filename))
				plot_errorbars(rescheduled, args.ccrs, args.algorithms, title=title, ylabel='Average of rescheduled tasks', plot_filename='%s/rescheduled-%s' % (output_dir, plot_filename))
				

	#runtimes = calculate_stats(runtimes, args.app_names, args.algorithms)
	#output_dir = '%s/plots/xaxis-ccr/' % (args.data_dir)
	#plot_bars(runtimes, args.app_names, algorithms, xlabel='Workflow Applications', ylabel='Average execution time (ms)', title='', plot_filename='%s/general-runtime' % (output_dir))


				#### OLD BELLOW

				# ratio
				##cost_makespan_ratio = calculate_ratio(makespans, cost,  args.ccrs, args.algorithms)
				##call_plots(cost_makespan_ratio, args.ccrs, args.algorithms, title=title, ylabel='$\\frac{\\mbox{Execution cost}}{\\mbox{Makespan}}$ Ratio', plot_filename='%s/makespan-cost-ratio-%s' % (output_dir, plot_filename))

				##makespan_communication_ratio = calculate_ratio(communications, makespans, args.ccrs, args.algorithms)
				##call_plots(makespan_communication_ratio, args.ccrs, args.algorithms, title=title, ylabel='$\\frac{\\mbox{Makespan}}{\mbox{Communication}}$ Ratio', plot_filename='%s/communication-makespan-ratio-%s' % (output_dir, plot_filename))

				


				#makespans_relative = calculate_relative(makespans, args.ccrs, args.algorithms, relative='HEFT')
				#call_plots(makespans_relative, args.ccrs, args.algorithms, title=title, ylabel='Normalised makespan', plot_filename='%s/relative-makespan-%s' % (output_dir, plot_filename))

				#communications_relative = calculate_relative(communications, args.ccrs, args.algorithms,  relative='HEFT')
				#call_plots(communications_relative, args.ccrs, args.algorithms, title=title, ylabel='Normalised communication cost', plot_filename='%s/relative-communication-%s' % (output_dir, plot_filename))

				#cost_relative = calculate_relative(cost, args.ccrs, args.algorithms,  relative='HEFT')
				#call_plots(cost_relative, args.ccrs, args.algorithms, title=title, ylabel='Normalised execution cost (\\$)', plot_filename='%s/relative-cost-%s' % (output_dir, plot_filename))

				
				
