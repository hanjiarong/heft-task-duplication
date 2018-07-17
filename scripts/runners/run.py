import os, shlex, subprocess,errno, sys, time
import itertools, random, argparse, json
import multiprocessing as mp

from functools import partial


from utils.utils import call_java
from utils.utils import countdown
from utils.utils import create_output_directory
from utils.utils import start_process
from utils.utils import query_yes_no




""" GLOBAL VARIABLES """
ROOT_DIR = '/local1/thiagogenez/heft-task-duplication'
APP_DIR = '%s/data/dags' % (ROOT_DIR)
VMs_DIR = '%s/data/vms' % (ROOT_DIR)




def commify(list):
	return ','.join(list)

def create_args(output_dir, app_names, app_sizes, nodes, ccrs, vm_files, bandwidth_values, processing_capacity, algorithms, it_begin=0, iterations=20):

	args = list()
	dirs_to_make = set()

	for app_name in app_names:
		for app_size in app_sizes:
			for vm_file in vm_files:
				arg = str()

				simulation_output_path= '%s/%s/%s/%s/results' % (output_dir, app_name, app_size, vm_file)
				simulation_filename = 'simulation'

				arg = arg + '-app %s.n.%s.0.dag -app-dir %s ' % (app_name, app_size, APP_DIR)
				arg = arg + '-vms-dir %s --vm %s ' % (VMs_DIR, vm_file)
				arg = arg + '--bandwidth %s ' % (commify(bandwidth_values))
				arg = arg + '--ccrs %s ' % (commify(ccrs))
				arg = arg + '--fromDAGNumber %s ' % (it_begin)
				arg = arg + '--toDAGNumber %s ' % (it_begin + iterations)
				arg = arg + '--nodes %s ' % (commify(nodes))
				arg = arg + '--processing-capacity %s ' % (commify(processing_capacity))
				arg = arg + '--algorithms %s ' % (commify(algorithms))
				arg = arg + '--output %s ' % (simulation_output_path)
				arg = arg + '--outputFilenamePrefix %s ' % (simulation_filename)
					
				args.append(arg)

				dirs_to_make.add(simulation_output_path)

	return (dirs_to_make, args)





def main(output_dir, algorithms, vm_files, processing_capacity, bandwidth_values,\
		app_names, app_sizes, nodes, ccrs, iter_begin, iterations, xms,\
		xmx, cpu, JAR):

	#creating params
	print '>>> creating simulation arguments...'
	
	(dirs, args) = create_args(output_dir, app_names, app_sizes, nodes, ccrs, vm_files, bandwidth_values, processing_capacity, algorithms, it_begin=iter_begin, iterations=iterations)


	if len(args) == 0:
		print '>>> No simulation was created to run'
		exit(0)


	print '>>> creating output directories...'
	create_output_directory(dirs)

	# creating (dead)pool =P of workes
	pool = mp.Pool(processes=cpu,initializer=start_process)

	func = partial(call_java, xms, xmx, '', JAR, 'java')

	it = pool.imap(func, args)
	
	for result in it:
		print '\t', result

	pool.close()
	pool.join()

if __name__ == '__main__':


	algorithms = ['HEFT','HEFT-TaskDuplication','HEFT-LookAhead-TaskDuplication','HEFT-Ilia-W-0.05', 'HEFT-Ilia-W-0.10']#, 'HEFT-Ilia-W-0.50', 'HEFT-Ilia-W-0.90']
	app_names =  ['MONTAGE', 'CYBERSHAKE', 'GENOME', 'LIGO']#, 'SIPHT']
	app_sizes = ['50', '100', '500', '1000']

	vm_files = ['heft.5.yaml', 'heft.10.yaml', 'heft.15.yaml']

	nodes = ['500', '5000']
	ccrs = ['0.1', '0.5', '1.0', '2.0', '5.0', '10.0']

	processing_capacity = ['10', '100']
	bandwidth_values = ['10', '100'] 

	parser = argparse.ArgumentParser(description='Simulator runner', add_help=True, prog='run.py', usage='python %(prog)s [options]', epilog='Mail me (thiagogenez@ic.unicamp.br) for more details', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--app_names', nargs='+', type=str, help='App names for simulations', action='store', default=app_names)
	parser.add_argument('--app_sizes', nargs='+', help='Size of the application', type=int, action='store', default=app_sizes)
	parser.add_argument('--vm_files', nargs='+', help='VM files', type=int, action='store', default=vm_files)
	parser.add_argument('--algorithms', nargs='+', type=str, help='Scheduling policies', action='store', default=algorithms)
	parser.add_argument('--xms', nargs='?', help='The initial memory allocation pool for a Java Virtual Machine (JVM)', type=int, action='store', default=4)
	parser.add_argument('--xmx', nargs='?', help='The maximum memory allocation pool for a Java Virtual Machine (JVM)', type=int, action='store', default=8)
	parser.add_argument('--cpu', nargs='?', help='Number of CPUs', type=int,  action='store', default=mp.cpu_count())
	parser.add_argument('--iterations', nargs='?', help='Simulation iterations', type=int, action='store', default=20)
	parser.add_argument('--iter_begin', nargs='?', help='Simulation iterations', type=int, action='store', default=0)
	parser.add_argument('--version', action='version', version='%(prog)s 1.0')
	parser.add_argument('--nodes', nargs='+', help='Nodes range', type=int, action='store', default=nodes)
	parser.add_argument('--ccrs', nargs='+', help='communication to computation ratios (CCR)', type=float, action='store', default=ccrs)
	parser.add_argument('--bandwidth_values', nargs='+', help='Bandwidth values', type=int, action='store', default=bandwidth_values)
	parser.add_argument('--processing_capacity', nargs='+', help='Processing Capacities', type=int, action='store', default=processing_capacity)

	# parsing arguments
	try:
		args = parser.parse_args()
	except IOError as ioerr:
		parser.print_usage()


	JAR = '%s/jar/HEFTs.jar-jar-with-dependencies.jar' % (ROOT_DIR)
	output_dir = '%s/simulations-outputs' % (ROOT_DIR)

	print args
	#countdown(5)



	
	main(output_dir, args.algorithms, args.vm_files, args.processing_capacity, args.bandwidth_values,\
		args.app_names, args.app_sizes, args.nodes, args.ccrs, args.iter_begin, args.iterations, args.xms,\
		args.xmx, args.cpu, JAR)



	# python call
	# python run.py --iterations 100 --iter_begin 0 