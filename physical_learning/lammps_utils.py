import numpy as np
import os
import glob
from plot_imports import *
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from allosteric_utils import Allosteric

def read_dump(filename):
	'''Read a LAMMPS dumpfile.
	   
	Parameters
	----------
	filename : str
		The name of the file to read.

	Returns
	-------
	ndarray
		The (x,y) coordinates for each of n points over all output frames.
	'''

	with open(filename, 'r') as f:
		lines = f.readlines()
		n = int(lines[0].split()[0])
		m = n+2
		frames = len(lines)//m

	traj = np.zeros((frames,n,3))
	for fr in range(frames):
		for i in range(n):
			line = lines[m*fr+2+i]
			x, y, z = np.array(line.strip().split()).astype(float)
			traj[fr,i,0] = x
			traj[fr,i,1] = y
			traj[fr,i,2] = z
	return traj

def read_log(filename):
	'''Read a LAMMPS log file.
	   
	Parameters
	----------
	filename : str
		The name of the file to read.

	Returns
	-------
	data : ndarray
		The log data (typically printed to screen) of the simulation at each
		integration timestep.
	cols : list of str
		The column names associated with each data column.
	'''

	with open(filename, 'r') as f:
		lines = f.readlines()
		i = 0
		line  = lines[i].strip().split()
		while (len(line) < 1) or not ((line[0] == 'variable') and (line[1] == 'frames')):
			i += 1
			line  = lines[i].strip().split()
		nstep = int(line[3]) + 1
		while (len(line) < 1) or (line[0] != 'Step'):
			i += 1
			line  = lines[i].strip().split()
		cols = line
		ncol = len(cols)
		data = np.zeros((nstep, ncol))
		for j in range(nstep):
			data[j] = np.array(lines[i+j+1].strip().split()).astype(float)
	return data, cols

def read_data(filename, graph):
	'''Read a LAMMPS data file and update a graph based on its contents.

	The bond stiffnesses and rest lengths are set based on the datafile specifications.
	   
	Parameters
	----------
	filename : str
		The name of the file to read.
	graph : networkx.graph
		The graph to update.
	'''

	with open(filename) as f:
		line = f.readline()
		while len(line.strip().split()) < 1 or line.strip().split()[0] != 'Bond':
			line = f.readline()
		f.readline() # empty space
		for i, edge in enumerate(graph.edges(data=True)):
			line = f.readline()
			idx, hk, l = np.array(line.strip().split())[:3].astype(float)
			k = 2*hk
			edge[2]['stiffness'] = k
			edge[2]['length'] = l

def read_dim(filename):
	'''Read a LAMMPS input file to parse out the dimension.

	Parameters
	----------
	filename : str
		The name of the file to read.

	Returns
	-------
	int
		The dimension of the system.
	'''
	with open(filename) as f:
		line = f.readline()
		while len(line.strip().split()) < 1 or line.strip().split()[0] != 'dimension':
			line = f.readline()
	return int(line.strip().split()[1])

def setup_run(allo, odir, prefix, lmp_path, duration, frames, applied_args, train=0, method=None, eta=1., alpha=1e-3, temp=0, dt=0.005, hours=24):
	'''Set up a complete LAMMPS simulation in a directory.
	   
	Parameters
	----------
	allo : Allosteric
		The Allosteric object to simulate.
	odir : str
		The path to the directory.
	prefix : str
		The file prefix to use for data, input, dump, and logfiles.
	lmp_path : str
		The path to the LAMMPS executable.
	duration : float
		The final integration time.
	frames : int
		The number of output frames to produce (excluding initial frame).
	applied_args : tuple
		Simulation arguments: the source strain(s), target strain(s), and pinning stiffness.
	train : int, optional
		Training mode. 0 = no training, 1 = l-model, 2 = k-model.
	method : str, optional
		Training method to use. Options are 'aging' or 'learning'.
	eta : float, optional
		The learning rate by which the clamped state target strain approaches the final desired strain.
	alpha : float, optional
		The aging rate.
	temp : float, optional
		The temperature setting, in LJ units. If zero (default), an athermal simulation is performed.
	dt : float, optional
		Integration step size.
	hours : int, optional
		The number of hours to allocate for the job.
	'''
	

	datafile = prefix+'.data'
	infile = prefix+'.in'
	dumpfile = prefix+'.dump'
	logfile = prefix+'.log'
	jobfile = 'job.sh'

	if odir[-1] != '/' : odir += '/'
	if not os.path.exists(odir):
		os.makedirs(odir)

	if train:
		allo.write_lammps_data_learning(odir+datafile, 'Allosteric network', applied_args,
										train=train, method=method, eta=eta, alpha=alpha, dt=dt)
	else:
		allo.write_lammps_data(odir+datafile, 'Allosteric network', applied_args)
	allo.write_lammps_input(odir+infile, datafile, dumpfile, duration, frames, temp=temp, method=method, dt=dt)
	allo.save(odir+'allo.txt') # do this last, because it resets init!!

	cmd = lmp_path+' -i '+infile+' -log '+logfile

	allo.write_job(odir+jobfile, prefix+'_test', hours, cmd)
	print("Simulation files written to: {:s}".format(odir))
	print("Navigate to directory and run: {:s}".format(cmd))

def load_run(odir):
	'''Load a complete LAMMPS simulation from its directory.

	The directory should contain an Allosteric network file, LAMMPS datafile,
	dumpfile, and logfile.
	   
	Parameters
	----------
	odir : str
		The path to the directory.

	Returns
	-------
	allo : Allosteric
		Allosteric Class object with network set up according to provided LAMMPS datafile,
		with simulation history loaded from dumpfile.
	data : ndarray
		The log data (typically printed to screen) of the simulation at each
		integration timestep.
	cols : list of str
		The column names associated with each data column.
	'''

	# collect all filenames
	if odir[-1] != '/' : odir += '/'
	netfile = glob.glob(odir+'*.txt')[0]
	datafile = glob.glob(odir+'*.data')[0]
	infile = glob.glob(odir+'*.in')[0]
	dumpfile = glob.glob(odir+'*.dump')[0]
	logfile = glob.glob(odir+'*.log')[0]

	allo = Allosteric(netfile)
	dim = read_dim(infile)
	if dim != allo.dim:
		raise ValueError("Dimension mismatch between LAMMPS simulation (d={:d}) and network file (d={:d}).".format(dim,allo.dim))
	read_data(datafile, allo.graph)
	data, cols = read_log(logfile)
	allo.t_eval = data[:,cols.index('Time')]
	contents = read_dump(dumpfile)
	if contents.shape[1] == allo.n: allo.traj = np.copy(contents)
	else:
		allo.traj = np.copy(contents[:,1::2,:])
		allo.traj_clamped = np.copy(contents[:,::2,:])
	return allo, data, cols

def get_clusters(data, n, seed=12):
	'''Get k-means clusters.

	Parameters
	----------
	data : ndarray
		The data to cluster.
	n : int
		The number of clusters.
	seed : int, optional
		The random seed.

	Returns
	-------
	ndarray
		The cluster id to which each data point belongs.
	float
		The silhouette score of the clustering.

	'''
	km = KMeans(n_clusters=n, random_state=seed)
	labels = km.fit_predict(data.reshape(-1,1))
	score = silhouette_score(data.reshape(-1,1), labels)
	return labels, score

