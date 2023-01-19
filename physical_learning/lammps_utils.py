import numpy as np
import glob
from plot_imports import *
import matplotlib.pyplot as plt

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

	traj = np.zeros((frames,n,2))
	for fr in range(frames):
		for i in range(n):
			line = lines[m*fr+2+i]
			a, x, y, z = np.array(line.strip().split()).astype(float)
			traj[fr,i,0] = x
			traj[fr,i,1] = y
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
	dumpfile = glob.glob(odir+'*.dump')[0]
	logfile = glob.glob(odir+'*.log')[0]

	allo = Allosteric(netfile)
	read_data(datafile, allo.graph)
	data, cols = read_log(logfile)
	allo.t_eval = data[:,cols.index('Time')]
	contents = read_dump(dumpfile)
	if len(contents) == allo.n: allo.traj = np.copy(contents)
	else: allo.traj = np.copy(contents[1::2])
	return allo, data, cols
