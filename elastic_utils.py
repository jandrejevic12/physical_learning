import numpy as np

from plot_imports import *
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import collections  as mc
import matplotlib.animation as animation

from scipy.linalg import orthogonal_procrustes as procrustes

import networkx as nx
from scipy.integrate import solve_ivp
from tqdm import tqdm

from numba import jit

class Elastic:
	'''Class to simulate an elastic network with trainable bonds and rest lengths.
		
	Attributes
	----------
	graph : networkx graph object
		Graph specifying the nodes and edges in the network. A stiffness, rest length,
		and "trainable" parameter are associated with each edge. A trainable edge means
		it will be updated during training.
	seed : int
		A random seed used for selecting sources and targets at random.
	n : int
		Number of nodes in the network.
	ne : int
		Number of edges in the network.
	params : dict
    	Specifies system parameters. Required keywords are:
    	rfac : factor of shortest edge length that should correspond to node radius (used for plotting)
    	drag : coefficient of isotropic drag
    	dashpot : coefficient of dashpot damping at each edge
    	stiffness : initial stiffness assigned to each edge spring
    pts : float array, shape (n,2)
		(x,y) coordinates for each node in the system.
	degree : int array, shape (n,)
		The degree (number of neighbors) of each node.
	Z : float
		The average coordination number, defined as 2*ne/nc, where nc is the number of nodes in the
		biggest connected component of the system.
	dZ : float
		The excess coordination, defined as Z - Ziso, where Ziso is the average coordination required
		for isostaticity (Ziso = 4 - 6/nc in 2D).
	traj : float array, shape (frames+1, n, 2)
		The simulated trajectory of the network produced after a call to the solve() routine. Frames is
		the number of output frames specified for solve, plus one for the initial condition.
	t_eval : float array, shape (frames+1,)
		The corresponding time at each simulated frame.
	
	Methods
	-------
	save(filename)
		Save network to a file.
	solve(duration, frames, T, applied_args, train=0, method='learning', eta=1., alpha=1e-3, pbar=True)
		A multipurpose routine for numerical integration of the network. When train = 0, simply
		integrates the system under specified applied strain. When train is nonzero, adjusts the edge
		rest lengths or stiffnesses according to directed aging or coupled learning update rules.
	reset_init()
		Resets the network node positions to the initial, relaxed state.
	set_frame(fr)
		Set the network state to a particular frame of the simulation.
	plot_network(ax)
		Plot the network of nodes and edges.
	'''
	def __init__(self, graph, params={'rfac':0.05, 'drag':0.005, 'dashpot':10., 'stiffness':1.}):
		'''
		Parameters
    	----------
    	graph : string (filename) or networkx graph object
    		Graph specifying the nodes and edge connections of the elastic network.
    	params : dict, optional
    		Specifies system parameters. Required keywords are:
    		rfac : factor of shortest edge length that should correspond to node radius (used for plotting)
    		drag : coefficient of isotropic drag
    		dashpot : coefficient of dashpot damping at each edge
    		stiffness : initial stiffness assigned to each edge spring
    	'''
		self.params = params
		self.graph = graph

		if 'stiffness' not in list(self.graph.edges(data=True))[0][2]:
			nx.set_edge_attributes(self.graph, self.params['stiffness'], 'stiffness')
		if 'length' not in list(self.graph.edges(data=True))[0][2]:
			nx.set_edge_attributes(self.graph, 1., 'length')
		if 'trainable' not in list(self.graph.edges(data=True))[0][2]:
			nx.set_edge_attributes(self.graph, True, 'trainable')

		# set network node positions and elastic properties
		self.n = len(self.graph.nodes())
		self.pts = np.zeros((self.n,2)) # free state, main memory
		self._pts = np.zeros((self.n,2)) # clamped state
		for i in range(self.n):
			self.pts[i,:] = self.graph.nodes[i]['pos']
			self._pts[i,:] = self.graph.nodes[i]['pos']
		self.set_attributes()
		self.set_coordination()

		self.pts_init = np.copy(self.pts)

	'''
	*****************************************************************************************************
	*****************************************************************************************************

											NETWORK INITIALIZATION

	*****************************************************************************************************
	*****************************************************************************************************
	'''

	def set_attributes(self):
		'''Set the rest length attribute of each edge.'''
		min_l = 0
		for edge in self.graph.edges(data=True):
			i = edge[0]; j = edge[1]
			l = self.distance(self.pts[i], self.pts[j])
			edge[2]['length'] = l
			if l > min_l:
				min_l = l
		self.params['radius'] = self.params['rfac']*min_l

	def set_degree(self):
		'''Compute the degree (number of neighbors) of each node.'''
		self.degree = np.array(list(self.graph.degree[i] for i in range(self.n)), dtype=int)

	def set_coordination(self):
		'''Compute the average coordination and excess coordination of the network.'''
		self.set_degree()
		self.ne = len(self.graph.edges())
		self.nc = self.n-np.sum(self.degree==0).astype(int) # connected nodes
		self.Z = 2*self.ne/float(self.nc)
		self.Ziso = 4. - 6./self.nc
		self.dZ = self.Z - self.Ziso

	def reset_init(self):
		'''Reset the network to its initial, relaxed state.'''
		self.pts = np.copy(self.pts_init)
		self._pts = np.copy(self.pts_init)

	def remove_dangling_edges(self):
		'''Remove connections to nodes with only one bond.'''
		remove = []
		for i in range(self.n):
			if self.graph.degree[i] == 1:
				for edge in self.graph.edges(i):
					remove += [[edge[0], edge[1]]]
		for edge in remove:
			try:
				self.graph.remove_edge(edge[0],edge[1])
			except:
				pass

	def prune_bonds(self, tol=1e-8):
		'''Prune bonds that are within a tolerance of 0 in stiffness. Dangling edges
			are subsequently removed, and coordination is recomputed.
		'''
		remove = []
		for edge in self.graph.edges(data=True):
			if edge[2]['stiffness'] < tol:
				remove += [[edge[0], edge[1]]]
		for edge in remove:
				self.graph.remove_edge(edge[0],edge[1])

		self.remove_dangling_edges()
		self.set_coordination()

	def save(self, filename):
		'''Save network to a file.
		   File contents are written as follows:
		   n : int, number of nodes
		   next n entries : floats, (x,y) positions of nodes
		   ne : int, number of edges
		   next ne entries : ints, (i,j,k) node indices connecting edge, and properties
		
		Parameters
		----------
		filename : string
			The name of the text file to write.
		'''
		self.reset_init()
		with open(filename, 'w') as f:
			f.write(str(self.n)+'\n')
			for i in range(self.n):
				f.write('{:.12g} {:.12g}\n'.format(self.pts[i,0],self.pts[i,1]))
			f.write(str(len(self.graph.edges()))+'\n')
			for edge in self.graph.edges(data=True):
				f.write('{:d} {:d} {:.12g} {:.12g} {:d}\n'.format(edge[0],edge[1],edge[2]['stiffness'], edge[2]['length'], edge['trainable']))

	def distance(self, p, q):
		'''Compute the distance between two coordinate pairs p and q.

		Parameters
		----------
		p, q : float arrays
			The positions, or array of positions, of two points.

		Returns
		-------
		d : float or float array
			The distance between the two points.
		'''
		return np.sqrt(np.sum(np.square(p - q), axis=-1))

	'''
	*****************************************************************************************************
	*****************************************************************************************************

										NUMERICAL INTEGRATION

	*****************************************************************************************************
	*****************************************************************************************************
	'''

	def solve(self, duration, frames, T, applied_args, train=0, method='learning', eta=1., alpha=1e-3, pbar=True):
		'''Numerically integrate the elastic network in time. Optionally train edge stiffnesses or rest
			lengths using directed aging or coupled learning. Produces an output trajectory of shape
			(frames+1, n, 2) stored in the attribute 'traj', and corresponding times in 't_eval'.
		
		Parameters
		----------
		duration : float
			The final integration time.
		frames : int
			The number of output frames to produce (excluding initial frame).
		T : float
			Period for oscillatory force. If T = 0, nodes with an applied force are held stationary.
		applied_args : tuple
			Simulation arguments, which can vary by problem.
		train : int, optional
			The type of training to perform. If train = 0 (default), no training is done. If train = 1,
			train lengths using method 'aging' or 'learning'. If train = 2, train stiffnesses using
			method 'aging' or 'learning'.
		method : string, optional, 'aging' or 'learning'
			Used only if train is nonzero. Specifies the type of training approach to use. Default is
			'learning'.
		eta : float, optional
			Learning rate by which to increment applied strain towards the target. Default is 1, which
			corresponds to pinning directly at the target.
		alpha : float, optional
			Aging rate of each learning degree of freedom (stiffnesses or rest lengths). Default is 1e-3.
		pbar : bool, optional
			Whether to display a progress bar. Default is True. 
		'''
		edge_i, edge_j, edge_k, edge_l, edge_t = self.edge_lists()
		network = (edge_i, edge_j, edge_k, edge_l, edge_t)
		n = self.n

		q = np.hstack([self.pts.ravel(),np.zeros(2*n),
					   self._pts.ravel(),np.zeros(2*n)])
		if train == 1:
			q = np.hstack([q, edge_l])
		elif train == 2:
			q = np.hstack([q, edge_k])

		ti = 0; tf = duration
		t_span = [ti, tf]
		self.t_eval = np.linspace(ti, tf, frames+1)
		self.tp = ti

		if pbar:
			with tqdm(total=tf-ti, unit='sim. time', initial=ti, ascii=True, 
					  bar_format='{l_bar}{bar}| {n:.2f}/{total:.2f} [{elapsed}<{remaining}]', desc='progress') as self.pbar:
				if train:
					sol = solve_ivp(self.ff, t_span, q, t_eval=self.t_eval,
									args=(T, network, applied_args, train, method, eta, alpha, pbar),
									method='RK23')
				else:
					sol = solve_ivp(self.ff, t_span, q, t_eval=self.t_eval, jac=self.jj,
									args=(T, network, applied_args, train, method, eta, alpha, pbar),
									method='BDF')

		else:
			if train:
				sol = solve_ivp(self.ff, t_span, q, t_eval=self.t_eval,
								args=(T, network, applied_args, train, method, eta, alpha, pbar),
								method='RK23')
			else:
				sol = solve_ivp(self.ff, t_span, q, t_eval=self.t_eval, jac=self.jj,
								args=(T, network, applied_args, train, method, eta, alpha, pbar),
								method='BDF')


		q = sol.y.T
		self.traj = np.copy(q[:,:2*n].reshape(frames+1, n, 2))
		self._traj = np.copy(q[:,4*n:6*n].reshape(frames+1, n, 2))
		self.pts = np.copy(self.traj[-1])
		self._pts = np.copy(self._traj[-1])

		if train == 1:
			edge_l = q[-1,8*n:]
			for e, edge in enumerate(self.graph.edges(data=True)):
				edge[2]['length'] = edge_l[e]
		elif train == 2:
			edge_k = q[-1,8*n:]
			for e, edge in enumerate(self.graph.edges(data=True)):
				edge[2]['stiffness'] = edge_k[e]

	def edge_lists(self):
		'''Copy edge properties stored in networkx object into numpy arrays for easy access.

		Returns
		-------
		edge_i : int array
			The first neighbor in each pairwise interaction.
		edge_j : int array
			The second neighbor in each pairwise interaction.
		edge_k : float array
			The stiffness of each bond.
		edge_l : float array
			The rest length of each bond.
		edge_t : bool array
			Whether each bond is trainable or not.
		'''
		edge_i = np.zeros(self.ne, dtype=int)
		edge_j = np.zeros(self.ne, dtype=int)
		edge_k = np.zeros(self.ne)
		edge_l = np.zeros(self.ne)
		edge_t = np.zeros(self.ne, dtype=bool)
		for e, edge in enumerate(self.graph.edges(data=True)):
			edge_i[e] = edge[0]
			edge_j[e] = edge[1]
			edge_k[e] = edge[2]['stiffness']
			edge_l[e] = edge[2]['length']
			edge_t[e] = edge[2]['trainable']
		return edge_i, edge_j, edge_k, edge_l, edge_t

	def energy(self, t, q, *args):
		'''Compute the energy of the spring network.
		
		Parameters
		----------
		t : float
			The current time.
		q : float array
			The degrees of freedom.
		args : tuple
			Collection of simulation arguments:
			T : float
				Period for oscillatory force. If T = 0, nodes with an applied force are held stationary.
			network : tuple of arrays
				Network edge properties obtained from edge_lists().
			applied_args : tuple
				Simulation arguments, which can vary by problem.
			train : int, optional
				The type of training to perform. If train = 0 (default), no training is done. If train = 1,
				train lengths using method 'aging' or 'learning'. If train = 2, train stiffnesses using
				method 'aging' or 'learning'.
			method : string, optional, 'aging' or 'learning'
				Used only if train is nonzero. Specifies the type of training approach to use. Default is
				'learning'.
			eta : float, optional
				Learning rate by which to increment applied strain towards the target. Default is 1, which
				corresponds to pinning directly at the target.
			alpha : float, optional
				Aging rate of each learning degree of freedom (stiffnesses or rest lengths). Default is 1e-3.
			pbar : bool, optional
				Whether to display a progress bar. Default is True. 

		Returns
		-------
		en : float
			Total energy of the network.
		'''
		T, network, applied_args, train, method, eta, alpha, pbar = args
		edge_i, edge_j, edge_k, edge_l, edge_t = network
		n = self.n

		if train == 1:
			l = q[8*n:]
			k = edge_k
		elif train == 2:
			l = edge_l
			k = q[8*n:]
		else:
			l = edge_l
			k = edge_k

		en = 0.
		en += self.elastic_energy(t, n, q, l, k, network)
		en += self.applied_energy(t, n, q, T, applied_args, eta)
		return en

	def applied_energy(self, t, n, q, _q, T, applied_args, eta):
		'''Contribution to total energy from applied forces. Problem specific and implemented
		by subclass.'''
		raise NotImplementedError

	def ff(self, t, q, *args):
		'''Compute the derivative of the degrees of freedom of the spring network to integrate.
		
		Parameters
		----------
		t : float
			The current time.
		q : float array
			The degrees of freedom.
		args : tuple
			Collection of simulation arguments:
			T : float
				Period for oscillatory force. If T = 0, nodes with an applied force are held stationary.
			network : tuple of arrays
				Network edge properties obtained from edge_lists().
			applied_args : tuple
				Simulation arguments, which can vary by problem.
			train : int, optional
				The type of training to perform. If train = 0 (default), no training is done. If train = 1,
				train lengths using method 'aging' or 'learning'. If train = 2, train stiffnesses using
				method 'aging' or 'learning'.
			method : string, optional, 'aging' or 'learning'
				Used only if train is nonzero. Specifies the type of training approach to use. Default is
				'learning'.
			eta : float, optional
				Learning rate by which to increment applied strain towards the target. Default is 1, which
				corresponds to pinning directly at the target.
			alpha : float, optional
				Aging rate of each learning degree of freedom (stiffnesses or rest lengths). Default is 1e-3.
			pbar : bool, optional
				Whether to display a progress bar. Default is True. 

		Returns
		-------
		fun : float array, shape (len(q),)
			Derivative of the degrees of freedom.
		'''
		T, network, applied_args, train, method, eta, alpha, pbar = args
		edge_i, edge_j, edge_k, edge_l, edge_t = network
		n = self.n

		_q = q[4*n:8*n]
		fun = np.zeros_like(q)
		acc = fun[2*n:4*n]
		_fun = fun[4*n:8*n]
		_acc = _fun[2*n:4*n]

		if train:
			if train == 1:
				l = q[8*n:]
				k = edge_k
				dl = fun[8*n:]

				if method == 'learning': self.length_update_learning(t, n, q, _q, l, dl, eta, alpha, network)
				else: self.length_update_aging(t, n, q, _q, l, dl, eta, alpha, network)
				
			elif train == 2:
				l = edge_l
				k = q[8*n:]
				dk = fun[8*n:]

				if method == 'learning': self.stiffness_update_learning(t, n, q, _q, k, dk, eta, alpha, network)
				else: self.stiffness_update_aging(t, n, q, _q, k, dk, eta, alpha, network)
		else:
			l = edge_l
			k = edge_k
		
		# free state
		self.drag_force(t, n, q, fun, network, self.params['drag'])
		self.dashpot_force(t, n, q, l, acc, network, self.params['dashpot'])
		self.elastic_force(t, n, q, l, k, acc, network)

		# clamped state
		self.drag_force(t, n, _q, _fun, network, self.params['drag'])
		self.dashpot_force(t, n, _q, l, _acc, network, self.params['dashpot'])
		self.elastic_force(t, n, _q, l, k, _acc, network)

		self.applied_force(t, n, q, _q, acc, _acc, T, applied_args, eta)

		# update progress bar
		if pbar:
			dt = t - self.tp
			self.pbar.update(dt)
			self.tp = t

		return fun

	def jj(self, t, q, *args):
		'''Compute the jacobian of the derivative of the degrees of freedom of the spring network.
			Only needed by implicit integrators, which are used when train = 0.
		
		Parameters
		----------
		t : float
			The current time.
		q : float array
			The degrees of freedom.
		args : tuple
			Collection of simulation arguments:
			T : float
				Period for oscillatory force. If T = 0, nodes with an applied force are held stationary.
			network : tuple of arrays
				Network edge properties obtained from edge_lists().
			applied_args : tuple
				Simulation arguments, which can vary by problem.
			train : int, optional
				The type of training to perform. If train = 0 (default), no training is done. If train = 1,
				train lengths using method 'aging' or 'learning'. If train = 2, train stiffnesses using
				method 'aging' or 'learning'.
			method : string, optional, 'aging' or 'learning'
				Used only if train is nonzero. Specifies the type of training approach to use. Default is
				'learning'.
			eta : float, optional
				Learning rate by which to increment applied strain towards the target. Default is 1, which
				corresponds to pinning directly at the target.
			alpha : float, optional
				Aging rate of each learning degree of freedom (stiffnesses or rest lengths). Default is 1e-3.
			pbar : bool, optional
				Whether to display a progress bar. Default is True. 

		Returns
		-------
		jac : float array, shape (len(q), len(q))
			Jacobian of the derivative.
		'''
		T, network, applied_args, train, method, eta, alpha, pbar = args
		edge_i, edge_j, edge_k, edge_l, edge_t = network
		n = self.n

		_q = q[4*n:]
		jac = np.zeros((len(q),len(q)))
		_jac = jac[4*n:,4*n:]
		for i in range(2*n):
			jac[i,2*n+i] = 1
			_jac[i,2*n+i] = 1
		dfdx = jac[2*n:4*n,:2*n]
		dfdv = jac[2*n:4*n,2*n:4*n]
		_dfdx = _jac[2*n:4*n,:2*n]
		_dfdv = _jac[2*n:4*n,2*n:4*n]
		l = edge_l
		k = edge_k

		# free state
		self.drag_jacobian(t, n, q, dfdv, network, self.params['drag'])
		self.dashpot_jacobian(t, n, q, l, dfdx, dfdv, network, self.params['dashpot'])
		self.elastic_jacobian(t, n, q, l, k, dfdx, network)

		# clamped state
		self.drag_jacobian(t, n, _q, _dfdv, network, self.params['drag'])
		self.dashpot_jacobian(t, n, _q, l, _dfdx, _dfdv, network, self.params['dashpot'])
		self.elastic_jacobian(t, n, _q, l, k, _dfdx, network)

		self.applied_jacobian(t, n, q, _q, dfdx, _dfdx, T, applied_args, eta)

		return jac

	def applied_force(self, t, n, q, _q, acc, _acc, T, applied_args, eta):
		'''Applied forces on the elastic network. Problem specific and implemented by subclass.'''
		raise NotImplementedError

	def applied_jacobian(self, t, n, q, _q, dfdx, _dfdx, T, applied_args, eta):
		'''Jacobian of applied forces on the elastic network. Problem specific and implemented
		by subclass.'''
		raise NotImplementedError

	@staticmethod
	@jit(nopython=True)
	def drag_force(t, n, q, fun, network, b):
		'''Apply isotropic drag force on each node.

		Parameters
		----------
		t : float
			The current time.
		n : int
			The number of nodes.
		q : float array, shape (4*n,)
			The positions and velocities of nodes in the system.
		fun : float array, shape (4*n,)
			Derivative array in which to store velocities and drag forces.
		network : tuple of arrays
			Network edge properties obtained from edge_lists().
		b : float
			The drag coefficient.
		'''
		vel = q[2*n:4*n]
		acc = fun[2*n:4*n]
		for i in range(n):
			vx, vy = vel[2*i], vel[2*i+1]
			acc[2*i] -= b*vx
			acc[2*i+1] -= b*vy
			fun[2*i] = vx
			fun[2*i+1] = vy

	@staticmethod
	@jit(nopython=True)
	def drag_jacobian(t, n, q, jac, network, b):
		'''Compute the jacobian of the isotropic drag force on each node.

		Parameters
		----------
		t : float
			The current time.
		n : int
			The number of nodes.
		q : float array, shape (4*n,)
			The positions and velocities of nodes in the system.
		jac : float array, shape (2*n,2*n)
			Subblock of jacobian in which to store derivative of drag force.
		network : tuple of arrays
			Network edge properties obtained from edge_lists().
		b : float
			The drag coefficient.
		'''
		for i in range(2*n):
			jac[i,i] -= b

	@staticmethod
	@jit(nopython=True)
	def elastic_energy(t, n, q, l, k, network):
		'''Compute the energy contribution due to pairwise interactions of bonded nodes.

		Parameters
		----------
		t : float
			The current time.
		n : int
			The number of nodes.
		q : float array, shape (2*n,)
			The positions of the nodes.
		l : float array, shape (ne,)
			The rest length of each bond. Different from network lists if it is a
			learning degree of freedom.
		k : float array, shape (ne,)
			The stiffness of each bond. Different from network lists if it is a
			learning degree of freedom.
		network : tuple of arrays
			Network edge properties obtained from edge_lists().

		Returns
		-------
		en : float
			Contribution to total energy due to elastic bonds.
		'''
		en = 0.
		edge_i, edge_j, edge_k, edge_l, edge_t = network
		for e,(i, j) in enumerate(zip(edge_i, edge_j)):
			xi, yi = q[2*i], q[2*i+1]
			xj, yj = q[2*j], q[2*j+1]
			dx = xi-xj; dy = yi-yj
			r = np.sqrt(dx**2 + dy**2)
			en += k[e]*(r-l[e])**2
		return 0.5*en

	@staticmethod
	@jit(nopython=True)
	def elastic_force(t, n, q, l, k, acc, network):
		'''Apply elastic forces between bonded nodes.

		Parameters
		----------
		t : float
			The current time.
		n : int
			The number of nodes.
		q : float array, shape (2*n,)
			The positions of the nodes.
		l : float array, shape (ne,)
			The rest length of each bond. Different from network lists if it is a
			learning degree of freedom.
		k : float array, shape (ne,)
			The stiffness of each bond. Different from network lists if it is a
			learning degree of freedom.
		acc : float array, shape (2*n,)
			The acceleration of each node, populated as output.
		network : tuple of arrays
			Network edge properties obtained from edge_lists().
		'''
		edge_i, edge_j, edge_k, edge_l, edge_t = network
		for e,(i, j) in enumerate(zip(edge_i, edge_j)):
			xi, yi = q[2*i], q[2*i+1]
			xj, yj = q[2*j], q[2*j+1]
			dx = xi-xj; dy = yi-yj
			r = np.sqrt(dx**2 + dy**2)
			fac = -k[e]*(1 - l[e]/r)
			fx = fac*dx
			fy = fac*dy
			acc[2*i] += fx; acc[2*i+1] += fy
			acc[2*j] -= fx; acc[2*j+1] -= fy

	@staticmethod
	@jit(nopython=True)
	def elastic_jacobian(t, n, q, l, k, jac, network):
		'''Compute the jacobian of elastic forces between bonded nodes.

		Parameters
		----------
		t : float
			The current time.
		n : int
			The number of nodes.
		q : float array, shape (2*n,)
			The positions of the nodes.
		l : float array, shape (ne,)
			The rest length of each bond. Different from network lists if it is a
			learning degree of freedom.
		k : float array, shape (ne,)
			The stiffness of each bond. Different from network lists if it is a
			learning degree of freedom.
		jac : float array, shape (2*n,2*n)
			The jacobian of elastic forces, populated as output.
		network : tuple of arrays
			Network edge properties obtained from edge_lists().
		'''
		edge_i, edge_j, edge_k, edge_l, edge_t = network
		for e,(i, j) in enumerate(zip(edge_i, edge_j)):
			xi, yi = q[2*i], q[2*i+1]
			xj, yj = q[2*j], q[2*j+1]
			dx = xi-xj; dy = yi-yj
			r2 = dx**2 + dy**2
			r = np.sqrt(r2); r3 = r2*r
			xx = -k[e]*(l[e]/r*(dx*dx/r2-1)+1)
			yy = -k[e]*(l[e]/r*(dy*dy/r2-1)+1)
			xy = -k[e]*l[e]*dx*dy/r3
			
			jac[2*i,2*i] += xx # xixi
			jac[2*i+1,2*i+1] += yy # yiyi
			jac[2*i,2*i+1] += xy # xiyi
			jac[2*i+1,2*i] += xy # yixi

			jac[2*j,2*j] += xx # xjxj
			jac[2*j+1,2*j+1] += yy # yjyj
			jac[2*j,2*j+1] += xy # xjyj
			jac[2*j+1,2*j] += xy # yjxj

			jac[2*i,2*j] -= xx # xixj
			jac[2*j,2*i] -= xx # xjxi
			jac[2*i+1,2*j+1] -= yy # yiyj
			jac[2*j+1,2*i+1] -= yy # yjyi
			jac[2*i,2*j+1] -= xy # xiyj
			jac[2*j,2*i+1] -= xy # xjyi
			jac[2*i+1,2*j] -= xy # yixj
			jac[2*j+1,2*i] -= xy # yjxi

	@staticmethod
	@jit(nopython=True)
	def dashpot_force(t, n, q, l, acc, network, b):
		'''Apply dashpot forces between bonded nodes.

		Parameters
		----------
		t : float
			The current time.
		n : int
			The number of nodes.
		q : float array, shape (2*n,)
			The positions of the nodes.
		l : float array, shape (ne,)
			The rest length of each bond. Different from network lists if it is a
			learning degree of freedom.
		k : float array, shape (ne,)
			The stiffness of each bond. Different from network lists if it is a
			learning degree of freedom.
		acc : float array, shape (2*n,)
			The acceleration of each node, populated as output.
		network : tuple of arrays
			Network edge properties obtained from edge_lists().
		b : float
			The dashpot damping coefficient.
		'''
		edge_i, edge_j, edge_k, edge_l, edge_t = network
		for e,(i, j) in enumerate(zip(edge_i, edge_j)):
			xi, yi = q[2*i], q[2*i+1]
			xj, yj = q[2*j], q[2*j+1]
			vxi, vyi = q[2*i+2*n], q[2*i+1+2*n]
			vxj, vyj = q[2*j+2*n], q[2*j+1+2*n]
			dx = xi-xj; dy = yi-yj
			dvx = vxi-vxj; dvy = vyi-vyj
			r2 = dx**2 + dy**2
			r = np.sqrt(r2)
			fac = -(1 - l[e]/r)
			vfac = -l[e]*(dx*dvx + dy*dvy)/(r2*r)
			fvx = b*(fac*dvx + vfac*dx)
			fvy = b*(fac*dvy + vfac*dy)
			acc[2*i] += fvx; acc[2*i+1] += fvy
			acc[2*j] -= fvx; acc[2*j+1] -= fvy

	@staticmethod
	@jit(nopython=True)
	def dashpot_jacobian(t, n, q, l, jacx, jacv, network, b):
		'''Compute the jacobian of dashpot forces between bonded nodes.

		Parameters
		----------
		t : float
			The current time.
		n : int
			The number of nodes.
		q : float array, shape (2*n,)
			The positions of the nodes.
		l : float array, shape (ne,)
			The rest length of each bond. Different from network lists if it is a
			learning degree of freedom.
		k : float array, shape (ne,)
			The stiffness of each bond. Different from network lists if it is a
			learning degree of freedom.
		jacx, jacv : float arrays, shape (2*n,2*n)
			The jacobian subblocks to populate.
		network : tuple of arrays
			Network edge properties obtained from edge_lists().
		b : float
			The dashpot damping coefficient.
		'''
		edge_i, edge_j, edge_k, edge_l, edge_t = network
		for e,(i, j) in enumerate(zip(edge_i, edge_j)):
			xi, yi = q[2*i], q[2*i+1]
			xj, yj = q[2*j], q[2*j+1]
			vxi, vyi = q[2*i+2*n], q[2*i+1+2*n]
			vxj, vyj = q[2*j+2*n], q[2*j+1+2*n]
			dx = xi-xj; dy = yi-yj
			dvx = vxi-vxj; dvy = vyi-vyj
			xvdot = dx*dvx + dy*dvy
			r2 = dx**2 + dy**2
			r = np.sqrt(r2)
			r3 = r*r2

			xx = -b*l[e]/r3*(2*dx*dvx-xvdot*(3*dx*dx/r2-1))
			yy = -b*l[e]/r3*(2*dy*dvy-xvdot*(3*dy*dy/r2-1))
			xy = -b*l[e]/r3*(dx*dvy+dy*dvx-3*xvdot*dx*dy/r2)

			vxx = -b*(l[e]/r*(dx*dx/r2-1)+1)
			vyy = -b*(l[e]/r*(dy*dy/r2-1)+1)
			vxy = -b*l[e]*dx*dy/r3
			
			jacx[2*i,2*i] += xx # xixi
			jacx[2*i+1,2*i+1] += yy # yiyi
			jacx[2*i,2*i+1] += xy # xiyi
			jacx[2*i+1,2*i] += xy # yixi

			jacx[2*j,2*j] += xx # xjxj
			jacx[2*j+1,2*j+1] += yy # yjyj
			jacx[2*j,2*j+1] += xy # xjyj
			jacx[2*j+1,2*j] += xy # yjxj

			jacx[2*i,2*j] -= xx # xixj
			jacx[2*j,2*i] -= xx # xjxi
			jacx[2*i+1,2*j+1] -= yy # yiyj
			jacx[2*j+1,2*i+1] -= yy # yjyi
			jacx[2*i,2*j+1] -= xy # xiyj
			jacx[2*j,2*i+1] -= xy # xjyi
			jacx[2*i+1,2*j] -= xy # yixj
			jacx[2*j+1,2*i] -= xy # yjxi

			jacv[2*i,2*i] += vxx # xixi
			jacv[2*i+1,2*i+1] += vyy # yiyi
			jacv[2*i,2*i+1] += vxy # xiyi
			jacv[2*i+1,2*i] += vxy # yixi

			jacv[2*j,2*j] += vxx # xjxj
			jacv[2*j+1,2*j+1] += vyy # yjyj
			jacv[2*j,2*j+1] += vxy # xjyj
			jacv[2*j+1,2*j] += vxy # yjxj

			jacv[2*i,2*j] -= vxx # xixj
			jacv[2*j,2*i] -= vxx # xjxi
			jacv[2*i+1,2*j+1] -= vyy # yiyj
			jacv[2*j+1,2*i+1] -= vyy # yjyi
			jacv[2*i,2*j+1] -= vxy # xiyj
			jacv[2*j,2*i+1] -= vxy # xjyi
			jacv[2*i+1,2*j] -= vxy # yixj
			jacv[2*j+1,2*i] -= vxy # yjxi

	@staticmethod
	@jit(nopython=True)
	def length_update_learning(t, n, q, _q, l, dl, eta, alpha, network):
		'''Apply an update to edge rest lengths using coupled learning.
		
		Parameters
		----------
		t : float
			The current time.
		n : int
			The number of nodes.
		q : float array, shape (2*n,)
			The positions of the nodes in the free state.
		_q : float array, shape (2*n,)
			The positions of the nodes in the clamped state.
		l : float array, shape (ne,)
			The rest length of each bond.
		dl : float array, shape (ne,)
			The derivative of the rest lengths with time, populated on output.
		eta : float
			The learning rate.
		alpha : float
			The aging rate.
		network : tuple of arrays
			Network edge properties obtained from edge_lists().
		'''
		edge_i, edge_j, edge_k, edge_l, edge_t = network
		for e, (i, j, k, train) in enumerate(zip(edge_i, edge_j, edge_k, edge_t)):
			if train:
				# free state
				xi, yi = q[2*i], q[2*i+1]
				xj, yj = q[2*j], q[2*j+1]
				dx = xi-xj; dy = yi-yj
				r = np.sqrt(dx**2 + dy**2)

				# clamped state
				_xi, _yi = _q[2*i], _q[2*i+1]
				_xj, _yj = _q[2*j], _q[2*j+1]
				_dx = _xi-_xj; _dy = _yi-_yj
				_r = np.sqrt(_dx**2 + _dy**2)

				if l[e] > 0:
					dl[e] = alpha/eta*k*((r-l[e])-(_r-l[e]))
				else:
					l[e] = dl[e] = 0.

	@staticmethod
	@jit(nopython=True)
	def length_update_aging(t, n, q, _q, l, dl, eta, alpha, network):
		'''Apply an update to edge rest lengths using directed aging.
		
		Parameters
		----------
		t : float
			The current time.
		n : int
			The number of nodes.
		q : float array, shape (2*n,)
			The positions of the nodes in the free state.
		_q : float array, shape (2*n,)
			The positions of the nodes in the clamped state.
		l : float array, shape (ne,)
			The rest length of each bond.
		dl : float array, shape (ne,)
			The derivative of the rest lengths with time, populated on output.
		eta : float
			The learning rate.
		alpha : float
			The aging rate.
		network : tuple of arrays
			Network edge properties obtained from edge_lists().
		'''
		edge_i, edge_j, edge_k, edge_l, edge_t = network
		for e, (i, j, k, train) in enumerate(zip(edge_i, edge_j, edge_k, edge_t)):
			if train:
				# clamped state
				_xi, _yi = _q[2*i], _q[2*i+1]
				_xj, _yj = _q[2*j], _q[2*j+1]
				_dx = _xi-_xj; _dy = _yi-_yj
				_r = np.sqrt(_dx**2 + _dy**2)
				if l[e] > 0:
					dl[e] = alpha/eta*k*(_r-l[e])
				else:
					l[e] = dl[e] = 0.

	@staticmethod
	@jit(nopython=True)
	def stiffness_update_learning(t, n, q, _q, k, dk, eta, alpha, network):
		'''Apply an update to edge stiffnesses using coupled learning.
		
		Parameters
		----------
		t : float
			The current time.
		n : int
			The number of nodes.
		q : float array, shape (2*n,)
			The positions of the nodes in the free state.
		_q : float array, shape (2*n,)
			The positions of the nodes in the clamped state.
		k : float array, shape (ne,)
			The stiffness of each bond.
		dk : float array, shape (ne,)
			The derivative of the stiffnesses with time, populated on output.
		eta : float
			The learning rate.
		alpha : float
			The aging rate.
		network : tuple of arrays
			Network edge properties obtained from edge_lists().
		'''
		edge_i, edge_j, edge_k, edge_l, edge_t = network
		for e, (i, j, l, train) in enumerate(zip(edge_i, edge_j, edge_l, edge_t)):
			if train:
				# free state
				xi, yi = q[2*i], q[2*i+1]
				xj, yj = q[2*j], q[2*j+1]
				dx = xi-xj; dy = yi-yj
				r = np.sqrt(dx**2 + dy**2)

				# clamped state
				_xi, _yi = _q[2*i], _q[2*i+1]
				_xj, _yj = _q[2*j], _q[2*j+1]
				_dx = _xi-_xj; _dy = _yi-_yj
				_r = np.sqrt(_dx**2 + _dy**2)

				if k[e] > 0:
					dk[e] = 0.5*alpha/eta*((r-l)**2-(_r-l)**2)
				else:
					k[e] = dk[e] = 0.

	@staticmethod
	@jit(nopython=True)
	def stiffness_update_aging(t, n, q, _q, k, dk, eta, alpha, network):
		'''Apply an update to edge stiffnesses using directed aging.
		
		Parameters
		----------
		t : float
			The current time.
		n : int
			The number of nodes.
		q : float array, shape (2*n,)
			The positions of the nodes in the free state.
		_q : float array, shape (2*n,)
			The positions of the nodes in the clamped state.
		k : float array, shape (ne,)
			The stiffness of each bond.
		dk : float array, shape (ne,)
			The derivative of the stiffnesses with time, populated on output.
		eta : float
			The learning rate.
		alpha : float
			The aging rate.
		network : tuple of arrays
			Network edge properties obtained from edge_lists().
		'''
		edge_i, edge_j, edge_k, edge_l, edge_t = network
		for e, (i, j, l, train) in enumerate(zip(edge_i, edge_j, edge_l, edge_t)):
			if train:
				# clamped state
				_xi, _yi = _q[2*i], _q[2*i+1]
				_xj, _yj = _q[2*j], _q[2*j+1]
				_dx = _xi-_xj; _dy = _yi-_yj
				_r = np.sqrt(_dx**2 + _dy**2)

				if k[e] > 0:
					dk[e] = -alpha/eta*k[e]*(_r-l)**2
				else:
					k[e] = dk[e] = 0.

	def sine_pulse(self, t, T):
		'''A sine pulse in time.

		Parameters
		----------
		t : float
			The current time.
		T : float
			The period of oscillation.

		Returns
		-------
		p : The value of the sine pulse at the current time.
		'''
		return np.sin(2*np.pi*t/T)

	def cosine_pulse(self, t, T):
		'''A cosine pulse in time.

		Parameters
		----------
		t : float
			The current time.
		T : float
			The period of oscillation.

		Returns
		-------
		p : The value of the cosine pulse at the current time.
		'''
		return np.cos(2*np.pi*t/T)

	def ramp(self, t, T):
		'''A smooth ramp function in time.

		Parameters
		----------
		t : float
			The current time.
		T : float
			The ramp time to final value.

		Returns
		-------
		p : The value of the ramp function at the current time.
		'''
		q = t/T
		if q > 1: q=1
		return q*q*q*(q*(q*6-15)+10)

	'''
	*****************************************************************************************************
	*****************************************************************************************************

												ANALYSIS

	*****************************************************************************************************
	*****************************************************************************************************
	'''

	def compute_modes(self):
		'''Compute the normal modes of the elastic network at equilibrium.

		Returns
		-------
		evals : float array, shape (2*n,)
			Array of eigenvalues.
		evecs : float array, shape (2*n,2*n)
			Array of unit eigenvectors, with each column corresponding to a different
			eigenvector.
		'''
		self.reset_init()
		
		t = 0
		hess = np.zeros((2*self.n,2*self.n))
		q = np.hstack([self.pts.ravel(),np.zeros(2*self.n)])
		mask = np.zeros(2*self.n, dtype=bool)
		mask[::2] = self.degree > 0
		mask[1::2] = self.degree > 0

		edge_i, edge_j, edge_k, edge_l, edge_t = self.edge_lists()
		network = (edge_i, edge_j, edge_k, edge_l, edge_t)

		self.elastic_jacobian(t, self.n, q, edge_l, edge_k, hess, network)

		hess = hess[mask]
		hess = hess.T[mask].T

		evals, evecs = np.linalg.eigh(-hess)
		return evals, evecs

	def rigid_correction(self):
		'''Find the nearest procrustes transformation (translation + rotation) to the first frame.

		Returns
		-------
		traj : float array, shape (frames+1,n,2)
			The particles' trajectories corrected for rigid translation and rotation.
		'''
		b = self.traj[0] - np.mean(self.traj[0], axis=0)
		traj = np.zeros_like(self.traj)
		for i in range(len(self.traj)):
			a = self.traj[i]-np.mean(self.traj[i], axis=0)
			R, sca = procrustes(a, b, check_finite=False)
			traj[i] = a @ R
		return traj

	def mean_square_displacement(self, min_degree=1):
		'''Compute the mean square displacement of the nodes due to thermal fluctuations.
		This routine first removes rigid rotations and translations relative to the first frame,
		then finds the mean square displacement for each particle over time, and then averages over
		all particles.

		Parameters
		----------
		min_degree : int, optional
			Only include particles with a minimum degree set by min_degree. Default is 1,
			which includes all connected particles.

		Returns
		-------
		l_ms : float
			The mean square displacement.
		'''
		# remove rigid rotations and translations relative to first frame
		traj = self.rigid_correction()
		# average position of each particle
		p_avg = np.mean(traj, axis=0)
		# displacement over time for each particle
		disp = traj - p_avg
		# mean square displacement for each particle (average over time)
		msd = np.mean(np.square(np.linalg.norm(disp, axis=2)), axis=0)
		# mean square displacement over all particles with at least min_degree
		l_ms = np.mean(msd[self.degree>=min_degree])
		return l_ms

	'''
	*****************************************************************************************************
	*****************************************************************************************************

										PLOTTING AND ANIMATION

	*****************************************************************************************************
	*****************************************************************************************************
	'''

	def set_frame(self, fr):
		'''Select a specific simulation frame as the network state.
		
		Parameters
		----------
		fr : int
			The frame number.
		'''
		frames = len(self.traj) - 1
		if fr < 0 or fr > frames:
			raise ValueError("Invalid frame number: Select frame from 0 to {:d}.".format(frames))
		self.pts = self.traj[fr]

	def set_axes(self, ax):
		'''Set up axis limits based on extent of network.

		Parameters
		----------
		ax : matplotlib axes object
			The axes to set up.
		'''
		lim = 1.1*np.max(np.abs(self.pts))
		ax.set_xlim(-lim,lim)
		ax.set_ylim(-lim,lim)
		ax.axis('off')

	def collect_edges(self):
		'''Prepare edges for matplotlib collections plotting.
		
		Returns
		-------
		edges : float array, shape (ne,2,2)
			Array storing the endpoints of each edge.
		'''
		edges = np.zeros((len(self.graph.edges()),2,2))
		for i,edge in enumerate(self.graph.edges()):
			edges[i,0,:] = self.pts[edge[0]]
			edges[i,1,:] = self.pts[edge[1]]
		return edges

	def plot_network(self, ax):
		'''Plot the network.

		Parameters
		----------
		ax : matplotlib axes object
			The axes on which to plot.

		Returns
		-------
		ec, dc : matplotlib LineCollection and EllipseCollection objects, respectively
			Handles to the plotted edges and disks.
		'''
		e = self.collect_edges()
		ec = mc.LineCollection(e, colors='k', linewidths=1)
		ax.add_collection(ec)

		r = 2*self.params['radius']*np.ones(self.n)[self.degree>0]
		dc = mc.EllipseCollection(r, r, np.zeros_like(r), offsets=self.pts[self.degree>0],
									  transOffset=ax.transData, units='x',
									  edgecolor='k', facecolor='k', linewidths=0.5)
		ax.add_collection(dc)
		return ec, dc

