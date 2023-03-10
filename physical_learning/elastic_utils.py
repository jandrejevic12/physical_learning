import numpy as np

from plot_imports import *
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import collections as mc
import matplotlib.animation as animation

from scipy.linalg import orthogonal_procrustes as procrustes
from scipy.spatial import ConvexHull

import networkx as nx
from scipy.integrate import solve_ivp
from tqdm import tqdm

from numba import jit
from vapory import *

class Elastic(object):
	'''Class to simulate an elastic network with trainable bonds and rest lengths.
	
	Parameters
	----------
	graph : str or networkx.graph
		If string, filename of saved graph specifying the nodes and edge connections of
		the elastic network. If networkx.graph object, the graph to use.
	dim : int
		The dimensionality of the system. Valid options are 2 and 3.
	params : dict, optional
		Specifies system parameters. Required keywords are :

		- 'rfac': factor of shortest edge length that should correspond to node radius (used for plotting)
		- 'drag': coefficient of isotropic drag
		- 'dashpot': coefficient of dashpot damping at each edge
		- 'stiffness': initial stiffness assigned to each edge spring
	
	Attributes
	----------
	graph : networkx.graph
		Graph specifying the nodes and edges in the network. A stiffness, rest length,
		and "trainable" parameter are associated with each edge. A trainable edge will
		be updated during training.
	dim : int
		The dimensionality of the system. Valid options are 2 and 3.
	seed : int
		A random seed used for selecting sources and targets at random.
	n : int
		Number of nodes in the network.
	ne : int
		Number of edges in the network.
	pts : ndarray
		position coordinates for each node in the system.
	vel : ndarray
		velocities for each node in the system.
	degree : ndarray
		The degree (number of neighbors) of each node.
	Z : float
		The average coordination number, defined as 2*ne/nc, where nc is the number of nodes in the
		biggest connected component of the system.
	dZ : float
		The excess coordination, defined as Z - Ziso, where Ziso is the average coordination required
		for isostaticity.
	traj : ndarray
		The simulated trajectory of the network produced after a call to the solve() routine.
	vtraj : ndarray
		The simulated set of velocities produced after a call to the solve() routine.
	t_eval : ndarray
		The corresponding time at each simulated frame.
	'''

	def __init__(self, graph, dim=2, params={'rfac':0.05, 'drag':0.005, 'dashpot':10., 'stiffness':1.}):

		if (dim != 2) and (dim != 3):
			raise ValueError("Dimension must be 2 or 3.")

		self.params = params
		self.graph = graph
		self.dim = dim

		if 'stiffness' not in list(self.graph.edges(data=True))[0][2]:
			nx.set_edge_attributes(self.graph, self.params['stiffness'], 'stiffness')
		if 'length' not in list(self.graph.edges(data=True))[0][2]:
			nx.set_edge_attributes(self.graph, 1., 'length')
		if 'trainable' not in list(self.graph.edges(data=True))[0][2]:
			nx.set_edge_attributes(self.graph, True, 'trainable')

		# set network node positions and elastic properties
		self.n = len(self.graph.nodes())
		self.pts = np.zeros((self.n,3))
		self.vel = np.zeros((self.n,3))
		for i in range(self.n):
			self.pts[i,:] = self.graph.nodes[i]['pos']
		self._set_attributes()
		self._set_coordination()

		self.pts_init = np.copy(self.pts)

		# Set up some default 3D visualization properties.
		self.reset_view()

	'''
	*****************************************************************************************************
	*****************************************************************************************************

											NETWORK INITIALIZATION

	*****************************************************************************************************
	*****************************************************************************************************
	'''

	def _set_attributes(self):
		'''Set the rest length attribute of each edge.'''

		min_l = 0
		for edge in self.graph.edges(data=True):
			i = edge[0]; j = edge[1]
			l = self._distance(self.pts[i], self.pts[j])
			edge[2]['length'] = l
			if l > min_l:
				min_l = l
		self.params['radius'] = self.params['rfac']*min_l

	def _set_degree(self):
		'''Compute the degree (number of neighbors) of each node.'''

		self.degree = np.array(list(self.graph.degree[i] for i in range(self.n)), dtype=int)

	def _set_coordination(self):
		'''Compute the average coordination and excess coordination of the network.'''

		self._set_degree()
		self.ne = len(self.graph.edges())
		self.nc = self.n-np.sum(self.degree==0).astype(int) # connected nodes
		self.Z = 2*self.ne/float(self.nc)
		self.Ziso = 2*self.dim - self.dim*(self.dim+1)/float(self.nc)
		self.dZ = self.Z - self.Ziso

	def reset_init(self):
		'''Reset the network to its initial, relaxed state.'''

		self.pts = np.copy(self.pts_init)
		self.pts_c = np.copy(self.pts_init)

	def reset_equilibrium(self):
		'''Set the current network state to its equilibrium state.'''

		# reset equilibrium node positions
		self.pts_init = np.copy(self.pts)
		self.vel *= 0.
		
		# reset edge rest lengths
		for edge in self.graph.edges(data=True):
			i, j = edge[0], edge[1]
			edge[2]['length'] = self._distance(self.pts[i], self.pts[j])

	def _remove_dangling_edges(self):
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

	def _prune_bonds(self, tol=1e-8):
		'''Prune bonds that are within a tolerance of 0 in stiffness.

		Dangling edges are subsequently removed, and coordination information
		is recomputed.
		'''

		remove = []
		for edge in self.graph.edges(data=True):
			if edge[2]['stiffness'] < tol:
				remove += [[edge[0], edge[1]]]
		for edge in remove:
				self.graph.remove_edge(edge[0],edge[1])

		self._remove_dangling_edges()
		self._set_coordination()

	def save(self, filename):
		'''Save the network to a file.
		
		Parameters
		----------
		filename : str
			The name of the text file to write.
		'''

		with open(filename, 'w') as f:
			f.write(str(self.dim)+'\n')
			f.write(str(self.n)+'\n')
			for i in range(self.n):
				f.write('{:.15g} {:.15g} {:.15g}\n'.format(self.pts_init[i,0],self.pts_init[i,1],self.pts_init[i,2]))
			f.write(str(len(self.graph.edges()))+'\n')
			for edge in self.graph.edges(data=True):
				f.write('{:d} {:d} {:.15g} {:.15g} {:d}\n'.format(edge[0],edge[1],edge[2]['stiffness'], edge[2]['length'], edge['trainable']))

	def _distance(self, p, q):
		'''Compute the distance between two coordinate pairs p and q.

		Parameters
		----------
		p : ndarray
			The positions, or array of positions, of the first point.
		q : ndarray
			The positions, or array of positions, of the second point.

		Returns
		-------
		float or ndarray
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

	def solve(self, duration, frames, T, applied_args, train=0, method='learning', eta=1., alpha=1e-3, vmin=1e-3, pbar=True):	
		'''Numerically integrate the elastic network in time.

		This routine ptionally trains edge stiffnesses or rest lengths using directed aging or
		coupled learning. Upon completion, an output trajectory of frames+1 snapshots is stored
		in the attribute 'traj', and corresponding times in 't_eval'.
		
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
		method : str, optional, 'aging' or 'learning'
			Used only if train is nonzero. Specifies the type of training approach to use. Default is
			'learning'.
		eta : float, optional
			Learning rate by which to increment applied strain towards the target. Default is 1, which
			corresponds to pinning directly at the target.
		alpha : float, optional
			Aging rate of each learning degree of freedom (stiffnesses or rest lengths). Default is 1e-3.
		vmin : float, optional
			The smallest allowed value for each learning degree of freedom. Default is 1e-3.
		pbar : bool, optional
			Whether to display a progress bar. Default is True. 
		'''

		edge_i, edge_j, edge_k, edge_l, edge_t = self._edge_lists()
		network = (edge_i, edge_j, edge_k, edge_l, edge_t)
		n = self.n

		q = np.hstack([self.pts.ravel(),np.zeros(3*n)])

		# if training, augment with one additional network:
		# base network is the free strained state.
		# second is the clamped strained state.
		if train:
			q = np.hstack([q,self.pts.ravel(),np.zeros(3*n)])
			if train == 1:
				q = np.hstack([q, edge_l]) # train lengths
			else:
				q = np.hstack([q, edge_k]) # train stiffnesses

		ti = 0; tf = duration
		t_span = [ti, tf]
		self.t_eval = np.linspace(ti, tf, frames+1)
		self.tp = ti

		if pbar:
			with tqdm(total=tf-ti, unit='sim. time', initial=ti, ascii=True, 
					  bar_format='{l_bar}{bar}| {n:.2f}/{total:.2f} [{elapsed}<{remaining}]', desc='progress') as self.pbar:
				if train:
					sol = solve_ivp(self._ff, t_span, q, t_eval=self.t_eval,
									args=(T, network, applied_args, train, method, eta, alpha, vmin, pbar),
									method='RK23')
				else:
					sol = solve_ivp(self._ff, t_span, q, t_eval=self.t_eval, jac=self._jj,
									args=(T, network, applied_args, train, method, eta, alpha, vmin, pbar),
									method='BDF')

		else:
			if train:
				sol = solve_ivp(self._ff, t_span, q, t_eval=self.t_eval,
								args=(T, network, applied_args, train, method, eta, alpha, vmin, pbar),
								method='RK23')
			else:
				sol = solve_ivp(self._ff, t_span, q, t_eval=self.t_eval, jac=self._jj,
								args=(T, network, applied_args, train, method, eta, alpha, vmin, pbar),
								method='BDF')


		q = sol.y.T
		self.traj = np.copy(q[:,:3*n].reshape(frames+1, n, 3))
		self.vtraj = np.copy(q[:,3*n:6*n].reshape(frames+1, n, 3))
		self.pts = np.copy(self.traj[-1])
		self.vel = np.copy(self.vtraj[-1])

		if train:
			self.traj_c = np.copy(q[:,6*n:9*n].reshape(frames+1, n, 3))
			self.vtraj_c = np.copy(q[:,9*n:12*n].reshape(frames+1, n, 3))
			self.pts_c = np.copy(self.traj_c[-1])
			self.vel_c = np.copy(self.vtraj_c[-1])

			if train == 1:
				edge_l = q[-1,12*n:]
				for e, edge in enumerate(self.graph.edges(data=True)):
					edge[2]['length'] = edge_l[e]
			else:
				edge_k = q[-1,12*n:]
				for e, edge in enumerate(self.graph.edges(data=True)):
					edge[2]['stiffness'] = edge_k[e]

		else:
			self.pts_c = np.copy(self.pts)
			self.vel_c = np.copy(self.vel)
			self.traj_c = np.copy(self.traj)
			self.vtraj_c = np.copy(self.vtraj)

	def _edge_lists(self):
		'''Copy edge properties stored in networkx object into numpy arrays for easy access.

		Returns
		-------
		edge_i : ndarray
			Integer array of the first nodes in each pairwise interaction.
		edge_j : ndarray
			Integer array of the second nodes in each pairwise interaction.
		edge_k : ndarray
			The stiffness of each bond.
		edge_l : ndarray
			The rest length of each bond.
		edge_t : ndarray
			Boolean array indicating whether each bond is trainable or not.
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
		q : ndarray
			The degrees of freedom.
		args : tuple
			Collection of simulation arguments :
			
			- T: Period for oscillatory force. If T = 0, nodes with an applied force are held stationary.
			- network: Network edge properties obtained from _edge_lists().
			- applied_args: Simulation arguments, which can vary by problem.
			- train: The type of training to perform. If train = 0 (default), no training is done. If train = 1,
			  train lengths using method 'aging' or 'learning'. If train = 2, train stiffnesses using
			  method 'aging' or 'learning'.
			- method: Used only if train is nonzero. Specifies the type of training approach to use. Default is
			  'learning'.
			- eta: Learning rate by which to increment applied strain towards the target. Default is 1, which
			  corresponds to pinning directly at the target.
			- alpha: Aging rate of each learning degree of freedom (stiffnesses or rest lengths). Default is 1e-3.
			- vmin: The smallest allowed value for each learning degree of freedom.
			- pbar: Whether to display a progress bar. Default is True. 

		Returns
		-------
		float
			Total energy of the network.
		'''

		T, network, applied_args, train, method, eta, alpha, vmin, pbar = args
		edge_i, edge_j, edge_k, edge_l, edge_t = network
		n = self.n

		if train == 1:
			l = q[12*n:]
			k = edge_k
		elif train == 2:
			l = edge_l
			k = q[12*n:]
		else:
			l = edge_l
			k = edge_k

		en = 0.
		en += self._elastic_energy(t, n, q, l, k, network)
		en += self._applied_energy(t, n, q, T, applied_args)
		return en

	def _applied_energy(self, t, n, q, T, applied_args):
		raise NotImplementedError

	def _ff(self, t, q, *args):
		'''Compute the derivative of the degrees of freedom of the spring network to integrate.
		
		Parameters
		----------
		t : float
			The current time.
		q : ndarray
			The degrees of freedom.
		args : tuple
			Collection of simulation arguments :
			
			- T: Period for oscillatory force. If T = 0, nodes with an applied force are held stationary.
			- network: Network edge properties obtained from _edge_lists().
			- applied_args: Simulation arguments, which can vary by problem.
			- train: The type of training to perform. If train = 0 (default), no training is done. If train = 1,
			  train lengths using method 'aging' or 'learning'. If train = 2, train stiffnesses using
			  method 'aging' or 'learning'.
			- method: Used only if train is nonzero. Specifies the type of training approach to use. Default is
			  'learning'.
			- eta: Learning rate by which to increment applied strain towards the target. Default is 1, which
			  corresponds to pinning directly at the target.
			- alpha: Aging rate of each learning degree of freedom (stiffnesses or rest lengths). Default is 1e-3.
			- vmin: The smallest allowed value for each learning degree of freedom.
			- pbar: Whether to display a progress bar. Default is True. 

		Returns
		-------
		ndarray
			Derivative of the degrees of freedom.
		'''

		T, network, applied_args, train, method, eta, alpha, vmin, pbar = args
		edge_i, edge_j, edge_k, edge_l, edge_t = network
		n = self.n

		fun = np.zeros_like(q)
		acc = fun[3*n:6*n]

		if train:
			q_c = q[6*n:12*n]
			fun_c = fun[6*n:12*n]
			acc_c = fun_c[3*n:6*n]

			if train == 1:
				l = q[12*n:]
				k = edge_k
				dl = fun[12*n:]

				if method == 'learning':
					self._length_update_learning(t, n, q, q_c, l, dl, eta, alpha, vmin, network)
				else:
					self._length_update_aging(t, n, q, q_c, l, dl, eta, alpha, vmin, network)
				
			else:
				l = edge_l
				k = q[12*n:]
				dk = fun[12*n:]

				if method == 'learning':
					self._stiffness_update_learning(t, n, q, q_c, k, dk, eta, alpha, vmin, network)
				else:
					self._stiffness_update_aging(t, n, q, q_c, k, dk, eta, alpha, vmin, network)
		else:
			l = edge_l
			k = edge_k
		
		# base network
		self._drag_force(t, n, q, fun, network, self.params['drag'])
		self._dashpot_force(t, n, q, l, acc, network, self.params['dashpot'])
		self._elastic_force(t, n, q, l, k, acc, network)

		if train:
			# clamped state
			self._drag_force(t, n, q_c, fun_c, network, self.params['drag'])
			self._dashpot_force(t, n, q_c, l, acc_c, network, self.params['dashpot'])
			self._elastic_force(t, n, q_c, l, k, acc_c, network)

			self._applied_force(t, n, q, q_c, acc, acc_c, T, applied_args, train, eta)
		else:
			self._applied_force(t, n, q, q, acc, acc, T, applied_args, train, eta)


		# update progress bar
		if pbar:
			dt = t - self.tp
			self.pbar.update(dt)
			self.tp = t

		return fun

	def _jj(self, t, q, *args):
		'''Compute the jacobian of the derivative of the degrees of freedom of the spring network.
		
		This routine is only needed by implicit integrators, which are used when train = 0.
		
		Parameters
		----------
		t : float
			The current time.
		q : ndarray
			The degrees of freedom.
		args : tuple
			Collection of simulation arguments :
			
			- T: Period for oscillatory force. If T = 0, nodes with an applied force are held stationary.
			- network: Network edge properties obtained from _edge_lists().
			- applied_args: Simulation arguments, which can vary by problem.
			- train: The type of training to perform. If train = 0 (default), no training is done. If train = 1,
			  train lengths using method 'aging' or 'learning'. If train = 2, train stiffnesses using
			  method 'aging' or 'learning'.
			- method: Used only if train is nonzero. Specifies the type of training approach to use. Default is
			  'learning'.
			- eta: Learning rate by which to increment applied strain towards the target. Default is 1, which
			  corresponds to pinning directly at the target.
			- alpha: Aging rate of each learning degree of freedom (stiffnesses or rest lengths). Default is 1e-3.
			- vmin: The smallest allowed value for each learning degree of freedom.
			- pbar: Whether to display a progress bar. Default is True. 

		Returns
		-------
		ndarray
			Jacobian of the derivative.
		'''

		# Note: this method is only used when training is off. So, auxiliary networks are not used.

		T, network, applied_args, train, method, eta, alpha, vmin, pbar = args
		edge_i, edge_j, edge_k, edge_l, edge_t = network
		n = self.n

		jac = np.zeros((len(q),len(q)))
		for i in range(3*n):
			jac[i,3*n+i] = 1
		dfdx = jac[3*n:6*n,:3*n]
		dfdv = jac[3*n:6*n,3*n:6*n]

		l = edge_l
		k = edge_k

		self._drag_jacobian(t, n, q, dfdv, network, self.params['drag'])
		self._dashpot_jacobian(t, n, q, l, dfdx, dfdv, network, self.params['dashpot'])
		self._elastic_jacobian(t, n, q, l, k, dfdx, network)
		self._applied_jacobian(t, n, q, dfdx, T, applied_args)

		return jac

	def _applied_force(self, t, n, q, q_c, acc, acc_c, T, applied_args, train, eta):
		raise NotImplementedError

	def _applied_jacobian(self, t, n, q, dfdx, T, applied_args):
		raise NotImplementedError

	@staticmethod
	@jit(nopython=True)
	def _drag_force(t, n, q, fun, network, b):
		'''Apply an isotropic drag force on each node.

		Parameters
		----------
		t : float
			The current time.
		n : int
			The number of nodes.
		q : ndarray
			The positions and velocities of nodes in the system.
		fun : ndarray
			Derivative array in which to store velocities and drag forces.
		network : tuple of ndarrays
			Network edge properties obtained from _edge_lists().
		b : float
			The drag coefficient.
		'''

		vel = q[3*n:6*n]
		acc = fun[3*n:6*n]
		for i in range(n):
			vx, vy, vz = vel[3*i], vel[3*i+1], vel[3*i+2]
			acc[3*i] -= b*vx
			acc[3*i+1] -= b*vy
			acc[3*i+2] -= b*vz
			fun[3*i] = vx
			fun[3*i+1] = vy
			fun[3*i+2] = vz

	@staticmethod
	@jit(nopython=True)
	def _drag_jacobian(t, n, q, jac, network, b):
		'''Compute the jacobian of the isotropic drag force on each node.

		Parameters
		----------
		t : float
			The current time.
		n : int
			The number of nodes.
		q : ndarray
			The positions and velocities of nodes in the system.
		jac : ndarray
			Subblock of jacobian in which to store derivative of drag force.
		network : tuple of ndarrays
			Network edge properties obtained from _edge_lists().
		b : float
			The drag coefficient.
		'''

		for i in range(3*n):
			jac[i,i] -= b

	@staticmethod
	@jit(nopython=True)
	def _elastic_energy(t, n, q, l, k, network):
		'''Compute the energy contribution due to pairwise interactions of bonded nodes.

		Parameters
		----------
		t : float
			The current time.
		n : int
			The number of nodes.
		q : ndarray
			The positions of the nodes.
		l : ndarray
			The rest length of each bond. Different from network lists if it is a
			learning degree of freedom.
		k : ndarray
			The stiffness of each bond. Different from network lists if it is a
			learning degree of freedom.
		network : tuple of ndarrays
			Network edge properties obtained from _edge_lists().

		Returns
		-------
		float
			Contribution to total energy due to elastic bonds.
		'''

		en = 0.
		edge_i, edge_j, edge_k, edge_l, edge_t = network
		for e,(i, j) in enumerate(zip(edge_i, edge_j)):
			xi, yi, zi = q[3*i], q[3*i+1], q[3*i+2]
			xj, yj, zj = q[3*j], q[3*j+1], q[3*j+2]
			dx = xi-xj; dy = yi-yj; dz = zi-zj
			r = np.sqrt(dx**2 + dy**2 + dz**2)
			en += k[e]*(r-l[e])**2
		return 0.5*en

	@staticmethod
	@jit(nopython=True)
	def _elastic_force(t, n, q, l, k, acc, network):
		'''Apply elastic forces between bonded nodes.

		Parameters
		----------
		t : float
			The current time.
		n : int
			The number of nodes.
		q : ndarray
			The positions of the nodes.
		l : ndarray
			The rest length of each bond. Different from network lists if it is a
			learning degree of freedom.
		k : ndarray
			The stiffness of each bond. Different from network lists if it is a
			learning degree of freedom.
		acc : ndarray
			The acceleration of each node, populated as output.
		network : tuple of ndarrays
			Network edge properties obtained from _edge_lists().
		'''

		edge_i, edge_j, edge_k, edge_l, edge_t = network
		for e,(i, j) in enumerate(zip(edge_i, edge_j)):
			xi, yi, zi = q[3*i], q[3*i+1], q[3*i+2]
			xj, yj, zj = q[3*j], q[3*j+1], q[3*j+2]
			dx = xi-xj; dy = yi-yj; dz = zi-zj
			r = np.sqrt(dx**2 + dy**2 + dz**2)
			fac = -k[e]*(1 - l[e]/r)
			fx = fac*dx
			fy = fac*dy
			fz = fac*dz
			acc[3*i] += fx; acc[3*i+1] += fy; acc[3*i+2] += fz
			acc[3*j] -= fx; acc[3*j+1] -= fy; acc[3*j+2] -= fz

	@staticmethod
	@jit(nopython=True)
	def _elastic_jacobian(t, n, q, l, k, jac, network):
		'''Compute the jacobian of elastic forces between bonded nodes.

		Parameters
		----------
		t : float
			The current time.
		n : int
			The number of nodes.
		q : ndarray
			The positions of the nodes.
		l : ndarray
			The rest length of each bond. Different from network lists if it is a
			learning degree of freedom.
		k : ndarray
			The stiffness of each bond. Different from network lists if it is a
			learning degree of freedom.
		jac :ndarray
			The jacobian of elastic forces, populated as output.
		network : tuple of ndarrays
			Network edge properties obtained from _edge_lists().
		'''

		edge_i, edge_j, edge_k, edge_l, edge_t = network
		for e,(i, j) in enumerate(zip(edge_i, edge_j)):
			xi, yi, zi = q[3*i], q[3*i+1], q[3*i+2]
			xj, yj, zj = q[3*j], q[3*j+1], q[3*j+2]
			dx = xi-xj; dy = yi-yj; dz = zi-zj
			r2 = dx**2 + dy**2 + dz**2
			r = np.sqrt(r2); r3 = r2*r
			xx = -k[e]*(l[e]/r*(dx*dx/r2-1)+1)
			yy = -k[e]*(l[e]/r*(dy*dy/r2-1)+1)
			zz = -k[e]*(l[e]/r*(dz*dz/r2-1)+1)
			xy = -k[e]*l[e]*dx*dy/r3
			xz = -k[e]*l[e]*dx*dz/r3
			yz = -k[e]*l[e]*dy*dz/r3
			
			jac[3*i,3*i] += xx # xixi
			jac[3*i+1,3*i+1] += yy # yiyi
			jac[3*i+2,3*i+2] += zz # zizi
			jac[3*i,3*i+1] += xy # xiyi
			jac[3*i+1,3*i] += xy # yixi
			jac[3*i,3*i+2] += xz # xizi
			jac[3*i+2,3*i] += xz # zixi
			jac[3*i+1,3*i+2] += yz # yizi
			jac[3*i+2,3*i+1] += yz # ziyi

			jac[3*j,3*j] += xx # xjxj
			jac[3*j+1,3*j+1] += yy # yjyj
			jac[3*j+2,3*j+2] += zz # zjzj
			jac[3*j,3*j+1] += xy # xjyj
			jac[3*j+1,3*j] += xy # yjxj
			jac[3*j,3*j+2] += xz # xjzj
			jac[3*j+2,3*j] += xz # zjxj
			jac[3*j+1,3*j+2] += yz # yjzj
			jac[3*j+2,3*j+1] += yz # zjyj

			jac[3*i,3*j] -= xx # xixj
			jac[3*j,3*i] -= xx # xjxi
			jac[3*i+1,3*j+1] -= yy # yiyj
			jac[3*j+1,3*i+1] -= yy # yjyi
			jac[3*i+2,3*j+2] -= zz # zizj
			jac[3*j+2,3*i+2] -= zz # zjzi

			jac[3*i,3*j+1] -= xy # xiyj
			jac[3*j,3*i+1] -= xy # xjyi
			jac[3*i+1,3*j] -= xy # yixj
			jac[3*j+1,3*i] -= xy # yjxi

			jac[3*i,3*j+2] -= xz # xizj
			jac[3*j,3*i+2] -= xz # xjzi
			jac[3*i+2,3*j] -= xz # zixj
			jac[3*j+2,3*i] -= xz # zjxi

			jac[3*i+1,3*j+2] -= yz # yizj
			jac[3*j+1,3*i+2] -= yz # yjzi
			jac[3*i+2,3*j+1] -= yz # ziyj
			jac[3*j+2,3*i+1] -= yz # zjyi

	@staticmethod
	@jit(nopython=True)
	def _dashpot_force(t, n, q, l, acc, network, b):
		'''Apply dashpot forces between bonded nodes.

		Parameters
		----------
		t : float
			The current time.
		n : int
			The number of nodes.
		q : ndarray
			The positions of the nodes.
		l : ndarray
			The rest length of each bond. Different from network lists if it is a
			learning degree of freedom.
		acc : ndarray
			The acceleration of each node, populated as output.
		network : tuple of ndarrays
			Network edge properties obtained from _edge_lists().
		b : float
			The dashpot damping coefficient.
		'''

		edge_i, edge_j, edge_k, edge_l, edge_t = network
		for e,(i, j) in enumerate(zip(edge_i, edge_j)):
			xi, yi, zi = q[3*i], q[3*i+1], q[3*i+2]
			xj, yj, zj = q[3*j], q[3*j+1], q[3*j+2]
			vxi, vyi, vzi = q[3*i+3*n], q[3*i+1+3*n], q[3*i+2+3*n]
			vxj, vyj, vzj = q[3*j+3*n], q[3*j+1+3*n], q[3*j+2+3*n]
			dx = xi-xj; dy = yi-yj; dz = zi-zj
			dvx = vxi-vxj; dvy = vyi-vyj; dvz = vzi-vzj
			r2 = dx**2 + dy**2 + dz**2
			r = np.sqrt(r2)
			fac = -(1 - l[e]/r)
			vfac = -l[e]*(dx*dvx + dy*dvy + dz*dvz)/(r2*r)
			fvx = b*(fac*dvx + vfac*dx)
			fvy = b*(fac*dvy + vfac*dy)
			fvz = b*(fac*dvz + vfac*dz)
			acc[3*i] += fvx; acc[3*i+1] += fvy; acc[3*i+2] += fvz
			acc[3*j] -= fvx; acc[3*j+1] -= fvy; acc[3*j+2] -= fvz

	@staticmethod
	@jit(nopython=True)
	def _dashpot_jacobian(t, n, q, l, jacx, jacv, network, b):
		'''Compute the jacobian of dashpot forces between bonded nodes.

		Parameters
		----------
		t : float
			The current time.
		n : int
			The number of nodes.
		q : ndarray
			The positions of the nodes.
		l : ndarray
			The rest length of each bond. Different from network lists if it is a
			learning degree of freedom.
		jacx : ndarray
			The jacobian subblock to populate.
		jacv : ndarray
			The jacobian subblock to populate.
		network : tuple of ndarrays
			Network edge properties obtained from _edge_lists().
		b : float
			The dashpot damping coefficient.
		'''

		edge_i, edge_j, edge_k, edge_l, edge_t = network
		for e,(i, j) in enumerate(zip(edge_i, edge_j)):
			xi, yi, zi = q[3*i], q[3*i+1], q[3*i+2]
			xj, yj, zj = q[3*j], q[3*j+1], q[3*j+2]
			vxi, vyi, vzi = q[3*i+3*n], q[3*i+1+3*n], q[3*i+2+3*n]
			vxj, vyj, vzj = q[3*j+3*n], q[3*j+1+3*n], q[3*j+2+3*n]
			dx = xi-xj; dy = yi-yj; dz = zi-zj
			dvx = vxi-vxj; dvy = vyi-vyj; dvz = vzi-vzj
			xvdot = dx*dvx + dy*dvy + dz*dvz
			r2 = dx**2 + dy**2 + dz**2
			r = np.sqrt(r2)
			r3 = r*r2

			xx = -b*l[e]/r3*(2*dx*dvx-xvdot*(3*dx*dx/r2-1))
			yy = -b*l[e]/r3*(2*dy*dvy-xvdot*(3*dy*dy/r2-1))
			zz = -b*l[e]/r3*(2*dz*dvz-xvdot*(3*dz*dz/r2-1))
			xy = -b*l[e]/r3*(dx*dvy+dy*dvx-3*xvdot*dx*dy/r2)
			xz = -b*l[e]/r3*(dx*dvz+dz*dvx-3*xvdot*dx*dz/r2)
			yz = -b*l[e]/r3*(dy*dvz+dz*dvy-3*xvdot*dy*dz/r2)

			vxx = -b*(l[e]/r*(dx*dx/r2-1)+1)
			vyy = -b*(l[e]/r*(dy*dy/r2-1)+1)
			vzz = -b*(l[e]/r*(dz*dz/r2-1)+1)
			vxy = -b*l[e]*dx*dy/r3
			vxz = -b*l[e]*dx*dz/r3
			vyz = -b*l[e]*dy*dz/r3
			
			jacx[3*i,3*i] += xx # xixi
			jacx[3*i+1,3*i+1] += yy # yiyi
			jacx[3*i+2,3*i+2] += zz # zizi
			jacx[3*i,3*i+1] += xy # xiyi
			jacx[3*i+1,3*i] += xy # yixi
			jacx[3*i,3*i+2] += xz # xizi
			jacx[3*i+2,3*i] += xz # zixi
			jacx[3*i+1,3*i+2] += yz # yizi
			jacx[3*i+2,3*i+1] += yz # ziyi

			jacx[3*j,3*j] += xx # xjxj
			jacx[3*j+1,3*j+1] += yy # yjyj
			jacx[3*j+2,3*j+2] += zz # zjzj
			jacx[3*j,3*j+1] += xy # xjyj
			jacx[3*j+1,3*j] += xy # yjxj
			jacx[3*j,3*j+2] += xz # xjzj
			jacx[3*j+2,3*j] += xz # zjxj
			jacx[3*j+1,3*j+2] += yz # yjzj
			jacx[3*j+2,3*j+1] += yz # zjyj

			jacx[3*i,3*j] -= xx # xixj
			jacx[3*j,3*i] -= xx # xjxi
			jacx[3*i+1,3*j+1] -= yy # yiyj
			jacx[3*j+1,3*i+1] -= yy # yjyi
			jacx[3*i+2,3*j+2] -= zz # zizj
			jacx[3*j+2,3*i+2] -= zz # zjzi

			jacx[3*i,3*j+1] -= xy # xiyj
			jacx[3*j,3*i+1] -= xy # xjyi
			jacx[3*i+1,3*j] -= xy # yixj
			jacx[3*j+1,3*i] -= xy # yjxi

			jacx[3*i,3*j+2] -= xz # xizj
			jacx[3*j,3*i+2] -= xz # xjzi
			jacx[3*i+2,3*j] -= xz # zixj
			jacx[3*j+2,3*i] -= xz # zjxi

			jacx[3*i+1,3*j+2] -= yz # yizj
			jacx[3*j+1,3*i+2] -= yz # yjzi
			jacx[3*i+2,3*j+1] -= yz # ziyj
			jacx[3*j+2,3*i+1] -= yz # zjyi

			jacv[3*i,3*i] += vxx # xixi
			jacv[3*i+1,3*i+1] += vyy # yiyi
			jacv[3*i+2,3*i+2] += vzz # zizi
			jacv[3*i,3*i+1] += vxy # xiyi
			jacv[3*i+1,3*i] += vxy # yixi
			jacv[3*i,3*i+2] += vxz # xizi
			jacv[3*i+2,3*i] += vxz # zixi
			jacv[3*i+1,3*i+2] += vyz # yizi
			jacv[3*i+2,3*i+1] += vyz # ziyi

			jacv[3*j,3*j] += vxx # xjxj
			jacv[3*j+1,3*j+1] += vyy # yjyj
			jacv[3*j+2,3*j+2] += vzz # zjzj
			jacv[3*j,3*j+1] += vxy # xjyj
			jacv[3*j+1,3*j] += vxy # yjxj
			jacv[3*j,3*j+2] += vxz # xjzj
			jacv[3*j+2,3*j] += vxz # zjxj
			jacv[3*j+1,3*j+2] += vyz # yjzj
			jacv[3*j+2,3*j+1] += vyz # zjyj

			jacv[3*i,3*j] -= vxx # xixj
			jacv[3*j,3*i] -= vxx # xjxi
			jacv[3*i+1,3*j+1] -= vyy # yiyj
			jacv[3*j+1,3*i+1] -= vyy # yjyi
			jacv[3*i+2,3*j+2] -= vzz # zizj
			jacv[3*j+2,3*i+2] -= vzz # zjzi

			jacv[3*i,3*j+1] -= vxy # xiyj
			jacv[3*j,3*i+1] -= vxy # xjyi
			jacv[3*i+1,3*j] -= vxy # yixj
			jacv[3*j+1,3*i] -= vxy # yjxi

			jacv[3*i,3*j+2] -= vxz # xizj
			jacv[3*j,3*i+2] -= vxz # xjzi
			jacv[3*i+2,3*j] -= vxz # zixj
			jacv[3*j+2,3*i] -= vxz # zjxi

			jacv[3*i+1,3*j+2] -= vyz # yizj
			jacv[3*j+1,3*i+2] -= vyz # yjzi
			jacv[3*i+2,3*j+1] -= vyz # ziyj
			jacv[3*j+2,3*i+1] -= vyz # zjyi

	@staticmethod
	@jit(nopython=True)
	def _length_update_learning(t, n, q, q_c, l, dl, eta, alpha, lmin, network):
		'''Apply an update to edge rest lengths using coupled learning.
		
		Parameters
		----------
		t : float
			The current time.
		n : int
			The number of nodes.
		q : ndarray
			The positions of the nodes in the source clamp only state.
		q_c : ndarray
			The positions of the nodes in the clamped state.
		l : ndarray
			The rest length of each bond.
		dl : ndarray
			The derivative of the rest lengths with time, populated on output.
		eta : float
			The learning rate.
		alpha : float
			The aging rate.
		lmin : float
			The minimum allowed rest length.
		network : tuple of ndarrays
			Network edge properties obtained from _edge_lists().
		'''

		edge_i, edge_j, edge_k, edge_l, edge_t = network
		for e, (i, j, k, train) in enumerate(zip(edge_i, edge_j, edge_k, edge_t)):
			if train:
				# free state
				xi, yi, zi = q[3*i], q[3*i+1], q[3*i+2]
				xj, yj, zj = q[3*j], q[3*j+1], q[3*j+2]
				dx = xi-xj; dy = yi-yj; dz = zi-zj
				r = np.sqrt(dx**2 + dy**2 + dz**2)

				# clamped state
				xi_c, yi_c, zi_c = q_c[3*i], q_c[3*i+1], q_c[3*i+2]
				xj_c, yj_c, zj_c = q_c[3*j], q_c[3*j+1], q_c[3*j+2]
				dx_c = xi_c-xj_c; dy_c = yi_c-yj_c; dz_c = zi_c-zj_c
				r_c = np.sqrt(dx_c**2 + dy_c**2 + dz_c**2)
				dl[e] = alpha/eta*k*((r-l[e])-(r_c-l[e]))

				if (l[e] <= lmin) and (dl[e] < 0):
					l[e] = lmin
					dl[e] = 0

	@staticmethod
	@jit(nopython=True)
	def _length_update_aging(t, n, q, q_c, l, dl, eta, alpha, lmin, network):
		'''Apply an update to edge rest lengths using directed aging.
		
		Parameters
		----------
		t : float
			The current time.
		n : int
			The number of nodes.
		q : ndarray
			The positions of the nodes in the source only clamped state.
		q_c : ndarray
			The positions of the nodes in the clamped state.
		l : ndarray
			The rest length of each bond.
		dl : ndarray
			The derivative of the rest lengths with time, populated on output.
		eta : float
			The learning rate.
		alpha : float
			The aging rate.
		lmin : float
			The minimum allowed rest length.
		network : tuple of ndarrays
			Network edge properties obtained from _edge_lists().
		'''

		edge_i, edge_j, edge_k, edge_l, edge_t = network
		for e, (i, j, k, train) in enumerate(zip(edge_i, edge_j, edge_k, edge_t)):
			if train:
				# clamped state
				xi_c, yi_c, zi_c = q_c[3*i], q_c[3*i+1], q_c[3*i+2]
				xj_c, yj_c, zj_c = q_c[3*j], q_c[3*j+1], q_c[3*j+2]
				dx_c = xi_c-xj_c; dy_c = yi_c-yj_c; dz_c = zi_c-zj_c
				r_c = np.sqrt(dx_c**2 + dy_c**2 + dz_c**2)
				dl[e] = alpha/eta*k*(r_c-l[e])
				
				if (l[e] <= lmin) and (dl[e] < 0):
					l[e] = lmin
					dl[e] = 0

	@staticmethod
	@jit(nopython=True)
	def _stiffness_update_learning(t, n, q, q_c, k, dk, eta, alpha, kmin, network):
		'''Apply an update to edge stiffnesses using coupled learning.
		
		Parameters
		----------
		t : float
			The current time.
		n : int
			The number of nodes.
		q : ndarray
			The positions of the nodes in the source only clamped state.
		q_c : ndarray
			The positions of the nodes in the clamped state.
		k : ndarray
			The stiffness of each bond.
		dk : ndarray
			The derivative of the stiffnesses with time, populated on output.
		eta : float
			The learning rate.
		alpha : float
			The aging rate.
		kmin : float
			The minimum allowed stiffness.
		network : tuple of ndarrays
			Network edge properties obtained from _edge_lists().
		'''

		edge_i, edge_j, edge_k, edge_l, edge_t = network
		for e, (i, j, l, train) in enumerate(zip(edge_i, edge_j, edge_l, edge_t)):
			if train:
				# free state
				xi, yi, zi = q[3*i], q[3*i+1], q[3*i+2]
				xj, yj, zj = q[3*j], q[3*j+1], q[3*j+2]
				dx = xi-xj; dy = yi-yj; dz = zi-zj
				r = np.sqrt(dx**2 + dy**2 + dz**2)

				# clamped state
				xi_c, yi_c, zi_c = q_c[3*i], q_c[3*i+1], q_c[3*i+2]
				xj_c, yj_c, zj_c = q_c[3*j], q_c[3*j+1], q_c[3*j+2]
				dx_c = xi_c-xj_c; dy_c = yi_c-yj_c; dz_c = zi_c-zj_c
				r_c = np.sqrt(dx_c**2 + dy_c**2 + dz_c**2)
				dk[e] = 0.5*alpha/eta*((r-l)**2-(r_c-l)**2)
				
				if (k[e] <= kmin) and (dk[e] < 0):
					k[e] = kmin
					dk[e] = 0

	@staticmethod
	@jit(nopython=True)
	def _stiffness_update_aging(t, n, q, q_c, k, dk, eta, alpha, kmin, network):
		'''Apply an update to edge stiffnesses using directed aging.
		
		Parameters
		----------
		t : float
			The current time.
		n : int
			The number of nodes.
		q : ndarray
			The positions of the nodes in the source only clamped state.
		q_c : ndarray
			The positions of the nodes in the clamped state.
		k : ndarray
			The stiffness of each bond.
		dk : ndarray
			The derivative of the stiffnesses with time, populated on output.
		eta : float
			The learning rate.
		alpha : float
			The aging rate.
		kmin : float
			The minimum allowed stiffness.
		network : tuple of ndarrays
			Network edge properties obtained from _edge_lists().
		'''

		edge_i, edge_j, edge_k, edge_l, edge_t = network
		for e, (i, j, l, train) in enumerate(zip(edge_i, edge_j, edge_l, edge_t)):
			if train:
				# clamped state
				xi_c, yi_c, zi_c = q_c[3*i], q_c[3*i+1], q_c[3*i+2]
				xj_c, yj_c, zj_c = q_c[3*j], q_c[3*j+1], q_c[3*j+2]
				dx_c = xi_c-xj_c; dy_c = yi_c-yj_c; dz_c = zi_c-zj_c
				r_c = np.sqrt(dx_c**2 + dy_c**2 + dz_c**2)
				dk[e] = -alpha/eta*k[e]*(r_c-l)**2
				
				if (k[e] <= kmin) and (dk[e] < 0):
					k[e] = kmin
					dk[e] = 0

	def _sine_pulse(self, t, T):
		'''A sine pulse in time.

		Parameters
		----------
		t : float
			The current time.
		T : float
			The period of oscillation.

		Returns
		-------
		float
			The value of the sine pulse at the current time.
		'''

		return np.sin(2*np.pi*t/T)

	def _cosine_pulse(self, t, T):
		'''A cosine pulse in time.

		Parameters
		----------
		t : float
			The current time.
		T : float
			The period of oscillation.

		Returns
		-------
		float
			The value of the cosine pulse at the current time.
		'''

		return np.cos(2*np.pi*t/T)

	def _ramp(self, t, T):
		'''A smooth ramp function in time.

		Parameters
		----------
		t : float
			The current time.
		T : float
			The ramp time to final value.

		Returns
		-------
		float
			The value of the ramp function at the current time.
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

	def compute_modes(self, applied_args=None):
		'''Compute the normal modes of the elastic network at equilibrium.

		Parameters
		----------
		applied_args : tuple, optional
			Arguments for applied forces. If None, no applied force is added.
		Returns
		-------
		evals : ndarray
			Eigenvalues.
		evecs : ndarray
			Unit eigenvectors stored in each column.
		'''

		t = 0
		hess = np.zeros((3*self.n,3*self.n))
		q = np.hstack([self.pts.ravel(),np.zeros(3*self.n)])
		mask = np.zeros(3*self.n, dtype=bool)
		mask[::3] = self.degree > 0
		mask[1::3] = self.degree > 0
		mask[2::3] = self.degree > 0

		edge_i, edge_j, edge_k, edge_l, edge_t = self._edge_lists()
		network = (edge_i, edge_j, edge_k, edge_l, edge_t)

		self._elastic_jacobian(t, self.n, q, edge_l, edge_k, hess, network)
		if applied_args is not None:
			self._applied_jacobian(t, self.n, q, hess, 0, applied_args)

		hess = hess[mask]
		hess = hess.T[mask].T

		evals, evecs = np.linalg.eigh(-hess)
		return evals, evecs

	def rigid_correction(self):
		'''Find the nearest Procrustes transformation (translation + rotation) to the first frame.

		Returns
		-------
		ndarray
			The particles' trajectories corrected for rigid translation and rotation.
		'''

		b = self.traj[0] - np.mean(self.traj[0], axis=0)
		traj = np.zeros_like(self.traj)
		for i in range(len(self.traj)):
			a = self.traj[i]-np.mean(self.traj[i], axis=0)
			R, sca = procrustes(a, b, check_finite=False)
			traj[i] = a @ R
		return traj

	def mean_square_displacement(self, nodes=None):
		'''Compute the mean square displacement of the nodes over time.

		This routine first removes rigid rotations and translations relative to the first frame,
		then finds the mean square displacement for each particle over time, and then averages over
		all particles.

		Parameters
		----------
		nodes : int, optional
			Only include the node indices provided in the MSD. If None (default), include
			all nodes in the calculation.

		Returns
		-------
		float
			The mean square displacement.
		'''
		if nodes is None:
			nodes = np.arange(self.n).astype(int)

		# remove rigid rotations and translations relative to first frame
		traj = self.rigid_correction()
		# average position of each particle
		p_avg = np.mean(traj, axis=0)
		# displacement over time for each particle
		disp = traj - p_avg
		# mean square displacement for each particle (average over time)
		msd = np.mean(np.square(np.linalg.norm(disp, axis=2)), axis=0)
		# mean square displacement over all particles in nodes.
		l_ms = np.mean(msd[nodes])
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
		self.vel = self.vtraj[fr]
		self.pts_c = self.traj_c[fr]
		self.vel_c = self.vtraj_c[fr]

	def set_axes(self, ax):
		'''Set up axis limits based on extent of network.

		Parameters
		----------
		ax : matplotlib.axes.Axes
			The axes to set up.
		'''

		lim = 1.1*np.max(np.abs(self.pts))
		ax.set_xlim(-lim,lim)
		ax.set_ylim(-lim,lim)
		ax.axis('off')

	def _collect_edges(self):
		'''Prepare edges for matplotlib collections plotting.
		
		Returns
		-------
		ndarray
			Array storing the endpoints of each edge.
		'''

		edges = np.zeros((len(self.graph.edges()),2,3))
		for i,edge in enumerate(self.graph.edges()):
			edges[i,0,:] = self.pts[edge[0]]
			edges[i,1,:] = self.pts[edge[1]]
		return edges

	def rotate_view(self, R):
		'''Redefine the perspective for 3D visualization.
		
		Parameters
		----------
		R : float or ndarray
			If float, rotation angle about z-axis in radians.
			If ndarray, a 3D rotation matrix.
		'''

		if not hasattr(R, '__len__'): R = np.array([[np.cos(R),-np.sin(R),0],
													[np.sin(R),np.cos(R),0],
													[0,0,1]])
		self._povray_props['light1'] = R @ self._povray_props['light1']
		self._povray_props['light2'] = R @ self._povray_props['light2']
		self._povray_props['camera'] = R @ self._povray_props['camera']

	def reset_view(self):
		'''Reset the 3D view orientation.'''
		self._povray_props = {'light1':np.array([-10,10,10]),
							  'light2':np.array([10,10,10]),
							  'camera':np.array([0,0.75,0.25])}

	def _povray_setup(self, R=np.eye(3)):
		bg = Background("color", [1.0,1.0,1.0])
		l1pos = R @ self._povray_props['light1']
		l2pos = R @ self._povray_props['light2']
		cpos = R @ self._povray_props['camera']

		#lights = [LightSource(l1pos, 'color', 'rgb <0.7,0.65,0.65>',
		#					  'area_light', [10,0,0], [0,10,0], 5, 5,
		#					  'adaptive', 1, 'jitter'),
		#		  LightSource(l2pos, 'color', 'rgb <0.7,0.65,0.65>',
		#					  'area_light', [10,0,0], [0,10,0], 5, 5,
		#					  'adaptive', 1, 'jitter')]
		lights = [LightSource(l1pos, 'color', 'rgb <0.7,0.65,0.65>'),
				  LightSource(l2pos, 'color', 'rgb <0.7,0.65,0.65>')]
		camera = Camera('location', cpos, 'sky', 'z', 'look_at', [0,0,0])
		return bg, lights, camera

	def _povray_spheres(self, nodes):
		spheres = [0 for _ in range(len(nodes))]
		c = 'rgb<0.3,0.4,0.5>'
		r = 2*self.params['radius']
		for i,node in enumerate(nodes):
			spheres[i] = Sphere(self.pts[node], r,
						 Texture(Pigment('color',c),
						 Finish('ambient',0.24,'diffuse',0.88,
						 'specular',0.1,'phong',0.2,'phong_size',5)))
		return spheres

	def _povray_edges(self, pairs):
		edges = [0 for _ in range(len(pairs))]
		c = 'rgb<0.3,0.4,0.5>'
		r = self.params['radius']
		for i,pair in enumerate(pairs):
			edges[i] = Cylinder(self.pts[pair[0]], self.pts[pair[1]], r,
						   Texture(Pigment('color',c),
						   Finish('ambient',0.24,'diffuse',0.88,
						   'specular',0.1,'phong',0.2,'phong_size',5)))
		return edges

	def _povray_hull(self):
		hull = ConvexHull(self.pts)
		hull_pts = sorted(list(set([i for simplex in hull.simplices for i in simplex])))
		simplices = [0 for _ in range(2*len(hull.simplices))]
		for s,simplex in enumerate(hull.simplices):
			i, j, k = simplex
			simplices[2*s] = [hull_pts.index(i),hull_pts.index(j),hull_pts.index(k)]

		c = 'rgb<0.65,0.6,0.85>'
		hull = Mesh2(VertexVectors(len(hull_pts), *[self.pts[i] for i in hull_pts]),
					 TextureList(1, Texture(Pigment('color',c,'transmit',0.7),
									Finish('ambient',0.24,'diffuse',0.88,
									'specular',0.1,'phong',0.2,'phong_size',5))),
					 FaceIndices(len(simplices)//2, *[simplex for simplex in simplices]))
		return hull


	def _povray_color_edges(self, pairs, colors, alphas):
		edges = [0 for _ in range(len(pairs))]
		r = self.params['radius']
		for i,pair in enumerate(pairs):
			edges[i] = Cylinder(self.pts[pair[0]], self.pts[pair[1]], r,
						   Texture(Pigment('color', colors[i], 'transmit', 1-alphas[i]),
						   Finish('ambient',0.24,'diffuse',0.88,
						   'specular',0.1,'phong',0.2,'phong_size',5)))
		return edges

	def plot_network(self, ax):
		'''Plot the network.

		Parameters
		----------
		ax : matplotlib.axes.Axes
			The axes on which to plot.
		'''

		if self.dim == 2: return self._plot_network_2d(ax)
		else: return self._plot_network_3d(ax)

	def _plot_network_2d(self, ax):
		'''Plot the network.

		Parameters
		----------
		ax : matplotlib.axes.Axes
			The axes on which to plot.

		Returns
		-------
		ec : matplotlib.collections.LineCollection
			Handle to the plotted edges.

		dc : matplotlib.collections.EllipseCollection
			Handle to the plotted nodes.
		'''

		e = self._collect_edges()
		ec = mc.LineCollection(e[:,:,:self.dim], colors='k', linewidths=1)
		ax.add_collection(ec)

		r = 2*self.params['radius']*np.ones(self.n)[self.degree>0]
		dc = mc.EllipseCollection(r, r, np.zeros_like(r), offsets=self.pts[self.degree>0,:self.dim],
									  transOffset=ax.transData, units='x',
									  edgecolor='k', facecolor='k', linewidths=0.5)
		ax.add_collection(dc)
		return ec, dc

	def _plot_network_3d(self, ax):
		'''Plot the network.

		Parameters
		----------
		ax : matplotlib.axes.Axes
			The axes on which to plot.

		Returns
		-------
		spheres : list of vapory.Sphere objects
			Nodes to plot.
		edges : list of vapory.Cylinder objects
			Network edges to plot.
		'''
		return self._povray_spheres(np.arange(self.n).astype(int)), self._povray_edges(self.graph.edges())



