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

		

		# set network node positions and elastic properties
		self.n = len(self.graph.nodes())
		self.pts = np.zeros((self.n,3))
		self.vel = np.zeros((self.n,3))
		for i in range(self.n):
			self.pts[i,:] = self.graph.nodes[i]['pos']
		self.pts_c = np.copy(self.pts)
		self.vel_c = np.copy(self.vel)
		self.pts_s = np.copy(self.pts)
		self.pts_sc = np.copy(self.pts)
		self.vel_s = np.copy(self.vel)
		self.vel_sc = np.copy(self.vel)

		if 'stiffness' not in list(self.graph.edges(data=True))[0][2]:
			nx.set_edge_attributes(self.graph, self.params['stiffness'], 'stiffness')
		if 'length' not in list(self.graph.edges(data=True))[0][2]:
			nx.set_edge_attributes(self.graph, 1., 'length')
			min_l = self._set_attributes()
		else:
			min_l = np.min([edge[2]['length'] for edge in self.graph.edges(data=True)])
		self.params['radius'] = self.params['rfac']*min_l
		if 'trainable' not in list(self.graph.edges(data=True))[0][2]:
			nx.set_edge_attributes(self.graph, True, 'trainable')

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
		return min_l

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
		self.pts_s = np.copy(self.pts_init)
		self.pts_sc = np.copy(self.pts_init)
		self.vel *= 0
		self.vel_c *= 0
		self.vel_s *= 0
		self.vel_sc *= 0

	def reset_equilibrium(self):
		'''Set the current network state to its equilibrium state.'''

		# reset equilibrium node positions
		self.pts_init = np.copy(self.pts)
		self.vel *= 0.
		self.vel_c *= 0.
		self.vel_s *= 0.
		self.vel_sc *= 0.
		
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

	def solve(self, duration, frames, T, applied_args, train=0, method='learning', eta=1., alpha=1e-3, vmin=1e-3, vsmooth=None, fix=0, symmetric=False, pbar=True, integrator='LSODA', rtol=1e-6, atol=1e-8):	
		'''Numerically integrate the elastic network in time.

		This routine optionally trains edge stiffnesses or rest lengths using directed aging or
		coupled learning. Upon completion, an output trajectory of frames+1 snapshots is stored
		in the attribute 'traj', and corresponding times in 't_eval'.
		
		Parameters
		----------
		duration : float
			The final integration time.
		frames : int or ndarray
			If integer, the number of evenly-spaced output frames to produce (excluding initial frame).
			If array, the time points at which to output snapshots.
		T : float
			Period for oscillatory force. If T = 0, nodes with an applied force are held stationary.
		applied_args : tuple
			Simulation arguments, which can vary by problem.
		train : int, optional
			The type of training to perform. If train = 0 (default), no training is done. If train = 1,
			train lengths using method 'aging' or 'learning'. If train = 2, train stiffnesses using
			method 'aging' or 'learning'. If train = 3, train both.
		method : str, optional, 'aging' or 'learning'
			Used only if train is nonzero. Specifies the type of training approach to use. Default is
			'learning'.
		eta : float, optional
			Nudge factor by which to increment applied strain towards the target. Default is 1, which
			corresponds to pinning directly at the target.
		alpha : float or ndarray, optional
			Learning rate of each learning degree of freedom (stiffnesses or rest lengths). Default is 1e-3.
			If reported as an ndarray of 2 entries, provides the alpha for rest lengths, then stiffnesses.
		vmin : float or ndarray, optional
			The smallest allowed value for each learning degree of freedom. Default is 1e-3. If reported as
			an ndarray of 2 entries, provides the vmin for rest lengths, then stiffnesses.
		vsmooth : float or ndarray, optional
			The value of the learning degree of freedom at which to begin smooth ramping to vmin. If reported as
			an ndarray of 2 entries, provides the vsmooth for rest lengths, then stiffnesses.
		fix : ndarray, optional
			An (n,3) array indicating which degrees of freedom should remain fixed and not integrated. If default (0),
			all degrees of freedom are integrated.
		symmetric : bool, optional
			Whether to introduce a symmetric state for training with a different set of boundary conditions. Default is False.
		pbar : bool, optional
			Whether to display a progress bar. Default is True.
		integrator : str, optional
			Type of integrator to use, as available for scipy.integrate.solve_ivp. Default is 'LSODA'.
		rtol, atol : float, optional
			The relative and absolute error tolerance for the integrator, respectively. Default values are 1e-6 and 1e-8.


		Returns
		-------
		dict
			The scipy.integrate.solve_ivp return object, only for diagnostic purposes; the solution is internally parsed.
		'''

		edge_i, edge_j, edge_k, edge_l, edge_t = self._edge_lists()
		network = (edge_i, edge_j, edge_k, edge_l, edge_t)
		n = self.n

		q = np.hstack([self.pts.ravel(),self.vel.ravel()])

		if not hasattr(fix, '__len__'): fix = np.zeros((n,3), dtype=bool)

		# if training, augment with one additional network:
		# base network is the free strained state.
		# second is the clamped strained state.
		# if symmetric, augment with an additional free and clamped state.
		if train:
			q = np.hstack([q,self.pts_c.ravel(),self.vel_c.ravel()])
			if symmetric:
				q = np.hstack([q,self.pts_s.ravel(),self.vel_s.ravel()])
				q = np.hstack([q,self.pts_sc.ravel(),self.vel_sc.ravel()])
			if train & 1:
				q = np.hstack([q, edge_l]) # train lengths
			if train & 2:
				q = np.hstack([q, edge_k]) # train stiffnesses (potentially with lengths)

		if not hasattr(alpha, '__len__'): alpha = [alpha, alpha]
		if not hasattr(vmin, '__len__'): vmin = [vmin, vmin]

		if vsmooth is None:
			vsmooth = [vmin[0] + 0.1*np.mean(edge_l), vmin[1] + 0.1*np.mean(edge_k)]

		if not hasattr(vsmooth, '__len__'): vsmooth = [vsmooth, vsmooth]

		ti = 0; tf = duration
		t_span = [ti, tf]
		if not hasattr(frames, '__len__'):
			self.t_eval = np.linspace(ti, tf, frames+1)
		else:
			self.t_eval = np.copy(frames)
			frames = len(self.t_eval)-1
		self.tp = ti

		if pbar:
			with tqdm(total=tf-ti, unit='sim. time', initial=ti, ascii=True, 
					  bar_format='{l_bar}{bar}| {n:.2f}/{total:.2f} [{elapsed}<{remaining}]', desc='progress') as self.pbar:
				
				sol = solve_ivp(self._ff, t_span, q, t_eval=self.t_eval, jac=self._jj,
								args=(T, fix.ravel(), network, applied_args, train, method, eta, alpha, vmin, vsmooth, symmetric, pbar),
								method=integrator, rtol=rtol, atol=atol)

		else:
			sol = solve_ivp(self._ff, t_span, q, t_eval=self.t_eval, jac=self._jj,
							args=(T, fix.ravel(), network, applied_args, train, method, eta, alpha, vmin, vsmooth, symmetric, pbar),
							method=integrator, rtol=rtol, atol=atol)

		if sol.status != 0:
			return sol

		else:

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

				offset = 12*n

				if symmetric:
					self.traj_s = np.copy(q[:,12*n:15*n].reshape(frames+1, n, 3))
					self.vtraj_s = np.copy(q[:,15*n:18*n].reshape(frames+1, n, 3))
					self.pts_s = np.copy(self.traj_s[-1])
					self.vel_s = np.copy(self.vtraj_s[-1])
					self.traj_sc = np.copy(q[:,18*n:21*n].reshape(frames+1, n, 3))
					self.vtraj_sc = np.copy(q[:,21*n:24*n].reshape(frames+1, n, 3))
					self.pts_sc = np.copy(self.traj_sc[-1])
					self.vel_sc = np.copy(self.vtraj_sc[-1])

					offset += 12*n

				if train & 1:
					edge_l = q[-1,offset:offset+self.ne]
					for e, edge in enumerate(self.graph.edges(data=True)):
						edge[2]['length'] = edge_l[e]
					self.ltraj = np.copy(q[:,offset:offset+self.ne])
					offset += self.ne

				if train & 2:
					edge_k = q[-1,offset:offset+self.ne]
					for e, edge in enumerate(self.graph.edges(data=True)):
						edge[2]['stiffness'] = edge_k[e]
					self.ktraj = np.copy(q[:,offset:offset+self.ne])

			else:
				self.pts_c = np.copy(self.pts)
				self.vel_c = np.copy(self.vel)
				self.traj_c = np.copy(self.traj)
				self.vtraj_c = np.copy(self.vtraj)
				self.traj_s = np.copy(self.traj)
				self.vtraj_s = np.copy(self.vtraj)
				self.traj_sc = np.copy(self.traj)
				self.vtraj_sc = np.copy(self.vtraj)
				self.ltraj = np.tile(edge_l, (frames+1,1))
				self.ktraj = np.tile(edge_k, (frames+1,1))

			return sol

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
			- fix: Boolean indicating which degrees of freedom are held fixed.
			- network: Network edge properties obtained from _edge_lists().
			- applied_args: Simulation arguments, which can vary by problem.
			- train: The type of training to perform.
			- method: Used only if train is nonzero. Specifies the type of learning rule to use. Default is
			  'learning'.
			- eta: Nudge factor by which to increment applied strain towards the target. Default is 1, which
			  corresponds to pinning directly at the target.
			- alpha: Learning rate of each learning degree of freedom (stiffnesses or rest lengths). Default is 1e-3.
			- vmin: The smallest allowed value for each learning degree of freedom.
			- vsmooth: The value of each learning degree of freedom at which to begin smooth ramp down to vmin.
			- symmetric: Whether to train symmetrically with an additional free and clamped state.
			- pbar: Whether to display a progress bar. Default is True. 

		Returns
		-------
		float
			Total energy of the network.
		'''

		T, fix, network, applied_args, train, method, eta, alpha, vmin, vsmooth, symmetric, pbar = args
		edge_i, edge_j, edge_k, edge_l, edge_t = network
		n = self.n
		ne = self.ne
		offset = 12*n
		if symmetric: offset += 12*n

		if train & 1:
			l = q[offset:offset+ne]
			offset += ne
		else:
			l = edge_l
		if train & 2:
			k = q[offset:offset+ne]
		else:
			k = edge_k

		en = 0.
		en += self._elastic_energy(t, n, q, k, l, network)
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
			- fix: Boolean indicating which degrees of freedom are held fixed.
			- network: Network edge properties obtained from _edge_lists().
			- applied_args: Simulation arguments, which can vary by problem.
			- train: The type of training to perform.
			- method: Used only if train is nonzero. Specifies the type of learning rule to use. Default is
			  'learning'.
			- eta: Nudge factor by which to increment applied strain towards the target. Default is 1, which
			  corresponds to pinning directly at the target.
			- alpha: Learning rate of each learning degree of freedom (stiffnesses or rest lengths). Default is 1e-3.
			- vmin: The smallest allowed value for each learning degree of freedom.
			- vsmooth: The value of each learning degree of freedom at which to begin smooth ramp down to vmin.
			- symmetric: Whether to train symmetrically with an additional free and clamped state.
			- pbar: Whether to display a progress bar. Default is True. 

		Returns
		-------
		ndarray
			Derivative of the degrees of freedom.
		'''

		T, fix, network, applied_args, train, method, eta, alpha, vmin, vsmooth, symmetric, pbar = args
		edge_i, edge_j, edge_k, edge_l, edge_t = network
		n = self.n
		ne = self.ne

		fun = np.zeros_like(q)
		acc = fun[3*n:6*n]

		if train:
			q_c = q[6*n:12*n]
			fun_c = fun[6*n:12*n]
			acc_c = fun_c[3*n:6*n]
			offset = 12*n

			if symmetric:
				q_s = q[12*n:18*n]
				fun_s = fun[12*n:18*n]
				acc_s = fun_s[3*n:6*n]
				q_sc = q[18*n:24*n]
				fun_sc = fun[18*n:24*n]
				acc_sc = fun_sc[3*n:6*n]
				offset += 12*n

			if train & 1:
				l = q[offset:offset+ne]
				dl = fun[offset:offset+ne]
				offset += ne
			else:
				l = edge_l
			if train & 2:
				k = q[offset:offset+ne]
				dk = fun[offset:offset+ne]
			else:
				k = edge_k

			if train & 1:
				if method == 'learning':
					self._length_update_learning(t, n, q, q_c, k, l, dl, eta, alpha[0], vmin[0], vsmooth[0], network)
					if symmetric:
						self._length_update_learning(t, n, q_s, q_sc, k, l, dl, eta, alpha[0], vmin[0], vsmooth[0], network)
				else:
					self._length_update_aging(t, n, q, q_c, k, l, dl, eta, alpha[0], vmin[0], vsmooth[0], network)
					if symmetric:
						self._length_update_aging(t, n, q_s, q_sc, k, l, dl, eta, alpha[0], vmin[0], vsmooth[0], network)
			if train & 2:
				if method == 'learning':
					self._stiffness_update_learning(t, n, q, q_c, k, l, dk, eta, alpha[1], vmin[1], vsmooth[1], network)
					if symmetric:
						self._stiffness_update_learning(t, n, q_s, q_sc, k, l, dk, eta, alpha[1], vmin[1], vsmooth[1], network)
				else:
					self._stiffness_update_aging(t, n, q, q_c, k, l, dk, eta, alpha[1], vmin[1], vsmooth[1], network)
					if symmetric:
						self._stiffness_update_aging(t, n, q_s, q_sc, k, l, dk, eta, alpha[1], vmin[1], vsmooth[1], network)

		else:
			l = edge_l
			k = edge_k
		
		# base network
		self._drag_force(t, n, q, fun, network, self.params['drag'])
		self._dashpot_force(t, n, q, l, acc, network, self.params['dashpot'])
		self._elastic_force(t, n, q, k, l, acc, network)

		if train:

			# clamped state
			self._drag_force(t, n, q_c, fun_c, network, self.params['drag'])
			self._dashpot_force(t, n, q_c, l, acc_c, network, self.params['dashpot'])
			self._elastic_force(t, n, q_c, k, l, acc_c, network)
			self._applied_force(t, n, q, q_c, acc, acc_c, T, applied_args, train, eta, False)

			# zero out force on any fixed nodes
			acc[fix] = 0.
			acc_c[fix] = 0.

			if symmetric:
				# second free state
				self._drag_force(t, n, q_s, fun_s, network, self.params['drag'])
				self._dashpot_force(t, n, q_s, l, acc_s, network, self.params['dashpot'])
				self._elastic_force(t, n, q_s, k, l, acc_s, network)
				
				# second clamped state
				self._drag_force(t, n, q_sc, fun_sc, network, self.params['drag'])
				self._dashpot_force(t, n, q_sc, l, acc_sc, network, self.params['dashpot'])
				self._elastic_force(t, n, q_sc, k, l, acc_sc, network)
				self._applied_force(t, n, q_s, q_sc, acc_s, acc_sc, T, applied_args, train, eta, symmetric)
				
				acc_s[fix] = 0.
				acc_sc[fix] = 0.

		else:
			self._applied_force(t, n, q, q, acc, acc, T, applied_args, train, eta, False)
			
			# zero out force on any fixed nodes
			acc[fix] = 0.

		# update progress bar
		if pbar:
			dt = t - self.tp
			self.pbar.update(dt)
			self.tp = t

		return fun

	def _jj(self, t, q, *args):
		'''Compute the jacobian of the derivative of the degrees of freedom of the spring network.
		
		Parameters
		----------
		t : float
			The current time.
		q : ndarray
			The degrees of freedom.
		args : tuple
			Collection of simulation arguments :
			
			- T: Period for oscillatory force. If T = 0, nodes with an applied force are held stationary.
			- fix: Boolean indicating which degrees of freedom are held fixed.
			- network: Network edge properties obtained from _edge_lists().
			- applied_args: Simulation arguments, which can vary by problem.
			- train: The type of training to perform.
			- method: Used only if train is nonzero. Specifies the type of learning rule to use. Default is
			  'learning'.
			- eta: Nudge factor by which to increment applied strain towards the target. Default is 1, which
			  corresponds to pinning directly at the target.
			- alpha: Learning rate of each learning degree of freedom (stiffnesses or rest lengths). Default is 1e-3.
			- vmin: The smallest allowed value for each learning degree of freedom.
			- vsmooth: The value of each learning degree of freedom at which to begin smooth ramp down to vmin.
			- symmetric: Whether to train symmetrically with an additional free and clamped state.
			- pbar: Whether to display a progress bar. Default is True. 

		Returns
		-------
		ndarray
			Jacobian of the derivative.
		'''

		T, fix, network, applied_args, train, method, eta, alpha, vmin, vsmooth, symmetric, pbar = args
		edge_i, edge_j, edge_k, edge_l, edge_t = network
		n = self.n
		ne = self.ne

		jac = np.zeros((len(q),len(q)))
		for i in range(3*n):
			jac[i,3*n+i] = 1
		dfdx = jac[3*n:6*n,:3*n]
		dfdv = jac[3*n:6*n,3*n:6*n]

		if train:
			q_c = q[6*n:12*n]
			for i in range(3*n):
				jac[6*n+i,9*n+i] = 1
			dfdx_c = jac[9*n:12*n,6*n:9*n]
			dfdv_c = jac[9*n:12*n,9*n:12*n]
			dfdx_f = jac[9*n:12*n,:3*n] # deriv of f_c with respect to x_f
			offset = 12*n

			if symmetric:
				q_s = q[12*n:18*n]
				q_sc = q[18*n:24*n]
				for i in range(3*n):
					jac[12*n+i,15*n+i] = 1
					jac[18*n+i,21*n+i] = 1
				dfdx_s = jac[15*n:18*n,12*n:15*n]
				dfdv_s = jac[15*n:18*n,15*n:18*n]
				dfdx_sc = jac[21*n:24*n,18*n:21*n]
				dfdv_sc = jac[21*n:24*n,21*n:24*n]
				dfdx_sf = jac[21*n:24*n,12*n:15*n] # deriv of f_sc with respect to x_sf
				offset += 12*n

			if train & 1:
				l = q[offset:offset+ne]
				dfdl = jac[3*n:6*n,offset:offset+ne]
				dfdl_c = jac[9*n:12*n,offset:offset+ne]
				dgldx = jac[offset:offset+ne,:3*n]
				dgldx_c = jac[offset:offset+ne,6*n:9*n]
				dgldl = jac[offset:offset+ne,offset:offset+ne]

				if symmetric:
					dfdl_s = jac[15*n:18*n,offset:offset+ne]
					dfdl_sc = jac[21*n:24*n,offset:offset+ne]
					dgldx_s = jac[offset:offset+ne,12*n:15*n]
					dgldx_sc = jac[offset:offset+ne,18*n:21*n]
				
				if train & 2:
					dgldk = jac[offset:offset+ne,offset+ne:offset+2*ne]
				else:
					dgldk = np.zeros((ne,ne))
				offset += ne
			else:
				l = edge_l

			if train & 2:
				k = q[offset:offset+ne]
				dfdk = jac[3*n:6*n,offset:offset+ne]
				dfdk_c = jac[9*n:12*n,offset:offset+ne]
				dgkdx = jac[offset:offset+ne,:3*n]
				dgkdx_c = jac[offset:offset+ne,6*n:9*n]
				dgkdk = jac[offset:offset+ne,offset:offset+ne]

				if symmetric:
					dfdk_s = jac[15*n:18*n,offset:offset+ne]
					dfdk_sc = jac[21*n:24*n,offset:offset+ne]
					dgkdx_s = jac[offset:offset+ne,12*n:15*n]
					dgkdx_sc = jac[offset:offset+ne,18*n:21*n]

				if train & 1:
					dgkdl = jac[offset:offset+ne,offset-ne:offset]
				else:
					dgkdl = np.zeros((ne,ne))
			else:
				k = edge_k

			if train & 1:
				if method == 'learning':
					self._length_jacobian_learning(t, n, q, q_c, k, l, dgldx, dgldx_c, dgldk, dgldl, eta, alpha[0], vmin[0], vsmooth[0], network)
					if symmetric:
						self._length_jacobian_learning(t, n, q_s, q_sc, k, l, dgldx_s, dgldx_sc, dgldk, dgldl, eta, alpha[0], vmin[0], vsmooth[0], network)
				else:
					self._length_jacobian_aging(t, n, q, q_c, k, l, dgldx, dgldx_c, dgldk, dgldl, eta, alpha[0], vmin[0], vsmooth[0], network)
					if symmetric:
						self._length_jacobian_aging(t, n, q_s, q_sc, k, l, dgldx_s, dgldx_sc, dgldk, dgldl, eta, alpha[0], vmin[0], vsmooth[0], network)
				
				self._length_jacobian(t, n, q, k, l, dfdl, network)
				self._length_jacobian(t, n, q_c, k, l, dfdl_c, network)

				if symmetric:
					self._length_jacobian(t, n, q_s, k, l, dfdl_s, network)
					self._length_jacobian(t, n, q_sc, k, l, dfdl_sc, network)
					dfdl_s[fix] = 0.
					dfdl_sc[fix] = 0.
				
				# zero out jacobian for any fixed nodes
				dfdl[fix] = 0.
				dfdl_c[fix] = 0.

			if train & 2:
				if method == 'learning':
					self._stiffness_jacobian_learning(t, n, q, q_c, k, l, dgkdx, dgkdx_c, dgkdk, dgkdl, eta, alpha[1], vmin[1], vsmooth[1], network)
					if symmetric:
						self._stiffness_jacobian_learning(t, n, q_s, q_sc, k, l, dgkdx_s, dgkdx_sc, dgkdk, dgkdl, eta, alpha[1], vmin[1], vsmooth[1], network)
				else:
					self._stiffness_jacobian_aging(t, n, q, q_c, k, l, dgkdx, dgkdx_c, dgkdk, dgkdl, eta, alpha[1], vmin[1], vsmooth[1], network)
					if symmetric:
						self._stiffness_jacobian_aging(t, n, q_s, q_sc, k, l, dgkdx_s, dgkdx_sc, dgkdk, dgkdl, eta, alpha[1], vmin[1], vsmooth[1], network)

				self._stiffness_jacobian(t, n, q, k, l, dfdk, network)
				self._stiffness_jacobian(t, n, q_c, k, l, dfdk_c, network)

				if symmetric:
					self._stiffness_jacobian(t, n, q_s, k, l, dfdk_s, network)
					self._stiffness_jacobian(t, n, q_sc, k, l, dfdk_sc, network)
					dfdk_s[fix] = 0.
					dfdk_sc[fix] = 0.

				# zero out jacobian for any fixed nodes
				dfdk[fix] = 0.
				dfdk_c[fix] = 0.

		else:
			l = edge_l
			k = edge_k

		# base network
		self._drag_jacobian(t, n, q, dfdv, network, self.params['drag'])
		self._dashpot_jacobian(t, n, q, l, dfdx, dfdv, network, self.params['dashpot'])
		self._elastic_jacobian(t, n, q, k, l, dfdx, network)

		if train:
			# clamped state
			self._drag_jacobian(t, n, q_c, dfdv_c, network, self.params['drag'])
			self._dashpot_jacobian(t, n, q_c, l, dfdx_c, dfdv_c, network, self.params['dashpot'])
			self._elastic_jacobian(t, n, q_c, k, l, dfdx_c, network)
			self._applied_jacobian(t, n, q, q_c, dfdx, dfdx_c, dfdx_f, T, applied_args, train, eta, False)

			if symmetric:
				# second free state
				self._drag_jacobian(t, n, q_s, dfdv_s, network, self.params['drag'])
				self._dashpot_jacobian(t, n, q_s, l, dfdx_s, dfdv_s, network, self.params['dashpot'])
				self._elastic_jacobian(t, n, q_s, k, l, dfdx_s, network)
				
				# second clamped state
				self._drag_jacobian(t, n, q_sc, dfdv_sc, network, self.params['drag'])
				self._dashpot_jacobian(t, n, q_sc, l, dfdx_sc, dfdv_sc, network, self.params['dashpot'])
				self._elastic_jacobian(t, n, q_sc, k, l, dfdx_sc, network)
				self._applied_jacobian(t, n, q_s, q_sc, dfdx_s, dfdx_sc, dfdx_sf, T, applied_args, train, eta, symmetric)
				
				dfdx_s[fix] = 0.; dfdx_s[:,fix] = 0.
				dfdv_s[fix] = 0.; dfdv_s[:,fix] = 0.
				dfdx_sc[fix] = 0.; dfdx_sc[:,fix] = 0.
				dfdv_sc[fix] = 0.; dfdv_sc[:,fix] = 0.
				dfdx_sf[fix] = 0.; dfdx_sf[:,fix] = 0.
				

			# zero out jacobian for any fixed nodes
			dfdx[fix] = 0.; dfdx[:,fix] = 0.
			dfdv[fix] = 0.; dfdv[:,fix] = 0.
			dfdx_c[fix] = 0.; dfdx_c[:,fix] = 0.
			dfdv_c[fix] = 0.; dfdv_c[:,fix] = 0.
			dfdx_f[fix] = 0.; dfdx_f[:,fix] = 0.

		else:
			self._applied_jacobian(t, n, q, q, dfdx, dfdx, dfdx, T, applied_args, train, eta, False)

			# zero out jacobian for any fixed nodes
			dfdx[fix] = 0.; dfdx[:,fix] = 0.
			dfdv[fix] = 0.; dfdv[:,fix] = 0.

		return jac

	def _applied_force(self, t, n, q, q_c, acc, acc_c, T, applied_args, train, eta, symmetric):
		raise NotImplementedError

	def _applied_jacobian(self, t, n, q, q_c, dfdx, dfdx_c, dfdx_f, T, applied_args, train, eta, symmetric):
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
	def _elastic_energy(t, n, q, k, l, network):
		'''Compute the energy contribution due to pairwise interactions of bonded nodes.

		Parameters
		----------
		t : float
			The current time.
		n : int
			The number of nodes.
		q : ndarray
			The positions of the nodes.
		k : ndarray
			The stiffness of each bond. Different from network lists if it is a
			learning degree of freedom.
		l : ndarray
			The rest length of each bond. Different from network lists if it is a
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
	def _elastic_force(t, n, q, k, l, acc, network):
		'''Apply elastic forces between bonded nodes.

		Parameters
		----------
		t : float
			The current time.
		n : int
			The number of nodes.
		q : ndarray
			The positions of the nodes.
		k : ndarray
			The stiffness of each bond. Different from network lists if it is a
			learning degree of freedom.
		l : ndarray
			The rest length of each bond. Different from network lists if it is a
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
	def _elastic_jacobian(t, n, q, k, l, jac, network):
		'''Compute the jacobian of elastic forces between bonded nodes.

		Parameters
		----------
		t : float
			The current time.
		n : int
			The number of nodes.
		q : ndarray
			The positions of the nodes.
		k : ndarray
			The stiffness of each bond. Different from network lists if it is a
			learning degree of freedom.
		l : ndarray
			The rest length of each bond. Different from network lists if it is a
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
	def _length_jacobian(t, n, q, k, l, dfdl, network):
		'''Compute the jacobian of elastic forces due to changes in stiffness.

		Parameters
		----------
		t : float
			The current time.
		n : int
			The number of nodes.
		q : ndarray
			The positions of the nodes.
		k : ndarray
			The stiffness of each bond. Different from network lists if it is a
			learning degree of freedom.
		l : ndarray
			The rest length of each bond. Different from network lists if it is a
			learning degree of freedom.
		dfdl :ndarray
			The jacobian of elastic forces, populated as output.
		network : tuple of ndarrays
			Network edge properties obtained from _edge_lists().
		'''

		edge_i, edge_j, edge_k, edge_l, edge_t = network
		for e,(i, j) in enumerate(zip(edge_i, edge_j)):
			xi, yi, zi = q[3*i], q[3*i+1], q[3*i+2]
			xj, yj, zj = q[3*j], q[3*j+1], q[3*j+2]
			dx = xi-xj; dy = yi-yj; dz = zi-zj
			r = np.sqrt(dx**2 + dy**2 + dz**2)
			rfac = k[e]/r

			dfdl[3*i,e] += rfac*dx
			dfdl[3*i+1,e] += rfac*dy
			dfdl[3*i+2,e] += rfac*dz
			dfdl[3*j,e] += -rfac*dx
			dfdl[3*j+1,e] += -rfac*dy
			dfdl[3*j+2,e] += -rfac*dz

	@staticmethod
	@jit(nopython=True)
	def _stiffness_jacobian(t, n, q, k, l, dfdk, network):
		'''Compute the jacobian of elastic forces due to changes in stiffness.

		Parameters
		----------
		t : float
			The current time.
		n : int
			The number of nodes.
		q : ndarray
			The positions of the nodes.
		k : ndarray
			The stiffness of each bond. Different from network lists if it is a
			learning degree of freedom.
		l : ndarray
			The rest length of each bond. Different from network lists if it is a
			learning degree of freedom.
		dfdk :ndarray
			The jacobian of elastic forces, populated as output.
		network : tuple of ndarrays
			Network edge properties obtained from _edge_lists().
		'''

		edge_i, edge_j, edge_k, edge_l, edge_t = network
		for e,(i, j) in enumerate(zip(edge_i, edge_j)):
			xi, yi, zi = q[3*i], q[3*i+1], q[3*i+2]
			xj, yj, zj = q[3*j], q[3*j+1], q[3*j+2]
			dx = xi-xj; dy = yi-yj; dz = zi-zj
			r = np.sqrt(dx**2 + dy**2 + dz**2)
			rfac = (1 - l[e]/r)

			dfdk[3*i,e] += -rfac*dx
			dfdk[3*i+1,e] += -rfac*dy
			dfdk[3*i+2,e] += -rfac*dz
			dfdk[3*j,e] += rfac*dx
			dfdk[3*j+1,e] += rfac*dy
			dfdk[3*j+2,e] += rfac*dz


	@staticmethod
	@jit(nopython=True)
	def _length_update_learning(t, n, q, q_c, k, l, dl, eta, alpha, lmin, lsmooth, network):
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
		k : ndarray
			The stiffness of each bond.
		l : ndarray
			The rest length of each bond.
		dl : ndarray
			The derivative of the rest lengths with time, populated on output.
		eta : float
			The nudge factor.
		alpha : float
			The learning rate.
		lmin : float
			The minimum allowed rest length.
		lsmooth : float
			Value at which to begin smoothly ramping to lmin.
		network : tuple of ndarrays
			Network edge properties obtained from _edge_lists().
		'''

		edge_i, edge_j, edge_k, edge_l, edge_t = network
		for e, (i, j, train) in enumerate(zip(edge_i, edge_j, edge_t)):
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

				s=(l[e]-lmin)/(lsmooth-lmin)
				if s < 0: s = 0
				if s > 1: s = 1
				sf = s*s*s*(s*(s*6-15)+10)
				dl[e] += -alpha/eta*k[e]*(r-r_c)*sf

	@staticmethod
	@jit(nopython=True)
	def _length_jacobian_learning(t, n, q, q_c, k, l, dgdx, dgdx_c, dgdk, dgdl, eta, alpha, lmin, lsmooth, network):
		'''Jacobian for edge length update.
		
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
		k : ndarray
			The stiffness of each bond.
		l : ndarray
			The rest length of each bond.
		dgdx : ndarray
			Memory for storing derivative of the learning rule with respect to free state positions.
		dgdx_c : ndarray
			Memory for storing the derivative of the learning rule with respect to clamped state positions.
		dgdk : ndarray
			Memory for storing the derivative of the learning rule with respect to stiffnesses.
		dgdl : ndarray
			Memory for storing the derivative of the learning rule with respect to rest lengths.
		eta : float
			The nudge factor.
		alpha : float
			The learning rate.
		lmin : float
			The minimum allowed rest length.
		lsmooth : float
			Value at which to begin smoothly ramping to lmin.
		network : tuple of ndarrays
			Network edge properties obtained from _edge_lists().
		'''

		edge_i, edge_j, edge_k, edge_l, edge_t = network
		for e, (i, j, train) in enumerate(zip(edge_i, edge_j, edge_t)):
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

				s=(l[e]-lmin)/(lsmooth-lmin)
				if s < 0: s = 0
				if s > 1: s = 1
				sf = s*s*s*(s*(s*6-15)+10)
				dsf = 30*s*s*(s*(s-2)+1)/(lsmooth-lmin)

				fac = -alpha/eta*k[e]*sf

				dgdx[e,3*i] += fac/r*dx
				dgdx[e,3*i+1] += fac/r*dy
				dgdx[e,3*i+2] += fac/r*dz
				dgdx[e,3*j] += -fac/r*dx
				dgdx[e,3*j+1] += -fac/r*dy
				dgdx[e,3*j+2] += -fac/r*dz

				dgdx_c[e,3*i] += -fac/r_c*dx_c
				dgdx_c[e,3*i+1] += -fac/r_c*dy_c
				dgdx_c[e,3*i+2] += -fac/r_c*dz_c
				dgdx_c[e,3*j] += fac/r_c*dx_c
				dgdx_c[e,3*j+1] += fac/r_c*dy_c
				dgdx_c[e,3*j+2] += fac/r_c*dz_c

				dgdk[e,e] += -alpha/eta*(r-r_c)*sf
				dgdl[e,e] += -alpha/eta*k[e]*(r-r_c)*dsf

	@staticmethod
	@jit(nopython=True)
	def _length_update_aging(t, n, q, q_c, k, l, dl, eta, alpha, lmin, lsmooth, network):
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
		k : ndarray
			The stiffness of each bond.
		l : ndarray
			The rest length of each bond.
		dl : ndarray
			The derivative of the rest lengths with time, populated on output.
		eta : float
			The nudge factor.
		alpha : float
			The learning rate.
		lmin : float
			The minimum allowed rest length.
		lsmooth : float
			Value at which to begin smooth ramp down to lmin.
		network : tuple of ndarrays
			Network edge properties obtained from _edge_lists().
		'''

		edge_i, edge_j, edge_k, edge_l, edge_t = network
		for e, (i, j, train) in enumerate(zip(edge_i, edge_j, edge_t)):
			if train:
				# clamped state
				xi_c, yi_c, zi_c = q_c[3*i], q_c[3*i+1], q_c[3*i+2]
				xj_c, yj_c, zj_c = q_c[3*j], q_c[3*j+1], q_c[3*j+2]
				dx_c = xi_c-xj_c; dy_c = yi_c-yj_c; dz_c = zi_c-zj_c
				r_c = np.sqrt(dx_c**2 + dy_c**2 + dz_c**2)

				s=(l[e]-lmin)/(lsmooth-lmin)
				if s < 0: s = 0
				if s > 1: s = 1
				sf = s*s*s*(s*(s*6-15)+10)
				dl[e] += alpha/eta*k[e]*(r_c-l[e])*sf

	@staticmethod
	@jit(nopython=True)
	def _length_jacobian_aging(t, n, q, q_c, k, l, dgdx, dgdx_c, dgdk, dgdl, eta, alpha, lmin, lsmooth, network):
		'''Jacobian for edge length update.
		
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
		l : ndarray
			The rest length of each bond.
		dgdx : ndarray
			Memory for storing derivative of the learning rule with respect to free state positions.
		dgdx_c : ndarray
			Memory for storing the derivative of the learning rule with respect to clamped state positions.
		dgdk : ndarray
			Memory for storing the derivative of the learning rule with respect to stiffnesses.
		dgdl : ndarray
			Memory for storing the derivative of the learning rule with respect to rest lengths.
		eta : float
			The nudge factor.
		alpha : float
			The learning rate.
		lmin : float
			The minimum allowed rest length.
		lsmooth : float
			Value at which to begin smooth ramp down to lmin.
		network : tuple of ndarrays
			Network edge properties obtained from _edge_lists().
		'''

		edge_i, edge_j, edge_k, edge_l, edge_t = network
		for e, (i, j, train) in enumerate(zip(edge_i, edge_j, edge_t)):
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

				s=(l[e]-lmin)/(lsmooth-lmin)
				if s < 0: s = 0
				if s > 1: s = 1
				sf = s*s*s*(s*(s*6-15)+10)
				dsf = 30*s*s*(s*(s-2)+1)/(lsmooth-lmin)

				fac = alpha/eta*k[e]*sf

				dgdx_c[e,3*i] += fac/r_c*dx_c
				dgdx_c[e,3*i+1] += fac/r_c*dy_c
				dgdx_c[e,3*i+2] += fac/r_c*dz_c
				dgdx_c[e,3*j] += -fac/r_c*dx_c
				dgdx_c[e,3*j+1] += -fac/r_c*dy_c
				dgdx_c[e,3*j+2] += -fac/r_c*dz_c

				dgdk[e,e] += alpha/eta*(r_c-l[e])*sf
				dgdl[e,e] += alpha/eta*k[e]*(-sf + (r_c-l[e])*dsf)

	@staticmethod
	@jit(nopython=True)
	def _stiffness_update_learning(t, n, q, q_c, k, l, dk, eta, alpha, kmin, ksmooth, network):
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
		l : ndarray
			The length of each bond.
		dk : ndarray
			The derivative of the stiffnesses with time, populated on output.
		eta : float
			The nudge factor.
		alpha : float
			The learning rate.
		kmin : float
			The minimum allowed stiffness.
		ksmooth : float
			Value at which to begin smooth ramp down to kmin.
		network : tuple of ndarrays
			Network edge properties obtained from _edge_lists().
		'''

		edge_i, edge_j, edge_k, edge_l, edge_t = network
		for e, (i, j, train) in enumerate(zip(edge_i, edge_j, edge_t)):
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

				# smoothstep
				s=(k[e]-kmin)/(ksmooth-kmin)
				if s < 0: s = 0
				if s > 1: s = 1
				sf = s*s*s*(s*(s*6-15)+10)
				dk[e] += 0.5*alpha/eta*((r-l[e])**2-(r_c-l[e])**2)*sf

	@staticmethod
	@jit(nopython=True)
	def _stiffness_jacobian_learning(t, n, q, q_c, k, l, dgdx, dgdx_c, dgdk, dgdl, eta, alpha, kmin, ksmooth, network):
		'''Jacobian for stiffness udpate.
		
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
		l : ndarray
			The length of each bond.
		dgdx : ndarray
			Memory for storing derivative of the learning rule with respect to free state positions.
		dgdx_c : ndarray
			Memory for storing the derivative of the learning rule with respect to clamped state positions.
		dgdk : ndarray
			Memory for storing the derivative of the learning rule with respect to stiffnesses.
		dgdl : ndarray
			Memory for storing the derivative of the learning rule with respect to rest lengths.
		eta : float
			The nudge factor.
		alpha : float
			The learning rate.
		kmin : float
			The minimum allowed stiffness.
		ksmooth : float
			Value at which to begin smooth ramp down to kmin.
		network : tuple of ndarrays
			Network edge properties obtained from _edge_lists().
		'''

		edge_i, edge_j, edge_k, edge_l, edge_t = network
		for e, (i, j, train) in enumerate(zip(edge_i, edge_j, edge_t)):
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
				
				s=(k[e]-kmin)/(ksmooth-kmin)
				if s < 0: s = 0
				if s > 1: s = 1
				sf = s*s*s*(s*(s*6-15)+10)
				dsf = 30*s*s*(s*(s-2)+1)/(ksmooth-kmin)

				fac = alpha/eta*sf
				rfac = (1 - l[e]/r)
				rfac_c = (1 - l[e]/r_c)

				dgdx[e,3*i] += fac*rfac*dx
				dgdx[e,3*i+1] += fac*rfac*dy
				dgdx[e,3*i+2] += fac*rfac*dz
				dgdx[e,3*j] += -fac*rfac*dx
				dgdx[e,3*j+1] += -fac*rfac*dy
				dgdx[e,3*j+2] += -fac*rfac*dz

				dgdx_c[e,3*i] += -fac*rfac_c*dx_c
				dgdx_c[e,3*i+1] += -fac*rfac_c*dy_c
				dgdx_c[e,3*i+2] += -fac*rfac_c*dz_c
				dgdx_c[e,3*j] += fac*rfac_c*dx_c
				dgdx_c[e,3*j+1] += fac*rfac_c*dy_c
				dgdx_c[e,3*j+2] += fac*rfac_c*dz_c

				dgdk[e,e] += 0.5*alpha/eta*((r-l[e])**2-(r_c-l[e])**2)*dsf
				dgdl[e,e] += -alpha/eta*(r-r_c)*sf

	@staticmethod
	@jit(nopython=True)
	def _stiffness_update_aging(t, n, q, q_c, k, l, dk, eta, alpha, kmin, ksmooth, network):
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
		l : ndarray
			The length of each bond.
		dk : ndarray
			The derivative of the stiffnesses with time, populated on output.
		eta : float
			The nudge factor.
		alpha : float
			The learning rate.
		kmin : float
			The minimum allowed stiffness.
		ksmooth : float
			Value at which to begin smooth ramp down to kmin.
		network : tuple of ndarrays
			Network edge properties obtained from _edge_lists().
		'''

		edge_i, edge_j, edge_k, edge_l, edge_t = network
		for e, (i, j, train) in enumerate(zip(edge_i, edge_j, edge_t)):
			if train:
				# clamped state
				xi_c, yi_c, zi_c = q_c[3*i], q_c[3*i+1], q_c[3*i+2]
				xj_c, yj_c, zj_c = q_c[3*j], q_c[3*j+1], q_c[3*j+2]
				dx_c = xi_c-xj_c; dy_c = yi_c-yj_c; dz_c = zi_c-zj_c
				r_c = np.sqrt(dx_c**2 + dy_c**2 + dz_c**2)

				s=(k[e]-kmin)/(ksmooth-kmin)
				if s < 0: s = 0
				if s > 1: s = 1
				sf = s*s*s*(s*(s*6-15)+10)
				dk[e] += -(alpha/eta*k[e]*(r_c-l[e])**2)*sf

	@staticmethod
	@jit(nopython=True)
	def _stiffness_jacobian_aging(t, n, q, q_c, k, l, dgdx, dgdx_c, dgdk, dgdl, eta, alpha, kmin, ksmooth, network):
		'''Jacobian for stiffness update.
		
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
		l : ndarray
			The length of each bond.
		dgdx : ndarray
			Memory for storing derivative of the learning rule with respect to free state positions.
		dgdx_c : ndarray
			Memory for storing the derivative of the learning rule with respect to clamped state positions.
		dgdk : ndarray
			Memory for storing the derivative of the learning rule with respect to stiffnesses.
		dgdl : ndarray
			Memory for storing the derivative of the learning rule with respect to rest lengths.
		eta : float
			The nudge factor.
		alpha : float
			The learning rate.
		kmin : float
			The minimum allowed stiffness.
		ksmooth : float
			Value at which to start smooth ramp down to kmin.
		network : tuple of ndarrays
			Network edge properties obtained from _edge_lists().
		'''

		edge_i, edge_j, edge_k, edge_l, edge_t = network
		for e, (i, j, train) in enumerate(zip(edge_i, edge_j, edge_t)):
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
				
				s=(k[e]-kmin)/(ksmooth-kmin)
				if s < 0: s = 0
				if s > 1: s = 1
				sf = s*s*s*(s*(s*6-15)+10)
				dsf = 30*s*s*(s*(s-2)+1)/(ksmooth-kmin)

				fac = 2*alpha/eta*k[e]*sf
				rfac_c = (1 - l[e]/r_c)

				dgdx_c[e,3*i] += -fac*rfac_c*dx_c
				dgdx_c[e,3*i+1] += -fac*rfac_c*dy_c
				dgdx_c[e,3*i+2] += -fac*rfac_c*dz_c
				dgdx_c[e,3*j] += fac*rfac_c*dx_c
				dgdx_c[e,3*j+1] += fac*rfac_c*dy_c
				dgdx_c[e,3*j+2] += fac*rfac_c*dz_c

				dgdk[e,e] += -(alpha/eta*(r_c-l[e])**2)*(sf +k[e]*dsf)
				dgdl[e,e] += 2*(alpha/eta*k[e]*(r_c-l[e]))*sf

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

	def compute_modes(self):
		'''Compute the normal modes of the elastic network at equilibrium.

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

		self._elastic_jacobian(t, self.n, q, edge_k, edge_l, hess, network)
		
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

	def _rigidity_pair(self, q, R, e, i, j):
		xi, yi, zi = q[3*i], q[3*i+1], q[3*i+2]
		xj, yj, zj = q[3*j], q[3*j+1], q[3*j+2]
		dx = xi-xj; dy = yi-yj; dz = zi-zj
		r = np.sqrt(dx**2 + dy**2 + dz**2)
		R[e,3*i] = dx/r; R[e,3*i+1] = dy/r; R[e,3*i+2] = dz/r
		R[e,3*j] = -dx/r; R[e,3*j+1] = -dy/r; R[e,3*j+2] = -dz/r

	def _pre_stress(self, q, Kp, Gx, Gy, Gz, e, i, j, k, l):
		xi, yi, zi = q[3*i], q[3*i+1], q[3*i+2]
		xj, yj, zj = q[3*j], q[3*j+1], q[3*j+2]
		dx = xi-xj; dy = yi-yj; dz = zi-zj
		r = np.sqrt(dx**2 + dy**2 + dz**2)
		tau = -k*(r-l) # tension (compression is positive)
		Kp[e,e] = tau/r
		Gx[e,3*i] = 1; Gx[e,3*j] = -1
		Gy[e,3*i+1] = 1; Gy[e,3*j+1] = -1
		Gz[e,3*i+2] = 1; Gz[e,3*j+2] = -1

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

		lim = (1+1./np.sqrt(self.n))*np.max(np.abs(self.pts))
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
		lx, ly, lz = np.max(self.pts, axis=0) - np.min(self.pts, axis=0)
		self._povray_props = {'light1':np.array([-10,10,10]),
							  'light2':np.array([10,10,10]),
							  'camera':np.array([1.25*lx,1.25*ly,1*lz])}

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
		c = 'rgb<0.35,0.7,0.7>'
		r = 5*self.params['radius']
		for i,node in enumerate(nodes):
			spheres[i] = Sphere(self.pts[node], r,
						 Texture(Pigment('color',c),
						 Finish('ambient',0.24,'diffuse',0.88,
						 'specular',0.3,'phong',0.2,'phong_size',5)))
		return spheres

	def _povray_edges(self, pairs):
		edges = [0 for _ in range(len(pairs))]
		c = 'rgb<0.1,0.3,0.3>'
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



