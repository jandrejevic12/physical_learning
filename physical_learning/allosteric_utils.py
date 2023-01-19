import numpy as np

from plot_imports import *
from elastic_utils import *

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import collections  as mc
import matplotlib.animation as animation

import networkx as nx

class Allosteric(Elastic):
	'''Class to train an allosteric response in a random elastic network.
	
	Parameters
	----------
	graph : str or networkx.graph
		If string, filename of saved graph specifying the nodes and edge connections of
		the elastic network. If networkx.graph object, the graph to use.
	auto : bool, optional
		Whether to select a source and target pair automatically at random. Default
		is False, which prompts user input for the source and target, unless the input
		graph already has at least one source specified. Additional sources and targets
		may be added with the methods add_sources(ns) and add_targets(nt).
	seed : int, optional
		The random seed to use if selecting sources and targets at random.
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
	seed : int
		A random seed used for selecting sources and targets at random.
	n : int
		Number of nodes in the network.
	ne : int
		Number of edges in the network.
	sources : list of dicts
		List of source node pairs. Each dict entry contains the following keys :

		- 'i': Index of the first node in the pair.
		- 'j': Index of the second node in the pair.
		- 'length': The rest separation of the nodes.
		- 'phase': +1 or -1, whether the source is stretched with positive or negative strain.
	targets : list of dicts
		List of target node pairs. Each dict entry contains the following keys :

		- 'i': Index of the first node in the pair.
		- 'j': Index of the second node in the pair.
		- 'length': The rest separation of the nodes.
		- 'phase': +1 or -1, whether the target is stretched with positive or negative strain.

	pts : ndarray
		(x,y) coordinates for each node in the system.
	degree : ndarray
		The degree (number of neighbors) of each node.
	Z : float
		The average coordination number, defined as 2*ne/nc, where nc is the number of nodes in the
		biggest connected component of the system.
	dZ : float
		The excess coordination, defined as Z - Ziso, where Ziso is the average coordination required
		for isostaticity (Ziso = 4 - 6/nc in 2D).
	traj : ndarray
		The simulated trajectory of the network produced after a call to the solve() routine.
	t_eval : ndarray
		The corresponding time at each simulated frame.
	'''
	
	def __init__(self, graph, auto=False, seed=12, params={'rfac':0.05, 'drag':0.01, 'dashpot':10., 'stiffness':1.}):
		self.seed = seed
		self.sources = []
		self.targets = []

		# Initialize the network
		if isinstance(graph, str):
			graph = self._load_network(graph)

		# Call superclass (Elastic) constructor and remove nodes
		# with a single bond.
		super().__init__(graph, params)
		self._remove_dangling_edges()

		# Choose initial set of source and target nodes, if none
		# are present initially.
		if len(self.sources) == 0:
			self._add_source_target(auto)

	'''
	*****************************************************************************************************
	*****************************************************************************************************

											NETWORK INITIALIZATION

	*****************************************************************************************************
	*****************************************************************************************************
	'''

	def _load_network(self, filename):
		'''Load a network from file.

		Parameters
		----------
		filename : str
			The name of the text file to read.

		Returns
		-------
		networkx.graph
			The loaded graph.
		'''

		with open(filename, 'r') as f:
			# Read nodes
			n = int(f.readline().rstrip('\n'))
			dpos = {}
			for i in range(n):
				line = f.readline().rstrip('\n').split()
				x, y = float(line[0]), float(line[1])
				dpos[i] = np.array([x,y])
			graph = nx.random_geometric_graph(n, 0, seed=self.seed, pos=dpos)
			
			# Read edges
			ne = int(f.readline().rstrip('\n'))
			for e in range(ne):
				line = f.readline().rstrip('\n').split()
				if len(line) > 2:
					i, j, k, l, train = int(line[0]), int(line[1]), float(line[2]), float(line[3]), bool(int(line[4]))
					graph.add_edge(i,j,stiffness=k,length=l,trainable=train)
				else:
					i, j = int(line[0]), int(line[1])
					graph.add_edge(i,j)
			
			flag = True
			while flag:
				try:
					ns = int(f.readline().rstrip('\n'))
				except:
					flag = False
				else:
					# Read sources
					for s in range(ns):
						line = f.readline().rstrip('\n').split()
						si, sj, sl, sp = int(line[0]), int(line[1]), float(line[2]), int(line[3])
						self.sources += [{'i':si, 'j':sj, 'length':sl, 'phase':sp}]

					# Read targets
					nt = int(f.readline().rstrip('\n'))
					for t in range(nt):
						line = f.readline().rstrip('\n').split()
						ti, tj, tl, sp = int(line[0]), int(line[1]), float(line[2]), int(line[3])
						self.targets += [{'i':ti, 'j':tj, 'length':tl, 'phase':sp}]
		return graph

	def _add_source_target(self, auto=False):
		'''Add a pair of source nodes and a pair of target nodes.
		
		Parameters
		----------
		auto : bool, optional
			Whether to select a source and target pair automatically at random. Default
			is False, which prompts user input for the source and target, unless the input
			graph already has at least one source specified. Additional sources and targets
			may be added with the methods add_sources(ns) and add_targets(nt).
		'''

		if auto:
			self._auto_source_target()
		else:
			self._pick_source_target()

	def _auto_source_target(self):
		'''Select a pair of source and target nodes at random.

		If any two connected nodes are selected, the edge between them is removed.
		'''

		np.random.seed(self.seed)
		edges = list(self.graph.edges())
		degi = 2; degj = 2
		while (degi <= 2) or (degj <= 2):
			se = np.random.randint(len(edges))
			si = edges[se][0]
			sj = edges[se][1]
			degi = self.graph.degree[si]
			degj = self.graph.degree[sj]
		self.sources += [{'i':si, 'j':sj, 'length':self._distance(self.pts[si], self.pts[sj]), 'phase':1}]
		ti = si; tj = sj; te = se
		while (te == se) or (ti == si) or (ti == sj) or (tj == si) or (tj == sj) or (degi <= 2) or (degj <= 2):
			te = np.random.randint(len(edges))
			ti = edges[te][0]
			tj = edges[te][1]
			degi = self.graph.degree[ti]
			degj = self.graph.degree[tj]
		phase = np.random.choice([-1,1])
		self.targets += [{'i':ti, 'j':tj, 'length':self._distance(self.pts[ti], self.pts[tj]), 'phase':phase}]

		# remove the selected bonds.
		self.graph.remove_edge(si,sj)
		self.graph.remove_edge(ti,tj)
		self._remove_dangling_edges()
		self._set_coordination()

		self.plot()

	def _pick_source_target(self):
		'''Select a pair of source and target nodes manually.

		A selection is made by clicking first each of two source nodes, then each of two target nodes,
		on the provided plot. If any two connected nodes are selected, the edge between them is removed.
		'''

		fig, ax = plt.subplots(1,1,figsize=(5,5))
		ax.set_title("Select two source nodes (blue), then two target nodes (red).", size=10)
		ec, dc = self.plot_network(ax)

		ip, = ax.plot(self.pts[:,0], self.pts[:,1], '.', color='k',
						picker=True, pickradius=5)

		# cover detached points in white
		r = 4*self.params['radius']*np.ones(self.n)[self.degree<2]
		dc2 = mc.EllipseCollection(r, r, np.zeros_like(r), offsets=self.pts[self.degree<2],
									  transOffset=ax.transData, units='x',
									  edgecolor='w', facecolor='w', linewidths=2, zorder=1000)
		ax.add_collection(dc2)

		colors = [pal['blue'],pal['blue'],pal['red'],pal['red']]
		self.pick_count = 0
		source = {'i':0, 'j':0, 'length':0, 'phase':1}
		target = {'i':0, 'j':0, 'length':0, 'phase':1}

		def onpick(event):
			artist = event.artist
			xdata = artist.get_xdata()
			ydata = artist.get_ydata()
			ind = event.ind
			pts = tuple(zip(xdata[ind], ydata[ind]))
			self.pick_count += 1
			if self.pick_count == 1:
				for pt in pts:
					ax.scatter(pt[0], pt[1], s=60, edgecolor='k', facecolor=colors[self.pick_count-1], lw=1, zorder=1000)
				source['i'] = ind[0]
			elif self.pick_count == 2:
				for pt in pts:
					ax.scatter(pt[0], pt[1], s=60, edgecolor='k', facecolor=colors[self.pick_count-1], lw=1, zorder=1000)
				source['j'] = ind[0]
				source['length'] = self._distance(self.pts[source['i']],self.pts[source['j']])
				if self.graph.has_edge(source['i'],source['j']):
					self.graph.remove_edge(source['i'],source['j'])
			elif self.pick_count == 3:
				for pt in pts:
					ax.scatter(pt[0], pt[1], s=60, edgecolor='k', facecolor=colors[self.pick_count-1], lw=1, zorder=1000)
				target['i'] = ind[0]
			elif self.pick_count == 4:
				for pt in pts:
					ax.scatter(pt[0], pt[1], s=60, edgecolor='k', facecolor=colors[self.pick_count-1], lw=1, zorder=1000)
				target['j'] = ind[0]
				target['length'] = self._distance(self.pts[target['i']],self.pts[target['j']])
				if self.graph.has_edge(target['i'],target['j']):
					self.graph.remove_edge(target['i'],target['j'])
				self.sources += [source]
				self.targets += [target]
				self._set_coordination()

		fig.canvas.mpl_connect('pick_event', onpick)

		self.set_axes(ax)
		fig.tight_layout()
		plt.show()

	def add_sources(self, ns):
		'''Add pairs of source nodes to the network.

		If any two connected nodes are selected, the edge between them is removed.

		Parameters
		----------
		ns : int
			The number of new pairs of source nodes to add.
		'''

		fig, ax = plt.subplots(1,1,figsize=(5,5))
		ax.set_title("Select {:d} source pairs.".format(ns), size=10)
		ec, dc = self.plot_network(ax)

		ip, = ax.plot(self.pts[:,0], self.pts[:,1], '.', color='k',
						picker=True, pickradius=5)

		# cover detached points in white
		r = 4*self.params['radius']*np.ones(self.n)[self.degree<2]
		dc2 = mc.EllipseCollection(r, r, np.zeros_like(r), offsets=self.pts[self.degree<2],
									  transOffset=ax.transData, units='x',
									  edgecolor='w', facecolor='w', linewidths=2, zorder=1000)
		ax.add_collection(dc2)

		for source in self.sources:
			self.plot_source(ax, source)

		for target in self.targets:
			self.plot_target(ax, target)


		color = pal['blue']
		self.pick_count = 0
		sources = [{'i':0, 'j':0, 'length':0, 'phase':1} for _ in range(ns)]

		def onpick(event):
			artist = event.artist
			xdata = artist.get_xdata()
			ydata = artist.get_ydata()
			ind = event.ind
			pts = tuple(zip(xdata[ind], ydata[ind]))
			self.pick_count += 1
			if self.pick_count > 0 and self.pick_count <= 2*ns:
				if self.pick_count % 2 == 1:
					for pt in pts:
						ax.scatter(pt[0], pt[1], s=60, edgecolor='k', facecolor=color, lw=1, zorder=1000)
					si = self.pick_count // 2
					sources[si]['i'] = ind[0]
				else:
					for pt in pts:
						ax.scatter(pt[0], pt[1], s=60, edgecolor='k', facecolor=color, lw=1, zorder=1000)
					si = self.pick_count // 2 - 1
					sources[si]['j'] = ind[0]
					sources[si]['length'] = self._distance(self.pts[sources[si]['i']],self.pts[sources[si]['j']])
					if self.graph.has_edge(sources[si]['i'],sources[si]['j']):
						self.graph.remove_edge(sources[si]['i'],sources[si]['j'])
				if self.pick_count == 2*ns:
					self.sources += sources
					self._set_coordination()

		fig.canvas.mpl_connect('pick_event', onpick)

		self.set_axes(ax)
		fig.tight_layout()
		plt.show()

	def add_targets(self, nt):
		'''Add pairs of target nodes to the network.

		If any two connected nodes are selected, the edge between them is removed.

		Parameters
		----------
		nt : int
			The number of new pairs of target nodes to add.
		'''

		fig, ax = plt.subplots(1,1,figsize=(5,5))
		ax.set_title("Select {:d} target pairs.".format(nt), size=10)
		ec, dc = self.plot_network(ax)

		ip, = ax.plot(self.pts[:,0], self.pts[:,1], '.', color='k',
						picker=True, pickradius=5)

		# cover detached points in white
		r = 4*self.params['radius']*np.ones(self.n)[self.degree<2]
		dc2 = mc.EllipseCollection(r, r, np.zeros_like(r), offsets=self.pts[self.degree<2],
									  transOffset=ax.transData, units='x',
									  edgecolor='w', facecolor='w', linewidths=2, zorder=1000)
		ax.add_collection(dc2)

		for source in self.sources:
			self.plot_source(ax, source)

		for target in self.targets:
			self.plot_target(ax, target)


		color = pal['red']
		self.pick_count = 0
		targets = [{'i':0, 'j':0, 'length':0, 'phase':1} for _ in range(nt)]

		def onpick(event):
			artist = event.artist
			xdata = artist.get_xdata()
			ydata = artist.get_ydata()
			ind = event.ind
			pts = tuple(zip(xdata[ind], ydata[ind]))
			self.pick_count += 1
			if self.pick_count > 0 and self.pick_count <= 2*nt:
				if self.pick_count % 2 == 1:
					for pt in pts:
						ax.scatter(pt[0], pt[1], s=60, edgecolor='k', facecolor=color, lw=1, zorder=1000)
					ti = self.pick_count // 2
					targets[ti]['i'] = ind[0]
				else:
					for pt in pts:
						ax.scatter(pt[0], pt[1], s=60, edgecolor='k', facecolor=color, lw=1, zorder=1000)
					ti = self.pick_count // 2 - 1
					targets[ti]['j'] = ind[0]
					targets[ti]['length'] = self._distance(self.pts[targets[ti]['i']],self.pts[targets[ti]['j']])
					if self.graph.has_edge(targets[ti]['i'],targets[ti]['j']):
						self.graph.remove_edge(targets[ti]['i'],targets[ti]['j'])
				if self.pick_count == 2*nt:
					self.targets += targets
					self._set_coordination()

		fig.canvas.mpl_connect('pick_event', onpick)

		self.set_axes(ax)
		fig.tight_layout()
		plt.show()

	def save(self, filename):
		'''Save the network to a file.
		   
		Parameters
		----------
		filename : str
			The name of the text file to write.
		'''

		self.reset_init()
		with open(filename, 'w') as f:
			# write nodes
			f.write(str(self.n)+'\n')
			for i in range(self.n):
				f.write('{:.12g} {:.12g}\n'.format(self.pts[i,0],self.pts[i,1]))

			# write edges
			f.write(str(len(self.graph.edges()))+'\n')
			for edge in self.graph.edges(data=True):
				f.write('{:d} {:d} {:.12g} {:.12g} {:d}\n'.format(edge[0],edge[1],edge[2]['stiffness'],edge[2]['length'],edge[2]['trainable']))

			# write sources
			f.write(str(len(self.sources))+'\n')
			for source in self.sources:
				f.write('{:d} {:d} {:.12g} {:d}\n'.format(source['i'],source['j'],source['length'],source['phase']))

			# write targets
			f.write(str(len(self.targets))+'\n')
			for target in self.targets:
				f.write('{:d} {:d} {:.12g} {:d}\n'.format(target['i'],target['j'],target['length'],target['phase']))

	'''
	*****************************************************************************************************
	*****************************************************************************************************

										NUMERICAL INTEGRATION

	*****************************************************************************************************
	*****************************************************************************************************
	'''

	def _applied_energy(self, t, n, q, _q, T, applied_args, eta):
		'''Compute the elastic energy due to applied forces at the source(s) and target(s).
		
		Parameters
		----------
		t : float
			The current time.
		n : int
			The number of nodes.
		q : ndarray
			The 2*n position coordinates of the nodes in the free state.
		_q : ndarray
			The 2*n position coordinates of the nodes in the clamped state.
		T : float
			Period for oscillatory force. If T = 0, sources and targets are held
			stationary.
		applied_args : tuple
			Simulation arguments: the source strain(s), target strain(s), and pinning stiffness.
		eta : float
			Learning rate by which to increment clamped strain towards the target.

		Returns
		-------
		float
			Energy contribution due to applied forces. 
		'''

		ess, ets, k = applied_args
		en = 0

		if not hasattr(ess, '__len__'): ess = len(self.sources)*[ess]
		
		for es, source in zip(ess, self.sources):
			if np.abs(es) > 0:
				en += self._applied_strain_energy(t, n, q, (source, es, k, T))
		return en

	def _applied_force(self, t, n, q, _q, acc, _acc, T, applied_args, eta):
		'''Compute the applied force at the source(s) and target(s).
		
		Parameters
		----------
		t : float
			The current time.
		n : int
			The number of nodes.
		q : ndarray
			The 2*n position coordinates of the nodes in the free state.
		_q : ndarray
			The 2*n position coordinates of the nodes in the clamped state.
		acc : ndarray
			The acceleration of each node in the free state. Applied forces are added
			into this array as output.
		_acc : ndarray
			The acceleration of each node in the clamped state. Applied forces are added
			into this array as output.
		T : float
			Period for oscillatory force. If T = 0, sources and targets are held
			stationary.
		applied_args : tuple
			Simulation arguments: the source strain(s), target strain(s), and pinning stiffness.
		eta : float
			Learning rate by which to increment clamped strain towards the target.
		'''

		ess, ets, k = applied_args

		if not hasattr(ess, '__len__'): ess = len(self.sources)*[ess]
		if not hasattr(ets, '__len__'): ets = len(self.targets)*[ets]

		for es, source in zip(ess, self.sources):
			if np.abs(es) > 0:
				self._applied_strain_force(t, n, q, acc, (source, es, k, T))
				self._applied_strain_force(t, n, _q, _acc, (source, es, k, T))

		for et, target in zip(ets, self.targets):
			if np.abs(et) > 0:
				i, j = target['i'], target['j']
				l0 = target['length']
				l = np.sqrt((q[2*i]-q[2*j])**2+(q[2*i+1]-q[2*j+1])**2)
				ef = (l - l0)/l0
				ec = ef + eta*(et - ef)
				self._applied_strain_force(t, n, _q, _acc, (target, ec, k, T))

	def _applied_jacobian(self, t, n, q, _q, dfdx, _dfdx, T, applied_args, eta):
		'''Compute the jacobian of the applied force at the source(s) and target(s).
		
		Parameters
		----------
		t : float
			The current time.
		n : int
			The number of nodes.
		q : ndarray
			The 2*n position coordinates of the nodes in the free state.
		_q : ndarray
			The 2*n position coordinates of the nodes in the clamped state.
		dfdx : ndarray
			The jacobian of the forces on the free state. Applied force jacobians are added
			into this array as output.
		_dfdx : ndarray
			The jacobian of the forces on the clamped state. Applied force jacobians are added
			into this array as output.
		T : float
			Period for oscillatory force. If T = 0, sources and targets are held
			stationary.
		applied_args : tuple
			Simulation arguments: the source strain(s), target strain(s), and pinning stiffness.
		eta : float
			Learning rate by which to increment clamped strain towards the target.
		'''

		ess, ets, k = applied_args

		if not hasattr(ess, '__len__'): ess = len(self.sources)*[ess]
		if not hasattr(ets, '__len__'): ets = len(self.targets)*[ets]

		for es, source in zip(ess, self.sources):
			if np.abs(es) > 0:
				self._applied_strain_jacobian(t, n, q, dfdx, (source, es, k, T))
				self._applied_strain_jacobian(t, n, _q, _dfdx, (source, es, k, T))

		for et, target in zip(ets, self.targets):
			if np.abs(et) > 0:
				self._applied_strain_jacobian(t, n, _q, _dfdx, (target, ec, k, T))

	def _applied_strain_energy(self, t, n, q, args):
		'''Compute the elastic energy due to applied force for a single node pair.
		
		Parameters
		----------
		t : float
			The current time.
		n : int
			The total number of nodes in the system (free or clamped).
		q : float array, shape (2*n,)
			The 2*n position coordinates of the nodes (free or clamped).
		args : tuple
			Applied force parameters: the node pair on which to apply the force,
			the amount of strain, the applied spring stiffness, and period T.

		Returns
		-------
		float
			Energy contribution due to applied force. 
		'''

		pair, eps, k, T = args
		i = pair['i']; j = pair['j']
		if T > 0:
			l = pair['length']*(1 + eps/2 - eps/2*(self._cosine_pulse(t,T)))
		else:
			l = pair['length']*(1 + pair['phase']*eps)
		xi, yi = q[2*i], q[2*i+1]
		xj, yj = q[2*j], q[2*j+1]
		dx = xi-xj; dy = yi-yj
		r = np.sqrt(dx**2 + dy**2)
		en = 0.5*k*(r-l)**2
		return en

	def _applied_strain_force(self, t, n, q, acc, args):
		'''Compute the applied force for a single node pair.
		
		Parameters
		----------
		t : float
			The current time.
		n : int
			The total number of nodes in the system (free or clamped).
		q : ndarray
			The 2*n position coordinates of the nodes (free or clamped).
		acc : ndarray
			The acceleration of each node in the system (free or clamped).
			Applied forces are added into this array as output.
		args : tuple
			Applied force parameters: the node pair on which to apply the force,
			the amount of strain, the applied spring stiffness, and period T.
		'''

		pair, eps, k, T = args
		i = pair['i']; j = pair['j']
		if T > 0:
			l = pair['length']*(1 + eps/2 - eps/2*(self._cosine_pulse(t,T)))
		else:
			l = pair['length']*(1 + pair['phase']*eps)
		xi, yi = q[2*i], q[2*i+1]
		xj, yj = q[2*j], q[2*j+1]
		dx = xi-xj; dy = yi-yj
		r = np.sqrt(dx**2 + dy**2)
		fac = -k*(1 - l/r)
		fx = fac*dx
		fy = fac*dy
		acc[2*i] += fx; acc[2*i+1] += fy
		acc[2*j] -= fx; acc[2*j+1] -= fy

	def _applied_strain_jacobian(self, t, n, q, jac, args):
		'''Compute the jacobian of the applied force for a single node pair.
		
		Parameters
		----------
		t : float
			The current time.
		n : int
			The total number of nodes in the system (free or clamped).
		q : ndarray
			The 2*n position coordinates of the nodes (free or clamped).
		jac : ndarray
			The jacobian of the forces, added into this array as output.
		args : tuple
			Applied force parameters: the node pair on which to apply the force,
			the amount of strain, the applied spring stiffness, and period T.
		'''

		pair, eps, k, T = args
		i = pair['i']; j = pair['j']
		if T > 0:
			l = pair['length']*(1 + eps/2 - eps/2*(self._cosine_pulse(t,T)))
		else:
			l = pair['length']*(1 + pair['phase']*eps)
		xi, yi = q[2*i], q[2*i+1]
		xj, yj = q[2*j], q[2*j+1]
		dx = xi-xj; dy = yi-yj
		r2 = dx**2 + dy**2
		r = np.sqrt(r2); r3 = r2*r
		xx = -k*(l/r*(dx*dx/r2-1)+1)
		yy = -k*(l/r*(dy*dy/r2-1)+1)
		xy = -k*l*dx*dy/r3
		
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

	'''
	*****************************************************************************************************
	*****************************************************************************************************

										PLOTTING AND ANIMATION

	*****************************************************************************************************
	*****************************************************************************************************
	'''

	def plot_source(self, ax, source):
		'''Plot a source pair.
		
		Parameters
		----------
		ax : matplotlib.axes.Axes
			The axes on which to plot.
		source : dict
			The source to visualize.

		Returns
		-------
		matplotlib.axes.Axes.scatter
			A handle to the plotted scatter object.
		'''

		six, siy = self.pts[source['i']]
		sjx, sjy = self.pts[source['j']]
		s = ax.scatter([six, sjx], [siy, sjy], s=60, edgecolor='k', facecolor=pal['blue'], lw=1, zorder=1000)
		return s

	def plot_target(self, ax, target):
		'''Plot a target pair.
		
		Parameters
		----------
		ax : matplotlib.axes.Axes
			The axes on which to plot.
		target : dict
			The target to visualize.

		Returns
		-------
		matplotlib.axes.Axes.scatter
			A handle to the plotted scatter object.
		'''

		tix, tiy = self.pts[target['i']]
		tjx, tjy = self.pts[target['j']]
		t = ax.scatter([tix, tjx], [tiy, tjy], s=60, edgecolor='k', facecolor=pal['red'], lw=1, zorder=1000)
		return t

	def color_plot(self, cmap, vmin, vmax, filename=None):
		'''Plot the network with edges colored according to the bond stiffness.
		
		Parameters
		----------
		cmap : str or matplotlib.colors.Colormap
			The colormap to use.
		vmin : float
			The lower bound for the mapped values.
		vmax : float
			The upper bound for the mapped values.
		filename : str, optional
			The name of the file for saving the plot.
		'''

		fig, ax = plt.subplots(1,1,figsize=(5,5))
		norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

		edgecolors = np.zeros((self.ne, 4))
		for i,edge in enumerate(self.graph.edges(data=True)):
			edgecolors[i] = cmap(norm(edge[2]['stiffness']))

		e = self._collect_edges()
		ec = mc.LineCollection(e, colors=edgecolors, linewidths=3)
		ax.add_collection(ec)

		for source in self.sources:
			s = self.plot_source(ax, source)
		for target in self.targets:
			t = self.plot_target(ax, target)

		self.set_axes(ax)
		fig.tight_layout()
		if filename:
			fig.savefig(filename, bbox_inches='tight')
		plt.show()

	def plot(self, filename=None):
		'''Plot the network.
		
		Parameters
		----------
		filename : str, optional
			The name of the file for saving the plot.
		'''

		fig, ax = plt.subplots(1,1,figsize=(5,5))
		ec, dc = self.plot_network(ax)
		for source in self.sources:
			s = self.plot_source(ax, source)
		for target in self.targets:
			t = self.plot_target(ax, target)
		  
		self.set_axes(ax)
		fig.tight_layout()
		if filename:
			fig.savefig(filename, bbox_inches='tight')
		plt.show()

	def animate(self, skip=1):
		'''Animate the network after a simulation.
		
		Parameters
		----------
		skip : int, optional
			Use every skip number of frames (skip=1 uses every frame).

		Returns
		-------
		matplotlib.animation.FuncAnimation
			The resulting animation. In a jupyter notebook, the animation
			may be visualized with the import from IPython.display import HTML,
			and running HTML(ani.to_html5_video()).
		'''

		frames = len(self.traj[::skip]) - 1
		fig, ax =plt.subplots(1,1,figsize=(5,5))
		ec, dc = self.plot_network(ax)

		s = []; t = []
		for source in self.sources:
			s += [self.plot_source(ax, source)]
		for target in self.targets:
			t += [self.plot_target(ax, target)]

		self.set_axes(ax)
		fig.tight_layout()

		def step(i):
			self.set_frame(i*skip)
			e = self._collect_edges()
			ec.set_segments(e)
			dc.set_offsets(self.pts[self.degree>0])
			for j, source in enumerate(self.sources):
				s[j].set_offsets(np.vstack([self.pts[source['i']],
											self.pts[source['j']]]))
			for j, target in enumerate(self.targets):
				t[j].set_offsets(np.vstack([self.pts[target['i']],
											self.pts[target['j']]]))
			return ec, dc, *s, *t,

		ani = animation.FuncAnimation(fig, step, frames=frames+1, interval=25, blit=True)
		plt.close(fig)
		return ani

	def time_lapse(self, filename=None):
		'''Overlay all snapshots of a simulation.
		
		Parameters
		----------
		filename : str, optional
			The name of the file for saving the plot.
		'''

		fig, ax = plt.subplots(1,1,figsize=(5,5))

		ax.scatter(self.traj[:,:,0], self.traj[:,:,1], color=[0.8,0.8,0.8], alpha=0.05, zorder=0, s=5, lw=0)

		for source in self.sources:
			si, sj = source['i'], source['j']
			ax.scatter(self.traj[:,si,0], self.traj[:,si,1], color=pal['blue'], alpha=0.05, s=5, lw=0)
			ax.scatter(self.traj[:,sj,0], self.traj[:,sj,1], color=pal['blue'], alpha=0.05, s=5, lw=0)

		for target in self.targets:
			ti, tj = target['i'], target['j']
			ax.scatter(self.traj[:,ti,0], self.traj[:,ti,1], color=pal['red'], alpha=0.05, s=5, lw=0)
			ax.scatter(self.traj[:,tj,0], self.traj[:,tj,1], color=pal['red'], alpha=0.05, s=5, lw=0)

		pts_init = np.copy(self.pts) # temporary storage
		self.pts = np.mean(self.traj, axis=0)

		ec, dc = self.plot_network(ax)

		for source in self.sources:
			s = self.plot_source(ax, source)
		for target in self.targets:
			t = self.plot_target(ax, target)

		self.pts = np.copy(pts_init) # reset
		  
		#self.set_axes(ax)
		ax.axis('off')
		fig.tight_layout()
		if filename:
			fig.savefig(filename, bbox_inches='tight', dpi=300)
		plt.show()

	def strain(self, pair):
		'''Compute the strain over time for a source or target pair.
		
		Parameters
		----------
		pair : dict
			The source or target pair whose strain to compute.

		Returns
		-------
		ndarray
			The strain over the course of the simulation.
		'''

		i, j = pair['i'], pair['j']
		l = pair['length']
		e = (self._distance(self.traj[:,i], self.traj[:,j]) - l)/l
		return e

	def strain_plot(self, filename=None):
		'''Make a line plot of source and target strains after a simulation.
		
		Parameters
		----------
		filename : str, optional
			The name of the file for saving the plot.
		'''

		fig, ax =plt.subplots(1,1,figsize=(3.5,2))
		emax = 0

		for source in self.sources:
			es = self.strain(source)
			ax.plot(self.t_eval, es, color=pal['blue'])
			es_max = np.max(np.abs(es))
			if es_max > emax:
				emax = es_max

		for target in self.targets:
			et = self.strain(target)
			ax.plot(self.t_eval, et, color=pal['red'])
			et_max = np.max(np.abs(et))
			if et_max > emax:
				emax = et_max

		ax.set_xlabel('time')
		ax.set_ylabel('strain')
		lim = 1.5*emax
		ax.set_ylim(-lim,lim)
		fig.tight_layout()
		if filename:
			fig.savefig(filename, bbox_inches='tight')
		plt.show()

	def strain_plot_thermal(self, es0, et0, filename=None):
		'''Make a scatter plot of source and target strains.

		Solid lines are used to denote reference source and target strain values.
		
		Parameters
		----------
		es0 : float
			Reference source strain.
		et0 : float
			Reference target strain.
		filename : str, optional
			The name of the file for saving the plot.
		'''

		fig, (ax1, ax2) =plt.subplots(2,1,figsize=(6,4), sharex=True)
		emax = 0

		ax1.axhline(es0, color=pal['blue'], lw=1.5, label='training strain')
		for source in self.sources:
			es = self.strain(source)
			ax1.scatter(self.t_eval, es, color=add_alpha(pal['blue'],0.7), s=2)
			es_max = np.max(np.abs(es))
			if es_max > emax:
				emax = es_max

		ax2.axhline(et0, color=pal['red'], lw=1.5, label='training strain')
		for target in self.targets:
			et = self.strain(target)
			ax2.scatter(self.t_eval, et, color=add_alpha(pal['red'],0.7), s=2)
			et_max = np.max(np.abs(et))
			if et_max > emax:
				emax = et_max

		ax2.set_xlabel('time')
		ax1.set_ylabel('strain')
		ax2.set_ylabel('strain')
		ax1.legend(frameon=False, ncol=2)
		ax2.legend(frameon=False, ncol=2)
		lim = 1.5*emax
		ax1.set_ylim(-lim,lim)
		ax2.set_ylim(-lim,lim)
		fig.tight_layout()
		if filename:
			fig.savefig(filename, bbox_inches='tight', dpi=300)
		plt.show()

	def mode_plot(self, v, scale, disks=True, filename=None):
		'''Plot a deformation mode (displacement) of the network.
		
		Parameters
		----------
		v : ndarray
			Unit displacement vector.
		scale : float
			Factor by which to scale the displacement values.
		disks : bool, optional
			Whether to plot disks at each node colored according to displacement
			direction, with size proportional to displacement magnitude.
		filename : str, optional
			The name of the file for saving the plot.
		'''

		# Plot the network with source and target edges and nodes.
		self.reset_init()
		fig, ax = plt.subplots(1,1,figsize=(5,5))
		fac = 4

		e = self._collect_edges()
		eci = mc.LineCollection(e, colors=[0.6,0.6,0.6], linewidths=0.5, linestyle='dashed')
		ax.add_collection(eci)

		# add offset
		self.pts[:,0][self.degree>0] += scale*v[::2]/fac
		self.pts[:,1][self.degree>0] += scale*v[1::2]/fac

		e = self._collect_edges()
		ec = mc.LineCollection(e, colors='k', linewidths=0.5)
		ax.add_collection(ec)

		if disks:
			col = np.arctan2(v[1::2], v[::2])
			col[col<0] += np.pi
			norm = mpl.colors.Normalize(vmin=0, vmax=np.pi)
			r = scale*np.sqrt(v[::2]**2+v[1::2]**2)/fac
			dc = mc.EllipseCollection(r, r, np.zeros_like(r), offsets=self.pts[self.degree>0],
										  transOffset=ax.transData, units='x',
										  edgecolor='k', facecolor=cyclic_cmap(norm(col)), linewidths=0.5, zorder=100)
			ax.add_collection(dc)

		# reset
		self.reset_init()

		lim = 1.5*np.max(np.abs(self.pts))
		ax.set_xlim(-lim,lim)
		ax.set_ylim(-lim,lim)
		ax.axis('off')

		fig.tight_layout()
		if filename:
			fig.savefig(filename, bbox_inches='tight')
		plt.show()

	def mode_tile(self, vs, scale, filename=None):
		'''Plot a collection of deformation modes (displacements) of the network.
		
		Parameters
		----------
		vs : ndarray
			Array of unit displacement vectors in each column.
		scale : float
			Factor by which to scale the displacement values.
		disks : bool, optional
			Whether to plot disks at each node colored according to displacement
			direction, with size proportional to displacement magnitude.
		filename : str, optional
			The name of the file for saving the plot.
		'''

		# Plot the network with source and target edges and nodes.
		self.reset_init()
		cols = 8
		rows = vs.shape[1]//cols
		if vs.shape[1]%cols:
			rows += 1

		fig, axes = plt.subplots(rows,cols,figsize=(16,2*rows))
		fac = 4

		for mode, ax in enumerate(axes.flat):
			if mode >= vs.shape[1]:
				ax.axis('off')
			else:

				e = self._collect_edges()
				eci = mc.LineCollection(e, colors=[0.6,0.6,0.6], linewidths=0.5, linestyle='dashed')
				ax.add_collection(eci)

				v = vs[:,mode]
				self.pts[:,0][self.degree>0] += scale*v[::2]/fac
				self.pts[:,1][self.degree>0] += scale*v[1::2]/fac

				col = np.arctan2(v[1::2], v[::2])
				col[col<0] += np.pi
				norm = mpl.colors.Normalize(vmin=0, vmax=np.pi)
				r = scale*np.sqrt(v[::2]**2+v[1::2]**2)/fac
				dc = mc.EllipseCollection(r, r, np.zeros_like(r), offsets=self.pts[self.degree>0],
											  transOffset=ax.transData, units='x',
											  edgecolor='k', facecolor=cyclic_cmap(norm(col)), linewidths=0.5, zorder=100)
				ax.add_collection(dc)

				e = self._collect_edges()
				ec = mc.LineCollection(e, colors='k', linewidths=0.5)
				ax.add_collection(ec)

				self.reset_init()

				self.set_axes(ax)
		fig.tight_layout()
		if filename:
			fig.savefig(filename, bbox_inches='tight')
		plt.show()

	def distribution_plot(self, kind='stiffness', vmin=0, vmax=2, nbins=25):
		'''Make a distribution plot of either bond stiffnesses or rest lengths.

		Parameters
		----------
		kind : str, optional
			Which quantity to plot. Valid options are 'stiffness' or 'length'.
		vmin : float, optional
			The lower bound of the distribution.
		vmax : float, optional
			The upper bound of the distribution.
		nbins : int, optional
			The number of evenly-spaced bins to use.

		Returns
		-------
		ndarray
			Array of values whose distribution is plotted.
		'''

		if kind not in ['stiffness','length']:
			raise ValueError("'kind' must be 'stiffness' or 'length'.")

		v = np.zeros(self.ne)
		for e,edge in enumerate(self.graph.edges(data=True)):
			v[e] = edge[2][kind]
		
		bins = np.linspace(vmin, vmax, nbins)
		x = 0.5*(bins[1:]+bins[:-1])
		y = np.histogram(v, bins)[0]

		fig, ax = plt.subplots(1,1,figsize=(2.5,2))
		ax.bar(x, y, width=0.8*np.diff(bins), color=pal['blue'])
		ax.set_xlabel(kind)
		ax.set_ylabel('count')
		fig.tight_layout()
		plt.show()

		return v

	'''
	*****************************************************************************************************
	*****************************************************************************************************

											LAMMPS FUNCTIONS

	*****************************************************************************************************
	*****************************************************************************************************
	'''

	def write_lammps_data(self, filename, title, applied_args, dashpot=0):
		'''Write the datafile of atoms and bonds for a simple LAMMPS simulation with harmonic bonds.
		
		Parameters
		----------
		filename : str
			The name of the file to write to.
		title : str
			Title string for the file.
		applied_args : tuple
			Simulation arguments: the source strain(s), target strain(s), and pinning stiffness.
		dashpot : float, optional
			Dashpot damping coefficient. If zero, no dashpots are used.
		'''

		es, et, ka = applied_args
		if np.abs(es) > 0: ns = len(self.sources)
		else: ns = 0
		if np.abs(et) > 0: nt = len(self.targets)
		else: nt = 0
		
		with open(filename, 'w') as f:
			f.write('{:s}\n\n'.format(title))

			nb = self.ne + ns + nt
			f.write('{:d} atoms\n'.format(self.n))
			f.write('{:d} bonds\n'.format(nb))
			f.write('0 angles\n')
			f.write('0 dihedrals\n')
			f.write('0 impropers\n\n')
			
			f.write('1 atom types\n')
			f.write('{:d} bond types\n\n'.format(nb))

			f.write('{:.12g} {:.12g} xlo xhi\n'.format(2*np.min(self.pts[:,0]),2*np.max(self.pts[:,0])))
			f.write('{:.12g} {:.12g} ylo yhi\n\n'.format(2*np.min(self.pts[:,1]),2*np.max(self.pts[:,1])))

			f.write('Masses\n\n')
			f.write('1 1\n\n')

			f.write('Bond Coeffs\n\n')
			for e,edge in enumerate(self.graph.edges(data=True)):
				if dashpot > 0:
					f.write('{:d} {:.12g} {:.12g} {:.12g}\n'.format(e+1,0.5*edge[2]['stiffness'],edge[2]['length'],dashpot))
				else:
					f.write('{:d} {:.12g} {:.12g}\n'.format(e+1,0.5*edge[2]['stiffness'],edge[2]['length']))
			e = self.ne
			if ns > 0:
				for source in self.sources:
					if dashpot > 0:
						f.write('{:d} {:.12g} {:.12g} {:.12g}\n'.format(e+1,0.5*ka,source['length']*(1 + source['phase']*es),dashpot))
					else:
						f.write('{:d} {:.12g} {:.12g}\n'.format(e+1,0.5*ka,source['length']*(1 + source['phase']*es)))
					e += 1
			if nt > 0:
				for target in self.targets:
					if dashpot > 0:
						f.write('{:d} {:.12g} {:.12g} {:.12g}\n'.format(e+1,0.5*ka,target['length']*(1 + target['phase']*et),dashpot))
					else:
						f.write('{:d} {:.12g} {:.12g}\n'.format(e+1,0.5*ka,target['length']*(1 + target['phase']*et)))
					e += 1
			f.write('\n')

			f.write('Atoms\n\n')
			for i in range(self.n):
				f.write('{:d} 1 1 {:.4g} {:.4g} 0.0\n'.format(i+1,self.pts[i,0],self.pts[i,1]))
			f.write('\n')

			f.write('Bonds\n\n')
			for e,edge in enumerate(self.graph.edges()):
				f.write('{:d} {:d} {:d} {:d}\n'.format(e+1,e+1,edge[0]+1,edge[1]+1))
			e = self.ne
			if ns > 0:
				for source in self.sources:
					f.write('{:d} {:d} {:d} {:d}\n'.format(e+1,e+1,source['i']+1,source['j']+1))
					e += 1
			if nt > 0:
				for target in self.targets:
					f.write('{:d} {:d} {:d} {:d}\n'.format(e+1,e+1,target['i']+1,target['j']+1))
					e += 1

	def write_lammps_data_aging(self, filename, title, applied_args, train=2, dashpot=0, alpha=0, dt=0.005):
		'''Write the datafile of atoms and bonds for a LAMMPS simulation with custom aging routine.
		
		Parameters
		----------
		filename : str
			The name of the file to write to.
		title : str
			Title string for the file.
		applied_args : tuple
			Simulation arguments: the source strain(s), target strain(s), and pinning stiffness.
		train : int, optional
			Training mode. 0 = no training, 1 = l-model, 2 = k-model.
		dashpot : float, optional
			Dashpot damping coefficient. If zero, no dashpots are used.
		alpha : float, optional
			The aging rate.
		dt : float, optional
			Integration step size.
		'''

		es, et, ka = applied_args
		if np.abs(es) > 0: ns = len(self.sources)
		else: ns = 0
		if np.abs(et) > 0: nt = len(self.targets)
		else: nt = 0

		with open(filename, 'w') as f:
			f.write('{:s}\n\n'.format(title))

			nb = self.ne + ns + nt
			f.write('{:d} atoms\n'.format(2*self.n))
			f.write('{:d} bonds\n'.format(nb))
			f.write('0 angles\n')
			f.write('0 dihedrals\n')
			f.write('0 impropers\n\n')
			
			f.write('1 atom types\n')
			f.write('{:d} bond types\n\n'.format(nb))

			f.write('{:.12g} {:.12g} xlo xhi\n'.format(2*np.min(self.pts[:,0]),2*np.max(self.pts[:,0])))
			f.write('{:.12g} {:.12g} ylo yhi\n\n'.format(2*np.min(self.pts[:,1]),2*np.max(self.pts[:,1])))

			f.write('Masses\n\n')
			f.write('1 1\n\n')

			f.write('Bond Coeffs\n\n')
			for e,edge in enumerate(self.graph.edges(data=True)):
				if dashpot > 0:
					f.write('{:d} {:.12g} {:.12g} {:.12g} {:.12g} {:d} {:d}\n'.format(e+1,0.5*edge[2]['stiffness'],edge[2]['length'],dashpot,alpha*dt,train*int(edge[2]['trainable']),0))
				else:
					f.write('{:d} {:.12g} {:.12g} {:.12g} {:d} {:d}\n'.format(e+1,0.5*edge[2]['stiffness'],edge[2]['length'],alpha*dt,train*int(edge[2]['trainable']),0))
			e = self.ne
			if ns > 0:
				for source in self.sources:
					if dashpot > 0:
						f.write('{:d} {:.12g} {:.12g} {:.12g} {:.12g} {:d} {:d}\n'.format(e+1,0.5*ka,source['length']*(1 + source['phase']*es),dashpot,0,0,0))
					else:
						f.write('{:d} {:.12g} {:.12g} {:.12g} {:d} {:d}\n'.format(e+1,0.5*ka,source['length']*(1 + source['phase']*es),0,0,0))
					e += 1
			if nt > 0:
				for target in self.targets:
					if dashpot > 0:
						f.write('{:d} {:.12g} {:.12g} {:.12g} {:.12g} {:d} {:d}\n'.format(e+1,0.5*ka,target['length']*(1 + target['phase']*et),dashpot,0,0,1))
					else:
						f.write('{:d} {:.12g} {:.12g} {:.12g} {:d} {:d}\n'.format(e+1,0.5*ka,target['length']*(1 + target['phase']*et),0,0,1))
					e += 1
			f.write('\n')

			f.write('Atoms\n\n')
			for i in range(self.n):
				f.write('{:d} 1 1 {:.12g} {:.12g} 0.0\n'.format(2*i+1,self.pts[i,0],self.pts[i,1])) # clamped
				f.write('{:d} 1 1 {:.12g} {:.12g} 0.0\n'.format(2*i+2,self.pts[i,0],self.pts[i,1])) # free
			f.write('\n')

			f.write('Bonds\n\n')
			for e,edge in enumerate(self.graph.edges()):
				f.write('{:d} {:d} {:d} {:d}\n'.format(e+1,e+1,2*edge[0]+1,2*edge[1]+1))
			e = self.ne
			if ns > 0:
				for source in self.sources:
					f.write('{:d} {:d} {:d} {:d}\n'.format(e+1,e+1,2*source['i']+1,2*source['j']+1))
					e += 1
			if nt > 0:
				for target in self.targets:
					f.write('{:d} {:d} {:d} {:d}\n'.format(e+1,e+1,2*target['i']+1,2*target['j']+1))
					e += 1

	def write_lammps_data_learning(self, filename, title, efs, applied_args, train=2, dashpot=0, eta=1e-1, alpha=1e-3, dt=0.005):
		'''Write the datafile of atoms and bonds for a LAMMPS simulation with custom coupled learning routine.
		
		Parameters
		----------
		filename : str
			The name of the file to write to.
		title : str
			Title string for the file.
		efs : list
			List of initial target strains in response to source strain (need to update)
		applied_args : tuple
			Simulation arguments: the source strain(s), target strain(s), and pinning stiffness.
		train : int, optional
			Training mode. 0 = no training, 1 = l-model, 2 = k-model.
		dashpot : float, optional
			Dashpot damping coefficient. If zero, no dashpots are used.
		eta : float, optional
			The learning rate by which the clamped state target strain approaches the final desired strain.
		alpha : float, optional
			The aging rate.
		dt : float, optional
			Integration step size.
		'''

		es, et, ka = applied_args
		if np.abs(es) > 0: ns = len(self.sources)
		else: ns = 0
		if np.abs(et) > 0: nt = len(self.targets)
		else: nt = 0
		
		with open(filename, 'w') as f:
			f.write('{:s}\n\n'.format(title))

			nb = self.ne + ns + nt
			f.write('{:d} atoms\n'.format(2*self.n))
			f.write('{:d} bonds\n'.format(nb))
			f.write('0 angles\n')
			f.write('0 dihedrals\n')
			f.write('0 impropers\n\n')
			
			f.write('1 atom types\n')
			f.write('{:d} bond types\n\n'.format(nb))

			f.write('{:.12g} {:.12g} xlo xhi\n'.format(2*np.min(self.pts[:,0]),2*np.max(self.pts[:,0])))
			f.write('{:.12g} {:.12g} ylo yhi\n\n'.format(2*np.min(self.pts[:,1]),2*np.max(self.pts[:,1])))

			f.write('Masses\n\n')
			f.write('1 1\n\n')

			f.write('Bond Coeffs\n\n')
			for e,edge in enumerate(self.graph.edges(data=True)):
				if dashpot > 0:
					f.write('{:d} {:.12g} {:.12g} {:.12g} {:.12g} {:.12g} {:.12g} {:d} {:d}\n'.format(e+1,0.5*edge[2]['stiffness'],edge[2]['length'],dashpot,0,eta,alpha*dt,train*int(edge[2]['trainable']),0))
				else:
					f.write('{:d} {:.12g} {:.12g} {:.12g} {:.12g} {:.12g} {:d} {:d}\n'.format(e+1,0.5*edge[2]['stiffness'],edge[2]['length'],0,eta,alpha*dt,train*int(edge[2]['trainable']),0))
			e = self.ne		

			if ns > 0:
				for source in self.sources:
					rs = source['length']*(1 + source['phase']*es)
					if dashpot > 0:
						f.write('{:d} {:.12g} {:.12g} {:.12g} {:.12g} {:.12g} {:.12g} {:d} {:d}\n'.format(e+1,0.5*ka,rs,dashpot,0,eta,alpha*dt,0,0))
					else:
						f.write('{:d} {:.12g} {:.12g} {:.12g} {:.12g} {:.12g} {:d} {:d}\n'.format(e+1,0.5*ka,rs,0,eta,alpha*dt,0,0))
					e += 1
			if nt > 0:
				for target, ef in zip(self.targets, efs):
					rf = target['length']*(1 + ef)
					rt = target['length']*(1 + target['phase']*et)
					if dashpot > 0:
						f.write('{:d} {:.12g} {:.12g} {:.12g} {:.12g} {:.12g} {:.12g} {:d} {:d}\n'.format(e+1,0.5*ka,rf,dashpot,rt,eta,alpha*dt,0,1))
					else:
						f.write('{:d} {:.12g} {:.12g} {:.12g} {:.12g} {:.12g} {:d} {:d}\n'.format(e+1,0.5*ka,rf,rt,eta,alpha*dt,0,1))
					e += 1
			f.write('\n')

			f.write('Atoms\n\n')
			for i in range(self.n):
				f.write('{:d} 1 1 {:.12g} {:.12g} 0.0\n'.format(2*i+1,self.pts[i,0],self.pts[i,1])) # clamped
				f.write('{:d} 1 1 {:.12g} {:.12g} 0.0\n'.format(2*i+2,self.pts[i,0],self.pts[i,1])) # free
			f.write('\n')

			f.write('Bonds\n\n')
			for e,edge in enumerate(self.graph.edges()):
				f.write('{:d} {:d} {:d} {:d}\n'.format(e+1,e+1,2*edge[0]+1,2*edge[1]+1))
			e = self.ne
			if ns > 0:
				for source in self.sources:
					f.write('{:d} {:d} {:d} {:d}\n'.format(e+1,e+1,2*source['i']+1,2*source['j']+1))
					e += 1
			if nt > 0:
				for target in self.targets:
					f.write('{:d} {:d} {:d} {:d}\n'.format(e+1,e+1,2*target['i']+1,2*target['j']+1))
					e += 1

	def write_lammps_input(self, filename, datafile, dumpfile, duration, frames, temp=0, method=None, dt=0.005):
		'''Write the input file for a LAMMPS simulation.
		
		Parameters
		----------
		filename : str
			The name of the file to write to.
		datafile : str
			The name of the datafile input, which will be overwritten on output. If path is included,
			must be relative to filename.
		dumpfile : str
			The name of the LAMMPS dumpfile for outputting node positions. If path is included, must
			be relative to filename.
		duration : float
			The final integration time.
		frames : int
			The number of output frames to produce (excluding initial frame).
		temp : float, optional
			The temperature setting, in LJ units. If zero (default), an athermal simulation is performed.
		method : str, optional
			The type of simulation to run. If left as default (None), bonds are not trainable and have no
			applied dashpots. Valid options are 'dashpot', 'aging' 'aging/dashpot', 'learning', and
			'learning/dashpot'.
		dt : float, optional
			Integration step size.
		'''

		with open(filename, 'w') as f:
			f.write('units				lj\n')
			f.write('timestep			{:.12g}\n'.format(dt))
			f.write('dimension			2\n')
			f.write('atom_style			bond\n')
			if temp == 0:
				f.write('boundary			f f p\n')
			else:
				f.write('boundary			s s p\n')
			if method == None:
				f.write('bond_style 		harmonic\n\n')
			else:
				f.write('bond_style 		harmonic/{:s}\n\n'.format(method))

			f.write('read_data			{:s}\n\n'.format(datafile))

			f.write('variable 			duration equal {:12g}/dt\n'.format(duration))
			f.write('variable			frames equal {:d}\n'.format(frames))
			f.write('variable			step equal ${duration}/${frames}\n')

			if temp > 0:
				f.write('fix				therm all langevin {:.12g} {:.12g} $(100.0*dt) 12 zero yes\n'.format(temp,temp))
			f.write('fix				intgr all nve\n')
			if temp == 0:
				f.write('fix			drag all viscous 2\n')
			f.write('fix				dim all enforce2d\n')

			f.write('dump				out all xyz ${step}'+' {:s}\n'.format(dumpfile))
			f.write('thermo_style    	custom step time temp press vol pe ke\n')
			f.write('thermo          	${step}\n')
			f.write('neigh_modify		once yes\n')
			f.write('run				${duration}\n')
			f.write('write_data 		{:s}\n'.format(datafile)) # overwrite existing

	def write_job(self, filename, jobname, hours, cmd):
		'''Write a shell script for submitting a job to SLURM.
		
		Parameters
		----------
		filename : str
			The name of the file to write to.
		jobname : str
			An identifier for the job.
		hours : int
			Number of hours to allocate for the job.
		cmd : str
			The command to be executed.
		'''

		with open(filename, 'w') as f:
			f.write('#!/bin/bash\n')
			f.write('#SBATCH --job-name="{:s}"\n'.format(jobname))
			f.write('#SBATCH --mail-type=FAIL\n')
			f.write('#SBATCH --mail-user=jovana@sas.upenn.edu\n')
			f.write('#SBATCH --partition=compute\n')
			f.write('#SBATCH --time="{:d}:00:00"\n\n'.format(hours))
			f.write('srun -n 1 {:s}\n'.format(cmd))


