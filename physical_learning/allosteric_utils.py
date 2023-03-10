import numpy as np

from plot_imports import *
from elastic_utils import Elastic

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import collections  as mc
import matplotlib.animation as animation

from scipy.spatial import Voronoi, voronoi_plot_2d, Delaunay

import networkx as nx
from vapory import *

class Allosteric(Elastic):
	'''Class to train an allosteric response in a random elastic network.
	
	Parameters
	----------
	graph : str or networkx.graph
		If string, filename of saved graph specifying the nodes and edge connections of
		the elastic network. If networkx.graph object, the graph to use.
	dim : int, optional
		The dimensionality of the system. Valid options are 2 and 3. If graph is a filename,
		the dimensionality will be read out from the file.
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
	dim : int
		The dimensionality of the system. Valid options are 2 and 3.
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
	
	def __init__(self, graph, dim=2, auto=False, seed=12, params={'rfac':0.05, 'drag':0.01, 'dashpot':10., 'stiffness':1.}):
		
		if (dim != 2) and (dim != 3):
			raise ValueError("Dimension must be 2 or 3.")

		self.seed = seed
		self.sources = []
		self.targets = []

		# Initialize the network
		if isinstance(graph, str):
			graph, dim = self._load_network(graph)

		# Call superclass (Elastic) constructor and remove nodes
		# with a single bond.
		super().__init__(graph, dim, params)
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
		int
			The system dimension.
		'''

		with open(filename, 'r') as f:
			# Read system dimension.
			dim = int(f.readline().rstrip('\n'))
			# Read nodes
			n = int(f.readline().rstrip('\n'))
			dpos = {}
			for i in range(n):
				line = f.readline().rstrip('\n').split()
				x, y, z = float(line[0]), float(line[1]), float(line[2])
				dpos[i] = np.array([x,y,z])
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
		return graph, dim

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

		The source and target will be chosen from unbonded neighboring nodes along
		the edge. If the routine fails to find suitable options, it will revert to
		manual selection.
		'''

		# Identify edge points using Voronoi tessellation.
		vor, tri, bound_pairs, interior_pairs = self._find_boundary_pairs()

		if (len(bound_pairs) < 1) or (len(interior_pairs) < 1):
			print("Unable to find suitable source and target; switching to manual mode.")
			self._pick_source_target()

		else:
			np.random.seed(self.seed)
			se = np.random.randint(len(bound_pairs))
			si = bound_pairs[se][0]
			sj = bound_pairs[se][1]

			'''
			# eliminate any overlapping pairs
			for i in range(len(interior_pairs)-1,-1,-1):
				if interior_pairs[i][0] == si or interior_pairs[i][0] == sj \
				or interior_pairs[i][1] == si or interior_pairs[i][1] == sj:
					interior_pairs.pop(i)

			if len(interior_pairs) < 1:
				print("Unable to find suitable source and target; switching to manual mode.")
				self._pick_source_target()
			'''

			# eliminate any overlapping pairs
			for i in range(len(bound_pairs)-1,-1,-1):
				if bound_pairs[i][0] == si or bound_pairs[i][0] == sj \
				or bound_pairs[i][1] == si or bound_pairs[i][1] == sj:
					bound_pairs.pop(i)

			if len(bound_pairs) < 1:
				print("Unable to find suitable source and target; switching to manual mode.")
				self._pick_source_target()

			else:
				# Choose the pair that is furthest away from the source by computing the
				# shortest path on the graph between the source nodes and all other
				# bound pair nodes.
				ti = bound_pairs[0][0]
				tj = bound_pairs[0][1]
				#ti = interior_pairs[0][0]
				#tj = interior_pairs[0][1]
				dmax = 0
				#for p in interior_pairs:
				for p in bound_pairs:
					dii = nx.shortest_path_length(self.graph,si,p[0])
					dij = nx.shortest_path_length(self.graph,si,p[1])
					dji = nx.shortest_path_length(self.graph,sj,p[0])
					djj = nx.shortest_path_length(self.graph,sj,p[1])
					d = 0.25*(dii+dij+dji+djj)
					if d > dmax:
						dmax = d
						ti, tj = p[0], p[1]

				self.sources += [{'i':si, 'j':sj, 'length':self._distance(self.pts[si], self.pts[sj]), 'phase':1}]
				self.targets += [{'i':ti, 'j':tj, 'length':self._distance(self.pts[ti], self.pts[tj]), 'phase':1}]

				# remove the selected bonds.
				if self.graph.has_edge(si,sj):
					self.graph.remove_edge(si,sj)
				if self.graph.has_edge(ti,tj):
					self.graph.remove_edge(ti,tj)
				self._remove_dangling_edges()
				self._set_coordination()

				'''
				fig, ax = plt.subplots(1,1,figsize=(5,5))
				voronoi_plot_2d(vor, ax, show_points=False, show_vertices=False, line_width=1, line_colors=[0.8,0.8,0.8])
				ec, dc = self.plot_network(ax)
				for source in self.sources:
					s = self.plot_source(ax, source)
				for target in self.targets:
					t = self.plot_target(ax, target)
				  
				self.set_axes(ax)
				plt.show()
				'''
				self.plot()

	def _find_boundary_pairs(self):
		'''Identify node pairs on the boundary of the network.'''

		if self.dim == 2:
			min_degree = self.dim+1
		else:
			min_degree = self.dim+2

		# Construct Voronoi diagram.
		vor = Voronoi(self.pts_init[:,:self.dim])
		tri = Delaunay(self.pts_init[:,:self.dim])

		# Vertex positions.
		verts = vor.vertices
		
		# Identify pairs of nodes that share a Voronoi edge with at least one vertex
		# inside and one vertex outside the convex hull. They may be connected or not.
		bound_pairs = []

		for nodes, ridge in zip(vor.ridge_points, vor.ridge_vertices):
			p1, p2 = nodes
			
			is_finite = np.zeros(len(ridge), dtype=bool)
			for i,v in enumerate(ridge):
				is_finite[i] = (v >= 0) and (tri.find_simplex(verts[v]) >= 0)

			if np.any(is_finite) and np.any(np.logical_not(is_finite)):
				bound_pairs += [[p1, p2]]

		#Filter out only nonbonded pairs with min_degree neighbors each,
		# or bonded pairs with min_degree+1 neighbors each.
		bound_pairs = [[p[0],p[1]] for p in bound_pairs
					   if (not self.graph.has_edge(p[0],p[1]) and self.graph.degree(p[0]) >= min_degree
					   and self.graph.degree(p[1]) >=min_degree)
					   or (self.graph.has_edge(p[0],p[1]) and self.graph.degree(p[0]) >= min_degree+1
					   and self.graph.degree(p[1]) >=min_degree+1)]

		print(len(bound_pairs))

		# next, identify inside pairs one layer in.
		_, _, bound_nodes = self._find_boundary_nodes()
		interior_nodes = []
		for node in bound_nodes:
			for edge in self.graph.edges(node):
				if edge[1] not in bound_nodes:
					interior_nodes += [edge[1]]

		interior_pairs = []
		for nodes in vor.ridge_points:
			p1, p2 = nodes
			if (p1 in interior_nodes) and (p2 in interior_nodes):
				interior_pairs += [[p1, p2]]
			
		interior_pairs = [[p[0],p[1]] for p in interior_pairs
					   if (not self.graph.has_edge(p[0],p[1]) and self.graph.degree(p[0]) >= min_degree
					   and self.graph.degree(p[1]) >=min_degree)
					   or (self.graph.has_edge(p[0],p[1]) and self.graph.degree(p[0]) >= min_degree+1
					   and self.graph.degree(p[1]) >=min_degree+1)]

		return vor, tri, bound_pairs, interior_pairs

	def _find_boundary_nodes(self):
		# Construct Voronoi diagram.
		vor = Voronoi(self.pts_init[:,:self.dim])
		tri = Delaunay(self.pts_init[:,:self.dim])

		# Vertex positions.
		verts = vor.vertices

		# Identify any nodes whose Voronoi cell has vertices outside the convex hull.
		bound_nodes = []

		for i,r in enumerate(vor.point_region):
			vs = vor.regions[r]
			outside = np.zeros(len(vs), dtype=bool)
			for j,v in enumerate(vs):
				outside[j] = (v < 0) or (tri.find_simplex(verts[v]) < 0)
			if np.any(outside):
				bound_nodes += [i]

		return vor, tri, bound_nodes

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
		dc2 = mc.EllipseCollection(r, r, np.zeros_like(r), offsets=self.pts[self.degree<2,:self.dim],
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
		dc2 = mc.EllipseCollection(r, r, np.zeros_like(r), offsets=self.pts[self.degree<2,:self.dim],
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
		dc2 = mc.EllipseCollection(r, r, np.zeros_like(r), offsets=self.pts[self.degree<2,:self.dim],
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

	def reset_equilibrium(self):
		'''Set the current network state to its equilibrium state.'''

		super().reset_equilibrium()

		# reset source separation lengths
		for source in self.sources:
			i, j = source['i'], source['j']
			source['length'] = self._distance(self.pts[i], self.pts[j])
		
		# reset target separation lengths
		for target in self.targets:
			i, j = target['i'], target['j']
			target['length'] = self._distance(self.pts[i], self.pts[j])

	def save(self, filename):
		'''Save the network to a file.
		   
		Parameters
		----------
		filename : str
			The name of the text file to write.
		'''

		with open(filename, 'w') as f:
			# write dimension
			f.write(str(self.dim)+'\n')
			# write nodes
			f.write(str(self.n)+'\n')
			for i in range(self.n):
				f.write('{:.15g} {:.15g} {:.15g}\n'.format(self.pts_init[i,0],self.pts_init[i,1],self.pts_init[i,2]))

			# write edges
			f.write(str(len(self.graph.edges()))+'\n')
			for edge in self.graph.edges(data=True):
				f.write('{:d} {:d} {:.15g} {:.15g} {:d}\n'.format(edge[0],edge[1],edge[2]['stiffness'],edge[2]['length'],edge[2]['trainable']))

			# write sources
			f.write(str(len(self.sources))+'\n')
			for source in self.sources:
				f.write('{:d} {:d} {:.15g} {:d}\n'.format(source['i'],source['j'],source['length'],source['phase']))

			# write targets
			f.write(str(len(self.targets))+'\n')
			for target in self.targets:
				f.write('{:d} {:d} {:.15g} {:d}\n'.format(target['i'],target['j'],target['length'],target['phase']))

	'''
	*****************************************************************************************************
	*****************************************************************************************************

										NUMERICAL INTEGRATION

	*****************************************************************************************************
	*****************************************************************************************************
	'''

	def _applied_energy(self, t, n, q, T, applied_args):
		'''Compute the elastic energy due to applied forces at the source(s) and target(s).
		
		Parameters
		----------
		t : float
			The current time.
		n : int
			The number of nodes.
		q : ndarray
			The position coordinates of the nodes.
		T : float
			Period for oscillatory force. If T = 0, sources and targets are held
			stationary.
		applied_args : tuple
			Simulation arguments: the source strain(s), target strain(s), and pinning stiffness.

		Returns
		-------
		float
			Energy contribution due to applied forces. 
		'''

		ess, ets, k = applied_args
		en = 0

		if not hasattr(ess, '__len__'): ess = len(self.sources)*[ess]
		if not hasattr(ets, '__len__'): ets = len(self.targets)*[ets]
		
		for es, source in zip(ess, self.sources):
			if np.abs(es) > 0:
				i, j, l0, phase = source['i'], source['j'], source['length'], source['phase']
				en += self._applied_strain_energy(t, n, q, (i, j, es, k, l0, phase, T))
		for et, target in zip(ets, self.targets):
			if np.abs(et) > 0:
				i, j, l0, phase = target['i'], target['j'], target['length'], target['phase']
				en += self._applied_strain_energy(t, n, q, (i, j, et, k, l0, phase, T))
		return en

	def _applied_force(self, t, n, q, q_c, acc, acc_c, T, applied_args, train, eta):
		'''Compute the applied force at the source(s) and target(s).
		
		Parameters
		----------
		t : float
			The current time.
		n : int
			The number of nodes.
		q : ndarray
			The position coordinates of the nodes in the free strained state.
		q_c : ndarray
			The position coordinates of the nodes in the clamped strained state.
		acc : ndarray
			The acceleration of each node in the free strained state. Applied forces are added
			into this array as output.
		acc_c : ndarray
			The acceleration of each node in the clamped strained state. Applied forces are added
			into this array as output.
		T : float
			Period for oscillatory force. If T = 0, sources and targets are held
			stationary.
		applied_args : tuple
			Simulation arguments: the source strain(s), target strain(s), and pinning stiffness.
		train : int
			The type of training to perform. If train = 0 (default), no training is done.
		eta : float
			Learning rate by which to increment clamped strain towards the target.
		'''

		ess, ets, k = applied_args

		if not hasattr(ess, '__len__'): ess = len(self.sources)*[ess]
		if not hasattr(ets, '__len__'): ets = len(self.targets)*[ets]

		for es, source in zip(ess, self.sources):
			if np.abs(es) > 0:
				i, j, l0, phase = source['i'], source['j'], source['length'], source['phase']
				if train:
					self._applied_strain_force(t, n, q, acc, (i, j, es, k, l0, phase, T))
					self._applied_strain_force(t, n, q_c, acc_c, (i, j, es, k, l0, phase, T))
				else:
					self._applied_strain_force(t, n, q, acc, (i, j, es, k, l0, phase, T))

		for et, target in zip(ets, self.targets):
			if np.abs(et) > 0:
				i, j, l0, phase = target['i'], target['j'], target['length'], target['phase']
				if train:
					l = np.sqrt((q[3*i]-q[3*j])**2+(q[3*i+1]-q[3*j+1])**2+(q[3*i+2]-q[3*j+2])**2)
					ef = (l - l0)/l0
					ec = ef + eta*(et - ef)
					self._applied_strain_force(t, n, q_c, acc_c, (i, j, ec, k, l0, phase, T))
				else:
					self._applied_strain_force(t, n, q, acc, (i, j, et, k, l0, phase, T))

	def _applied_jacobian(self, t, n, q, dfdx, T, applied_args):
		'''Compute the jacobian of the applied force at the source(s) and target(s).

		This routine is only used when train=0.
		
		Parameters
		----------
		t : float
			The current time.
		n : int
			The number of nodes.
		q : ndarray
			The position coordinates of the nodes.
		dfdx : ndarray
			The jacobian of the forces. Applied force jacobians are added
			into this array as output.
		T : float
			Period for oscillatory force. If T = 0, sources and targets are held
			stationary.
		applied_args : tuple
			Simulation arguments: the source strain(s), target strain(s), and pinning stiffness.
		'''

		ess, ets, k = applied_args

		if not hasattr(ess, '__len__'): ess = len(self.sources)*[ess]
		if not hasattr(ets, '__len__'): ets = len(self.targets)*[ets]

		for es, source in zip(ess, self.sources):
			if np.abs(es) > 0:
				i, j, l0, phase = source['i'], source['j'], source['length'], source['phase']
				self._applied_strain_jacobian(t, n, q, dfdx, (i, j, es, k, l0, phase, T))

		for et, target in zip(ets, self.targets):
			if np.abs(et) > 0:
				i, j, l0, phase = target['i'], target['j'], target['length'], target['phase']
				self._applied_strain_jacobian(t, n, q, dfdx, (i, j, et, k, l0, phase, T))

	def _applied_strain_energy(self, t, n, q, args):
		'''Compute the elastic energy due to applied force for a single node pair.
		
		Parameters
		----------
		t : float
			The current time.
		n : int
			The total number of nodes in the system (free or clamped).
		q : ndarray
			The position coordinates of the nodes (free or clamped).
		args : tuple
			Applied force parameters: the node indices on which to apply the force,
			the amount of strain, the applied spring stiffness, rest length, phase,
			and period T.

		Returns
		-------
		float
			Energy contribution due to applied force. 
		'''

		i, j, eps, k, l0, phase, T = args
		if T > 0:
			l = l0*(1 + phase*eps/2 - phase*eps/2*(self._cosine_pulse(t,T)))
		else:
			l = l0*(1 + phase*eps)
		xi, yi, zi = q[3*i], q[3*i+1], q[3*i+2]
		xj, yj, zj = q[3*j], q[3*j+1], q[3*j+2]
		dx = xi-xj; dy = yi-yj; dz = zi-zj
		r = np.sqrt(dx**2 + dy**2 + dz**2)
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
			The position coordinates of the nodes (free or clamped).
		acc : ndarray
			The acceleration of each node in the system (free or clamped).
			Applied forces are added into this array as output.
		args : tuple
			Applied force parameters: the node indices on which to apply the force,
			the amount of strain, the applied spring stiffness, rest length, phase,
			and period T.
		'''

		i, j, eps, k, l0, phase, T = args
		if T > 0:
			l = l0*(1 + phase*eps/2 - phase*eps/2*(self._cosine_pulse(t,T)))
		else:
			l = l0*(1 + phase*eps)
		xi, yi, zi = q[3*i], q[3*i+1], q[3*i+2]
		xj, yj, zj = q[3*j], q[3*j+1], q[3*j+2]
		dx = xi-xj; dy = yi-yj; dz = zi-zj
		r = np.sqrt(dx**2 + dy**2 + dz**2)
		fac = -k*(1 - l/r)
		fx = fac*dx
		fy = fac*dy
		fz = fac*dz
		acc[3*i] += fx; acc[3*i+1] += fy; acc[3*i+2] += fz
		acc[3*j] -= fx; acc[3*j+1] -= fy; acc[3*j+2] -= fz

	def _applied_strain_jacobian(self, t, n, q, jac, args):
		'''Compute the jacobian of the applied force for a single node pair.
		
		Parameters
		----------
		t : float
			The current time.
		n : int
			The total number of nodes in the system (free or clamped).
		q : ndarray
			The position coordinates of the nodes (free or clamped).
		jac : ndarray
			The jacobian of the forces, added into this array as output.
		args : tuple
			Applied force parameters: the node indices on which to apply the force,
			the amount of strain, the applied spring stiffness, rest length, phase,
			and period T.
		'''

		i, j, eps, k, l0, phase, T = args
		if T > 0:
			l = l0*(1 + phase*eps/2 - phase*eps/2*(self._cosine_pulse(t,T)))
		else:
			l = l0*(1 + phase*eps)
		xi, yi, zi = q[3*i], q[3*i+1], q[3*i+2]
		xj, yj, zj = q[3*j], q[3*j+1], q[3*j+2]
		dx = xi-xj; dy = yi-yj; dz = zi-zj
		r2 = dx**2 + dy**2 + dz**2
		r = np.sqrt(r2); r3 = r2*r
		xx = -k*(l/r*(dx*dx/r2-1)+1)
		yy = -k*(l/r*(dy*dy/r2-1)+1)
		zz = -k*(l/r*(dz*dz/r2-1)+1)
		xy = -k*l*dx*dy/r3
		xz = -k*l*dx*dz/r3
		yz = -k*l*dy*dz/r3

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
		'''

		if self.dim == 2: return self._plot_source_2d(ax, source)
		else: return self._plot_source_3d(ax, source)

	def _plot_source_2d(self, ax, source):
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

		six, siy, siz = self.pts[source['i']]
		sjx, sjy, sjz = self.pts[source['j']]
		s = ax.scatter([six, sjx], [siy, sjy], s=60, edgecolor='k', facecolor=pal['blue'], lw=1, zorder=1000)
		return s

	def _plot_source_3d(self, ax, source):
		'''Plot a source pair.
		
		Parameters
		----------
		ax : matplotlib.axes.Axes
			The axes on which to plot.
		source : dict
			The source to visualize.

		Returns
		-------
		list of vapory.Sphere objects
			Spheres to plot.
		'''

		r = 4*self.params['radius']
		c = 'rgb<0.0,0.4,0.8>'

		return [Sphere(self.pts[source['i']], r,
				Texture(Pigment('color',c),
				Finish('ambient',0.24,'diffuse',0.88,
				'specular',0.1,'phong',0.2,'phong_size',5))),

				Sphere(self.pts[source['j']], r,
				Texture(Pigment('color',c),
				Finish('ambient',0.24,'diffuse',0.88,
				'specular',0.1,'phong',0.2,'phong_size',5)))]

	def plot_target(self, ax, target):
		'''Plot a target pair.
		
		Parameters
		----------
		ax : matplotlib.axes.Axes
			The axes on which to plot.
		target : dict
			The target to visualize.
		'''

		if self.dim == 2: return self._plot_target_2d(ax, target)
		else: return self._plot_target_3d(ax, target)

	def _plot_target_2d(self, ax, target):
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

		tix, tiy, tiz = self.pts[target['i']]
		tjx, tjy, tjz = self.pts[target['j']]
		t = ax.scatter([tix, tjx], [tiy, tjy], s=60, edgecolor='k', facecolor=pal['red'], lw=1, zorder=1000)
		return t

	def _plot_target_3d(self, ax, target):
		'''Plot a target pair.
		
		Parameters
		----------
		ax : matplotlib.axes.Axes
			The axes on which to plot.
		target : dict
			The target to visualize.

		Returns
		-------
		list of vapory.Sphere objects
			Spheres to plot.
		'''

		r = 4*self.params['radius']
		c = 'rgb<0.9,0.1,0.1>'

		return [Sphere(self.pts[target['i']], r,
				Texture(Pigment('color',c),
				Finish('ambient',0.24,'diffuse',0.88,
				'specular',0.1,'phong',0.2,'phong_size',5))),

				Sphere(self.pts[target['j']], r,
				Texture(Pigment('color',c),
				Finish('ambient',0.24,'diffuse',0.88,
				'specular',0.1,'phong',0.2,'phong_size',5)))]

	def plot(self, spine=False, contour=False, figsize=(5,5), filename=None):
		'''Plot the network.
		
		Parameters
		----------
		spine : bool, optional
			Whether to plot only the spine of the network connecting the source
			and target nodes. Only used for 3D plotting. Default is false.
		contour : bool, optional
			Whether to plot the contour of the whole network. Helps with decluttering
			complicated networks when used with spine.
		figsize : tuple, optional
			The figure size.
		filename : str, optional
			The name of the file for saving the plot.
		'''
		if self.dim == 2: self._plot_2d(figsize, filename)
		else: self._plot_3d(spine, contour, figsize, filename)

	def _plot_2d(self, figsize=(5,5), filename=None):
		'''Plot a 2D network.'''

		fig, ax = plt.subplots(1,1,figsize=figsize)
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

	def _plot_3d(self, spine=False, contour=False, figsize=(5,5), filename=None):
		'''Plot a 3D network.'''

		fig, ax = plt.subplots(1,1,figsize=figsize)
		width = int(100*figsize[0]); height = int(100*figsize[1])

		bg, lights, camera = self._povray_setup()

		if spine:
			nodes, pairs = self._get_path()
			spheres = self._povray_spheres(nodes)
			edges = self._povray_edges(pairs)
		else:
			spheres, edges = self.plot_network(ax)

		if contour:
			hull = [self._povray_hull()]
		else:
			hull = []

		objects = [bg]+lights+spheres+edges+hull
		for source in self.sources:
			objects += self.plot_source(ax, source)
		for target in self.targets:
			objects += self.plot_target(ax, target)

		scene = Scene(camera,
					  objects = objects,
					  included = ["colors.inc", "textures.inc"])

		mat = scene.render(width=width, height=height, antialiasing=0.01)
		ax.imshow(mat)
		ax.axis('off')
		fig.tight_layout()

		if filename:
			fig.savefig(filename, bbox_inches='tight')
		plt.show()

	def _get_path(self):
		nodes = []
		pairs = []

		# First extract all nodes traversed in the shortest path from any
		# source node to any target node.
		for source in self.sources:
			for orig in [source['i'], source['j']]:
				for target in self.targets:
					for dest in [target['i'], target['j']]:
						p = nx.shortest_path(self.graph, source=orig, target=dest)
						nodes += p
						pairs += [(i,j) for (i,j) in zip(p[:-1],p[1:])]
		nodes = list(set(nodes))

		# Next include all the bonds of the existing nodes.
		neigh_nodes = []
		for i in nodes:
			for edge in self.graph.edges(i):
				j = edge[1]
				neigh_nodes += [j]
				pairs += [(i,j)]
		nodes = list(set(nodes+neigh_nodes))
		pairs = [list(p)for p in set(pairs)]

		# Strip any nodes and bonds that have a node only appearing once.
		nodes, counts = np.unique([node for pair in pairs for node in pair],
								   return_counts=True)
		nodes = list(nodes[counts > 1].astype(int))
		for e in range(len(pairs)-1,-1,-1):
			edge = pairs[e]
			i, j = edge[0], edge[1]
			if (i not in nodes) or (j not in nodes):
				pairs.pop(e)

		# Finally, include any additional edges for which both nodes are present.
		'''
		for edge in self.graph.edges():
			i, j = edge[0], edge[1]
			if (i in nodes) and (j in nodes):
				pairs += [(i,j)]
		pairs = [list(p)for p in set(pairs)]
		'''

		return nodes, pairs

	def rotate(self, spine=False, contour=False, skip=1):
		'''Animate camera and light rotation about a static scene.

		Parameters
		----------
		spine : bool, optional
			Whether to plot only the spine of the network connecting the source
			and target nodes. Only used for 3D plotting. Default is false.
		contour : bool, optional
			Whether to plot the contour of the whole network. Helps with decluttering
			complicated networks when used with spine.
		skip : int, optional
			Use every skip number of frames (skip=1 uses every frame).

		Returns
		-------
		matplotlib.animation.FuncAnimation
			The resulting animation. In a jupyter notebook, the animation
			may be visualized with the import from IPython.display import HTML,
			and running HTML(ani.to_html5_video()).
		'''

		if self.dim != 3:
			raise ValueError("System must be three-dimensional.")

		frames = 200//skip
		
		fig, ax = plt.subplots(1,1,figsize=(5,5))
		width = int(100*figsize[0]); height = int(100*figsize[1])

		bg, lights, camera = self._povray_setup()

		if spine:
			nodes, pairs = self._get_path()
			spheres = self._povray_spheres(nodes)
			edges = self._povray_edges(pairs)
		else:
			spheres, edges = self.plot_network(ax)

		if contour:
			hull = [self._povray_hull()]
		else:
			hull = []

		for source in self.sources:
			sources = self.plot_source(ax, source)
		for target in self.targets:
			targets = self.plot_target(ax, target)

		scene = Scene(camera,
					  objects = [bg]+lights+spheres+edges+sources+targets+hull,
					  included = ["colors.inc", "textures.inc"])

		mat = scene.render(width=width, height=height, antialiasing=0.01)
		im = ax.imshow(mat)
		ax.axis('off')
		fig.tight_layout()

		def step(i):
			print("Rendering frame {:d}/{:d}".format(i+1,frames+1), end="\r")
			theta = 2*i*np.pi/(frames+1)
			R = np.array([[np.cos(theta),-np.sin(theta),0],
						  [np.sin(theta),np.cos(theta),0],
						  [0,0,1]])
			bg, lights, camera = self._povray_setup(R)

			scene = Scene(camera,
					  objects = [bg]+lights+spheres+edges+sources+targets+hull,
					  included = ["colors.inc", "textures.inc"])
			mat = scene.render(width=width, height=height, antialiasing=0.01)
			im.set_array(mat)
			return im,

		ani = animation.FuncAnimation(fig, step, frames=frames+1, interval=50*skip, blit=True)
		plt.close(fig)
		return ani

	def animate(self, spine=False, contour=False, figsize=(5,5), skip=1):
		'''Animate the network after a simulation.
		
		Parameters
		----------
		spine : bool, optional
			Whether to plot only the spine of the network connecting the source
			and target nodes. Only used for 3D plotting. Default is false.
		contour : bool, optional
			Whether to plot the contour of the whole network. Helps with decluttering
			complicated networks when used with spine.
		figsize : tuple, optional
			The figure size.
		skip : int, optional
			Use every skip number of frames (skip=1 uses every frame).

		Returns
		-------
		matplotlib.animation.FuncAnimation
			The resulting animation. In a jupyter notebook, the animation
			may be visualized with the import from IPython.display import HTML,
			and running HTML(ani.to_html5_video()).
		'''
		if self.dim == 2: return self._animate_2d(figsize, skip)
		else: return self._animate_3d(spine, contour, figsize, skip)

	def _animate_2d(self, figsize=(5,5), skip=1):
		'''Animate a 2D system.'''

		frames = len(self.traj[::skip]) - 1
		fig, ax =plt.subplots(1,1,figsize=figsize)
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
			ec.set_segments(e[:,:,:self.dim])
			dc.set_offsets(self.pts[self.degree>0,:self.dim])
			for j, source in enumerate(self.sources):
				s[j].set_offsets(np.vstack([self.pts[source['i'],:self.dim],
											self.pts[source['j'],:self.dim]]))
			for j, target in enumerate(self.targets):
				t[j].set_offsets(np.vstack([self.pts[target['i'],:self.dim],
											self.pts[target['j'],:self.dim]]))
			return ec, dc, *s, *t,

		ani = animation.FuncAnimation(fig, step, frames=frames+1, interval=25, blit=True)
		plt.close(fig)
		return ani

	def _animate_3d(self, spine=False, contour=False, figsize=(5,5), skip=1):
		'''Animate a 3D system.'''
		frames = len(self.traj[::skip]) - 1
		
		fig, ax = plt.subplots(1,1,figsize=figsize)
		width = int(100*figsize[0]); height = int(100*figsize[1])

		bg, lights, camera = self._povray_setup()
		spheres = []; edges = []; hull = []
		nodes, pairs = self._get_path()

		if spine:
			spheres = self._povray_spheres(nodes)
			edges = self._povray_edges(pairs)
		else:
			spheres, edges = self.plot_network(ax)

		if contour:
			hull = [self._povray_hull()]

		for source in self.sources:
			sources = self.plot_source(ax, source)
		for target in self.targets:
			targets = self.plot_target(ax, target)

		scene = Scene(camera,
					  objects = [bg]+lights+spheres+edges+sources+targets+hull,
					  included = ["colors.inc", "textures.inc"])

		mat = scene.render(width=width, height=height, antialiasing=0.01)
		im = ax.imshow(mat)
		ax.axis('off')
		fig.tight_layout()

		def step(i):
			print("Rendering frame {:d}/{:d}".format(i+1,frames+1), end="\r")
			self.set_frame(i*skip)
			spheres = []; edges = []; hull = []
			if spine:
				spheres = self._povray_spheres(nodes)
				edges = self._povray_edges(pairs)
			else:
				spheres, edges = self.plot_network(ax)
			if contour:
				hull = [self._povray_hull()]
			for source in self.sources:
				sources = self.plot_source(ax, source)
			for target in self.targets:
				targets = self.plot_target(ax, target)

			scene = Scene(camera,
					  objects = [bg]+lights+spheres+edges+sources+targets+hull,
					  included = ["colors.inc", "textures.inc"])
			mat = scene.render(width=width, height=height, antialiasing=0.01)
			im.set_array(mat)
			return im,

		ani = animation.FuncAnimation(fig, step, frames=frames+1, interval=50*skip, blit=True)
		plt.close(fig)
		return ani

	def color_plot(self, cmap, vmin, vmax, spine=False, contour=False, figsize=(5,5), filename=None):
		'''Plot the network with edges colored according to the bond stiffness.
		
		Parameters
		----------
		cmap : str or matplotlib.colors.Colormap
			The colormap to use.
		vmin : float
			The lower bound for the mapped values.
		vmax : float
			The upper bound for the mapped values.
		spine : bool, optional
			Whether to plot only the spine of the network connecting the source
			and target nodes. Only used for 3D plotting. Default is false.
		contour : bool, optional
			Whether to plot the contour of the whole network. Helps with decluttering
			complicated networks when used with spine.
		figsize : tuple, optional
			The figure size.
		filename : str, optional
			The name of the file for saving the plot.
		'''

		if self.dim == 2: self._color_plot_2d(cmap, vmin, vmax, figsize, filename)
		else: self._color_plot_3d(cmap, vmin, vmax, spine, contour, figsize, filename)

	def _color_plot_2d(self, cmap, vmin, vmax, figsize=(5,5), filename=None):
		'''Plot a 2D network.'''

		fig, ax = plt.subplots(1,1,figsize=figsize)
		norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

		edgecolors = np.zeros((self.ne, 4))
		for i,edge in enumerate(self.graph.edges(data=True)):
			edgecolors[i] = cmap(norm(edge[2]['stiffness']))

		e = self._collect_edges()
		ec = mc.LineCollection(e[:,:,:self.dim], colors=edgecolors, linewidths=3)
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

	def _color_plot_3d(self, cmap, vmin, vmax, spine=False, contour=False, figsize=(5,5), filename=None):
		'''Plot a 3D network.'''

		fig, ax = plt.subplots(1,1,figsize=figsize)
		width = int(100*figsize[0]); height = int(100*figsize[1])
		norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

		bg, lights, camera = self._povray_setup()

		if spine:
			nodes, pairs = self._get_path()
		else:
			nodes = np.arange(self.n).astype(int)
			pairs = self.graph.edges()

		if contour:
			hull = [self._povray_hull()]
		else:
			hull = []

		edgecolors = np.zeros((len(pairs), 4))
		alphas = np.zeros(len(pairs))
		k = nx.get_edge_attributes(self.graph, 'stiffness')
		for i,edge in enumerate(pairs):
			edge = sorted(edge)
			edgecolors[i] = cmap(norm(k[(edge[0],edge[1])]))
			alphas[i] = min(1,max(1,np.abs(1-k[(edge[0],edge[1])])))
		spheres = self._povray_spheres(nodes)
		edges = self._povray_color_edges(pairs, edgecolors[:,:3], alphas)

		objects = [bg]+lights+edges+hull

		for source in self.sources:
			objects += self.plot_source(ax, source)
		for target in self.targets:
			objects += self.plot_target(ax, target)

		scene = Scene(camera,
					  objects = objects,
					  included = ["colors.inc", "textures.inc"])

		mat = scene.render(width=width, height=height, antialiasing=0.01)
		ax.imshow(mat)
		ax.axis('off')
		fig.tight_layout()

		if filename:
			fig.savefig(filename, bbox_inches='tight')
		plt.show()

	def time_lapse(self, figsize=(5,5), filename=None):
		'''Overlay all snapshots of a simulation.
		
		Parameters
		----------
		figsize : tuple, optional
			The figure size.
		filename : str, optional
			The name of the file for saving the plot.
		'''

		fig, ax = plt.subplots(1,1,figsize=figsize)

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

		i, j, l0 = pair['i'], pair['j'], pair['length']
		l = self._distance(self.traj[:,i], self.traj[:,j])
		e = (l - l0)/l0
		return e

	def strain_plot(self, figsize=(3.5,2), filename=None):
		'''Make a line plot of source and target strains after a simulation.
		
		Parameters
		----------
		figsize : tuple, optional
			The figure size.
		filename : str, optional
			The name of the file for saving the plot.
		'''

		fig, ax =plt.subplots(1,1,figsize=figsize)
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

	def strain_plot_thermal(self, es0, et0, figsize=(6,4), filename=None):
		'''Make a scatter plot of source and target strains.

		Solid lines are used to denote reference source and target strain values.
		
		Parameters
		----------
		es0 : float
			Reference source strain.
		et0 : float
			Reference target strain.
		figsize : tuple, optional
			The figure size.
		filename : str, optional
			The name of the file for saving the plot.
		'''

		fig, (ax1, ax2) =plt.subplots(2,1,figsize=figsize, sharex=True)
		emax = 0

		ax1.axhline(es0, color=pal['blue'], lw=1.5, label='training strain')
		for source in self.sources:
			es = self.strain(source)
			ax1.scatter(self.t_eval, es, color=add_alpha(pal['blue'],0.7), s=2)
			es_max = np.max(np.abs(es))
			if es_max > emax:
				emax = es_max
		if np.abs(es0) > emax:
			emax = np.abs(es0)

		ax2.axhline(et0, color=pal['red'], lw=1.5, label='training strain')
		for target in self.targets:
			et = self.strain(target)
			ax2.scatter(self.t_eval, et, color=add_alpha(pal['red'],0.7), s=2)
			et_max = np.max(np.abs(et))
			if et_max > emax:
				emax = et_max
		if np.abs(et0) > emax:
			emax = np.abs(et0)

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

	def mode_plot(self, v, scale, disks=True, figsize=(5,5), filename=None):
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
		figsize : tuple, optional
			The figure size.
		filename : str, optional
			The name of the file for saving the plot.
		'''

		# Plot the network with source and target edges and nodes.
		self.reset_init()
		fig, ax = plt.subplots(1,1,figsize=figsize)
		fac = 4

		e = self._collect_edges()
		eci = mc.LineCollection(e[:,:,:self.dim], colors=[0.6,0.6,0.6], linewidths=0.5, linestyle='dashed')
		ax.add_collection(eci)

		# add offset
		self.pts[:,0][self.degree>0] += scale*v[::3]/fac
		self.pts[:,1][self.degree>0] += scale*v[1::3]/fac

		e = self._collect_edges()
		ec = mc.LineCollection(e[:,:,:self.dim], colors='k', linewidths=0.5)
		ax.add_collection(ec)

		if disks:
			col = np.arctan2(v[1::3], v[::3])
			col[col<0] += np.pi
			norm = mpl.colors.Normalize(vmin=0, vmax=np.pi)
			r = scale*np.sqrt(v[::3]**2+v[1::3]**2)/fac
			dc = mc.EllipseCollection(r, r, np.zeros_like(r), offsets=self.pts[self.degree>0,:self.dim],
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
				eci = mc.LineCollection(e[:,:,:self.dim], colors=[0.6,0.6,0.6], linewidths=0.5, linestyle='dashed')
				ax.add_collection(eci)

				v = vs[:,mode]
				self.pts[:,0][self.degree>0] += scale*v[::3]/fac
				self.pts[:,1][self.degree>0] += scale*v[1::3]/fac

				col = np.arctan2(v[1::3], v[::3])
				col[col<0] += np.pi
				norm = mpl.colors.Normalize(vmin=0, vmax=np.pi)
				r = scale*np.sqrt(v[::3]**2+v[1::3]**2)/fac
				dc = mc.EllipseCollection(r, r, np.zeros_like(r), offsets=self.pts[self.degree>0,:self.dim],
											  transOffset=ax.transData, units='x',
											  edgecolor='k', facecolor=cyclic_cmap(norm(col)), linewidths=0.5, zorder=100)
				ax.add_collection(dc)

				e = self._collect_edges()
				ec = mc.LineCollection(e[:,:,:self.dim], colors='k', linewidths=0.5)
				ax.add_collection(ec)

				self.reset_init()

				self.set_axes(ax)
		fig.tight_layout()
		if filename:
			fig.savefig(filename, bbox_inches='tight')
		plt.show()

	def distribution_plot(self, kind='stiffness', vmin=0, vmax=2, nbins=25, figsize=(2.5,2), filename=None):
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
		figsize : tuple, optional
			The figure size.
		filename : str, optional
			The name of the file for saving the plot.

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

		fig, ax = plt.subplots(1,1,figsize=figsize)
		ax.bar(x, y, width=0.8*np.diff(bins), color=pal['blue'])
		ax.set_xlabel(kind)
		ax.set_ylabel('count')
		fig.tight_layout()
		if filename:
			fig.savefig(filename, bbox_inches='tight')
		plt.show()

		return v

	'''
	*****************************************************************************************************
	*****************************************************************************************************

											LAMMPS FUNCTIONS

	*****************************************************************************************************
	*****************************************************************************************************
	'''

	def write_lammps_data(self, filename, title, applied_args):
		'''Write the datafile of atoms and bonds for a simple LAMMPS simulation with harmonic bonds.
		
		Parameters
		----------
		filename : str
			The name of the file to write to.
		title : str
			Title string for the file.
		applied_args : tuple
			Simulation arguments: the source strain(s), target strain(s), and pinning stiffness.
		'''

		es, et, ka = applied_args
		if np.abs(es) > 0: ns = len(self.sources)
		else: ns = 0
		if np.abs(et) > 0: nt = len(self.targets)
		else: nt = 0

		xmax = np.max(np.abs(self.pts[:,0]))
		ymax = np.max(np.abs(self.pts[:,1]))
		zmax = np.max(np.abs(self.pts[:,2]))
		
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

			f.write('{:.15g} {:.15g} xlo xhi\n'.format(-2*xmax,2*xmax))
			f.write('{:.15g} {:.15g} ylo yhi\n\n'.format(-2*ymax,2*ymax))
			if self.dim == 3:
				f.write('{:.15g} {:.15g} zlo zhi\n\n'.format(-2*zmax,2*zmax))

			f.write('Masses\n\n')
			f.write('1 1\n\n')

			f.write('Bond Coeffs\n\n')
			for e,edge in enumerate(self.graph.edges(data=True)):
				f.write('{:d} {:.15g} {:.15g}\n'.format(e+1,0.5*edge[2]['stiffness'],edge[2]['length']))
			e = self.ne
			if ns > 0:
				for source in self.sources:
					f.write('{:d} {:.15g} {:.15g}\n'.format(e+1,0.5*ka,source['length']*(1 + source['phase']*es)))
					e += 1
			if nt > 0:
				for target in self.targets:
					f.write('{:d} {:.15g} {:.15g}\n'.format(e+1,0.5*ka,target['length']*(1 + target['phase']*et)))
					e += 1
			f.write('\n')

			f.write('Atoms\n\n')
			for i in range(self.n):
				f.write('{:d} 1 1 {:.15g} {:.15g} {:.15g}\n'.format(i+1,self.pts[i,0],self.pts[i,1],self.pts[i,2]))
			f.write('\n')

			f.write('Velocities\n\n')
			for i in range(self.n):
				f.write('{:d} {:.15g} {:.15g} {:.15g}\n'.format(i+1,self.vel[i,0],self.vel[i,1],self.vel[i,2]))
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

	def write_lammps_data_learning(self, filename, title, applied_args, train=2, method='learning', eta=1e-1, alpha=1e-3, vmin=1e-3, dt=0.005):
		'''Write the datafile of atoms and bonds for a LAMMPS simulation with custom coupled learning routine.
		
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
		method : str, optional
			Training method to use. Options are 'aging' or 'learning'.
		eta : float, optional
			The learning rate by which the clamped state target strain approaches the final desired strain.
		alpha : float, optional
			The aging rate.
		vmin : float, optional
			The smallest allowed value for each learning degree of freedom.
		dt : float, optional
			Integration step size.
		'''

		ess, ets, ka = applied_args

		if not hasattr(ess, '__len__'): ess = np.array(len(self.sources)*[ess])
		if not hasattr(ets, '__len__'): ets = np.array(len(self.targets)*[ets])

		ns = len(ess[np.abs(ess) > 0])
		nt = len(ets[np.abs(ets) > 0])

		if method == 'aging': mode = 1
		else: mode = 2

		xmax = np.max(np.abs(self.pts[:,0]))
		ymax = np.max(np.abs(self.pts[:,1]))
		zmax = np.max(np.abs(self.pts[:,2]))
		
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

			f.write('{:.15g} {:.15g} xlo xhi\n'.format(-2*xmax,2*xmax))
			f.write('{:.15g} {:.15g} ylo yhi\n\n'.format(-2*ymax,2*ymax))
			if self.dim == 3:
				f.write('{:.15g} {:.15g} zlo zhi\n\n'.format(-2*zmax,2*zmax))

			f.write('Masses\n\n')
			f.write('1 1\n\n')

			f.write('Bond Coeffs\n\n')
			for e,edge in enumerate(self.graph.edges(data=True)):
				f.write('{:d} {:.15g} {:.15g} {:.15g} {:.15g} {:.15g} {:.15g} {:d} {:d} {:d} {:d}\n'.format(e+1,0.5*edge[2]['stiffness'],edge[2]['length'],0,eta,alpha*dt,0.5*vmin,train*int(edge[2]['trainable']),mode,1,0))
			e = self.ne		

			for es, source in zip(ess, self.sources):
				if np.abs(es) > 0:
					rs = source['length']*(1 + source['phase']*es)
					f.write('{:d} {:.15g} {:.15g} {:.15g} {:.15g} {:.15g} {:.15g} {:d} {:d} {:d} {:d}\n'.format(e+1,0.5*ka,rs,0,eta,alpha*dt,0.5*vmin,0,0,1,0))
					e += 1
			for et, target in zip(ets, self.targets):
				if np.abs(et) > 0:
					rt = target['length']
					f.write('{:d} {:.15g} {:.15g} {:.15g} {:.15g} {:.15g} {:.15g} {:d} {:d} {:d} {:d}\n'.format(e+1,0.5*ka,rt,et,eta,alpha*dt,0.5*vmin,0,0,target['phase'],1))
					e += 1
			f.write('\n')

			f.write('Atoms\n\n')
			for i in range(self.n):
				f.write('{:d} 1 1 {:.15g} {:.15g} {:.15g}\n'.format(2*i+1,self.pts[i,0],self.pts[i,1],self.pts[i,2])) # clamped
				f.write('{:d} 1 1 {:.15g} {:.15g} {:.15g}\n'.format(2*i+2,self.pts_c[i,0],self.pts_c[i,1],self.pts_c[i,2])) # free
			f.write('\n')

			f.write('Velocities\n\n')
			for i in range(self.n):
				f.write('{:d} {:.15g} {:.15g} {:.15g}\n'.format(2*i+1,self.vel[i,0],self.vel[i,1],self.vel[i,2])) # clamped
				f.write('{:d} {:.15g} {:.15g} {:.15g}\n'.format(2*i+2,self.vel_c[i,0],self.vel_c[i,1],self.vel_c[i,2])) # free
			f.write('\n')

			f.write('Bonds\n\n')
			for e,edge in enumerate(self.graph.edges()):
				f.write('{:d} {:d} {:d} {:d}\n'.format(e+1,e+1,2*edge[0]+1,2*edge[1]+1))
			e = self.ne
			for es, source in zip(ess, self.sources):
				if np.abs(es) > 0:
					f.write('{:d} {:d} {:d} {:d}\n'.format(e+1,e+1,2*source['i']+1,2*source['j']+1))
					e += 1
			for et, target in zip(ets, self.targets):
				if np.abs(et) > 0:
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
			applied dashpots. Valid options are 'aging' or 'learning'.
		dt : float, optional
			Integration step size.
		'''

		with open(filename, 'w') as f:
			f.write('units				lj\n')
			f.write('timestep			{:.15g}\n'.format(dt))
			f.write('dimension			{:d}\n'.format(self.dim))
			f.write('atom_style			bond\n')

			if temp == 0:
				if self.dim == 2:
					f.write('boundary			f f p\n')
				else:
					f.write('boundary			f f f\n')
			else:
				if self.dim == 2:
					f.write('boundary			s s p\n')
				else:
					f.write('boundary			s s s\n')

			if method == None:
				f.write('bond_style 		harmonic\n\n')
			else:
				f.write('bond_style 		harmonic/learning\n\n')

			f.write('read_data			{:s}\n\n'.format(datafile))
			if temp > 0:
				f.write('velocity			all create {:.15g} 12 dist gaussian mom yes rot yes sum no\n\n'.format(temp))

			f.write('variable 			duration equal {:12g}/dt\n'.format(duration))
			f.write('variable			frames equal {:d}\n'.format(frames))
			f.write('variable			step equal ${duration}/${frames}\n')

			if temp > 0:
				f.write('fix				therm all langevin {:.15g} {:.15g} $(100.0*dt) 12 zero yes\n'.format(temp,temp))
				f.write('fix				intgr all nve\n')
				#f.write('fix				therm all nvt temp {:.15g} {:.15g} $(100.0*dt)\n'.format(temp,temp))
			
			if temp == 0:
				f.write('fix				intgr all nve\n')
				f.write('fix			drag all viscous 2\n')
			if self.dim == 2:
				f.write('fix				dim all enforce2d\n')

			f.write('dump				out all custom ${step}'+' {:s} x y z vx vy vz\n'.format(dumpfile))
			f.write('dump_modify		out format line "%.15g %.15g %.15g %.15g %.15g %.15g"\n')
			f.write('thermo_style    	custom step time temp press vol pe ke\n')
			f.write('thermo          	${step}\n')
			f.write('neigh_modify		once yes\n')
			f.write('run				${duration}\n')

			if method != None:
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


