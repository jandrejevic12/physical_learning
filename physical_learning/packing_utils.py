import numpy as np

from plot_imports import *
import matplotlib.pyplot as plt
from matplotlib import collections as mc
import matplotlib.animation as animation

from poisson_disc import Bridson_sampling
from scipy.spatial import Voronoi, Delaunay
import networkx as nx
from scipy.integrate import solve_ivp
from tqdm import tqdm
from numba import jit

class Packing:
	'''Class to model a contact network of 2D bidisperse disks.
	
	Parameters
	----------
	n : int
		The number of disks.
	radius : float, optional
		The radius of the larger disks. If 0 (default), a good choice is computed
		automatically.
	rfac : float, optional
		Factor by which the smaller disk radius is scaled relative to large disks.
	params : dict, optional
		Specifies system parameters. Required keywords are :

		- 'central': strength of the force pulling disks toward origin
		- 'drag': coefficient of isotropic drag
		- 'contact': strength of contact repulsion

	seed : int, optional
		A random seed used for initializing the positions of the disks, and assigning
		which are large or small.
	
	Attributes
	----------
	n : int
		The number of disks.
	pts : ndarray
		(x,y) coordinates for each disk.
	labels : ndarray
		Array of disk labels (0=large, 1=small).
	radii : list
		List of disk radii, indexed by disk type (0=large, 1=small).
	traj : ndarray
		The simulated trajectory of the network produced after a call to the generate() routine.
	t_eval : ndarray
		The corresponding time at each simulated frame.
	graph : networkx.graph
		Graph specifying the nodes and edges in the final contact network.
	'''

	def __init__(self, n, radius=0, rfac=0.8,
				 params={'central':0.0005, 'drag':0.05, 'contact':0.5}, seed=12):
		self.n = n
		self.radius = radius
		self.rfac = rfac
		self.params = params
		self.seed = seed
		self.graph = None
		self.initialize()

	'''
	*****************************************************************************************************
	*****************************************************************************************************

											DISK INITIALIZATION

	*****************************************************************************************************
	*****************************************************************************************************
	'''

	def initialize(self):
		'''Initialize the disk positions and radii.'''

		self._initialize_points()
		self._initialize_labels()
		self.radii = [self.radius, self.rfac*self.radius]

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

	def _set_radius(self):
		'''Automatically set the radius of the large disks.'''

		fac = np.pi/4. # ratio of inscribed circle area to square area
		self.radius = 0.25*np.sqrt(fac/self.n)

	def _initialize_points(self):
		'''Randomly initialize disks using Poisson disc sampling.'''

		np.random.seed(self.seed)
		if self.radius == 0:
			self._set_radius()
		pts = Bridson_sampling(radius=2*self.radius)
		if len(pts) < self.n:
			print("Found only {:d} points; retry with smaller separation distance.".format(len(pts)))
		else:
			com = np.mean(pts, axis=0)
			idx = np.argsort(self._distance(pts, com))
			pts = pts[idx[:self.n]]
			self.pts = pts - np.mean(pts, axis=0)

	def _initialize_labels(self):
		'''Label disks as large (0) or small (1) at random.'''

		np.random.seed(self.seed)
		self.labels = np.random.randint(2, size=self.n)

	'''
	*****************************************************************************************************
	*****************************************************************************************************

											GRAPH GENERATION

	*****************************************************************************************************
	*****************************************************************************************************
	'''

	def _generate_graph(self):
		'''Generate the disk contact graph.'''

		dpos = {i:self.pts[i] for i in range(self.n)}
		# first, create graph with no edges. We will add in all the edges individually.
		self.graph = nx.random_geometric_graph(self.n, 0, seed=self.seed, pos=dpos)
		for a,b in zip([0,0,1],[0,1,1]):
			d = self.radii[a] + self.radii[b]
			g = nx.random_geometric_graph(self.n, d, seed=self.seed, pos=dpos)
			for edge in g.edges():
				i = edge[0]; j = edge[1]
				if (self.labels[i] == a and self.labels[j] == b) or \
				   (self.labels[i] == b and self.labels[j] == a):
				   self.graph.add_edge(i,j)

	def coordination(self):
		'''Compute the coordination information of the network.

		Returns
		-------
		nc : int
			The number of nodes forming a connected network.
		Ziso : float
			The required average coordination for isostaticity.
		Z : float
			The average coordination of the network.
		dZ : float
			The excess coordination, defined as Z - Ziso.
		'''

		if self.graph is None:
			raise ValueError('Generate must be called first to produce a graph.')

		degree = np.array(list(self.graph.degree[i] for i in range(self.n)), dtype=int)
		ne = len(self.graph.edges())
		nc = self.n-np.sum(degree<2).astype(int) # connected nodes (excluding dangling nodes)
		if nc > 0:
			Z = 2*ne/float(nc)
			Ziso = 4. - 6./nc
			dZ = Z - Ziso
		else:
			Ziso = Z = dZ = 0

		return nc, Ziso, Z, dZ

	def save(self, filename):
		'''Save the contact network to a file.
		
		Parameters
		----------
		filename : str
			The name of the text file to write.
		'''

		ne = len(self.graph.edges())
		with open(filename, 'w') as f:
			f.write(str(self.n)+'\n')
			for i in range(self.n):
				f.write('{:.12g} {:.12g}\n'.format(self.pts[i,0],self.pts[i,1]))
			f.write(str(ne)+'\n')
			for edge in self.graph.edges():
				f.write('{:d} {:d}\n'.format(edge[0],edge[1]))

	'''
	*****************************************************************************************************
	*****************************************************************************************************

											PACKING SIMULATION

	*****************************************************************************************************
	*****************************************************************************************************
	'''

	def generate(self, duration=1000., frames=200):
		'''Run a packing simulation.

		Generates a graph of the resulting contact network at the end.

		Parameters
		----------
		duration : float, optional
			The final integration time.
		frames : int, optional
			The number of output frames to produce (excluding initial frame).
		'''

		self._generate_packing(duration, frames)
		self._generate_graph()


	def _generate_packing(self, duration, frames):
		'''Routine for integrating the ODE system of packing disks.

		Parameters
		----------
		duration : float
			The final integration time.
		frames : int
			The number of output frames to produce (excluding initial frame).
		'''

		q = np.hstack([self.pts.ravel(),np.zeros(2*self.n)])
		ti = 0; tf = duration
		t_span = [ti, tf]
		self.t_eval = np.linspace(ti, tf, frames+1)

		self.tp = ti
		with tqdm(total=tf-ti, unit='sim. time', initial=ti, ascii=True, 
				  bar_format='{l_bar}{bar}| {n:.2f}/{total:.2f} [{elapsed}<{remaining}]', desc='progress') as self.pbar:
			sol = solve_ivp(self._ff, t_span, q, t_eval=self.t_eval, jac=self._jj, method='LSODA')

		q = sol.y.T[:,:2*self.n]
		self.traj = q.reshape(frames+1, self.n, 2)
		self.pts = self.traj[-1]

	def _ff(self, t, q):
		'''ODE system for modeling packed disks.

		Parameters
		----------
		t : float
			The current time.
		q : ndarray
			The disk position and velocity degrees of freedom.
		'''

		n = self.n
		pos, vel = q[:2*n], q[2*n:]
		acc = np.zeros(2*n)
		dpos = {i:[pos[2*i],pos[2*i+1]] for i in range(n)}

		# compute central force and drag.
		b = self.params['drag']
		c = self.params['central']
		self._central_force(t, n, pos, vel, acc, b, c)
		
		# compute contact forces.
		self._contact_force(0, 0, dpos, acc)
		self._contact_force(0, 1, dpos, acc)
		self._contact_force(1, 1, dpos, acc)

		dq = np.hstack([vel, acc])

		# update progress bar
		dt = t - self.tp
		self.pbar.update(dt)
		self.tp = t

		return dq

	def _jj(self, t, q):
		'''The jacobian of the ODE system, required by implicit integrators.

		Parameters
		----------
		t : float
			The current time.
		q : ndarray
			The disk position and velocity degrees of freedom.
		'''

		n = self.n
		pos, vel = q[:2*n], q[2*n:]
		jac = np.zeros((4*n, 4*n))

		dpos = {i:[pos[2*i],pos[2*i+1]] for i in range(n)}

		# compute central force and drag jacobian.
		b = self.params['drag']
		c = self.params['central']
		self._central_jacobian(t, n, pos, vel, jac, b, c)
		
		# compute contact jacobian.
		self._contact_jacobian(0, 0, dpos, jac)
		self._contact_jacobian(0, 1, dpos, jac)
		self._contact_jacobian(1, 1, dpos, jac)

		return jac

	@staticmethod
	@jit(nopython=True)
	def _central_force(t, n, pos, vel, acc, b, c):
		'''Apply a central force pulling disks toward origin.

		Also stabilizes the simulation with isotropic drag.
		
		Parameters
		----------
		t : float
			The current time.
		n : int
			The number of disks.
		pos : ndarray
			The 2D disk positions.
		vel : ndarray
			The 2D disk velocities.
		acc : ndarray
			The array in which to populate disk accelerations on output.
		b : float
			Drag coefficient.
		c : float
			Central force strength.
		'''

		for i in range(n):
			x, y, vx, vy = pos[2*i], pos[2*i+1], vel[2*i], vel[2*i+1]
			acc[2*i] -= b*vx + c*x
			acc[2*i+1] -= b*vy + c*y

	@staticmethod
	@jit(nopython=True)
	def _central_jacobian(t, n, pos, vel, jac, b, c):
		'''Compute the jacobian of the central forces.
		
		Parameters
		----------
		t : float
			The current time.
		n : int
			The number of disks.
		pos : ndarray
			The 2D disk positions.
		vel : ndarray
			The 2D disk velocities.
		jac : ndarray
			The array in which to populate jacobian entries on output.
		b : float
			Drag coefficient.
		c : float
			Central force strength.
		'''

		for i in range(n):
			x, y, vx, vy = pos[2*i], pos[2*i+1], vel[2*i], vel[2*i+1]
			jac[2*i,2*n+2*i] += 1.
			jac[2*i+1,2*n+2*i+1] += 1.
			jac[2*n+2*i,2*i] -= c
			jac[2*n+2*i+1,2*i+1] -= c
			jac[2*n+2*i,2*n+2*i] -= b
			jac[2*n+2*i+1,2*n+2*i+1] -= b

	def _contact_force(self, a, b, dpos, acc):
		'''Apply a repulsive force between disks in contact.

		Parameters
		----------
		a : int
			The first type of interacting disk (0 or 1, for large or small).
		b : int
			The second type of interacting disk.
		dpos : dict
			Dictionary of disk positions required by networkx.
		acc : ndarray
			The array in which to populate disk accelerations on output.
		'''

		c = self.params['contact']
		d = self.radii[a] + self.radii[b]
		g = nx.random_geometric_graph(self.n, d, seed=self.seed, pos=dpos)
		for edge in g.edges():
			i = edge[0]; j = edge[1]
			if (self.labels[i] == a and self.labels[j] == b) or \
			   (self.labels[i] == b and self.labels[j] == a):
				xi, yi = g.nodes[i]['pos']
				xj, yj = g.nodes[j]['pos']
				dx = xi-xj; dy = yi-yj
				r = np.sqrt(dx**2 + dy**2)
				fac = -c*(1 - d/r)
				fx = fac*dx
				fy = fac*dy
				acc[2*i] += fx; acc[2*i+1] += fy
				acc[2*j] -= fx; acc[2*j+1] -= fy

	def _contact_jacobian(self, a, b, dpos, jac):
		'''Compute the jacobian of the repulsive forces.

		Parameters
		----------
		a : int
			The first type of interacting disk (0 or 1, for large or small).
		b : int
			The second type of interacting disk.
		dpos : dict
			Dictionary of disk positions required by networkx.
		jac : ndarray
			The array in which to populate jacobian entries on output.
		'''

		c = self.params['contact']
		d = self.radii[a] + self.radii[b]
		g = nx.random_geometric_graph(self.n, d, seed=self.seed, pos=dpos)
		for edge in g.edges():
			i = edge[0]; j = edge[1]
			if (self.labels[i] == a and self.labels[j] == b) or \
			   (self.labels[i] == b and self.labels[j] == a):
				xi, yi = g.nodes[i]['pos']
				xj, yj = g.nodes[j]['pos']
				dx = xi-xj; dy = yi-yj
				r2 = dx**2 + dy**2
				r = np.sqrt(r2); r3 = r2*r
				xx = -c*(d/r*(dx*dx/r2-1)+1)
				yy = -c*(d/r*(dy*dy/r2-1)+1)
				xy = -c*d*dx*dy/r3

				jac[2*self.n+2*i,2*i] += xx # xixi
				jac[2*self.n+2*i+1,2*i+1] += yy # yiyi
				jac[2*self.n+2*i,2*i+1] += xy # xiyi
				jac[2*self.n+2*i+1,2*i] += xy # yixi

				jac[2*self.n+2*j,2*j] += xx # xjxj
				jac[2*self.n+2*j+1,2*j+1] += yy # yjyj
				jac[2*self.n+2*j,2*j+1] += xy # xjyj
				jac[2*self.n+2*j+1,2*j] += xy # yjxj

				jac[2*self.n+2*i,2*j] -= xx # xixj
				jac[2*self.n+2*j,2*i] -= xx # xjxi
				jac[2*self.n+2*i+1,2*j+1] -= yy # yiyj
				jac[2*self.n+2*j+1,2*i+1] -= yy # yjyi
				jac[2*self.n+2*i,2*j+1] -= xy # xiyj
				jac[2*self.n+2*j,2*i+1] -= xy # xjyi
				jac[2*self.n+2*i+1,2*j] -= xy # yixj
				jac[2*self.n+2*j+1,2*i] -= xy # yjxi

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
		self._generate_graph()

	def _collect_edges(self):
		'''Prepare edges for matplotlib collections plotting.
		
		Returns
		-------
		ndarray
			Array storing the endpoints of each edge.
		'''

		edges = np.zeros((len(self.graph.edges()),2,2))
		for i,edge in enumerate(self.graph.edges()):
			edges[i,0,:] = self.pts[edge[0]]
			edges[i,1,:] = self.pts[edge[1]]
		return edges

	def plot(self, disks=True, edges=True, filename=None):
		'''Plot the network, optionally specifying whether to include disks and edges.

		Parameters
		----------
		disks : bool, optional
			Whether to plot the disks. Default is True.
		edges : bool, optional
			Whether to plot the edges connecting disks in contact. Default is true.
		filename : str, optional
			The name of the file for saving the plot.
		'''

		fig, ax = plt.subplots(1,1,figsize=(5,5))

		if disks:
			r = [2*self.radii[i] for i in self.labels]
			c = [add_alpha(pal['red'],0.4) if i else add_alpha(pal['blue'],0.4) for i in self.labels]
			dc = mc.EllipseCollection(r, r, np.zeros_like(r), offsets=self.pts,
									  transOffset=ax.transData, units='x',
									  edgecolor='k', facecolor=c, linewidths=0.5)
			ax.add_collection(dc)
				
		if edges:
			e = self._collect_edges()
			ec = mc.LineCollection(e, colors='k', linewidths=1)
			ax.add_collection(ec)
		
		lim = 1.2*np.max(np.abs(self.pts))
		ax.set_xlim(-lim,lim)
		ax.set_ylim(-lim,lim)
		ax.axis('off')

		if filename:
			fig.savefig(filename, bbox_inches='tight')
		plt.show()

	def animate(self, disks=True, edges=True, skip=1):
		'''Animate the system after a simulation.
		
		Parameters
		----------
		disks : bool, optional
			Whether to plot the disks. Default is True.
		edges : bool, optional
			Whether to plot the edges connecting disks in contact. Default is true.
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
		r = [2*self.radii[i] for i in self.labels]
		c = [add_alpha(pal['red'],0.4) if i else add_alpha(pal['blue'],0.4) for i in self.labels]
		dc = mc.EllipseCollection(r, r, np.zeros_like(r), offsets=self.pts,
									  transOffset=ax.transData, units='x',
									  edgecolor='k', facecolor=c, linewidths=0.5)
		e = self._collect_edges()
		ec = mc.LineCollection(e, colors='k', linewidths=0.5)

		if disks:
			ax.add_collection(dc)

		if edges:
			ax.add_collection(ec)

		ax.set_xlim(-0.4,0.4)
		ax.set_ylim(-0.4,0.4)
		ax.axis('off')
		fig.tight_layout()

		def step(i):
			self.set_frame(i*skip)
			if disks:
				dc.set_offsets(self.pts)
			if edges:
				e = self._collect_edges()
				ec.set_segments(e)
			return dc, ec,

		ani = animation.FuncAnimation(fig, step, frames=frames+1, interval=50*skip, blit=True)
		plt.close(fig)
		return ani




