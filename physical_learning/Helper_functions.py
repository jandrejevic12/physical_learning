import numpy as np

def get_energy(allo, applied_args):
    allo.reset_init()
    network = allo._edge_lists()
    sol = allo.solve(1e9, 10, 0, applied_args, pbar=False)
    en = allo._elastic_energy(0, allo.n, allo.pts.ravel(), network[2], network[3], network) + \
         allo._applied_energy(0, allo.n, allo.pts.ravel(), 0, applied_args)
    return en

def get_energy_fast(allo, applied_args):
    allo.reset_init()
    network = allo._edge_lists()
    sol = allo.solve(1E5, 10, 2*1E5, applied_args, pbar=False)
    print('you are')
    en = allo._elastic_energy(0, allo.n, allo.pts.ravel(), network[2], network[3], network) + \
         allo._applied_energy(0, allo.n, allo.pts.ravel(), 0, applied_args)
    return en

def get_energy_fast_J(allo, applied_args):
    allo.reset_init()
    network = allo._edge_lists()
    sol = allo.solve(1E7, 10, 2*1E7, applied_args, pbar=False)
    print('you are')
    en = allo._elastic_energy(0, allo.n, allo.pts.ravel(), network[2], network[3], network) + \
         allo._applied_energy(0, allo.n, allo.pts.ravel(), 0, applied_args)
    return en
	
def get_energy_adapted(allo, applied_args):
    print('getting energy')
    # allo.reset_init()
    network = allo._edge_lists()
    print('straining gradually')
    sol = allo.solve(1E5, 10, 2*1E5, applied_args, pbar=False)
    print('straining fixed')
    sol = allo.solve(1e9, 10, 0, applied_args, pbar=False)
    print('calculating energy')
    en = allo._elastic_energy(0, allo.n, allo.pts.ravel(), network[2], network[3], network) + \
         allo._applied_energy(0, allo.n, allo.pts.ravel(), 0, applied_args)
    return en

# def get_energy_reduced(allo, applied_args):
#     print('getting energy')
#     allo.reset_init()
#     network = allo._edge_lists()
#     print('calculating energy')
#     en = allo._elastic_energy(0, allo.n, allo.pts.ravel(), network[2], network[3], network) + \
#          allo._applied_energy(0, allo.n, allo.pts.ravel(), 0, applied_args)
#     return en
	
def get_energy_combine(allo, applied_args):
	#This is a combination of "get_energy_fast" and "get_energy"
	#First, move the node gradually with T=2*duration.
	#Second, keep the clamped in place with T=0
    allo.reset_init()
    network = allo._edge_lists()
    sol = allo.solve(1E5, 10, 2*1E5, applied_args, pbar=False) #First
    sol = allo.solve(1E7, 10, 0, applied_args, pbar=False) #Second
    print('you are')
    en = allo._elastic_energy(0, allo.n, allo.pts.ravel(), network[2], network[3], network) + \
         allo._applied_energy(0, allo.n, allo.pts.ravel(), 0, applied_args)
    return en


def get_energy_and_strains(allo, applied_args,test_type):
    '''same as before but also returns the achieved strain for debugging purposes'''
    allo.reset_init()
    network = allo._edge_lists()
    sol = allo.solve(1e9, 10, 0, applied_args, pbar=False)
	# sol = allo.solve(1E5, 10, 2*1E5, applied_args, pbar=False)
    en = allo._elastic_energy(0, allo.n, allo.pts.ravel(), network[2], network[3], network) + \
         allo._applied_energy(0, allo.n, allo.pts.ravel(), 0, applied_args)
    if test_type=='source':
        delta = allo.strain(allo.targets[0])[-1]
    if test_type=='target':
        delta = allo.strain(allo.sources[0])[-1]
    return en, delta

def get_energy_and_strains_fast(allo, applied_args,test_type):
    '''same as before but also returns the achieved strain for debugging purposes'''
    allo.reset_init()
    network = allo._edge_lists(); 
    sol = allo.solve(1E5, 10, 2*1E5, applied_args, pbar=False)
    en = allo._elastic_energy(0, allo.n, allo.pts.ravel(), network[2], network[3], network) + \
         allo._applied_energy(0, allo.n, allo.pts.ravel(), 0, applied_args)
    if test_type=='source':
        delta = allo.strain(allo.targets[0])[-1]
    if test_type=='target':
        delta = allo.strain(allo.sources[0])[-1]
    return en, delta

def get_energy_and_strains_fast_J(allo, applied_args,test_type):
    '''same as before but also returns the achieved strain for debugging purposes'''
    allo.reset_init()
    network = allo._edge_lists(); 
    print('im in, ',end='')
    sol = allo.solve(1E7, 10, 2*1E7, applied_args, pbar=False)
    print('finished')
    en = allo._elastic_energy(0, allo.n, allo.pts.ravel(), network[2], network[3], network) + \
         allo._applied_energy(0, allo.n, allo.pts.ravel(), 0, applied_args)
    if test_type=='source':
        delta = allo.strain(allo.targets[0])[-1]
    if test_type=='target':
        delta = allo.strain(allo.sources[0])[-1]
    return en, delta

def get_energy_and_strains_combined(allo, applied_args,test_type):
    '''same as before but also returns the achieved strain for debugging purposes'''
    print('im in, ',end='')
    allo.reset_init()
    network = allo._edge_lists(); 
    sol = allo.solve(1E5, 10, 2*1E5, applied_args, pbar=False)
    print('first finished, ',end='')
    sol = allo.solve(1e9, 10, 0, applied_args, pbar=False)
    print('second finished finished')
    en = allo._elastic_energy(0, allo.n, allo.pts.ravel(), network[2], network[3], network) + \
         allo._applied_energy(0, allo.n, allo.pts.ravel(), 0, applied_args)
    if test_type=='source':
        delta = allo.strain(allo.targets[0])[-1]
    if test_type=='target':
        delta = allo.strain(allo.sources[0])[-1]
    return en, delta

def get_weight_bias(allo, applied_args):
    allo.reset_init()

    ess, ets, ka = applied_args
    if not hasattr(ess, '__len__'): ess = len(allo.sources)*[ess]
    if not hasattr(ets, '__len__'): ets = len(allo.targets)*[ets]

    ne = allo.ne
    n = allo.n

    ls = np.zeros(ne)
    ls0 = np.zeros(ne)
    for e,edge in enumerate(allo.graph.edges(data=True)):
        i, j = edge[0], edge[1]
        ls[e] = edge[2]['length']
        ls0[e] = allo._distance(allo.pts[i],allo.pts[j])
    dl = ls-ls0

    # augment with source and target edges
    ns = len(allo.sources)
    nt = len(allo.targets)
    ntot = ne+ns+nt

    # geometric terms
    L = np.zeros((ntot,ntot))
    K = np.zeros((ntot,ntot))
    R = np.zeros((ntot,3*n))
    H = np.zeros((3*n,3*n))

    # pre-stress terms
    Kp = np.zeros((ntot,ntot))
    Gx = np.zeros((ntot,3*allo.n))
    Gy = np.zeros((ntot,3*allo.n))
    Gz = np.zeros((ntot,3*allo.n))

    q = allo.pts.ravel()
    edge_i, edge_j, edge_k, edge_l, edge_t = allo._edge_lists()
    network = (edge_i, edge_j, edge_k, edge_l, edge_t)

    for e, edge in enumerate(allo.graph.edges(data=True)):
        i, j = edge[0], edge[1]
        k, l = edge[2]['stiffness'], edge[2]['length']
        allo._rigidity_pair(q, R, e, i, j)
        K[e,e] = k
        L[e,e] = l
        allo._pre_stress(q, Kp, Gx, Gy, Gz, e, i, j, k, l)

    e = ne

    allo._elastic_jacobian(0, n, q, edge_k, edge_l, H, network)

    for es, pair in zip(ess, allo.sources):
        i, j, l, p = pair['i'], pair['j'], pair['length'], pair['phase']
        allo._rigidity_pair(q, R, e, i, j)
        L[e,e] = l
        if np.abs(es) > 0:
            allo._applied_strain_jacobian(0, n, q, H, (i, j, 0, ka, l, p, 0))
            allo._pre_stress(q, Kp, Gx, Gy, Gz, e, i, j, ka, l)
            K[e,e] = ka
        else:
            K[e,e] = 0.
        e += 1

    for et, pair in zip(ets, allo.targets):
        i, j, l, p = pair['i'], pair['j'], pair['length'], pair['phase']
        allo._rigidity_pair(q, R, e, i, j)
        L[e,e] = l
        if np.abs(et) > 0:
            allo._applied_strain_jacobian(0, n, q, H, (i, j, 0, ka, l, p, 0))
            allo._pre_stress(q, Kp, Gx, Gy, Gz, e, i, j, ka, l)
            K[e,e] = ka
        else:
            K[e,e] = 0.
        e += 1

    delta = np.zeros(ntot)
    delta[:ne] = np.copy(dl)

    H *= -1
    Hg = R.T@K@R
    Hp = R.T@Kp@R - (Gx.T@Kp@Gx + Gy.T@Kp@Gy + Gz.T@Kp@Gz)

    if not np.allclose(Hg+Hp,H): print("Hessian mismatch.")

    Hinv = np.linalg.pinv(H, hermitian=True)
    W = np.linalg.inv(L)@R@Hinv@R.T@K@L
    bvec = np.dot(np.linalg.inv(L)@R@Hinv@R.T@K, delta.reshape(-1,1)).ravel()

    return W, bvec, H, Hinv, R, K, L

def get_eigens_strained(allo, applied_args, allo_site):
    W, bvec, H, Hinv, R, K, L = get_weight_bias(allo, applied_args)
    evals, evecs = np.linalg.eigh(H)
    return evals, evecs
    
def get_strain_stats(allo, applied_args, allo_site):
    tol = 1e-12
    
    W, bvec, H, Hinv, R, K, L = get_weight_bias(allo, applied_args)
    evals, evecs = np.linalg.eigh(H)
    lA = allo.sources[0]['length']
    lB = allo.targets[0]['length']
    if allo_site == 'source':
        e_avg_fac = lA/lB*(R@Hinv@R.T@K)[-1,-2]
        r = R[-1,:].reshape(1,-1)
    else:
        e_avg_fac = lB/lA*(R@Hinv@R.T@K)[-2,-1]
        r = R[-2,:].reshape(1,-1)
    
    nz = len(evals[evals<tol])
    if (nz != 6):
        print("Number of nonzeros:",nz)
    
    if allo_site == 'source':
        e_std_fac = np.sqrt(1/(lB)**2*(np.sum(np.dot(r,evecs[:,nz:])**2/(evals[nz:]))))
    else:
        e_std_fac = np.sqrt(1/(lA)**2*(np.sum(np.dot(r,evecs[:,nz:])**2/(evals[nz:]))))
    
    return e_avg_fac, e_std_fac