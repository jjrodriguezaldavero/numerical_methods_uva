
"""
Exercise 4.1: Monte Carlo code for the 2D Ising model
Juan José Rodríguez Aldavero
"""

import numpy as np

def flip_spin(lattice, x, y):
    """
    Flips the spin in position (x,y) of the spin lattice.
    Returns the flipped lattice as well as the change in energy from the flip.
    """
    #Flip spin
    spin = lattice[x,y]
    spin_flip = spin * -1
    lattice[x,y] = spin_flip
    L = lattice.shape[0]
    
    #Calculate variation in energy
    spin_up = lattice[x, (y+1)%L]
    spin_down = lattice[x, (y-1)%L]
    spin_right = lattice[(x+1)%L, y]
    spin_left = lattice[(x-1)%L, y]
    
    E = -spin * (spin_up + spin_down + spin_right + spin_left)
    E_flip = -spin_flip * (spin_up + spin_down + spin_right + spin_left)
    delta_E = E_flip - E
    
    return lattice, delta_E

def binning_analysis(samples):
    """Perform a binning analysis over samples and return 
    errors: an array of the error estimate at each binning level, 
    tau: the estimated integrated autocorrelation time, 
    converged: a flag indicating if the binning has converged, and 
    bins: the last bin values"""
    minbins = 2**6 # minimum number of bins     
    maxlevel = int(np.log2(len(samples)/minbins)) # number of binning steps
    maxsamples = minbins * 2**(maxlevel)   # the maximal number of samples considered 
    bins = np.array(samples[-maxsamples:]) 
    errors = np.zeros(maxlevel+1)
    for k in range(maxlevel):
        errors[k] = np.std(bins)/np.sqrt(len(bins)-1.)
        bins = np.array((bins[::2] + bins[1::2])/2.)
        
    errors[maxlevel] = np.std(bins)/np.sqrt(len(bins)-1.)    
    tau = 0.5*((errors[-1]/errors[0])**2 - 1.)
    relchange = (errors[1:] - errors[:-1]) / errors[1:]
    meanlastchanges = np.mean(relchange[-3:])    # get the average over last changes
    converged = 1
    if meanlastchanges > 0.05:
        print("warning: binning maybe not converged, meanlastchanges:", meanlastchanges)
        converged = 0
    return [errors, tau, converged, bins]


def measure_observables(lattice):
    """
    Measures the observables asked for in the exercise and returns them in a list:
        Energy per site, E
        Magnetization per site, M
        Squared magnetization, M2
        Order parameter, m
    """
    L = lattice.shape[0]
    
    #Compute energy per site
    E = 0
    
    for i in range(L):
        for j in range(L):
            
            spin = lattice[i,j]
            
            spin_up = lattice[i, (j+1)%L]
            spin_down = lattice[i, (j-1)%L]
            spin_right = lattice[(i+1)%L, j]
            spin_left = lattice[(i-1)%L, j]
            
            E -= spin * (spin_up + spin_down + spin_left + spin_right)
            
    E /= 2*L**2 #Not counting twice each pair and dividing by the number of sites
    
    #Compute the other observables
    M = lattice.sum() / L**2
    M2 = lattice.sum() ** 2
    op = np.abs(M)
    
    return [E,M,M2,op]

def spin_flip_metropolis_algorithm(L, T, J=1, Nsteps=2**14, Nthermsteps=2**12):
    """
    Performs a spin-flip Metropolis algorithm over a lattice of size L and temperature T initialized
    with all spins down, using Nsteps, Nthermsteps and L**2 sweeps.
    Outputs the mean, error and integrated correlation time of the following observables:
        Energy per site, E
        Magnetization per site, M
        Squared magnetization, M2
        Order parameter, m
    """
    #Initial parameters
    Nsweeps = L**2
    Nstepstotal = Nsteps + Nthermsteps
    
    #Initialization of random number generator
    seed = 42
    np.random.seed(seed)
    
    #Initialization of spin lattice and observable matrix
    lattice = np.full((L,L), -1)
    obs = np.zeros((Nstepstotal, 4))
    
    #Main loop
    for k in range(Nstepstotal * Nsweeps):
        x_flip = np.random.randint(L)
        y_flip = np.random.randint(L)
        #We need to pass a copy of the lattice to have the two versions available
        lattice_new, delta_E = flip_spin(np.copy(lattice), x_flip, y_flip) 
        
        #Spin flip acceptance or rejection
        p_new = np.exp(-delta_E / T)

        rand = np.random.random()
        if (delta_E <= 0) or (delta_E > 0 and rand < p_new):
            lattice = lattice_new
        
        #Measure observables
        if k % Nsweeps == 0:
            obs[k // Nsweeps] = measure_observables(lattice)
            
        #Print loop progress
        if (100*k / (Nstepstotal * Nsweeps)) % 20 == 0:
            print("Progress:", 100*k // (Nstepstotal * Nsweeps), '%')
        
    #Remove thermalization steps
    obs = obs[Nthermsteps:]
    
    #Output mean, error and correlation time of observables using binning analysis
    E_binning = binning_analysis(obs[:,0])
    E = [np.mean(obs[:,0]), E_binning[0][-1], E_binning[1]]
    
    M_binning = binning_analysis(obs[:,1])
    M = [np.mean(obs[:,1]), M_binning[0][-1], M_binning[1]]
    
    M2_binning = binning_analysis(obs[:,2])
    M2 = [np.mean(obs[:,2]), M2_binning[0][-1], M2_binning[1]]
    
    m_binning = binning_analysis(obs[:,3])
    m = [np.mean(obs[:,3]), m_binning[0][-1], m_binning[1]]
    
    return (E, M, M2, m)


    
    