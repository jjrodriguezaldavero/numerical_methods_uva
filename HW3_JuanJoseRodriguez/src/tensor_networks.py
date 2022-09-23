import numpy as np
import math
from diagonalization import heisenberg_exact_diagonalization

def mps_decomposition(state, truncate=False, D=0):
    """
    Computes a Matrix Product State with bond dimension D for a vector state
    """
    L = state.shape[0]
    m = int(math.log2(L))
    d = 2
    
    #Stores the unitaries from the MPS
    mps = []
    
    #Reshapes the state vector into an initial (d, d**(m-1)) matrix
    state_matrix = np.reshape(state, (d, d**(m-1)))
    
    #Main loop
    for i in range(m-1):
        
        #First tensor
        if i==0:
            if truncate==True:
                n = min(D, d**min(i+1, m-i+1))
            else:
                n = d**min(i+1, m-i+1)
            state_matrix = np.reshape(state_matrix, (d, d**(m-1)))
            u,s,vd = np.linalg.svd(state_matrix, full_matrices = False)
            u_i = u[:, :n].reshape(d, n)
            mps.append(u_i)
            state_matrix = np.dot(np.diag(s[:n]), vd[:n, :])
            
        #Remaining tensors
        else:
            if truncate==True:
                n = min(D, d**min(i+1, L-i-1))
                N = min(D, d**min(i, L-i))
            else:
                n = d**min(i+1, L-i-1)
                N = d**min(i, L-i)
            state_matrix = np.reshape(state_matrix, (N*d, d**(m-1-i)))
            u,s,vd = np.linalg.svd(state_matrix, full_matrices = False)
            u_i = u[:, :n].reshape(N*d, n)
            mps.append(u_i)
            state_matrix = np.dot(np.diag(s[:m]), vd[:m, :])
    
    return mps

def mps_recomposition(mps):
    """
    Recomposes the vector state from its Matrix Product State

    """
    
    #Computes the first tensor product
    state = np.tensordot(mps[0], mps[1])
    
    #Computes the dimension of the state vector from the dimensions of the MPS matrices
    vector_dim = mps[0].shape[0] * mps[-1].shape[1]
    
    #Computes the remaining tensor products
    for i in range(len(mps)-1):
        state = np.tensordot(state, mps[i+1])
        
    #Flattens the state into its vector form and returns it
    state_vector = np.reshape(state, (vector_dim,))
    
    return state_vector

    
    
    
    
    
    
    