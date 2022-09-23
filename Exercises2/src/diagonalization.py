import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

def heisenberg_exact_diagonalization(s, n, bc="periodic",sparse=False, sparse_eigenvalues=6):
    """
    Performs an exact diagonalization of the Heisenberg model with spin s and n lattice sites for both open and periodic boundary conditions.
    Outputs the dense/sparse hamiltonian H, its eigenvalues and eigenvectors.

    """
    
    #Defines the spin matrices for spin s
    if s == "1/2":
        I = np.identity(2)
        Sz = (1/2)*np.array([[1,0],[0,-1]])
        Sup = np.array([[0,1],[0,0]])
        Sdown = np.array([[0,0],[1,0]])
        
    elif s=="1":
        I = np.identity(3)
        Sz = np.array([[1,0,0],[0,0,0],[0,0,-1]])
        Sup = np.sqrt(2)*np.array([[0,1,0],[0,0,1],[0,0,0]])
        Sdown = np.sqrt(2)*np.array([[0,0,0],[1,0,0],[0,1,0]])
        
    #Builds the hamiltonian for the dense case
    if sparse==False:
        
        #Builds the 2-site hamiltonian term
        H2 = np.kron(Sz,Sz) + (np.kron(Sup,Sdown) + np.kron(Sdown,Sup))/2
        
        #Main loop
        Hk = H2
        Ik = 1
        for i in range(n-2):
            Ik = np.kron(Ik,I)
            Hk = np.kron(Hk,I) + np.kron(Ik,H2)
        
        #Implements the periodic boundary condition term
        if bc=="periodic":
            H_boundary = np.kron(Sz, np.kron(Ik, Sz)) + (np.kron(Sup, np.kron(Ik, Sdown)) + np.kron(Sdown, np.kron(Ik, Sup)))/2
            Hk += H_boundary
            
        eigenval, eigenvec = np.linalg.eigh(Hk)
        memory = Hk.nbytes
    
    #Builds the hamiltonian for the sparse case
    elif sparse==True:
        
        #Defines the sparse spin matrices
        I_sparse = sp.csr_matrix(I)
        Sz_sparse = sp.csr_matrix(Sz)
        Sup_sparse = sp.csr_matrix(Sup)
        Sdown_sparse = sp.csr_matrix(Sdown)
        
        #Builds the 2-site hamiltonian term
        H2 = sp.kron(Sz_sparse,Sz_sparse,'csr') + (sp.kron(Sup_sparse,Sdown_sparse,'csr') + sp.kron(Sdown_sparse,Sup_sparse,'csr'))/2
    
        #Main loop
        Hk = H2
        Ik = 1
        for i in range(n-2):
            Ik = sp.kron(Ik, I_sparse, 'csr')
            Hk = sp.kron(Hk, I_sparse, 'csr') + sp.kron(Ik,H2,'csr')
          
        #Implements the periodic boundary condition term
        if bc=="periodic":
            H_boundary = sp.kron(Sz, sp.kron(Ik, Sz,'csr'),'csr') + (sp.kron(Sup, sp.kron(Ik, Sdown,'csr'),'csr') + sp.kron(Sdown, sp.kron(Ik, Sup,'csr'),'csr'))/2
            Hk += H_boundary
            
        eigenval, eigenvec = spl.eigsh(Hk, k=sparse_eigenvalues, which='SA')
        memory = Hk.data.nbytes
            
    return (Hk, eigenval, eigenvec, memory)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        