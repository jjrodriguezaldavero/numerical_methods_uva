# import numpy as np
# from diagonalization import heisenberg_exact_diagonalization
# from tensor_networks import mps_decomposition, mps_recomposition
# import matplotlib.pyplot as plt

# #Problem 9.1

# #1. Defines the spin matrices and the 2-site hamiltonian
# I = np.identity(2)
# Sz = (1/2)*np.array([[1,0],[0,-1]])
# Sup = np.array([[0,1],[0,0]])
# Sdown = np.array([[0,0],[1,0]])

# H2 = np.kron(Sz,Sz) + (np.kron(Sup,Sdown) + np.kron(Sdown,Sup))/2

# #Reshapes the hamiltonian into tensor form
# d = 2
# H2_tensor = np.reshape(H2, (d,d,d,d))

# #2. Computes the eigenvalues and eigenvectors of the hamiltonian
# eigenval, eigenvec = np.linalg.eigh(H2)
# psi = eigenvec[:,0]

# #Reshapes the eigenvector into matrix form
# psi_matrix = np.reshape(psi, (d,d))

# #3 .Computes the contraction the two terms 
# H_contraction = np.tensordot(psi_matrix, np.tensordot(H2_tensor,psi_matrix, axes=2), axes=2)
# psi_contraction = np.tensordot(psi_matrix, psi_matrix, axes=2)

# #4. Computes the transpose terms and checks if the tensor form of H is hermitian
# psi_matrix_transpose = np.transpose(psi_matrix)
# psi_contraction_transpose = np.tensordot(psi_matrix, psi_matrix_transpose)
# H2_tensor_transpose = np.transpose(H2_tensor)
# print(H2_tensor == H2_tensor_transpose)


# #Problem 9.2
# #1. Computes the first 100 eigenvalues and eigenvectors of the 16-site Heisenberg model
# n_states = 100
# exact_diagonalization = heisenberg_exact_diagonalization("1/2", 16, bc="open",sparse=True, sparse_eigenvalues=n_states)
# exact_energies = exact_diagonalization[1]
# exact_states = exact_diagonalization[2]

# #Selects and reshapes the ground state
# psi1 = exact_states[:, 0]
# vector_size = psi1.shape[0]
# matrix_size = int(np.sqrt(vector_size))
# psi1_matrix = np.reshape(psi1, (matrix_size,matrix_size))

# #Performs a SVD on the middle of the chain for the ground state and computes the eigenvalues of the reduced matrix
# u1,s1,v1 = np.linalg.svd(psi1_matrix, full_matrices=False)
# p1 = s1**2 / sum(s1**2)

# #Computes entanglement entropy for the ground state
# entanglement_entropy1 = -np.sum(p1*np.log(p1))

# #Computes and reshapes a normalized random state from a linear combination of basis states
# psi2 = np.zeros(vector_size)
# amplitudes = np.array([])
# for i in range(n_states):
#     amplitude = np.random.random()
#     amplitudes = np.append(amplitudes, amplitude)
#     psi2 += amplitude * exact_states[:, i]
# psi2 /= np.sqrt(psi2.dot(psi2))
# amplitudes /= np.sqrt(amplitudes.dot(amplitudes))
# psi2_matrix = np.reshape(psi2, (matrix_size,matrix_size))

# #Performs a SVD on the middle of the chain for the random state and computes the eigenvalues of the reduced matrix
# u2,s2,v2 = np.linalg.svd(psi2_matrix, full_matrices=False)
# p2 = s2**2 / sum(s2**2)

# #Computes entanglement entropy for the random state
# entanglement_entropy2 = -np.sum(p2*np.log(p2))

# # Creates the chart for the eigenvalues
# plt.figure()
# plt.yscale("log")
# plt.plot(p1[:100],label='Ground state')
# plt.plot(p2[:100],label='Random state')
# plt.xticks(np.arange(0, 100, step=10))
# plt.xlabel('Position')
# plt.ylabel('Eigenvalue')
# plt.legend()
# plt.show()

# #Truncation step

# delta_E1 = []
# delta_E2 = []

# for D in range(4,100):
#     #Truncates the SVD matrices for the ground state and random state
#     U1 = u1[:, :D]
#     S1 = np.diag(s1[:D])
#     V1 = v1[:D, :]
    
#     U2 = u2[:, :D]
#     S2 = np.diag(s2[:D])
#     V2 = v2[:D, :]
    
#     #Computes the approximate wavefunctions
#     psi1_approx = np.dot(U1, np.dot(S1, V1)).reshape((vector_size,))
#     psi1_approx /= np.sqrt(psi1_approx.dot(psi1_approx))
#     psi2_approx = np.dot(U2, np.dot(S2, V2)).reshape((vector_size,))
#     psi2_approx /= np.sqrt(psi2_approx.dot(psi2_approx))
    
#     #Computes the exact energies for the states from the eigenvalues of the hamiltonian
#     E1_exact = exact_energies[0]
#     E2_exact = sum([energy*amplitude**2 for amplitude, energy in zip(amplitudes, exact_energies)])
    
    
#     #Computes the approximated energies from the projections of the state onto the basis state of the hamiltonian
#     E1_approx = 0
#     E2_approx = 0
#     for i in range(n_states):
#         E1_approx += exact_energies[i]*(psi1_approx.dot(exact_states[:,i]))**2
#         E2_approx += exact_energies[i]*(psi2_approx.dot(exact_states[:,i]))**2
    
#     delta_E1.append((E1_exact - E1_approx)/E1_exact)
#     delta_E2.append((E2_exact - E2_approx)/E2_exact)
    
# # Creates the chart for the energy variations
# plt.figure()
# plt.yscale("log")
# plt.plot(delta_E1,label='Ground state')
# plt.plot(delta_E2,label='Random state')
# plt.xticks(np.arange(0, 100, step=10))
# plt.xlabel('Bond value D')
# plt.ylabel('Difference in energies')
# plt.legend()
# plt.show()


# #Problem 9.3 DRAFT, NOT WORKING AND NOT TESTED

# #Computes an MPS decomposition of the ground state
# psi1_mps = mps_decomposition(psi1, truncate="False")

# #Recomputes the ground state from its MPS without a limit on the bond dimension
# ps1_approx_mps = mps_recomposition(psi1_mps)

# #Checks if both states have the same energy. Calculates the approximate energy from projections onto basis states
# E1_approx_mps = 0
# for i in range(n_states):
#     E1_approx_mps += exact_energies[i]*(psi1_approx.dot(ps1_approx_mps[:,i]))**2

# #Appends the data points into a list
# delta_E1_mps = E1_exact - E1_approx_mps

# #Truncation step
# for bond in range(4,100):
#     #Computes an MPS decomposition of the ground state with a maximal bond dimension D
#     psi1_mps = mps_decomposition(psi1, truncate=True, D=bond)
    
#     #Reconstructs the approximate wavefunction
#     psi1_approx_mps = mps_recomposition(psi1_mps)
    
#     #Compares the energies between states. Calculates the approximate energy from projections onto basis states
#     E1_exact = exact_energies[0]
#     E1_approx = 0
#     for i in range(n_states):
#         E1_approx += exact_energies[i]*(psi1_approx_mps.dot(exact_states[:,i]))**2
    
#     #Appends the data points into a list
#     delta_E1.append((E1_exact - E1_approx)/E1_exact)

# # Creates the chart for the energy variations
# plt.figure()
# plt.yscale("log")
# plt.plot(delta_E1,label='Ground state')
# plt.xticks(np.arange(0, 100, step=10))
# plt.xlabel('Bond value D')
# plt.ylabel('Difference in energies')
# plt.legend()
# plt.show()
    


























