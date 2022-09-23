from diagonalization import heisenberg_exact_diagonalization
import matplotlib.pyplot as plt
import numpy as np
#Problem 7.2 - Uncomment as necessary

# #7.2.1
# solution1 = heisenberg_exact_diagonalization("1/2", 4, bc="open")
# print("Ground state energy with open BC: ", solution1[1][0])

# #7.2.2
# solution2 = heisenberg_exact_diagonalization("1/2", 4, bc="periodic")
# print("Ground state energy with periodic BC: ", solution2[1][0])

# #7.2.3
# n_sites = 10
# dense_memory_usage = heisenberg_exact_diagonalization("1/2", n_sites, sparse=False)[3]
# sparse_memory_usage = heisenberg_exact_diagonalization("1/2", n_sites, sparse=True)[3]
# difference = round((1-sparse_memory_usage/dense_memory_usage)*100,2)
# print("The dense memory usage:", dense_memory_usage)
# print("The sparse memory usage:", sparse_memory_usage, f"({difference}% smaller)")

# #7.2.4 
# ground_E = []
# for n in range(2,14):
#     ground_E.append(heisenberg_exact_diagonalization("1/2", n, bc="open", sparse=True, sparse_eigenvalues=n)[1][0])

# plt.figure()
# plt.plot([i for i in range(2,14)], ground_E)
# plt.xlabel('Lattice sites n')
# plt.ylabel('Ground energy E0')
# plt.show()
# plt.savefig('ground_energies.pdf')
    
# #7.2.5
# n_initial = 6
# n_final = 16
# delta_E = []

# for n in range(n_initial,n_final,2):
#     energies = heisenberg_exact_diagonalization("1/2", n, bc="periodic", sparse=True, sparse_eigenvalues=n)[1]
#     gap = energies[1]-energies[0]
#     delta_E.append(gap)
#     print("Lattice sites:", n, "/ Energy gap:", gap)

# x = [1/n for n in range(n_initial,n_final,2)]
# z = np.polyfit(x[:3], delta_E[:3], 1)
# p = np.poly1d(z)

# ground_gap = np.polyval(z, 0)
# print("Ground gap:", ground_gap)

# plt.figure()
# plt.plot(x, delta_E, '*', label='Computed values')
# xp = np.linspace(0, 1/n_initial, 100)
# plt.plot(xp, p(xp), '-', label='Linear fit')
# plt.xlabel('Inverse lattice sites 1/n')
# plt.ylabel('Energy gap')
# plt.legend()
# plt.show()
# plt.savefig('spin 1_2 energy gap.pdf')

# #7.2.6

# n_initial = 2
# n_final = 12
# delta_E = []

# for n in range(n_initial,n_final,2):
#     energies = heisenberg_exact_diagonalization("1", n, bc="periodic", sparse=True, sparse_eigenvalues=n)[1]
#     gap = energies[1]-energies[0]
#     delta_E.append(gap)
#     print("Lattice sites:", n, "/ Energy gap:", gap)

# x = [1/n for n in range(n_initial,n_final,2)]
# z = np.polyfit(x[:3], delta_E[:3], 2)
# p = np.poly1d(z)

# ground_gap = np.polyval(z, 0)
# print("Ground gap:", ground_gap)

# plt.figure()
# plt.plot(x, delta_E, '*', label='Computed values')
# xp = np.linspace(0, 1/n_initial, 100)
# plt.plot(xp, p(xp), '-', label='Quadratic fit')
# plt.xlabel('Inverse lattice sites 1/n')
# plt.ylabel('Energy gap')
# plt.legend()
# plt.show()
# plt.savefig('spin 1 energy gap.pdf')

