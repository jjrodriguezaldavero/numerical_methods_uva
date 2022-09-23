"""
Execution of the spin-flip Metropolis algorithm for exercises 4.1 and 4.2:
    1. Solution for exercise 4.1
    2. Generation of data for exercise 4.2
    3. Plotting of the data to create the charts of exercise 4.2
Uncomment when needed
Juan José Rodríguez Aldavero
"""


# import metropolis as m
# import numpy as np
# import time
# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.rcParams.update({'font.size': 14})


##########################################################
#1. Exercise 4.1
##########################################################

# values = m.spin_flip_metropolis_algorithm(4, 3)
# print("Mean energy per site:", values[0][0], " +/- ", values[0][1])
# print("Mean magnetization per site:", values[1][0], " +/- ", values[1][1])
# print("Mean squared magnetization per site:", values[2][0], " +/- ", values[2][1])
# print("Order parameter:", values[3][0], " +/- ", values[3][1])  

# #Exercise 4.2: algorithm run

##########################################################
#2. Generation of data for exercise 4.2
##########################################################

# lengths = [4,8,12,16,24]
# #temperatures = [0.1,1,2,2.269,2.5,3,4]
# temperatures = [0.1,1,2,2.1,2.2,2.269,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3,4]
# m_values = np.zeros((len(lengths), len(temperatures)))
# m_errors = np.zeros((len(lengths), len(temperatures)))
# tau_m_values = np.zeros((len(lengths), len(temperatures)))
# M_values = np.zeros((len(lengths), len(temperatures)))
# M_errors = np.zeros((len(lengths), len(temperatures)))
# times = np.zeros((len(lengths), len(temperatures)))

# #Extract the data
# k=0
# for i in range(len(lengths)):
#     for j in range(len(temperatures)):
#         print("Iteration number:", k, "/", len(lengths) * len(temperatures))
#         print("Length:", lengths[i], "Temperature:", temperatures[j])
        
#         t1 = time.time()
#         values = m.spin_flip_metropolis_algorithm(lengths[i], temperatures[j])
#         t2 = time.time()
#         M_values[i][j] = values[1][0]
#         M_errors[i][j] = values[1][1]
#         m_values[i][j] = values[3][0]
#         m_errors[i][j] = values[3][1]
#         tau_m_values[i][j] = values[3][2]
#         
#         times[i][j] = t2-t1
#         k += 1
#         print("Time:", t2-t1)
        
# np.savetxt('m_values.csv',m_values, delimiter=',')
# np.savetxt('m_errors.csv',m_errors, delimiter=',')
# np.savetxt('tau_m_values.csv',tau_m_values, delimiter=',')
# np.savetxt('Mg_values.csv',M_values, delimiter=',')
# np.savetxt('Mg_errors.csv',M_errors, delimiter=',')
# np.savetxt('times.csv',times, delimiter=',')

##########################################################
#3. Plotting of the data to create the charts of exercise 4.2
##########################################################

# m_values = np.loadtxt('m_values.csv', delimiter=',')
# m_errors = np.loadtxt('m_errors.csv', delimiter=',')
# tau_m_values = np.loadtxt('tau_m_values.csv', delimiter=',')
# M_values = np.loadtxt('Mg_values.csv', delimiter=',')
# M_errors = np.loadtxt('Mg_errors.csv', delimiter=',')
# times = np.loadtxt('times.csv', delimiter=',')

#1. Order parameter as a function of T
# plt.figure()
# plt.errorbar(temperatures, m_values[0], yerr=m_errors[0], label='L=4')
# plt.errorbar(temperatures, m_values[1], yerr=m_errors[1], label='L=8')
# plt.errorbar(temperatures, m_values[2], yerr=m_errors[2], label='L=12')
# plt.errorbar(temperatures, m_values[3], yerr=m_errors[3], label='L=16')
# plt.errorbar(temperatures, m_values[4], yerr=m_errors[4], label='L=24')
# plt.xlabel('Temperature T')
# plt.ylabel('Order parameter')
# plt.legend()
# plt.show()
# plt.savefig('Order parameter.pdf')

#2. Autocorrelation time
# plt.figure()
# plt.plot(temperatures, tau_m_values[0], label='L=4', marker='.')
# plt.plot(temperatures, tau_m_values[1], label='L=8', marker='.')
# plt.plot(temperatures, tau_m_values[2], label='L=12', marker='.')
# plt.plot(temperatures, tau_m_values[3], label='L=16', marker='.')
# plt.plot(temperatures, tau_m_values[4], label='L=24', marker='.')
# plt.xlabel('Temperature T')
# plt.ylabel('Self correlation time')
# plt.legend()
# plt.show()
# plt.savefig('Self correlation.pdf')

#3. Magnetization per site 
# plt.figure()
# plt.errorbar(temperatures, M_values[0], marker='.', yerr=M_errors[0], label='L=4')
# plt.errorbar(temperatures, M_values[1], marker='.', yerr=M_errors[1], label='L=8')
# plt.errorbar(temperatures, M_values[2], marker='.', yerr=M_errors[2], label='L=12')
# plt.errorbar(temperatures, M_values[3], marker='.', yerr=M_errors[3], label='L=16')
# plt.errorbar(temperatures, M_values[4], marker='.', yerr=M_errors[4], label='L=24')
# plt.xlabel('Temperature T')
# plt.ylabel('Magnetization')
# plt.legend()
# plt.show()
# plt.savefig('Magnetization2.pdf')

#4. Execution times
# plt.figure()
# plt.plot(temperatures, times[0], label='L=4', marker='.')
# plt.plot(temperatures, times[1], label='L=8', marker='.')
# plt.plot(temperatures, times[2], label='L=12', marker='.')
# plt.plot(temperatures, times[3], label='L=16', marker='.')
# plt.plot(temperatures, times[4], label='L=24', marker='.')
# plt.xlabel('Temperature T')
# plt.ylabel('Execution time')
# plt.legend()
# plt.show()
# plt.savefig('Execution time.pdf')







