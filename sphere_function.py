import numpy as np
import matplotlib.pyplot as plt

def sphere_function(individual):
    fitness_value = 0
    for i in individual:
        if i < -100:
            i = -100
        if i > 100:
            i = 100
        fitness_value+=i**2
    return fitness_value
          

def binary_social_group_optimization(N = 10, D = 30, LL = -100, UL = 100, g = 10, c=0.2):
    # 1. Initialize the population
    population = np.random.randint(LL, UL, size=(N, D))
    # print(population)
    gBest_array = []
    x_axis = []

    #2.Calculate Fitness
    for gen in range(g):
        fitness_values = []
        for individual in population:
            individual_fitness = sphere_function(individual)
            fitness_values.append(individual_fitness)
        fitness = np.array(fitness_values)

        # 3. Improving Phase
        gbest = population[np.argmin(fitness)]
        gBest_array.append(fitness[np.argmin(fitness)])
        x_axis.append(gen)
        for i in range(N):
            for j in range(D):
                temp = population[i]
                r = np.random.rand()
                Xnew = c * population[i, j] + r * (gbest[j] - population[i, j])
                temp[j] = Xnew
                if sphere_function(temp) < fitness[i]:
                    population[i, j] = Xnew

        # 4. Acquiring Phase
        for i in range(N):
            ind = np.random.randint(0, N)
            Xr = population[ind]
            temp = population[i]
            if fitness[i] < fitness[ind]:
                r1, r2 = np.random.rand(2)
                for j in range(D):
                    temp = population[i]
                    Xnew = population[i, j] + r1 * (population[i, j] - Xr[j]) + r2 * (gbest[j] - population[i, j])
                    temp[j] = Xnew
                    if sphere_function(temp) < fitness[i]:
                        population[i, j] = Xnew
            else:  
                r1, r2 = np.random.rand(2)
                for j in range(D):
                    temp = population[i]
                    Xnew = population[i, j] + r1 * (Xr[j] - population[i, j]) + r2 * (gbest[j] - population[i, j])
                    temp[j] = Xnew
                    if sphere_function(temp) < fitness[i]:
                        population[i, j] = Xnew
    print(gBest_array)
    print(x_axis)
    plt.plot(x_axis,gBest_array)
    plt.ylabel("G-best values")
    plt.xlabel("Generations")
    plt.title("Graph of SGO")
    plt.show()

binary_social_group_optimization()