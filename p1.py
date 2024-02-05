import numpy as np

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
    population = np.random.randint(LL, UL, size=(N, D))  # Generate binary population

    for gen in range(g):
        # 2. Calculate fitness of the population
        fitness_values = []

        # Loop through each individual in the population
        for individual in population:
            # Calculate the fitness value for the current individual
            individual_fitness = sphere_function(individual)
            # Append the calculated fitness value to the list
            fitness_values.append(individual_fitness)

        # Convert the list of fitness values into a NumPy array for efficient operations
        fitness = np.array(fitness_values)

        # print(fitness)

        # 3. Improving Phase
        gbest = population[np.argmin(fitness)]  # Find the best solution

        for i in range(N):
            temp = population[i]
            for j in range(D):
                r = np.random.rand()
                Xnew = c * population[i, j] + r * (gbest[j] - population[i, j])
                temp[j] = Xnew
                if sphere_function(temp) <= fitness[i]:  # Accept if better
                    population[i, j] = Xnew

        # 4. Acquiring Phase (Corrected while loop condition)
        for i in range(N):
            # Randomly select a different person (using np.any for element-wise comparison)
            ind = np.random.randint(0, N)
            Xr = population[ind]
            while np.any(Xr == population[i]):  # Ensure selection of a distinct individual
                ind = np.random.randint(0, N)
                Xr = population[ind]
            temp = population[i]
            if fitness[i] < fitness[ind]:  # Learn from more knowledgeable
                r1, r2 = np.random.rand(2)
                for j in range(D):
                    Xnew = population[i, j] + r1 * (population[i, j] - Xr[j]) + r2 * (gbest[j] - population[i, j])
                    temp[j] = Xnew
                    if sphere_function(temp) <= fitness[i]:
                        population[i, j] = Xnew
            else:  # Learn from those with better attributes
                r1, r2 = np.random.rand(2)
                for j in range(D):
                    Xnew = population[i, j] + r1 * (Xr[j] - population[i, j]) + r2 * (gbest[j] - population[i, j])
                    temp[j] = Xnew
                    if sphere_function(temp) <= fitness[i]:
                        population[i, j] = Xnew

    # 5. Termination criterion

    best_solution = population[np.argmin(fitness)]
    best_fitness = sphere_function(best_solution)
    print(best_solution)
    print(best_fitness)
    return best_solution, best_fitness

binary_social_group_optimization()