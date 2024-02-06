import numpy as np

def sphere_function(individual):
    fitness_value = 0
    for i in individual:
        if i < -100:
            i = -100
        elif i > 100:
            i = 100
        fitness_value += i**2
    return fitness_value

def sum_squares_function(individual):
    return sum(i * (xi ** 2) for i, xi in enumerate(individual, start=1))

def combined_fitness_function(individual):
    return sphere_function(individual), sum_squares_function(individual)

def binary_social_group_optimization(N = 10, D = 30, LL = -100, UL = 100, g = 10, c=0.2):
    # 1. Initialize the population
    population = np.random.randint(LL, UL, size=(N, D))
    print("Initial population:\n", population)
    gBest_array = []

    for gen in range(g):
        # 2. Calculate fitness of the population
        fitness_values = [combined_fitness_function(individual) for individual in population]
        fitness = np.array(fitness_values)

        # 3. Improving Phase
        gbest_indices = np.argmin(fitness, axis=0)
        gbest_sphere = population[gbest_indices[0]]
        gbest_sum_squares = population[gbest_indices[1]]

        for i in range(N):
            temp = population[i].copy()
            for j in range(D):
                r = np.random.rand()
                # Sphere function improvement
                Xnew_sphere = c * population[i, j] + r * (gbest_sphere[j] - population[i, j])
                temp[j] = Xnew_sphere
                if combined_fitness_function(temp)[0] < fitness[i][0]:  # Accept if better in sphere function
                    population[i, j] = Xnew_sphere

                # Sum squares function improvement
                Xnew_sum_squares = c * population[i, j] + r * (gbest_sum_squares[j] - population[i, j])
                temp[j] = Xnew_sum_squares
                if combined_fitness_function(temp)[1] < fitness[i][1]:  # Accept if better in sum squares function
                    population[i, j] = Xnew_sum_squares

        # 4. Acquiring Phase
        for i in range(N):
            ind = np.random.randint(0, N)
            Xr = population[ind]
            temp = population[i].copy()
            if fitness[i][0] < fitness[ind][0]:  # Learn from more knowledgeable in sphere function
                r1, r2 = np.random.rand(2)
                for j in range(D):
                    Xnew = population[i, j] + r1 * (population[i, j] - Xr[j]) + r2 * (gbest_sphere[j] - population[i, j])
                    temp[j] = Xnew
                    if combined_fitness_function(temp)[0] < fitness[i][0]:
                        population[i, j] = Xnew

            if fitness[i][1] < fitness[ind][1]:  # Learn from more knowledgeable in sum squares function
                r1, r2 = np.random.rand(2)
                for j in range(D):
                    Xnew = population[i, j] + r1 * (Xr[j] - population[i, j]) + r2 * (gbest_sum_squares[j] - population[i, j])
                    temp[j] = Xnew
                    if combined_fitness_function(temp)[1] < fitness[i][1]:
                        population[i, j] = Xnew

        gBest_array.append((gbest_sphere, gbest_sum_squares))

    # 5. Termination criterion
    best_sphere_solution = population[gbest_indices[0]]
    best_sum_squares_solution = population[gbest_indices[1]]
    best_sphere_fitness = sphere_function(best_sphere_solution)
    best_sum_squares_fitness = sum_squares_function(best_sum_squares_solution)

    print("gBest array:\n", gBest_array)
    return (best_sphere_solution, best_sphere_fitness), (best_sum_squares_solution, best_sum_squares_fitness)

# Running the optimization
best_sphere, best_sum_squares = binary_social_group_optimization()
print("Best Sphere Solution:", best_sphere)
print("Best Sum Squares Solution:", best_sum_squares)
