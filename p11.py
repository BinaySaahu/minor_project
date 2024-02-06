import numpy as np

def sphere_function(individual):
    return sum(min(max(xi, -100), 100) ** 2 for xi in individual)

def sum_squares_function(individual):
    return sum(i * (xi ** 2) for i, xi in enumerate(individual, start=1))

def step_function(individual):
    return sum(int(round(xi)) ** 2 for xi in individual)

def combined_fitness_function(individual):
    return sphere_function(individual), sum_squares_function(individual), step_function(individual)

def binary_social_group_optimization(N = 10, D = 30, LL = -100, UL = 100, g = 10, c=0.2):
    population = np.random.randint(LL, UL, size=(N, D))
    print("Initial population:\n", population)
    gBest_array = []

    for gen in range(g):
        fitness_values = [combined_fitness_function(individual) for individual in population]
        fitness = np.array(fitness_values)

        gbest_indices = np.argmin(fitness, axis=0)
        gbest_sphere = population[gbest_indices[0]]
        gbest_sum_squares = population[gbest_indices[1]]
        gbest_step = population[gbest_indices[2]]

        # Improving Phase
        for i in range(N):
            temp = population[i].copy()
            for j in range(D):
                r = np.random.rand()
                # Apply improvements for each function
                for func_index, gbest in enumerate([gbest_sphere, gbest_sum_squares, gbest_step]):
                    Xnew = c * population[i, j] + r * (gbest[j] - population[i, j])
                    temp[j] = Xnew
                    if combined_fitness_function(temp)[func_index] < fitness[i][func_index]:
                        population[i, j] = Xnew

        # Acquiring Phase
        for i in range(N):
            ind = np.random.randint(0, N)
            Xr = population[ind]
            temp = population[i].copy()
            for func_index in range(3):  # Considering all three functions
                if fitness[i][func_index] < fitness[ind][func_index]:
                    r1, r2 = np.random.rand(2)
                    for j in range(D):
                        Xnew = population[i, j] + r1 * (population[i, j] - Xr[j]) + r2 * (population[gbest_indices[func_index]][j] - population[i, j])
                        temp[j] = Xnew
                        if combined_fitness_function(temp)[func_index] < fitness[i][func_index]:
                            population[i, j] = Xnew

        gBest_array.append((gbest_sphere, gbest_sum_squares, gbest_step))

    # Termination criterion
    best_sphere_solution = population[gbest_indices[0]]
    best_sum_squares_solution = population[gbest_indices[1]]
    best_step_solution = population[gbest_indices[2]]
    best_sphere_fitness = sphere_function(best_sphere_solution)
    best_sum_squares_fitness = sum_squares_function(best_sum_squares_solution)
    best_step_fitness = step_function(best_step_solution)

    print("gBest array:\n", gBest_array)
    return (best_sphere_solution, best_sphere_fitness), (best_sum_squares_solution, best_sum_squares_fitness), (best_step_solution, best_step_fitness)

# Running the optimization
best_sphere, best_sum_squares, best_step = binary_social_group_optimization()
print("Best Sphere Solution:", best_sphere)
print("Best Sum Squares Solution:", best_sum_squares)
print("Best Step Solution:", best_step)