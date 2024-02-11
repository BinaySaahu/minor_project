import numpy as np
import matplotlib.pyplot as plt

def sum_squares_function(individual):
    return sum(i * (xi ** 2) for i, xi in enumerate(individual, start=1))

def combined_fitness_function(individual):
    return sum_squares_function(individual)

def binary_social_group_optimization(N = 10, D = 30, LL = -100, UL = 100, g = 10, c=0.2):
    population = np.random.randint(LL, UL, size=(N, D))
    print("Initial population:\n", population)
    
    best_fitness_over_generations = []  # To track the best fitness value over generations
    
    for gen in range(g):
        fitness_values = [combined_fitness_function(individual) for individual in population]
        fitness = np.array(fitness_values)
        
        gbest_index = np.argmin(fitness)
        best_fitness_over_generations.append(fitness[gbest_index])  # Track the best fitness
        
        gbest = population[gbest_index]
        
        for i in range(N):
            temp = population[i].copy()
            for j in range(D):
                r = np.random.rand()
                Xnew = c * population[i, j] + r * (gbest[j] - population[i, j])
                temp[j] = Xnew
                if combined_fitness_function(temp) < fitness[i]:
                    population[i, j] = Xnew

        for i in range(N):
            ind = np.random.randint(0, N)
            Xr = population[ind]
            temp = population[i].copy()
            if fitness[i] < fitness[ind]:
                r1, r2 = np.random.rand(2)
                for j in range(D):
                    Xnew = population[i, j] + r1 * (population[i, j] - Xr[j]) + r2 * (gbest[j] - population[i, j])
                    temp[j] = Xnew
                    if combined_fitness_function(temp) < fitness[i]:
                        population[i, j] = Xnew

    best_solution = population[gbest_index]
    best_fitness = sum_squares_function(best_solution)

    print("Best Solution:", best_solution)
    print("Best Fitness:", best_fitness)

    # Return the best solution, fitness, and the tracking of fitness over generations
    return best_solution, best_fitness, best_fitness_over_generations

# Running the optimization
best_solution, best_fitness, best_fitness_over_generations = binary_social_group_optimization()

# Plotting the best fitness over generations
plt.plot(best_fitness_over_generations, marker='o')
plt.title('Best Fitness Over Generations')
plt.xlabel('Generation')
plt.ylabel('Best Fitness')
plt.grid(True)
plt.show()
