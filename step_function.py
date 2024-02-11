import numpy as np
import matplotlib.pyplot as plt

# New step function for fitness evaluation
def step_function(individual):
    return sum(int(round(xi)) ** 2 for xi in individual)

def binary_social_group_optimization(N=10, D=30, LL=-100, UL=100, g=10, c=0.2):
    population = np.random.randint(LL, UL, size=(N, D))
    print("Initial Population:\n", population)
    
    best_fitness_over_generations = [] 
    
    for gen in range(g):
        # Calculate fitness of the population using the new step function
        fitness_values = [step_function(individual) for individual in population]
        fitness = np.array(fitness_values)
        
        gbest_index = np.argmin(fitness)
        gbest = population[gbest_index]
        best_fitness_over_generations.append(fitness[gbest_index])
        
        # Improving Phase
        for i in range(N):
            for j in range(D):
                r = np.random.rand()
                Xnew = c * population[i, j] + r * (gbest[j] - population[i, j])
                population[i, j] = Xnew  # Directly updating without re-evaluation step
        
        # Simplified for demonstration purposes; directly updating without re-evaluation step

    # Plotting the optimization process
    plt.plot(best_fitness_over_generations, marker='o', linestyle='-', color='blue')
    plt.title('Best Fitness over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Fitness Value')
    plt.grid(True)
    plt.show()
    
    best_solution = population[gbest_index]
    best_fitness = step_function(best_solution)
    print("Best Solution:", best_solution)
    print("Best Fitness:", best_fitness)
    return best_solution, best_fitness

binary_social_group_optimization()
