import random
from deap import base, creator, tools

random.seed(42)

num_items = 20
max_weight = 150

items_weight = [random.randint(5, 20) for _ in range(num_items)]
items_value = [random.randint(10, 50) for _ in range(num_items)]

POPULATION_SIZE = 100
GENERATIONS = 50
CX_PROB = 0.8
MUT_PROB = 0.1
TOURNAMENT_SIZE = 3

def evalKnapsack(individual):
    total_weight = sum(items_weight[i] for i in range(num_items) if individual[i] == 1)
    total_value = sum(items_value[i] for i in range(num_items) if individual[i] == 1)
    if total_weight > max_weight:
        return 0,
    return total_value,

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, num_items)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evalKnapsack)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)

def run_ga():
    population = toolbox.population(n=POPULATION_SIZE)

    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    for gen in range(GENERATIONS):
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CX_PROB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUT_PROB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        population[:] = offspring

    best_individual = tools.selBest(population, 1)[0]
    best_value = int(best_individual.fitness.values[0])
    best_weight = sum(items_weight[i] for i in range(num_items) if best_individual[i] == 1)
    selected_items = [i + 1 for i in range(num_items) if best_individual[i] == 1]

    return best_weight, best_value, selected_items

best_weight, best_value, selected_items = run_ga()

print(f"最优解总重量：{best_weight}kg，总价值：{best_value}元")
print(f"所选货物编号：{selected_items}")
