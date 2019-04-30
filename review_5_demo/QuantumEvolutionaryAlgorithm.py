def train(
        dataset,
        population_size,
        pc, # crossover probability
        pm, # mutation probability
        pcc, # crossover probability when premature criterion is satisfied
        pmm, # mutation probability when premature criterion is satisfied
        max_unchanged_iterations, # prematurity criterion is satisfied when the best chromosome saved by the algorithm does not change after max_unchanged_iterations
        total_max_iterations, # total maximum iterations after which training must stop. higher the better
) :
    pass