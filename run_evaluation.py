import sys
import os
import csv
from uuid import uuid4
from operator import add
import time

from population import Population
from simulation import Simulation, ThreadedSim

from helpers.misc_helpers import write_csv, clear_training_dir

write_csv('log.csv', ['iteration_id', 'total_run_time(s)'])

for j in range(50):  # run 50 evaluations
    generation_count = 50  # for 50 generations
    population_count = 100  # of 100 specimen each

    run_id = str(uuid4())[:8]

    write_csv(
        f'fit_specimen/{run_id}.csv',
        ['generation_id', 'generation_index', 'average_fitness'],
    )

    start_time = time.time()

    population = Population(population_count)
    simulation = ThreadedSim()

    for i in range(generation_count):
        print('==============')
        print(f'Evaluating generation {i}...')
        print('==============')

        generation_id = str(uuid4())[:8]

        simulation.run_population(population)

        population.calc_fits()
        population.cross()
        population.mutate()
        population.repopulate()

        pop_fitness = population.pop_fitness

        total_fitness = 0
        for fitness, _ in pop_fitness:
            total_fitness += fitness
        pop_avg_fitness = total_fitness / population_count

        fittest = pop_fitness[0]

        if fittest[0] > 0:
            fittest[1].save_specimen(generation_id)

        write_csv(
            f'fit_specimen/{run_id}.csv',
            [generation_id, i, pop_avg_fitness],
        )

        clear_training_dir

    write_csv('log.csv', [j, time.time() - start_time])