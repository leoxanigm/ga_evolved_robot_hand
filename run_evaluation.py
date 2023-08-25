import sys
import os
import csv
from uuid import uuid4
from operator import add

from population import Population
from simulation import Simulation, ThreadedSim

from helpers.misc_helpers import write_csv

# if len(sys.argv) == 2 and sys.argv[1] == 'DIRECT':
#     conn = 'DIRECT'
# else:
#     conn = 'GUI'

generation_count = 50
population_count = 100
run_id = str(uuid4())[:8]


write_csv(
    f'fit_specimen/{run_id}.csv',
    ['generation_id', 'generation_index', 'average_fitness'],
)

for i in range(generation_count):
    print('==============')
    print(f'Evaluating generation {i}...')
    print('==============')

    generation_id = str(uuid4())[:8]

    population = Population(population_count)
    simulation = ThreadedSim()
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
