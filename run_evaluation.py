import sys
import os
import csv
from uuid import uuid4
from operator import add

from population import Population
from simulation import Simulation, ThreadedSim

# if len(sys.argv) == 2 and sys.argv[1] == 'DIRECT':
#     conn = 'DIRECT'
# else:
#     conn = 'GUI'

generation_count = 50
population_count = 100
run_id = str(uuid4())[:8]

csv_file = open(f'fit_specimen/{run_id}.csv', 'w')
writer = csv.writer(csv_file)
writer.writerow(['generation_id', 'generation_index', 'average_fitness'])

for i in range(generation_count):
    print('==============')
    print(f'Evaluating generation {i}...')
    print('==============')

    generation_id = str(uuid4())[:8]

    population = Population(population_count)
    simulation = ThreadedSim()
    simulation.run_population(population)

    pop_fitness = [(s.fitness, s) for s in population.specimen]
    pop_fitness = sorted(pop_fitness, key=lambda x: x[0], reverse=True)

    total_fitness = 0
    for fitness, _ in pop_fitness:
        total_fitness += fitness
    pop_avg_fitness = total_fitness / population_count

    fittest = pop_fitness[0]

    if fittest[0] > 0:
        fittest[1].save_specimen(generation_id)

    writer.writerow([generation_id, i, pop_avg_fitness])

    # Clean up training urdf file
    # Source: https://www.tutorialspoint.com/How-to-delete-all-files-in-a-directory-with-Python
    training_urdf_files = os.listdir('intraining_specimen/')
    for urdf_file in training_urdf_files:
        file_path = os.path.join('intraining_specimen/', urdf_file)
        if os.path.isfile(file_path):
            os.remove(file_path)

csv_file.close()