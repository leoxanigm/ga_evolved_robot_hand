import sys
import csv

from population import Population
from simulation import Simulation, ThreadedSim

# if len(sys.argv) == 2 and sys.argv[1] == 'DIRECT':
#     conn = 'DIRECT'
# else:
#     conn = 'GUI'

population = Population(50)
simulation = ThreadedSim()
simulation.run_population(population)

pop_fitness = [s.fitness for s in population.specimen]
pop_fitness = sorted(pop_fitness, reverse=True)
print(pop_fitness)