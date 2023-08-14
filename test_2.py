import sys

from population import Population
from simulation import Simulation

if len(sys.argv) == 2 and sys.argv[1] == 'DIRECT':
    conn = 'DIRECT'
else:
    conn = 'GUI'

specimen = Population(1).specimen[0]
simulation = Simulation(conn)
simulation.run_specimen(specimen)