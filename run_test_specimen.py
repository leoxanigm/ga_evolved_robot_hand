import time

from simulation import Simulation
from specimen import Specimen
from constants import GeneDesc

from fitness_fun import FitnessFunction

start = time.time()

specimen = Specimen()
simulation = Simulation(conn_method='GUI')
# simulation = Simulation()
simulation.run_specimen(specimen)

print('#######################################')
print(time.time() - start)
print('#######################################')