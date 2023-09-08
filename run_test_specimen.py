from simulation import Simulation
from specimen import Specimen
from constants import GeneDesc

specimen = Specimen()
simulation = Simulation(conn_method='GUI')
simulation.run_specimen(specimen)