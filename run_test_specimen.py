from simulation import Simulation
from specimen import Specimen
from constants import GeneDesc

specimen = Specimen(GeneDesc, 'robot_hand.urdf')
simulation = Simulation(conn_method='GUI')
simulation.run_specimen(specimen)