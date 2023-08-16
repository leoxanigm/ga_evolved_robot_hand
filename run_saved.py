from specimen import Specimen
from simulation import Simulation
from constants import GeneDesc

specimen = Specimen(GeneDesc, 'robot_hand.urdf', 'd98cd111', '875f0b59')
sim = Simulation('GUI')
sim.run_specimen(specimen, in_training=False)