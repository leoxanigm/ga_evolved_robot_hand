from specimen import Specimen
from simulation import Simulation
from constants import GeneDesc

specimen = Specimen(generation_id='fb6cfddf', specimen_id='777a06ce')
sim = Simulation('GUI')
sim.run_specimen(specimen, in_training=False)