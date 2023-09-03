from specimen import Specimen
from simulation import Simulation
from constants import GeneDesc

gen_id = '07381b4d'
spe_id = '37eb7ae3'

specimen = Specimen(generation_id=gen_id, specimen_id=spe_id)
sim = Simulation('GUI')
sim.run_specimen(specimen, in_training=False)