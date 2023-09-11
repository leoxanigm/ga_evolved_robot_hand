import os


from specimen import Specimen
from simulation import Simulation
from constants import GeneDesc


saved_specimen = sorted(
    [(path[0:8], path[9:17]) for path in os.listdir('fit_specimen/urdf_files/')]
)

curr_spe = saved_specimen[3]

gen_id = curr_spe[0]
spe_id = curr_spe[1]


specimen = Specimen(generation_id=gen_id, specimen_id=spe_id)
sim = Simulation('GUI')
sim.run_specimen(specimen, in_training=False)
