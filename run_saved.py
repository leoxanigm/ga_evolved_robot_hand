import os


from specimen import Specimen
from simulation import Simulation
from constants import GeneDesc


# saved_specimen = [('e76a7090', '93889a26')]  # Table joint has to be prismatic
saved_specimen = [('154173b4', 'a7d5c5f4')]
# saved_specimen = sorted(
#     [(path[0:8], path[9:17]) for path in os.listdir('fit_specimen/urdf_files/')]
# )


for curr_spe in saved_specimen:
    gen_id = curr_spe[0]
    spe_id = curr_spe[1]

    specimen = Specimen(generation_id=gen_id, specimen_id=spe_id)
    with Simulation('GUI') as sim:
        sim.run_specimen(specimen, in_training=False)
