import os
import csv
import pybullet as p
import numpy as np
import warnings
from glob import glob

from genome import load_genome
from specimen import Specimen
from simulation import Simulation
from fitness_fun import FitnessFunction
import constants as c

warnings.filterwarnings('ignore')

iteration = 1
iteration = 3
iteration = 4
csv_path = glob(os.path.join(c.FIT_DIR, f'iteration_{iteration}', '*.csv'))[0]
id_list = []

with open(csv_path, 'r') as f:
    csv_file = csv.reader(f)
    for i, row in enumerate(csv_file):
        if i != 0 and len(row[3]) > 0:
            id_list.append(row[3])

for g_id in id_list:
    genome_path = os.path.join(c.FIT_DIR, f'iteration_{iteration}', f'{g_id}_')
    try:
        fingers_genome = load_genome(genome_path + 'fingers.pk')
        brain_genome = load_genome(genome_path + 'brain.pk')
    except:
        continue

    specimen = Specimen(fingers_genome=fingers_genome, brain_genome=brain_genome)
    with Simulation('GUI', training=False) as sim:
        sim.run_specimen(specimen)

    # decision = input(f'Do you want to keep genome {g_id}? ([y]/n): ').strip().lower()

    # if decision == 'y':
    #     pass
    # if decision == 'n':
    #     os.remove(genome_path + 'fingers.pk')
    #     os.remove(genome_path + 'brain.pk')
    #     print(f'Genome {g_id} was deleted.')
    # else:
    #     pass
