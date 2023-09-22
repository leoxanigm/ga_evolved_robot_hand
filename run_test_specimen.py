import time
import numpy as np
from copy import deepcopy
import warnings

from simulation import Simulation
from specimen import Specimen
from fitness_fun import FitnessFunction
from cross_mutate import Mutate

warnings.filterwarnings('ignore')

# conn_method = 'GUI'
conn_method = 'DIRECT'
with Simulation(conn_method=conn_method) as simulation:
    specimen = Specimen()
    simulation.run_specimen(specimen)
