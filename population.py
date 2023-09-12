import math
from collections import namedtuple
import numpy as np
import random

from specimen import Specimen
from fitness_fun import FitnessFunction
from cross_mutate import CrossMutate, Mutate

from constants import GeneDesc, ROBOT_HAND


Fits = namedtuple('Fits', ('f', 's'))
'''Tuple to hold state of fit specimen. f - fitness, s - specimen'''


class Population:
    def __init__(self, pop_size):
        self.pop_size = pop_size
        self._specimen = [Specimen() for _ in range(pop_size)]
        self.fits: list[Specimen] = None  # Top 30% fittest specimen

        self.crossed = False  # Check for crossing
        self.mutated = False  # Check for mutation

        self.pop_fitness: list[Fits] = None

    def calc_fits(self):
        # Get fitnesses with corresponding specimen
        pop_fitness = [Fits(s.fitness, s) for s in self._specimen]
        # Sort specimen by their fitness
        pop_fitness = sorted(pop_fitness, key=lambda x: x.f, reverse=True)

        self.pop_fitness = pop_fitness

        # We only take top 30% fittest, and an even number of fit, to get
        # even number we divide by two, round to ceil and multiply by two
        fit_index = math.ceil((self.pop_size // 3) / 2) * 2

        self.fits = [fits.s for fits in pop_fitness][:fit_index]

        self.crossed = False
        self.mutated = False

    def cross_mutate(self):
        assert self.fits is not None
        assert len(self.fits) % 2 == 0

        random.shuffle(self.fits)

        children = []
        for i in range(0, len(self.fits), 2):
            parent_1 = self.fits[i]
            parent_2 = self.fits[i + 1]

            # Get fitness maps for both parents
            fit_map_p_1 = FitnessFunction.get_fitness_map(
                parent_1.phalanges, parent_1.fingers.shape
            )
            fit_map_p_2 = FitnessFunction.get_fitness_map(
                parent_2.phalanges, parent_2.fingers.shape
            )

            # Cross the parents to create a new child
            f_g_child, b_g_child = CrossMutate.cross_mutate_genomes(
                parent_1.fingers,
                parent_1.brain.genome,
                fit_map_p_1,
                parent_2.fingers,
                parent_2.brain.genome,
                fit_map_p_2,
            )

            # Mutate parents
            parent_1_f_g_mut, parent_1_b_g_mut = Mutate.mutate(
                parent_1.fingers, parent_1.brain.genome, fit_map_p_1
            )
            parent_2_f_g_mut, parent_2_b_g_mut = Mutate.mutate(
                parent_2.fingers, parent_2.brain.genome, fit_map_p_2
            )

            mut_parent_1 = Specimen(
                fingers_genome=parent_1_f_g_mut, brain_genome=parent_1_b_g_mut
            )
            mut_parent_2 = Specimen(
                fingers_genome=parent_2_f_g_mut, brain_genome=parent_2_b_g_mut
            )
            child = Specimen(fingers_genome=f_g_child, brain_genome=b_g_child)

            children.extend([mut_parent_1, mut_parent_2, child])

        # New generation will have fit parents and their children
        self._specimen = children

        self.crossed = True
        self.mutated = True

    def repopulate(self):
        # Make sure we only repopulate after crossing and mutation
        assert self.crossed and self.mutated

        # Add new random specimen to repopulate to the population size
        curr_pop_size = len(self._specimen)
        for _ in range(self.pop_size - curr_pop_size):
            self._specimen.append(Specimen())

    @property
    def specimen(self):
        return self._specimen

    @specimen.setter
    def specimen(self, new_specimen):
        self._specimen = new_specimen
