import math
from collections import namedtuple
import numpy as np
import random
from copy import deepcopy

from specimen import Specimen
from fitness_fun import FitnessFunction
from cross_mutate import CrossMutate, Mutate


Fits = namedtuple('Fits', ('f', 's'))
'''Tuple to hold state of fit specimen. f - fitness, s - specimen'''


class Population:
    def __init__(self, pop_size):
        self.pop_size = pop_size
        self._specimen = [Specimen() for _ in range(pop_size)]
        self.top_fits: list[Specimen] = None  # Top 30% fittest specimen

        self.crossed = False  # Check for crossing
        self.mutated = False  # Check for mutation

        self.fittest: list[Fits] = None

    def calc_fits(self):
        # Get fitnesses with corresponding specimen
        fittest = [Fits(s.fitness, s) for s in self._specimen]
        # Sort specimen by their fitness
        fittest = sorted(fittest, key=lambda x: x.f, reverse=True)

        self.fittest = fittest

        # We only take top 30% fittest, and an even number of fit, to get
        # even number we divide by two, round to ceil and multiply by two
        fit_index = math.ceil((self.pop_size // 3) / 2) * 2

        self.top_fits = [fits.s for fits in fittest][:fit_index]

        self.crossed = False
        self.mutated = False

    def cross_mutate(self):
        assert self.top_fits is not None
        assert len(self.top_fits) % 2 == 0

        random.shuffle(self.top_fits)

        children = []
        for i in range(0, len(self.top_fits), 2):
            parent_1 = self.top_fits[i]
            parent_2 = self.top_fits[i + 1]

            f_g_parent_1 = parent_1.fingers_genome
            b_g_parent_1 = parent_1.brain_genome
            f_g_parent_2 = parent_2.fingers_genome
            b_g_parent_2 = parent_2.brain_genome

            # Get fitness maps for both parents
            fit_map_p_1 = FitnessFunction.get_fitness_map(
                parent_1.phalanges, f_g_parent_1.shape
            )
            fit_map_p_2 = FitnessFunction.get_fitness_map(
                parent_2.phalanges, f_g_parent_2.shape
            )

            # Cross the parents to create a new child
            f_g_child, b_g_child = CrossMutate.cross_mutate_genomes(
                f_g_parent_1,
                b_g_parent_1,
                fit_map_p_1,
                f_g_parent_2,
                b_g_parent_2,
                fit_map_p_2,
                mutate_factor=1 - parent_1.fitness,
            )

            # Mutate parents
            f_g_parent_1, b_g_parent_1 = Mutate.mutate(
                f_g_parent_1,
                b_g_parent_1,
                fit_map_p_1,
                mutate_factor=1 - parent_1.fitness,
            )
            f_g_parent_2, b_g_parent_2 = Mutate.mutate(
                f_g_parent_2,
                b_g_parent_2,
                fit_map_p_2,
                mutate_factor=1 - parent_2.fitness,
            )

            mut_parent_1 = Specimen(
                fingers_genome=f_g_parent_1, brain_genome=b_g_parent_1
            )
            mut_parent_2 = Specimen(
                fingers_genome=f_g_parent_2, brain_genome=b_g_parent_2
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
