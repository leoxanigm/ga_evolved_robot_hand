import math

from specimen import Specimen
from constants import GeneDesc, ROBOT_HAND


class Population:
    def __init__(self, pop_size):
        self.pop_size = pop_size
        self._specimen = [Specimen() for _ in range(pop_size)]
        self.fits: list[Specimen] = None  # Top 30% fittest specimen

        self.crossed = False # Check for crossing
        self.mutated = False # Check for mutation

        self.pop_fitness = None

    def calc_fits(self):
        pop_fitness = [(s.fitness, s) for s in self._specimen]
        pop_fitness = sorted(pop_fitness, key=lambda x: x[0], reverse=True)

        self.pop_fitness = pop_fitness

        # We only take top 30% fittest, and an even number of fit,
        # to get even number we divide by two, round to ceil and multipy
        # by two
        fit_index = math.ceil((len(self._specimen) // 3) / 2) * 2

        self.fits = [s for _, s in pop_fitness][:fit_index]

        self.crossed = False
        self.mutated = False

    def cross(self):
        assert self.fits is not None
        assert len(self.fits) % 2 == 0

        children = []
        for i in range(0, len(self.fits), 2):
            parent_1 = self.fits[i]
            parent_2 = self.fits[i + 1]

            for _ in range(2):  # Two children from the parents
                child_fingers_genome = CrossFingers.cross_genomes(
                    parent_1.fingers_genome, parent_2.fingers_genome
                )
                child_brain_genome = CrossBrain.cross_genomes(
                    parent_1.brain_genome, parent_2.brain_genome
                )

                child_specimen = Specimen(
                    fingers_genome=child_fingers_genome, brain_genome=child_brain_genome
                )

                children.append(child_specimen)

        self.specimen = children

        self.crossed = True

    def mutate(self):
        # Make sure we only mutate after crossing
        assert self.crossed and not self.mutated

        for specimen in self._specimen:
            specimen.fingers_genome = MutateFingers.mutate(specimen.fingers_genome)
            # specimen.brain_genome = MutateBrain.mutate(specimen.brain_genome)

        self.mutated = True

    def repopulate(self):
        # Make sure we only repopulate after crossing and mutation
        assert self.crossed and self.mutated

        # Add new random specimen to repopulate to the population size
        curr_pop_size = len(self.specimen)
        for _ in range(self.pop_size - curr_pop_size):
            self.specimen.append(Specimen())

    @property
    def specimen(self):
        return self._specimen

    @specimen.setter
    def specimen(self, new_specimen):
        self._specimen = new_specimen
