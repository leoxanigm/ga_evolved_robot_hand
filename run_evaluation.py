import sys
import os
import csv
from uuid import uuid4
from operator import add
import time

from population import Population
from simulation import Simulation, ThreadedSim

from helpers.misc_helpers import write_csv

# Current time for the name of log file
t = time.localtime()
t_log = f'{t.tm_mon}{t.tm_mday}{t.tm_hour}{t.tm_min}'

# If there's an error while running evaluation, we should clean
# files created for that iteration. These two, will keep track.
written_files = []
run_success = False

# Log file
write_csv(f'log/{t_log}_log.csv', ['iteration_id', 'total_run_time(s)', 'pop_count'])
written_files.append(f'log/{t_log}_log.csv')

try:
    for j in range(50):  # run 50 evaluations
        generation_count = 20  # for 50 generations
        population_count = 100  # of 100 specimen each

        run_id = str(uuid4())[:8]

        write_csv(
            f'fit_specimen/{run_id}.csv',
            ['generation_id', 'generation_index', 'top_fitness', 'saved_specimen'],
        )
        written_files.append(f'fit_specimen/{run_id}.csv')

        start_time = time.time()

        population = Population(population_count)
        simulation = ThreadedSim(pool_size=10)

        for i in range(generation_count):
            print('==============')
            print(f'Evaluating generation {i} at {time.ctime()[11:19]} ...')
            print('==============')

            generation_id = str(uuid4())[:8]

            simulation.run_population(population)

            population.calc_fits()
            population.cross_mutate()
            population.repopulate()

            # We just want the fitness information
            # Where population.fittest is a list of (fitness, specimen)
            fittest = population.fittest

            # Save the fittest specimen of this generation
            saved_specimen_id = ''
            if fittest[0].f > 0.85:
                saved_specimen_id = fittest[0].s.save_specimen()

            print(f'Max fitness for generation {i} = {fittest[0].f}')

            # Write log for this generation
            write_csv(
                f'fit_specimen/{run_id}.csv',
                [generation_id, i, fittest[0].f, saved_specimen_id],
            )

        # Write log for current iteration
        write_csv(
            f'log/{t_log}_log.csv',
            [
                time.ctime(),
                time.time() - start_time,
                generation_count * population_count,
            ],
        )

    run_success = True

except Exception as e:
    # Clear log files
    if not run_success:
        for file_path in written_files:
            if os.path.isfile(file_path):
                os.remove(file_path)

    raise e
