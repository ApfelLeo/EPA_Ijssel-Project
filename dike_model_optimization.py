from __future__ import (unicode_literals, print_function, absolute_import,
                        division)


from ema_workbench import (Model, MultiprocessingEvaluator,
                           ScalarOutcome, IntegerParameter, optimize, Scenario)
from ema_workbench.em_framework.optimization import EpsilonProgress, HyperVolume
from ema_workbench.util import ema_logging

from problem_formulation import get_model_for_problem_formulation
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == '__main__':
    ema_logging.log_to_stderr(ema_logging.INFO)

    dike_model, planning_steps = get_model_for_problem_formulation(1)

    #reference_values = {'Bmax': 175, 'Brate': 1.5, 'pfail': 0.5,
    #                    'discount rate': 3.5,
    #                    'ID flood wave shape': 4}
    reference_values = {'Bmax': 175, 'Brate': 1.5, 'pfail': 0.5,
                        'ID flood wave shape': 4, 'planning steps': 2}
    reference_values.update({'discount rate {}'.format(n): 3.5 for n in planning_steps})
    scen1 = {}

    for key in dike_model.uncertainties:
        name_split = key.name.split('_')

        if len(name_split) == 1:
            scen1.update({key.name: reference_values[key.name]})

        else:
            scen1.update({key.name: reference_values[name_split[1]]})

    ref_scenario = Scenario('reference', **scen1)

    convergence_metrics = [EpsilonProgress()]

    espilon = [10] * len(dike_model.outcomes) #why?

    nfe = 10000

# OPTIMIZATION:
#    results, convergence = optimize(dike_model, nfe=nfe, searchover='levers',
#                                    epsilons=espilon, convergence = convergence_metrics,
#                                    reference = ref_scenario)
#
    with MultiprocessingEvaluator(dike_model) as evaluator:
        results, convergence = evaluator.optimize(nfe=nfe,
                                                  searchover='levers',
                                                  epsilons=espilon,
                                                  convergence=convergence_metrics,
                                                  reference=ref_scenario
                                                  )

    fig, (ax1, ax2) = plt.subplots(ncols=2, sharex=True)
    fig, ax1 = plt.subplots(ncols=1)
    ax1.plot(convergence.epsilon_progress)
    ax1.set_xlabel('nr. of generations')
    ax1.set_ylabel('$\epsilon$ progress')
#    ax2.plot(convergence.hypervolume)
#    ax2.set_ylabel('hypervolume')
    sns.despine()