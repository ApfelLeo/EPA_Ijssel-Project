from __future__ import (unicode_literals, print_function, absolute_import,
                        division)


from ema_workbench import (Model, IpyparallelEvaluator, MultiprocessingEvaluator, SequentialEvaluator, Policy,
                           Scenario)

from ema_workbench.em_framework.evaluators import perform_experiments
from ema_workbench.em_framework.samplers import sample_uncertainties
from ema_workbench.util import ema_logging
from problem_formulation import get_model_for_problem_formulation
from ema_workbench.analysis import prim

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#%%
ema_logging.log_to_stderr(ema_logging.INFO)

dike_model, planning_steps = get_model_for_problem_formulation(1)

# singleprocessing
with SequentialEvaluator(dike_model) as evaluator:
    results = evaluator.perform_experiments(scenarios=100, policies=10)


## multiprocessing
#with MultiprocessingEvaluator(dike_model) as evaluator:
#    results = evaluator.perform_experiments(scenarios=100, policies=4)
 
#%%
experiments, outcomes = results

experiments.to_csv('./results/exp_unc_10p_100s.csv')
pd.DataFrame(outcomes).to_csv('./results/out_unc_10p_100s.csv')

classification = outcomes['Expected Number of Deaths']!=0

exp1 = experiments.loc[:,list(dike_model.uncertainties.keys())]

#%%
prim_alg = prim.Prim(exp1.astype(float),classification,threshold=0.7, peel_alpha=0.1)
box1 = prim_alg.find_box()

fig = box1.show_pairs_scatter()

plt.title('Only uncertainties')
plt.show()
fig.savefig('./results/prim_unc_10p_100s.png',dpi=300)

#%% With rotated
rotated_experiments, rotation_matrix = prim.pca_preprocess(exp1.astype(float),classification)

prim_alg = prim.Prim(rotated_experiments,classification,threshold=0.7, peel_alpha=0.1)
box2 = prim_alg.find_box()

fig = box2.show_pairs_scatter()

plt.title('Only uncertainties')
plt.show()
fig.savefig('./results/prim_unc_10p_100s_rot.png',dpi=300)