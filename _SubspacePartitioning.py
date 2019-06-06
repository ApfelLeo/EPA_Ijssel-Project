#!/usr/bin/env python
# coding: utf-8

# In[6]:


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


# In[7]:


run = True



# In[8]:
    # no dike increase, no warning, none of the rfr
zero_policy = {'DaysToThreat': 0}
zero_policy.update({'DikeIncrease {}'.format(n): 0 for n in planning_steps})
zero_policy.update({'RfR {}'.format(n): 0 for n in planning_steps})
pol0 = {}

for key in dike_model.levers:
    s1, s2 = key.name.split('_')
    pol0.update({key.name: zero_policy[s2]})

policy0 = Policy('Policy 0', **pol0)

if run:
    ema_logging.log_to_stderr(ema_logging.INFO)
    dike_model, planning_steps = get_model_for_problem_formulation(1)
    
    if __name__ == '__main__':
        with SequentialEvaluator(dike_model) as evaluator:
            results_sens = evaluator.perform_experiments(scenarios=1000) #,policies=Policy(**{'A.1 DikeIncrease 0': 0})


# In[ ]:




