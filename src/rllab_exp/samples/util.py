#! /usr/bin/env python
#################################################################################
#     File Name           :     util.py
#     Created By          :     yang
#     Creation Date       :     [2017-04-06 19:21]
#     Last Modified       :     [2017-04-06 19:23]
#     Description         :      
#################################################################################

from rllab.sampler.stateful_pool import singleton_pool, SharedGlobal
from rllab.misc import logger
from rllab.sampler.parallel_sampler import _worker_populate_task, _get_scoped_G 
import pickle
# my populate things

def populate_task(env, policy, scope=None):
    logger.log("Populating workers...")
    if singleton_pool.n_parallel > 1:
        singleton_pool.run_each(
            _worker_populate_task,
            [(pickle.dumps(env), pickle.dumps(policy), scope)] * singleton_pool.n_parallel
        )
    else:
        # avoid unnecessary copying
        # still some issues when doing multiple copies
        G = _get_scoped_G(singleton_pool.G, scope)
        G.env = pickle.loads(pickle.dumps(env))
        G.policy = pickle.loads(pickle.dumps(policy))
    logger.log("Populated")

