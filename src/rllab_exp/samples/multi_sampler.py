#! /usr/bin/env python
#################################################################################
#     File Name           :     multi_sampler.py
#     Created By          :     yang
#     Creation Date       :     [2017-03-26 16:19]
#     Last Modified       :     [2017-04-06 19:28]
#     Description         :      
#################################################################################


from rllab.sampler.base import BaseSampler
import rllab.sampler.parallel_sampler as parallel_sampler
from rllab.algos.batch_polopt import BatchPolopt
from rllab.misc import special, tensor_utils, ext
from rllab.core.serializable import Serializable
from rllab.optimizers.first_order_optimizer import FirstOrderOptimizer
from ..optimizer.optimizer import MyFirstOrderOptimizer
from rllab.algos import util
import rllab.misc.logger as logger
import theano
import theano.tensor as TT
import pickle
import numpy as np
from .util import populate_task

class BatchSampler_Multi(BaseSampler):
    def __init__(self, algo, id, with_critic):
        """
        :type algo: BatchPolopt
        """
        self.algo = algo
        self.id = id
        self.with_critic = with_critic

    def start_worker(self):
        populate_task(self.algo.env, self.algo.policy_list[self.id], scope=self.algo.scope)

    def shutdown_worker(self):
        parallel_sampler.terminate_task(scope=self.algo.scope)

    def obtain_samples(self, itr):
        cur_params = self.algo.policy_list[self.id].get_param_values()
        paths = parallel_sampler.sample_paths(
            policy_params=cur_params,
            max_samples=self.algo.batch_size,
            max_path_length=self.algo.max_path_length,
            scope=self.algo.scope,
        )
        if self.algo.whole_paths:
            return paths
        else:
            paths_truncated = parallel_sampler.truncate_paths(paths, self.algo.batch_size)
            return paths_truncated

    def process_samples(self, itr, paths):
        baselines = []
        returns = []
        if self.with_critic: 
            if hasattr(self.algo.baseline_list[self.id], "predict_n"):
                all_path_baselines = self.algo.baseline_list[self.id].predict_n(paths)
            else:
                all_path_baselines = [self.algo.baseline_list[self.id].predict(path) for path in paths]

            for idx, path in enumerate(paths):
                path_baselines = np.append(all_path_baselines[idx], 0)
                deltas = path["rewards"] + \
                         self.algo.discount * path_baselines[1:] - \
                         path_baselines[:-1]
                path["advantages"] = special.discount_cumsum(
                    deltas, self.algo.discount * self.algo.gae_lambda)
                path["returns"] = special.discount_cumsum(path["rewards"], self.algo.discount)
                baselines.append(path_baselines[:-1])
                returns.append(path["returns"])
            ev = special.explained_variance_1d(
                np.concatenate(baselines),
                np.concatenate(returns)
            )
        else:
            for id, path in enumerate(paths):
                path["returns"] = special.discount_cumsum(path["rewards"], self.algo.discount)
                path['advantages'] = path['returns']


        if not self.algo.policy.recurrent:
            observations = tensor_utils.concat_tensor_list([path["observations"] for path in paths])
            actions = tensor_utils.concat_tensor_list([path["actions"] for path in paths])
            rewards = tensor_utils.concat_tensor_list([path["rewards"] for path in paths])
            returns = tensor_utils.concat_tensor_list([path["returns"] for path in paths])
            advantages = tensor_utils.concat_tensor_list([path["advantages"] for path in paths])
            env_infos = tensor_utils.concat_tensor_dict_list([path["env_infos"] for path in paths])
            agent_infos = tensor_utils.concat_tensor_dict_list([path["agent_infos"] for path in paths])

            if self.algo.center_adv:
                advantages = util.center_advantages(advantages)

            if self.algo.positive_adv:
                advantages = util.shift_advantages_to_positive(advantages)

            average_discounted_return = \
                np.mean([path["returns"][0] for path in paths])

            undiscounted_returns = [sum(path["rewards"]) for path in paths]

            ent = np.mean(self.algo.policy.distribution.entropy(agent_infos))

            samples_data = dict(
                observations=observations,
                actions=actions,
                rewards=rewards,
                returns=returns,
                advantages=advantages,
                env_infos=env_infos,
                agent_infos=agent_infos,
                paths=paths,
            )
        else:
            max_path_length = max([len(path["advantages"]) for path in paths])

            # make all paths the same length (pad extra advantages with 0)
            obs = [path["observations"] for path in paths]
            obs = tensor_utils.pad_tensor_n(obs, max_path_length)

            if self.algo.center_adv:
                raw_adv = np.concatenate([path["advantages"] for path in paths])
                adv_mean = np.mean(raw_adv)
                adv_std = np.std(raw_adv) + 1e-8
                adv = [(path["advantages"] - adv_mean) / adv_std for path in paths]
            else:
                adv = [path["advantages"] for path in paths]

            adv = np.asarray([tensor_utils.pad_tensor(a, max_path_length) for a in adv])

            actions = [path["actions"] for path in paths]
            actions = tensor_utils.pad_tensor_n(actions, max_path_length)

            rewards = [path["rewards"] for path in paths]
            rewards = tensor_utils.pad_tensor_n(rewards, max_path_length)

            returns = [path["returns"] for path in paths]
            returns = tensor_utils.pad_tensor_n(returns, max_path_length)

            agent_infos = [path["agent_infos"] for path in paths]
            agent_infos = tensor_utils.stack_tensor_dict_list(
                [tensor_utils.pad_tensor_dict(p, max_path_length) for p in agent_infos]
            )

            env_infos = [path["env_infos"] for path in paths]
            env_infos = tensor_utils.stack_tensor_dict_list(
                [tensor_utils.pad_tensor_dict(p, max_path_length) for p in env_infos]
            )

            valids = [np.ones_like(path["returns"]) for path in paths]
            valids = tensor_utils.pad_tensor_n(valids, max_path_length)

            average_discounted_return = \
                np.mean([path["returns"][0] for path in paths])

            undiscounted_returns = [sum(path["rewards"]) for path in paths]

            ent = np.sum(self.algo.policy.distribution.entropy(agent_infos) * valids) / np.sum(valids)

            samples_data = dict(
                observations=obs,
                actions=actions,
                advantages=adv,
                rewards=rewards,
                returns=returns,
                valids=valids,
                agent_infos=agent_infos,
                env_infos=env_infos,
                paths=paths,
            )
        
        if self.with_critic:
            logger.log("fitting baseline...")
            if hasattr(self.algo.baseline_list[self.id], 'fit_with_samples'):
                self.algo.baseline_list[self.id].fit_with_samples(paths, samples_data)
            else:
                self.algo.baseline_list[self.id].fit(paths)
            logger.log("fitted")
        else:
            pass

        average_return = np.mean(undiscounted_returns)
        logger.record_tabular('Iteration', itr)
        logger.record_tabular('#{:} AverageDiscountedReturn'.format(self.id),
                              average_discounted_return)
        logger.record_tabular('#{:} AverageReturn'.format(self.id), average_return)
        if self.with_critic:
            logger.record_tabular('#{:} ExplainedVariance'.format(self.id), ev)
        logger.record_tabular('#{:} NumTrajs'.format(self.id), len(paths))
        logger.record_tabular('#{:} Entropy'.format(self.id), ent)
        logger.record_tabular('#{:} Perplexity'.format(self.id), np.exp(ent))
        logger.record_tabular('#{:} StdReturn'.format(self.id), np.std(undiscounted_returns))
        logger.record_tabular('#{:} MaxReturn'.format(self.id), np.max(undiscounted_returns))
        logger.record_tabular('#{:} MinReturn'.format(self.id), np.min(undiscounted_returns))

        return samples_data, average_return

