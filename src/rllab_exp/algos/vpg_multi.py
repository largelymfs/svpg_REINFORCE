#! /usr/bin/env python
#################################################################################
#     File Name           :     vpg_multi.py
#     Created By          :     yang
#     Creation Date       :     [2017-03-18 19:00]
#     Last Modified       :     [2017-03-26 17:06]
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

from  ..samples.multi_sampler import BatchSampler_Multi


class VPG_multi(BatchPolopt, Serializable):
    def __init__(
            self,
            num_of_agents,
            env,
            policy,
            policy_list,
            baseline,
            baseline_list,
            optimizer=None,
            optimizer_args=None,
            with_critic = True,
            **kwargs):
        Serializable.quick_init(self, locals())
        if optimizer is None:
            default_args = dict(
                batch_size=None,
                max_epochs=1,
            )
            if optimizer_args is None:
                optimizer_args = default_args
            else:
                optimizer_args = dict(default_args, **optimizer_args)
            optimizer = FirstOrderOptimizer(**optimizer_args)
            #optimizer = MyFirstOrderOptimizer(**optimizer_args)
        self.optimizer = optimizer
        self.opt_info = None
        self.num_of_agents = num_of_agents
        self.sampler_list = [BatchSampler_Multi(self, i, with_critic) for i in range(self.num_of_agents)]
        self.optimizer_list = [pickle.loads(pickle.dumps(self.optimizer)) for _ in range(self.num_of_agents)]
        super(VPG_multi, self).__init__(env=env, policy=policy, baseline=baseline, **kwargs)
        self.policy_list = policy_list
        self.baseline_list = baseline_list

    def start_worker(self):
        for i in range(self.num_of_agents):
            self.sampler_list[i].start_worker()

    def shutdown_worker(self):
        for i in range(self.num_of_agents):
            self.sampler_list[i].shutdown_worker()

    def train(self):
        self.start_worker()
        self.init_opt()
        for itr in range(self.current_itr, self.n_itr):
            with logger.prefix('itr #%d | ' % itr):
                average_return_list = []
                for i in range(self.num_of_agents):
                    paths = self.sampler_list[i].obtain_samples(itr)
                    samples_data, average_return = self.sampler_list[i].process_samples(itr, paths)
                    average_return_list.append(average_return)
                    # self.log_diagnostics(paths)
                    self.optimize_policy(itr, samples_data, i)
                logger.record_tabular('AverageReturn', np.max(average_return_list))
                logger.log("saving snapshot...")
                params = self.get_itr_snapshot(itr)
                self.current_itr = itr + 1
                params["algo"] = self
                if self.store_paths:
                    pass
                logger.save_itr_params(itr, params)
                logger.log("saved")
                logger.dump_tabular(with_prefix=False)

        self.shutdown_worker()

    def get_itr_snapshot(self, itr):
        return dict(
            itr=itr,
            policy_list=self.policy_list,
            baseline_list=self.baseline_list,
            env=self.env,
        )

    def optimize_policy(self, itr, samples_data, id):
        logger.log("optimizing policy")
        inputs = ext.extract(
            samples_data,
            "observations", "actions", "advantages"
        )
        agent_infos = samples_data["agent_infos"]
        state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]
        inputs += tuple(state_info_list)
        if self.policy.recurrent:
            inputs += (samples_data["valids"],)
        dist_info_list = [agent_infos[k] for k in self.policy.distribution.dist_info_keys]
        loss_before = self.optimizer_list[id].loss(inputs)
        self.optimizer_list[id].optimize(inputs)

        loss_after = self.optimizer_list[id].loss(inputs)
        logger.record_tabular("#{:} LossBefore".format(id), loss_before)
        logger.record_tabular("#{:} LossAfter".format(id), loss_after)

        mean_kl, max_kl = self.opt_info['f_kl_list'][id](*(list(inputs) + dist_info_list))
        logger.record_tabular('#{:} MeanKL'.format(id), mean_kl)
        logger.record_tabular('#{:} MaxKL'.format(id), max_kl)

    def update_policies(self, gradient_list):
        ## naive way do sgd with a learning rate
        for i in range(self.num_of_agents):
            param = self.policy_list[i].get_param_values()
            self.policy_list[i].set_param_values(param - 1e-3 * gradient_list[i])

    def init_opt(self):
        is_recurrent = int(self.policy.recurrent)

        obs_var = self.env.observation_space.new_tensor_variable(
            'obs',
            extra_dims=1 + is_recurrent,
        )
        action_var = self.env.action_space.new_tensor_variable(
            'action',
            extra_dims=1 + is_recurrent,
        )
        advantage_var = ext.new_tensor(
            'advantage',
            ndim=1 + is_recurrent,
            dtype=theano.config.floatX
        )
        dist = self.policy.distribution
        old_dist_info_vars = {
            k: ext.new_tensor(
                'old_%s' % k,
                ndim=2 + is_recurrent,
                dtype=theano.config.floatX
            ) for k in dist.dist_info_keys
            }
        old_dist_info_vars_list = [old_dist_info_vars[k] for k in dist.dist_info_keys]

        if is_recurrent:
            valid_var = TT.matrix('valid')
        else:
            valid_var = None

        state_info_vars = {
            k: ext.new_tensor(
                k,
                ndim=2 + is_recurrent,
                dtype=theano.config.floatX
            ) for k in self.policy.state_info_keys
            }
        state_info_vars_list = [state_info_vars[k] for k in self.policy.state_info_keys]

        ## different policies should have different loss
        logli_list = []
        dist_info_vars_list = []
        kl_list = []
        for id in range(self.num_of_agents):
            dist_info_vars = self.policy_list[id].dist_info_sym(obs_var, state_info_vars)
            logli = dist.log_likelihood_sym(action_var, dist_info_vars)
            kl = dist.kl_sym(old_dist_info_vars, dist_info_vars)
            logli_list.append(logli)
            dist_info_vars_list.append(dist_info_vars)
            kl_list.append(kl)

        # formulate as a minimization problem
        # The gradient of the surrogate objective is the policy gradient
        mean_kl_list = []
        max_kl_list = []
        surr_obj_list = []

        if is_recurrent:
            for id in range(self.num_of_agents):
                surr_obj = - TT.sum(logli_list[id] * advantage_var * valid_var) / TT.sum(valid_var)
                mean_kl = TT.sum(kl_list[id] * valid_var) / TT.sum(valid_var)
                max_kl = TT.max(kl_list[id] * valid_var)
                mean_kl_list.append(mean_kl)
                max_kl_list.append(max_kl)
                surr_obj_list.append(surr_obj)
        else:
            for id in range(self.num_of_agents):
                surr_obj = - TT.mean(logli_list[id] * advantage_var)
                mean_kl = TT.mean(kl_list[id])
                max_kl = TT.max(kl_list[id])
                mean_kl_list.append(mean_kl)
                max_kl_list.append(max_kl)
                surr_obj_list.append(surr_obj)

        input_list = [obs_var, action_var, advantage_var] + state_info_vars_list
        if is_recurrent:
            input_list.append(valid_var)

        for id in range(self.num_of_agents):
            self.optimizer_list[id].update_opt(surr_obj_list[id], target=self.policy_list[id], inputs=input_list)

        f_kl_list = []
        for id in range(self.num_of_agents):
            f_kl = ext.compile_function(
                inputs=input_list + old_dist_info_vars_list,
                outputs=[mean_kl_list[id], max_kl_list[id]],
            )
            f_kl_list.append(f_kl)

        self.opt_info = dict(
            f_kl_list=f_kl_list,
        )
