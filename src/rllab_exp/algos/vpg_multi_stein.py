#! /usr/bin/env python
#################################################################################
#     File Name           :     vpg_multi.py
#     Created By          :     yang
#     Creation Date       :     [2017-03-18 19:00]
#     Last Modified       :     [2017-04-06 19:17]
#     Description         :      
#################################################################################

from rllab.sampler.base import BaseSampler
import rllab.sampler.parallel_sampler as parallel_sampler
from rllab.algos.batch_polopt import BatchPolopt
from rllab.misc import special, tensor_utils, ext
from rllab.core.serializable import Serializable
from ..optimizer.optimizer import MyFirstOrderOptimizer
from ..samples.multi_sampler import BatchSampler_Multi
from rllab.algos import util
import rllab.misc.logger as logger
import theano
import theano.tensor as TT
import pickle
import numpy as np
import sys


class VPG_multi_Stein(BatchPolopt, Serializable):
    def __init__(
            self,
            num_of_agents,
            temp,
            env,
            policy,
            baseline,
            policy_list,
            baseline_list,
            anneal_temp_start = 500,
            anneal_temp=False,
            anneal_method = 'linear',
            anneal_discount_epoch=1,
            anneal_discount_factor=0.02,
            temp_min = 1e-2,
            optimizer=None,
            optimizer_args=None,
            learning_rate = 1e-3,
            optimization_method = "adam",
            adaptive_kernel = False,
            policy_weight_decay = 0.0,
            with_critic = True,
            include_kernel = True,
            evolution = False,
            evolution_ratio = 0.25,
            evolution_epsilon = 0.01,
            evolution_update_steps = 20,
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
            optimizer = MyFirstOrderOptimizer(**optimizer_args)
        self.optimizer = optimizer
        self.opt_info = None
        self.temp = temp
        self.anneal_temp = anneal_temp
        self.anneal_method = anneal_method
        self.anneal_discount_epoch = anneal_discount_epoch
        self.anneal_discount_factor = anneal_discount_factor
        self.temp_min = temp_min
        self.anneal_temp_start = anneal_temp_start
        self.num_of_agents = num_of_agents
        self.sampler_list = [BatchSampler_Multi(self, i, with_critic) for i in range(self.num_of_agents)]
        self.optimizer_list = [pickle.loads(pickle.dumps(self.optimizer)) for _ in range(self.num_of_agents)]
        super(VPG_multi_Stein, self).__init__(env=env, policy=policy, baseline=baseline, **kwargs)
        self.policy_list = policy_list
        self.baseline_list = baseline_list
        self.stein_learning_rate = learning_rate
        self.stein_optimization_method = optimization_method
        self.adaptive_kernel = adaptive_kernel
        self.search_space = np.linspace(0.1, 2.0, num=20)
        self.policy_weight_decay = policy_weight_decay
        self.include_kernel = include_kernel
        self.evolution = evolution
        self.evolution_ratio = evolution_ratio
        self.evolution_epsilon = evolution_epsilon
        self.evolution_update_steps = evolution_update_steps

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
            if self.anneal_temp and (itr + 1) % self.anneal_discount_epoch == 0 and itr >= self.anneal_temp_start:
                if self.anneal_method == 'loglinear':
                    self.temp *= self.anneal_discount_factor
                elif self.anneal_method == 'linear':
                    self.temp -= self.anneal_discount_factor
                if self.temp < self.temp_min:
                    self.temp = self.temp_min
                logger.log("Current Temperature {:}".format(self.temp))
            with logger.prefix('itr #%d | ' % itr):
                average_return_list = []
                gradient_list = []
                for i in range(self.num_of_agents):
                    paths = self.sampler_list[i].obtain_samples(itr)
                    samples_data, average_return = self.sampler_list[i].process_samples(itr, paths)
                    average_return_list.append(average_return)
                    gradient = self.optimize_policy(itr, samples_data, i)
                    gradient_list.append(gradient)
                logger.log("Update Policy {BEGIN}")
                self.update_policies(gradient_list)
                logger.log("Update Policy {END}")
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
            if self.evolution and (itr + 1) % self.evolution_update_steps == 0:
                logger.log(">>>>>>>>>>>>>>>>>>>>>>> Evolution START <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
                num_of_update = int(self.evolution_ratio * self.num_of_agents)
                sorted_id = np.argsort(average_return_list)
                deleted_id = sorted_id[:num_of_update]
                sampled_id = sorted_id[num_of_update:]
                for i in range(len(deleted_id)):
                    current_id = np.random.choice(sampled_id, 1)
                    current_params = self.policy_list[current_id].get_param_values()
                    current_epsilon = self.evolution_epsilon * (np.random.random(current_params.shape) - 0.5)
                    self.policy_list[deleted_id[i]].set_param_values(current_params + current_epsilon)
                logger.log(">>>>>>>>>>>>>>>>>>>>>>> Evolution FINISH <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

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
        return self.optimizer_list[id].gradient

    def update_policies(self, gradient_list):
        gradient = -np.array(gradient_list)
        params = np.array([self.policy_list[i].get_param_values() for i in range(self.num_of_agents)])
        ## get distance matrix
        distance_matrix = np.sum(np.square(params[None, :, :] - params[:, None, :]), axis=-1)
        # get median
        distance_vector = distance_matrix.flatten()
        distance_vector.sort()
        median = 0.5 * (
        distance_vector[int(len(distance_vector) / 2)] + distance_vector[int(len(distance_vector) / 2) - 1])
        h = median / (2 * np.log(self.num_of_agents + 1))
        if self.adaptive_kernel:
            L_min = None
            alpha_best = None
            for alpha in self.search_space:
                kernel_alpha = np.exp(distance_matrix * (-alpha / h))
                mean_kernel = np.sum(kernel_alpha, axis = 1)
                L = np.mean(np.square(mean_kernel - 2.0 * np.ones_like(mean_kernel)))
                logger.log("Current Loss {:} and Alpha : {:}".format(L, alpha))
                if L_min is None:
                    L_min = L
                    alpha_best = alpha
                elif L_min > L:
                    L_min = L
                    alpha_best = alpha
            logger.record_tabular('Best Alpha', alpha_best)
            h =  h / alpha_best
        
        kernel = np.exp(distance_matrix[:, :] * (-1.0 / h))
        kernel_gradient = kernel[:, :, None] * (2.0 / h) * (params[None, :, :] - params[:, None, :])
        if self.include_kernel:
            weights = (1.0 / self.temp) * kernel[:, :, None] * gradient[:, None, :] + kernel_gradient[:, :, :]
        else:
            weights = kernel[:, :, None] * gradient[:, None, :]

        weights = -np.mean(weights[:, :, :], axis=0)
        # adam update
        if self.stein_optimization_method == 'adam':
            if self.stein_m is None:
                self.stein_m = np.zeros_like(params)
            if self.stein_v is None:
                self.stein_v = np.zeros_like(params)
            self.stein_t += 1.0
            self.stein_m = self.stein_beta1 * self.stein_m + (1.0 - self.stein_beta1) * weights
            self.stein_v = self.stein_beta2 * self.stein_v + (1.0 - self.stein_beta2) * np.square(weights)
            m_hat = self.stein_m / (1.0 - self.stein_beta1 ** self.stein_t)
            v_hat = self.stein_v / (1.0 - self.stein_beta2 ** self.stein_t)
            params = params - self.stein_learning_rate * (m_hat / (np.sqrt(v_hat) + self.stein_epsilon))
        elif self.stein_optimization_method == 'adagrad':
            if self.stein_m is None:
                self.stein_m = np.zeros_like(params)
            self.stein_m = self.stein_m + np.square(weights)
            params = params - self.stein_learning_rate * (weights / (np.sqrt(self.stein_m + self.stein_epsilon)))
            

        for i in range(self.num_of_agents):
            self.policy_list[i].set_param_values(params[i, :])
        logger.record_tabular('Median', median)
        logger.record_tabular('KGradient_Max', np.max(kernel_gradient.flatten()))
        logger.record_tabular('Kernal_Max', np.max(kernel.flatten()))
        logger.record_tabular('PolicyGradient_Max', np.max(gradient.flatten()))

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
                surr_obj_raw = - TT.mean(logli_list[id] * advantage_var)
                policy_weight_decay_term = 0.5 * self.policy_weight_decay * sum([TT.sum(TT.square(param)) for param in self.policy_list[id].get_params(regularizable=True)])
                surr_obj = surr_obj_raw + policy_weight_decay_term
                mean_kl = TT.sum(kl_list[id] * valid_var) / TT.sum(valid_var)
                max_kl = TT.max(kl_list[id] * valid_var)
                mean_kl_list.append(mean_kl)
                max_kl_list.append(max_kl)
                surr_obj_list.append(surr_obj)
        else:
            for id in range(self.num_of_agents):
                surr_obj_raw = - TT.mean(logli_list[id] * advantage_var)
                policy_weight_decay_term = 0.5 * self.policy_weight_decay * sum([TT.sum(TT.square(param)) for param in self.policy_list[id].get_params(regularizable=True)])
                surr_obj = surr_obj_raw + policy_weight_decay_term
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

        self.stein_m = None
        self.stein_v = None
        self.stein_epsilon = 1e-8
        self.stein_beta1 = 0.9
        self.stein_beta2 = 0.999
        self.stein_t = 0

