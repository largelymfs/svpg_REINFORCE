#! /usr/bin/env python
#################################################################################
#     File Name           :     main.py
#     Created By          :     yang
#     Creation Date       :     [2017-03-02 11:03]
#     Last Modified       :     [2017-04-08 16:38]
#     Last Modified       :     [2017-04-08 16:38]
#     Description         :      
#################################################################################

import argparse

# algorithms
from rllab.algos.vpg import VPG
from util import get_date

## rllab
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.algos.ddpg import DDPG
from rllab.envs.box2d.cartpole_swingup_env import CartpoleSwingupEnv
from rllab.envs.box2d.cartpole_env import CartpoleEnv 
from rllab.envs.box2d.double_pendulum_env import DoublePendulumEnv
from rllab.envs.box2d.car_parking_env import CarParkingEnv
from rllab.envs.box2d.mountain_car_env import MountainCarEnv

from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.exploration_strategies.ou_strategy import OUStrategy
from rllab.policies.deterministic_mlp_policy import DeterministicMLPPolicy
from rllab.q_functions.continuous_mlp_q_function import ContinuousMLPQFunction
from rllab.misc.instrument import run_experiment_lite
from rllab_exp.algos.vpg_multi import VPG_multi
from rllab_exp.algos.vpg_multi_stein import  VPG_multi_Stein
from rllab_exp.samples.sampler_no_critic import BatchSampler_no_critic

# main functions

prefix_map = {
    'c': 'cartpole_swing_up',
    'd': 'double_pendulum_env',
    'car': 'cartpole',
    'mou': 'mountain_car',
    'swim' : 'swim',
    'ant'  : 'ant',
    'hopper' : 'hopper'
}

env_map = {
    'c': CartpoleSwingupEnv,
    'd': DoublePendulumEnv,
    'car': CartpoleEnv,
    'mou': MountainCarEnv,
#    'swim' : SwimmerEnv,
#    'ant' : AntEnv,
#    'hopper' : HopperEnv
}

n_epochs_map = {
    'c': 1000,
    'd': 1000,
    'car': 100,
    'mou': 100,
    'swim' : 500,
    'ant' : 500,
    'hopper' : 500,
}

learning_rate_map = {
        'c': 5e-3,
        'd': 5e-3,
        'car': 5e-3,
        'mou': 5e-3,
        'swim':0.01,
        'ant':5e-3,
        'hopper' : 5e-3
        }

def run_vpg_baseline_large_batch_size_no_critic(*_):
    env = normalize(env_name())
    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(100, 50, 25,),
        adaptive_std=False,
    )
    baseline = LinearFeatureBaseline(env_spec=env.spec)
    print("Iteration Number: {:}".format(n_itr))
    print("Learning Rate : {:}".format(learning_rate))
    algo = VPG(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=batch_size* num_of_agents,
        max_path_length=500,
        n_itr=n_itr,
        discount=0.99,
        optimizer_args = {'learning_rate':learning_rate},
        sampler_cls = BatchSampler_no_critic,
    )
    algo.train()

def run_vpg_baseline_long_training_no_critic(*_):
    env = normalize(env_name())
    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(100, 50, 25,),
        adaptive_std=False,
    )
    baseline = LinearFeatureBaseline(env_spec=env.spec)
    print("Iteration Number: {:}".format(n_itr))
    print("Learning Rate : {:}".format(learning_rate))
    algo = VPG(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=batch_size,
        max_path_length=500,
        n_itr=n_itr * num_of_agents,
        discount=0.99,
        optimizer_args = {'learning_rate':learning_rate},
        sampler_cls = BatchSampler_no_critic,
    )
    algo.train()

def run_multi_vpg_baseline_no_critic(*_):
    env = normalize(env_name())
    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(100, 50, 25),
        adaptive_std=False,
    )
    policy_list = [GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(100, 50, 25,),
        adaptive_std=False,
    ) for i in range(num_of_agents)]
    baseline = LinearFeatureBaseline(env_spec=env.spec)
    baseline_list = [LinearFeatureBaseline(env_spec=env.spec) for i in range(num_of_agents)]
    print("Iteration Number: {:}".format(n_itr))
    print("Learning Rate : {:}".format(learning_rate))
    algo = VPG_multi(
        num_of_agents = num_of_agents,
        env=env,
        policy=policy,
        policy_list=policy_list,
        baseline=baseline,
        baseline_list=baseline_list,
        batch_size=batch_size,
        max_path_length=500,
        n_itr=n_itr,
        discount=0.99,
        optimizer_args = {'learning_rate':learning_rate},
        with_critic = False,
    )
    algo.train()

def run_multi_vpg_stein_no_critic(*_):
    env = normalize(env_name())
    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(100, 50, 25,),
        adaptive_std=False,
    )
    policy_list = [GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(100, 50, 25,),
        adaptive_std=False,
    ) for i in range(num_of_agents)]
    baseline = LinearFeatureBaseline(env_spec=env.spec)
    baseline_list = [LinearFeatureBaseline(env_spec=env.spec) for i in range(num_of_agents)]
    print("Iteration Number: {:}".format(n_itr))
    print("Learning Rate : {:}".format(learning_rate))
    algo = VPG_multi_Stein(
        num_of_agents = num_of_agents,
        temp = temperature,
        env=env,
        policy=policy,
        policy_list = policy_list,
        baseline=baseline,
        baseline_list=baseline_list,
        batch_size=batch_size,
        max_path_length=500,
        n_itr=n_itr,
        discount=0.99,
        learning_rate = learning_rate,
        with_critic = False,
    )
    algo.train()

function_map = {
    "REINFORCE_baseline_large_batch_size": run_vpg_baseline_large_batch_size_no_critic,
    "REINFORCE_baseline_long_training": run_vpg_baseline_long_training_no_critic,
    "multi_REINFORCE_baseline": run_multi_vpg_baseline_no_critic,
    "multi_REINFORCE_stein": run_multi_vpg_stein_no_critic,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multiple Agents with Actor-Critic Algorithm.')
    parser.add_argument("algo", type=str,
                        help='available algorithms')
    parser.add_argument("env_name", type=str,
                        help='available env_name:')
    parser.add_argument("random_seed", type=int)
    parser.add_argument("num_of_agents", type=int)
    parser.add_argument("temperature", type=float)
    parser.add_argument("batch_size", type=int, default = 5000)

    args = parser.parse_args()
    env_name = env_map[args.env_name]
    prefix = prefix_map[args.env_name]
    n_epochs = n_epochs_map[args.env_name]
    random_seed = int(args.random_seed)
    run_function = function_map[args.algo]
    n_itr = n_epochs_map[args.env_name]
    num_of_agents = int(args.num_of_agents)
    temperature = float(args.temperature)
    learning_rate = learning_rate_map[args.env_name]
    batch_size = int(args.batch_size) 

    if args.algo == "multi_REINFORCE_stein" or args.algo == "multi_REINFORCE_stein_anneal" or args.algo == 'multi_REINOFRCE_stein_reg' or args.algo == "multi_REINFORCE_stein_no_critic" or args.algo == 'multi_REINFORCE_baseline_no_critic' or args.algo == 'multi_REINFORCE_stein_evolution':
        args.algo = "{:}#{:}_temp={:}".format(args.algo, num_of_agents, args.temperature)

    run_experiment_lite(
        run_function,
        n_parallel=4,
        snapshot_mode="last",
        seed=random_seed,
        log_dir="./../exp_log/{:}_seed={:}_iter=500_env={:}_{:}".format(args.algo, random_seed, prefix, get_date()),
    )
