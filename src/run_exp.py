#! /usr/bin/env python
#################################################################################
#     File Name           :     run_exp.py
#     Created By          :     yang
#     Creation Date       :     [2017-02-17 16:24]
#     Last Modified       :     [2017-04-09 14:26]
#     Description         :      
#################################################################################
import multiprocessing
import subprocess
import sys

gpu_ids = [0, 1, 2, 3, 4, 5, 6, 7]
seeds = [85, 152, 1486, 1688, 4312]
games = ['c', 'd']
algorithm = "multi_REINFORCE_stein"
batch_size = 5000
num_of_agents = 16
temperature = 10.0


def worker_function(i):
    subprocess.check_output(i, shell = True)

if __name__=="__main__":
    gpu_index =  0
    commands = []
    for game in games:
        for seed in seeds:
            gpu_id = gpu_ids[gpu_index]
            gpu_index += 1
            if gpu_index >= len(gpu_ids):
                gpu_index = 0
            command = "THEANO_FLAGS='floatX=float32,device=cpu,allow_gc=False' python main_benchmark.py {:} {:} {:} {:} {:} {:} > /dev/null".format(
                algorithm, game, seed, num_of_agents, temperature,batch_size)
           # command = 'python main_benchmark.py {:} {:} {:} 16 10.0>/dev/null'.format(
           #     algorithm, game, seed
           # )
            commands.append(command)
            print(command)
    pool = multiprocessing.Pool(len(commands))
    result = pool.imap(worker_function, commands)
    pool.close()
    pool.join()
