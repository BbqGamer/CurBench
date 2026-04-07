import math
import os


WANDB_ARGS = ' --wandb --wandb_project CurBench --wandb_mode online --wandb_group run-wandb-test'


## Image
methods = ['base', 'spl', 'ttcl', 'mcl', 'screener_net', 'lre', 'mw_net',
           'dcl', 'lgl', 'dds', 'dihcl', 'superloss', 'cbs', 'c2f', 'adaptive_cl']
datasets = ['cifar10', 'cifar100', 'tinyimagenet', 'animal10n']
models = ['lenet', 'resnet18', 'vit']
settings = ['']
seeds = [42]
epoch = 2
gpu = 0
for setting in settings:
    for dataset in datasets:
        for model in models:
            for method in methods:
                for seed in seeds:
                    dir_name = 'runs/%s-%s%s-%s-%d-%d' % (method, dataset, setting, model, epoch, seed)
                    if not os.path.exists(dir_name):
                        cmd = 'python examples/%s.py --data %s%s --net %s --seed %d --epochs %d --gpu %d%s' % (
                            method, dataset, setting, model, seed, epoch, gpu, WANDB_ARGS)
                        print(cmd)
                        os.system(cmd)
                    else:
                        print('Already run: %s' % dir_name)


## Text
methods = ['base', 'spl', 'ttcl', 'mcl', 'screener_net', 'lre', 'mw_net',
           'dcl', 'dds', 'dihcl', 'superloss', 'adaptive_cl']
datasets = ['rte', 'mrpc', 'stsb', 'cola', 'sst2', 'qnli', 'qqp', 'mnli']
models = ['lstm', 'bert', 'gpt']
settings = ['']
seeds = [42]
epoch = 2
gpu = 0
for setting in settings:
    for dataset in datasets:
        for model in models:
            for method in methods:
                for seed in seeds:
                    dir_name = 'runs/%s-%s%s-%s-%d-%d' % (method, dataset, setting, model, epoch, seed)
                    if not os.path.exists(dir_name):
                        cmd = 'python examples/%s.py --data %s%s --net %s --seed %d --epochs %d --gpu %d%s' % (
                            method, dataset, setting, model, seed, epoch, gpu, WANDB_ARGS)
                        if method in ['spl', 'ttcl']:
                            cmd += ' --grow_epochs %d' % (int(math.ceil(epoch / 2)))
                        if method == 'mcl':
                            cmd += ' --warm_epoch 1 --schedule_epoch 1'
                        if method == 'dihcl':
                            cmd += ' --warm_epoch 1'
                        if method == 'adaptive_cl':
                            cmd += ' --inv 1'
                        print(cmd)
                        os.system(cmd)
                    else:
                        print('Already run: %s' % dir_name)


## Graph
methods = ['base', 'spl', 'ttcl', 'mcl', 'screener_net', 'lre', 'mw_net',
           'dcl', 'dds', 'dihcl', 'superloss', 'adaptive_cl']
datasets = ['mutag', 'proteins', 'nci1', 'molhiv']
models = ['gcn', 'gat', 'gin']
settings = ['']
seeds = [42]
epoch = 2
gpu = 0
for setting in settings:
    for dataset in datasets:
        for model in models:
            for method in methods:
                for seed in seeds:
                    dir_name = 'runs/%s-%s%s-%s-%d-%d' % (method, dataset, setting, model, epoch, seed)
                    if not os.path.exists(dir_name):
                        cmd = 'python examples/%s.py --data %s%s --net %s --seed %d --epochs %d --gpu %d%s' % (
                            method, dataset, setting, model, seed, epoch, gpu, WANDB_ARGS)
                        print(cmd)
                        os.system(cmd)
                    else:
                        print('Already run: %s' % dir_name)
