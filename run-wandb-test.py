import math
import os


WANDB_ARGS = ' --wandb --wandb_project CurBench --wandb_mode online --wandb_group run-wandb-test'
SEED = 42
EPOCHS = 20
GPU = 0
SETTING = ''


def method_args(method, epochs):
    args = ''
    if method in ['spl', 'ttcl']:
        args += ' --grow_epochs %d' % (int(math.ceil(epochs / 2)))
    if method == 'mcl':
        args += ' --warm_epoch 1 --schedule_epoch 1'
    if method == 'dihcl':
        args += ' --warm_epoch 1'
    if method == 'adaptive_cl':
        args += ' --inv 1'
    return args


def run_plan(task_name, methods, datasets, model):
    print('\n## %s' % task_name)
    for i, method in enumerate(methods):
        dataset = datasets[i % len(datasets)]
        dir_name = 'runs/%s-%s%s-%s-%d-%d' % (method, dataset, SETTING, model, EPOCHS, SEED)
        if os.path.exists(dir_name):
            print('Already run: %s' % dir_name)
            continue

        cmd = 'python examples/%s.py --data %s%s --net %s --seed %d --epochs %d --gpu %d%s%s' % (
            method,
            dataset,
            SETTING,
            model,
            SEED,
            EPOCHS,
            GPU,
            WANDB_ARGS,
            method_args(method, EPOCHS),
        )
        print(cmd)
        os.system(cmd)


# Text: all methods, all datasets, but only one model and a round-robin pairing.
run_plan(
    'Text',
    ['base', 'spl', 'ttcl', 'mcl', 'screener_net', 'lre', 'mw_net',
     'dcl', 'dds', 'dihcl', 'superloss', 'adaptive_cl'],
    ['rte', 'mrpc', 'stsb', 'cola', 'sst2', 'qnli', 'qqp', 'mnli'],
    'bert',
)


# Graph: all methods, all datasets, but only one model and a round-robin pairing.
run_plan(
    'Graph',
    ['base', 'spl', 'ttcl', 'mcl', 'screener_net', 'lre', 'mw_net',
     'dcl', 'dds', 'dihcl', 'superloss', 'adaptive_cl'],
    ['mutag', 'proteins', 'nci1', 'molhiv'],
    'gcn',
)

# Image: all methods, all datasets, but only one model and a round-robin pairing.
run_plan(
    'Image',
    ['base', 'spl', 'ttcl', 'mcl', 'screener_net', 'lre', 'mw_net',
     'dcl', 'lgl', 'dds', 'dihcl', 'superloss', 'cbs', 'c2f', 'adaptive_cl'],
    ['cifar10', 'cifar100', 'tinyimagenet', 'animal10n'],
    'lenet',
)


