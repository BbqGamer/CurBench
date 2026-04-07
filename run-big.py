import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--wandb', action='store_true')
parser.add_argument('--wandb_project', type=str, default='CurBench')
parser.add_argument('--wandb_entity', type=str, default=None)
parser.add_argument('--wandb_mode', type=str, default='online')
parser.add_argument('--wandb_name', type=str, default=None)
parser.add_argument('--wandb_group', type=str, default=None)
args = parser.parse_args()

wandb_env = ''
if args.wandb:
    wandb_env = f'CURBENCH_WANDB=1 CURBENCH_WANDB_PROJECT={args.wandb_project} CURBENCH_WANDB_MODE={args.wandb_mode}'
    if args.wandb_entity is not None:
        wandb_env += f' CURBENCH_WANDB_ENTITY={args.wandb_entity}'
    if args.wandb_name is not None:
        wandb_env += f' CURBENCH_WANDB_NAME={args.wandb_name}'
    if args.wandb_group is not None:
        wandb_env += f' CURBENCH_WANDB_GROUP={args.wandb_group}'

## Image
methods = ['base']
# datasets = ['cifar100', 'tinyimagenet', 'animal10n']
datasets = ['cifar100']
# models = ['lenet', 'resnet18', 'vit']
models = ['resnet18', 'vit']
settings = ['', '-noise-0.4', '-imbalance-50']
seeds = [42]
epoch = 200
gpu = 0
for setting in settings:
    for dataset in datasets:
        for model in models:
            for method in methods:
                for seed in seeds:
                    dir_name = 'runs/%s-%s%s-%s-%d-%d' % (method, dataset, setting, model, epoch, seed)
                    if not os.path.exists(dir_name):
                        cmd = ' '.join(filter(None, [wandb_env, 'python examples/%s.py --data %s%s --net %s --seed %d --epochs %d --gpu %d' % (method, dataset, setting, model, seed, epoch, gpu)]))
                        print(cmd)
                        os.system(cmd)
                    else:
                        print('Already run: %s' % dir_name)
