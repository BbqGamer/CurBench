import os


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
                        cmd = 'python examples/%s.py --data %s%s --net %s --seed %d --epochs %d --gpu %d' % (method, dataset, setting, model, seed, epoch, gpu)
                        print(cmd)
                        os.system(cmd)
                    else:
                        print('Already run: %s' % dir_name)
