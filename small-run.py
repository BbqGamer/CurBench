import os


## Image
methods = ["base", "spl", "superloss"]
datasets = ['cifar10']
models = ['lenet']
settings = ['', '-noise-0.4', '-imbalance-50']
seeds = [42, 666, 777, 888, 999]
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
