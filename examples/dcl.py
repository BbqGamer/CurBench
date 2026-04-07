import argparse
import os

from curbench.algorithms import DCLTrainer


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='cifar10')
parser.add_argument('--net', type=str, default='lenet')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--wandb', action='store_true')
parser.add_argument('--wandb_project', type=str, default='CurBench')
parser.add_argument('--wandb_entity', type=str, default=None)
parser.add_argument('--wandb_mode', type=str, default='online')
parser.add_argument('--wandb_name', type=str, default=None)
parser.add_argument('--wandb_group', type=str, default=None)
parser.add_argument('--init_class_param', type=float, default=0.0)
parser.add_argument('--lr_class_param', type=float, default=0.1)
parser.add_argument('--wd_class_param', type=float, default=0.0)
parser.add_argument('--init_data_param', type=float, default=1.0)
parser.add_argument('--lr_data_param', type=float, default=0.1)
parser.add_argument('--wd_data_param', type=float, default=0.0)
args = parser.parse_args()

if args.wandb:
    os.environ["CURBENCH_WANDB"] = "1"
    os.environ["CURBENCH_WANDB_PROJECT"] = args.wandb_project
    os.environ["CURBENCH_WANDB_MODE"] = args.wandb_mode
    if args.wandb_entity is not None:
        os.environ["CURBENCH_WANDB_ENTITY"] = args.wandb_entity
    if args.wandb_name is not None:
        os.environ["CURBENCH_WANDB_NAME"] = args.wandb_name
    if args.wandb_group is not None:
        os.environ["CURBENCH_WANDB_GROUP"] = args.wandb_group


trainer = DCLTrainer(
    data_name=args.data,
    net_name=args.net,
    gpu_index=args.gpu,
    num_epochs=args.epochs,
    random_seed=args.seed,
    init_class_param=args.init_class_param,
    lr_class_param=args.lr_class_param,
    wd_class_param=args.wd_class_param,
    init_data_param=args.init_data_param,
    lr_data_param=args.lr_data_param,
    wd_data_param=args.wd_data_param,
)
trainer.fit()
trainer.evaluate()