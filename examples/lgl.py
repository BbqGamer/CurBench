import argparse
import os

from curbench.algorithms import LGLTrainer


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
parser.add_argument('--start_rate', type=float, default=0.1)
parser.add_argument('--grow_rate', type=float, default=0.3)
parser.add_argument('--grow_interval', type=int, default=20)
parser.add_argument('--strategy', type=str, default='random')
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


trainer = LGLTrainer(
    data_name=args.data,
    net_name=args.net,
    gpu_index=args.gpu,
    num_epochs=args.epochs,
    random_seed=args.seed,
    start_rate=args.start_rate,
    grow_rate=args.grow_rate,
    grow_interval=args.grow_interval,
    strategy=args.strategy,
)
trainer.fit()
trainer.evaluate()