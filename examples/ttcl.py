import argparse
import os

from curbench.algorithms import BaseTrainer, TTCLTrainer


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
parser.add_argument('--start_rate', type=float, default=0.0)
parser.add_argument('--grow_epochs', type=int, default=100)
parser.add_argument('--grow_fn', type=str, default='linear')
parser.add_argument('--weight_fn', type=str, default='hard')
parser.add_argument('--teacher_dir', type=str, default=None)
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


pretrainer = BaseTrainer(
    data_name=args.data,
    net_name=args.net,
    gpu_index=args.gpu,
    num_epochs=args.epochs,
    random_seed=args.seed,
)
# if args.teacher_dir is None:
#     pretrainer.fit()
args.teacher_dir = 'runs/base-%s-%s-%d-%d' % (args.data.split('-')[0], args.net, args.epochs, 42)
# pretrainer.evaluate(args.teacher_dir)
teacher_net = pretrainer.export(args.teacher_dir)


trainer = TTCLTrainer(
    data_name=args.data,
    net_name=args.net,
    gpu_index=args.gpu,
    num_epochs=args.epochs,
    random_seed=args.seed,
    start_rate=args.start_rate,
    grow_epochs=args.grow_epochs,
    grow_fn=args.grow_fn,
    weight_fn=args.weight_fn,
    teacher_net=teacher_net,
)
trainer.fit()
trainer.evaluate()