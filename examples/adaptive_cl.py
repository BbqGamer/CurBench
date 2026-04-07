import argparse
import os

from curbench.algorithms import BaseTrainer, AdaptiveCLTrainer


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
parser.add_argument('--pace_p', type=float, default=0.1)
parser.add_argument('--pace_q', type=float, default=2.5)
parser.add_argument('--pace_r', type=int, default=15)
parser.add_argument('--inv', type=int, default=20)
parser.add_argument('--alpha', type=float, default=0.7)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--gamma_decay', type=float, default=None)
parser.add_argument('--bottom_gamma', type=float, default=0.1)
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


trainer = AdaptiveCLTrainer(
    data_name=args.data,
    net_name=args.net,
    gpu_index=args.gpu,
    num_epochs=args.epochs,
    random_seed=args.seed,
    pace_p=args.pace_p,
    pace_q=args.pace_q,
    pace_r=args.pace_r,
    inv=args.inv,
    alpha=args.alpha,
    gamma=args.gamma,
    gamma_decay=args.gamma_decay,
    bottom_gamma=args.bottom_gamma,
    pretrained_net=teacher_net,
)
trainer.fit()
trainer.evaluate()