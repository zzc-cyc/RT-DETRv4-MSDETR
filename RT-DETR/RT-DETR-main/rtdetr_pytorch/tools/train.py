"""by lyuwenyu
"""

import os 
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import argparse

import src.misc.dist as dist 
from src.core import YAMLConfig 
from src.solver import TASKS


def build_dataloader_overrides(args):
    overrides = {}
    if args.train_batch_size is not None:
        overrides.setdefault('train_dataloader', {})['batch_size'] = args.train_batch_size
    if args.val_batch_size is not None:
        overrides.setdefault('val_dataloader', {})['batch_size'] = args.val_batch_size
    if args.train_num_workers is not None:
        overrides.setdefault('train_dataloader', {})['num_workers'] = args.train_num_workers
    if args.val_num_workers is not None:
        overrides.setdefault('val_dataloader', {})['num_workers'] = args.val_num_workers
    return overrides


def main(args, ) -> None:
    '''main
    '''
    dist.init_distributed()
    if args.seed is not None:
        dist.set_seed(args.seed)

    assert not all([args.tuning, args.resume]), \
        'Only support from_scrach or resume or tuning at one time'

    cfg = YAMLConfig(
        args.config,
        resume=args.resume, 
        use_amp=args.amp,
        tuning=args.tuning,
        checkpoint_step=args.checkpoint_step,
        checkpoint_name_style=args.checkpoint_name_style,
        **build_dataloader_overrides(args),
    )

    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    
    if args.test_only:
        solver.val()
    else:
        solver.fit()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, )
    parser.add_argument('--resume', '-r', type=str, )
    parser.add_argument('--tuning', '-t', type=str, )
    parser.add_argument('--test-only', action='store_true', default=False,)
    parser.add_argument('--amp', action='store_true', default=False,)
    parser.add_argument('--seed', type=int, help='seed',)
    parser.add_argument('--checkpoint-step', type=int, default=1, help='save an epoch checkpoint every N epochs')
    parser.add_argument(
        '--checkpoint-name-style',
        choices=['official', 'underscore'],
        default='official',
        help='official saves checkpoint0000.pth; underscore saves checkpoint_0000.pth',
    )
    parser.add_argument('--train-batch-size', type=int, default=None, help='override train dataloader batch_size')
    parser.add_argument('--val-batch-size', type=int, default=None, help='override val dataloader batch_size')
    parser.add_argument('--train-num-workers', type=int, default=None, help='override train dataloader num_workers')
    parser.add_argument('--val-num-workers', type=int, default=None, help='override val dataloader num_workers')
    args = parser.parse_args()

    main(args)
