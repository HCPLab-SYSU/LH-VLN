import argparse
import random
import numpy as np
import torch
import os
import datetime
import yaml
from easydict import EasyDict
from .distributed import world_info_from_env, init_distributed_device
from .common_utils import create_logger, log_config_to_file
from pathlib import Path

root_path = os.getcwd()

if not os.path.exists(root_path + '/output'):    # for LH-VLN task
    os.makedirs(root_path + '/output')
if not os.path.exists(root_path + '/run'):    # for step-by-step task
    os.makedirs(root_path + '/run')
if not os.path.exists(root_path + '/save_checkpoints'):    # for logs
    os.makedirs(root_path + '/save_checkpoints')

def random_seed(seed=0, rank=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def read_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--base_dir', type=str, default=root_path, help="base path")
    parser.add_argument('--data_dir', type=str, default=root_path + '/data', help="dataset root path")
    parser.add_argument('--cfg_file', type=str, default=root_path + '/configs/lh_vln.yaml', help='dataset configs')

    # local fusion
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument("--num_epochs", type=int, default=30)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", default=3e-5, type=float)
    parser.add_argument("--gradient_accumulation_step", type=int, default=2)
    parser.add_argument(
        "--precision",
        choices=["amp_bf16", "amp_bfloat16", "bf16", "fp16", "fp32"],
        default="amp_bf16",
        help="Floating point precision.",
    )
    parser.add_argument("--workers", type=int, default=0)

    # distributed training args
    parser.add_argument('--world_size', type=int, default=4, help='number of gpus')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--horovod",
        default=False,
        action="store_true",
        help="Use horovod for distributed training.",
    )
    parser.add_argument(
        "--no-set-device-rank",
        default=False,
        action="store_true",
        help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc).",
    )

    # Save checkpoints
    parser.add_argument('--output_dir', type=str, default=root_path + '/output', help="output logs and ckpts")
    parser.add_argument("--save_ckpt_per_epochs", type=int, default=10)
    parser.add_argument("--save_latest_states", action='store_true')

    # training
    parser.add_argument('--mode', type=str, default="test", choices=["train", "test"])
    parser.add_argument('--ignoreid', default=-100, type=int, help="criterion: ignore label")

    # datasets
    parser.add_argument('--scene', type=str, default='/data2/songxinshuai/nav_gen/data/hm3d/', help="scene path")
    parser.add_argument('--scene_dataset', type=str, default='/data2/songxinshuai/nav_gen/data/hm3d/hm3d_annotated_basis.scene_dataset_config.json', help="scene dataset path")
    parser.add_argument('--episode_data', type=str, default=None, help='episode data')
    parser.add_argument('--task_data', type=str, default=None, help='task data')
    parser.add_argument('--step_task_data', type=str, default=None, help='step task data')
    parser.add_argument('--train_batch', type=list, default=['/batch_1', '/batch_2', '/batch_3', '/batch_4', '/batch_5'], help='train batch data')
    parser.add_argument('--val_batch', type=list, default=['/batch_6'], help='validation batch data')
    parser.add_argument('--test_batch', type=list, default=['/batch_7', '/batch_8'], help='test batch data')

    args = parser.parse_args()

    args.local_rank, args.rank, args.world_size = world_info_from_env()

    ###################### configurations #########################
    # single-gpu or multi-gpu
    device_id = init_distributed_device(args)
    global_cfg = EasyDict(yaml.safe_load(open(str(Path(args.cfg_file).resolve()))))

    # global config for data path
    args.scene = global_cfg.Path.scene
    args.scene_dataset = global_cfg.Path.scene_dataset
    args.model_name = global_cfg.Model.model_name

    args.episode_data = global_cfg.Path.episode_data
    args.task_data = global_cfg.Path.task_data
    args.step_task_data = global_cfg.Path.step_task_data
    args.train_batch = global_cfg.Path.train_batch
    args.val_batch = global_cfg.Path.val_batch
    args.test_batch = global_cfg.Path.test_batch

    args.save_checkpoints = global_cfg.Path.save_checkpoints

    args.tensorboard_path = global_cfg.Path.tensorboard_path
    args.output_dir = global_cfg.Path.output_dir

    # global config for training
    args.mode = global_cfg.Train.mode
    args.tensorboard = global_cfg.Train.tensorboard
    args.best_checkpoint = global_cfg.Train.best_checkpoint
    args.load_checkpoint = global_cfg.Train.load_checkpoint
    args.save_ckpt_per_epochs = global_cfg.Train.save_ckpt_per_epochs

    args.batch_size = global_cfg.Train.batch_size
    args.num_epochs = global_cfg.Train.num_epochs
    args.ignoreid = global_cfg.Train.ignoreid
    args.max_step = global_cfg.Train.max_step
    args.gradient_accumulation_step = global_cfg.Train.gradient_accumulation_step
    args.num_warmup_steps = global_cfg.Train.num_warmup_steps

    args.success_dis = global_cfg.Train.success_dis

    os.makedirs(args.output_dir, exist_ok=True)
    log_file = Path(args.output_dir) / 'log.txt'

    logger = create_logger(log_file, rank=args.rank)
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(global_cfg, logger=logger)

    print(" + rank: {}, + device_id: {}".format(args.local_rank, device_id))
    print(f"Start running training on rank {args.rank}.")

    if os.path.exists(os.path.join(args.output_dir, "latest_states.pt")):
        state_path = os.path.join(args.output_dir, "latest_states.pt")
        logger.info("Resume checkponit from {}".format(state_path))
        args.resume_from_checkpoint = state_path

    return args, global_cfg, logger, device_id
