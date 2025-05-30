import os
import torch


def is_global_master(args):
    return args.rank == 0


def is_local_master(args):
    return args.local_rank == 0


def is_master(args, local=False):
    return is_local_master(args) if local else is_global_master(args)


def is_using_horovod():
    # NOTE w/ horovod run, OMPI vars should be set, but w/ SLURM PMI vars will be set
    # Differentiating between horovod and DDP use via SLURM may not be possible, so horovod arg still required...
    ompi_vars = ["OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_SIZE"]
    pmi_vars = ["PMI_RANK", "PMI_SIZE"]
    if all([var in os.environ for var in ompi_vars]) or all(
        [var in os.environ for var in pmi_vars]
    ):
        return True
    else:
        return False


def is_using_distributed():
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"]) > 1
    if "SLURM_NTASKS" in os.environ:
        return int(os.environ["SLURM_NTASKS"]) > 1
    return False


def world_info_from_env():
    local_rank = 0
    for v in (
        "LOCAL_RANK",
        "MPI_LOCALRANKID",
        "SLURM_LOCALID",
        "OMPI_COMM_WORLD_LOCAL_RANK",
    ):
        if v in os.environ:
            local_rank = int(os.environ[v])
            break
    global_rank = 0
    for v in ("RANK", "PMI_RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK"):
        if v in os.environ:
            global_rank = int(os.environ[v])
            break
    world_size = 1
    for v in ("WORLD_SIZE", "PMI_SIZE", "SLURM_NTASKS", "OMPI_COMM_WORLD_SIZE"):
        if v in os.environ:
            world_size = int(os.environ[v])
            break

    return local_rank, global_rank, world_size

from torch import distributed as torch_dist
import subprocess
def _init_dist_slurm(backend, port=None) -> None:
    """Initialize slurm distributed training environment.

    If argument ``port`` is not specified, then the master port will be system
    environment variable ``MASTER_PORT``. If ``MASTER_PORT`` is not in system
    environment variable, then a default port ``29500`` will be used.

    Args:
        backend (str): Backend of torch.distributed.
        port (int, optional): Master port. Defaults to None.
    """
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    # Not sure when this environment variable could be None, so use a fallback
    local_rank_env = os.environ.get('SLURM_LOCALID', None)
    if local_rank_env is not None:
        local_rank = int(local_rank_env)
    else:
        num_gpus = torch.cuda.device_count()
        local_rank = proc_id % num_gpus
    torch.cuda.set_device(local_rank)
    addr = subprocess.getoutput(
        f'scontrol show hostname {node_list} | head -n1')
    # specify master port
    if port is not None:
        os.environ['MASTER_PORT'] = str(port)
    elif 'MASTER_PORT' in os.environ:
        pass  # use MASTER_PORT in the environment variable
    else:
        # 29500 is torch.distributed default port
        os.environ['MASTER_PORT'] = '29500'
    # use MASTER_ADDR in the environment variable if it already exists
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['LOCAL_RANK'] = str(local_rank)
    os.environ['RANK'] = str(proc_id)
    torch_dist.init_process_group(backend=backend)


def init_distributed_device(args):
    # Distributed training = training on more than one GPU.
    # Works in both single and multi-node scenarios.
    args.distributed = False
    args.world_size = 1
    args.rank = 0  # global rank
    args.local_rank = 0
    # if args.horovod:
    #     assert hvd is not None, "Horovod is not installed"
    #     hvd.init()
    #     args.local_rank = int(hvd.local_rank())
    #     args.rank = hvd.rank()
    #     args.world_size = hvd.size()
    #     args.distributed = True
    #     os.environ["LOCAL_RANK"] = str(args.local_rank)
    #     os.environ["RANK"] = str(args.rank)
    #     os.environ["WORLD_SIZE"] = str(args.world_size)
    # elif is_using_distributed():
    if is_using_distributed():
        if "SLURM_PROCID" in os.environ:
            # DDP via SLURM
            args.local_rank, args.rank, args.world_size = world_info_from_env()
            # SLURM var -> torch.distributed vars in case needed
            os.environ["LOCAL_RANK"] = str(args.local_rank)
            os.environ["RANK"] = str(args.rank)
            os.environ["WORLD_SIZE"] = str(args.world_size)
            torch.distributed.init_process_group(
                backend=args.dist_backend,
                init_method=args.dist_url,
                world_size=args.world_size,
                rank=args.rank,
            )
        else:
            # DDP via torchrun, torch.distributed.launch
            args.local_rank, _, _ = world_info_from_env()
            torch.distributed.init_process_group(
                backend=args.dist_backend, init_method=args.dist_url
            )
            args.world_size = torch.distributed.get_world_size()
            args.rank = torch.distributed.get_rank()
        args.distributed = True
    else:
        DistSingleGPU = False
        if DistSingleGPU:
            # TODO in S2: "torchrun --nnodes=1 --nproc_per_node=1" Bug
            # DistSingleGPU = False
            os.environ['MASTER_ADDR'] = '127.0.0.1'
            from socket import socket
            with socket() as s:
                s.bind(('', 0))
                free_port = str(s.getsockname()[1])
            os.environ['MASTER_PORT'] = free_port
            # TODO 注释此处 for debug
            # needed to run on single gpu
            torch.distributed.init_process_group(
                backend=args.dist_backend,
                init_method=args.dist_url,
                world_size=1,
                rank=0,
            )
            print('[INFO] single gpu run')
            # args.distributed = False
        else:
            args.distributed = False
            print('[INFO] single gpu: Not distributed')

    if torch.cuda.is_available():
        if args.distributed and not args.no_set_device_rank:
            device = "cuda:%d" % args.local_rank
        else:
            device = "cuda:0"
        torch.cuda.set_device(device)
    else:
        device = "cpu"
    args.device = device
    device = torch.device(device)
    if args.distributed:
        print('[INFO] distributed: True')
    return device
