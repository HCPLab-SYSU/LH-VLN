import os
import torch
import glob
from transformers import get_constant_schedule_with_warmup


def check_checkpoint(args, model, optimizer, lr_scheduler, logger) -> int:
    resume_from_epoch = 0
    if args.resume_from_checkpoint is not None:
        if args.rank == 0:
            logger.info(f"Loading checkpoint from {args.resume_from_checkpoint}")
        checkpoint = torch.load(args.resume_from_checkpoint, map_location="cpu")
        model_state_dict = model.state_dict()
        state_disk = {k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()}
        update_model_state = {}
        for key, val in state_disk.items():
            if key in model_state_dict and model_state_dict[key].shape == val.shape:
                update_model_state[key] = val
            else:
                logger.info(
                    'Ignore weight %s: %s' % (key, str(val.shape))
                )
        msg = model.load_state_dict(update_model_state, strict=False)
        logger.info(msg)

        if 'epoch' in checkpoint:
            resume_from_epoch = checkpoint['epoch'] + 1
            logger.info("Resume from Epoch {}".format(resume_from_epoch))
            optimizer.load_state_dict(checkpoint['optimizer'])


    return resume_from_epoch


def dist_models(args, model, logger):
    logger.info("*************** init model *************** ")
    # args.rank: global rank.
    total_gpus = torch.cuda.device_count()
    device_id = args.rank % total_gpus

    model.to(device_id)
    
    optimizer = torch.optim.AdamW([p for n, p in model.named_parameters() if p.requires_grad], lr=args.lr)

    lr_scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=args.num_warmup_steps)

    resume_from_epoch = check_checkpoint(
        args, model, optimizer, lr_scheduler, logger,
    )
    param_sums = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("model initialized with {:.2f} M trainable parameters".format(param_sums/1000**2))
    if args.distributed:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[device_id], find_unused_parameters=True)

        # args.batch_size: BATCH_SIZE_PER_GPU
        logger.info('Training in distributed mode : total_batch_size: %d' % (total_gpus * args.batch_size))
    else:
        total_gpus = 1
        logger.info('Training with a single process')

    return model, optimizer, resume_from_epoch, lr_scheduler


def save_checkpoint(model, model_path, optimizer=None, epoch: int=0, save_states: bool=False):
    if hasattr(model, 'module'):
        model = model.module
    
    state_dict = {
        "model_state_dict": model.state_dict()
    }
    if save_states:
        state_dict.update({
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        })

    torch.save(state_dict, model_path)