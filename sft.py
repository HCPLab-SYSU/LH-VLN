from utils.agent import HabitatAgent, save_checkpoint
from utils.SFT_agent import SFTAgent
from utils.metrics import NavigationMetrics
from utils.dataset import TaskDataset, EpisodeDataset, SFTDataset, create_split_datasets
from utils.parser import read_args, random_seed
from torch.utils.data import DataLoader
from NavModel.RandomNav import RandomAgent
from NavModel.LLMModel.continuous_nav import ContinuousNav
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import numpy as np
import tqdm
import time
import torch


def train_one_sft_epoch(
        args, 
        epoch, 
        dataloader, 
        optimizer,
        lr_scheduler,
        criterion,
        nav_model,
        logger
        ):
    nav_model.model.train()
    lh_losses = []
    st_losses = []

    num_batches_per_epoch = len(dataloader)
    total_training_steps = num_batches_per_epoch * args.num_epochs
    pbar = tqdm.tqdm(
        range(num_batches_per_epoch),
        disable=True,
        total=total_training_steps,
        initial=(epoch * num_batches_per_epoch)
    )
    if args.tensorboard:
        writer_step = SummaryWriter(log_dir=args.tensorboard_path, filename_suffix='epoch_'+str(epoch))
    else:
        writer_step = None
    
    for step, (config, step_configs) in enumerate(dataloader):
        logger.info(f"****** statrt training in step: {step} ******")
        logger.info(config)

        # SFT for LH task
        agent = SFTAgent(args, config, nav_model)
        lh_loss = agent.train(criterion)

        # #training for step task
        st_loss = []
        for step_config in step_configs:
            logger.info(step_config)
            agent = SFTAgent(args, step_config, nav_model)
            st = agent.train(criterion)
            st_loss.append(st)

        if (step + 1) % args.gradient_accumulation_step == 0:
            torch.nn.utils.clip_grad_norm_(nav_model.model.parameters(), 40.)
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
        
        verbose_dict = dict(
            step=step,
            loss=lh_loss,
            lr=lr_scheduler.get_last_lr()[0],
        )
        pbar.set_postfix(verbose_dict)
        pbar.update()

        lh_losses.append(lh_loss)
        st_losses.append(sum(st_loss)/len(st_loss) if len(st_loss) > 0 else [])

        if writer_step:
            writer_step.add_scalars('Training loss', {
                'imiation loss': lh_loss,
                'st_loss': sum(st_loss)/len(st_loss) if len(st_loss) > 0 else float("inf"),
            }, step)

        if step % args.save_ckpt_per_epochs == 0 and step != 0:
            checkpoint_path = args.save_checkpoints + '/' + 'epoch_' + str(epoch) + 'step_' + str(step) + '.pth'
            save_checkpoint(nav_model.model, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")

    if writer_step:
        writer_step.close()

    return sum(lh_losses)/len(lh_losses) if len(lh_losses) > 0 else float("inf"), sum(st_losses)/len(st_losses) if len(st_losses) > 0 else float("inf")


def main():
    args, global_cfg, logger, device_id = read_args()
    random_seed(args.seed)

    if args.tensorboard:
        writer_epoch = SummaryWriter(log_dir=args.tensorboard_path, filename_suffix='epoch')
    else:
        writer_epoch = None
    
    def custom_collate_fn(batch):
        if args.batch_size == 1:
            return batch[0]
        else:
            return batch

    # nav_model = RandomAgent()
    nav_model = ContinuousNav(args, global_cfg, logger, device_id)

    criterion = nn.CrossEntropyLoss(ignore_index=args.ignoreid, reduction='sum')

    train_dataset = SFTDataset(args, mode='train')
    val_dataset = SFTDataset(args, mode='val')
    test_dataset = SFTDataset(args, mode='test')
    if args.split_by_scene:
        train_dataset, _, _ = create_split_datasets([train_dataset, val_dataset, test_dataset], args)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=custom_collate_fn)

    for epoch in range(args.num_epochs):
        lh_loss, st_loss = train_one_sft_epoch(
                args,
                epoch,
                train_dataloader,
                nav_model.optimizer,
                nav_model.lr_scheduler,
                criterion,
                nav_model,
                logger
            )
        
        logger.info(f"###### Epoch: {epoch} ######")
        logger.info("###### Training ######")
        logger.info(f"Imiation learning Loss: {lh_loss:.4f}, dagger training Loss: {st_loss:.4f}")

        if epoch % args.save_ckpt_per_epochs == 0:
            checkpoint_path = args.save_checkpoints + '/' + 'epoch_' + str(epoch) + '.pth'
            save_checkpoint(nav_model.model, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")

        if writer_epoch:
            writer_epoch.add_scalars('Training metric', {
                "lh loss": lh_loss,
                "st loss": st_loss,
            }, epoch)
    
    if writer_epoch:
        writer_epoch.close()


if __name__ == '__main__':
    main()
