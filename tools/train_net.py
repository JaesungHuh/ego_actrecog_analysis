#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Train a video classification model."""

import numpy as np
from scipy.stats import gmean
import pprint
from timm.data import Mixup
import torch
import torch.nn as nn
from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats
import pdb
import slowfast.models.losses as losses
import slowfast.models.optimizer as optim
import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.metrics as metrics
import slowfast.utils.misc as misc
from slowfast.datasets import loader
from slowfast.models import build_model
from slowfast.utils.meters import TrainMeter, ValMeter, EPICTrainMeter, EPICValMeter
from timm.utils import NativeScaler

import wandb

logger = logging.get_logger(__name__)


def train_epoch(train_loader, model, optimizer, train_meter, cur_epoch, cfg, wandb_log=False, loss_scaler=None, mixup_fn=None):
    """
    Perform the video training for one epoch.
    Args:
        train_loader (loader): video training loader.
        model (model): the video model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Enable train mode.
    model.train()
    if cfg.BN.FREEZE:
        model.module.freeze_fn('bn_statistics') if cfg.NUM_GPUS > 1 else model.freeze_fn('bn_statistics')

    train_meter.iter_tic()
    data_size = len(train_loader)

    for cur_iter, (inputs, labels, _, meta) in enumerate(train_loader):
        # Transfer the data to the current GPU device.
        if isinstance(inputs, (list,)):
            for i in range(len(inputs)):
                inputs[i] = inputs[i].cuda(non_blocking=True)
        else:
            inputs = inputs.cuda(non_blocking=True)
        if isinstance(labels, (dict,)):
            labels = {k: v.cuda() for k, v in labels.items()}
        else:
            labels = labels.cuda()
        
        train_meter.data_toc()

        # Update the learning rate.
        lr = optim.get_epoch_lr(cur_epoch + float(cur_iter) / data_size, cfg)
        optim.set_lr(optimizer, lr)
        with torch.cuda.amp.autocast(enabled=cfg.SOLVER.USE_MIXED_PRECISION):
            if cfg.DETECTION.ENABLE:
                for key, val in meta.items():
                    if isinstance(val, (list,)):
                        for i in range(len(val)):
                            val[i] = val[i].cuda(non_blocking=True)
                    else:
                        meta[key] = val.cuda(non_blocking=True)
                # Compute the predictions.
                preds = model(inputs, meta["boxes"])

            else:
                # Perform the forward pass.
                preds = model(inputs)
            if mixup_fn is None:
                if isinstance(labels, (dict,)):
                    labels = {k: v.cuda() for k, v in labels.items()}
                else:
                    labels = labels.cuda()

            if isinstance(labels, (dict,)):
                # Explicitly declare reduction to mean.
                #loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")
                loss_fun=nn.CrossEntropyLoss(reduction="mean", label_smoothing=0.2)
                
                # Compute the loss.
                loss_verb = loss_fun(preds[0], labels['verb'])
                loss_noun = loss_fun(preds[1], labels['noun'])
                loss = 0.5 * (loss_verb + loss_noun)

                # check Nan Loss.
                misc.check_nan_losses(loss)
            else:
                # Explicitly declare reduction to mean.
                #loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")
                loss_fun=nn.CrossEntropyLoss(reduction="mean", label_smoothing=0.2)

                # Compute the loss.
                loss = loss_fun(preds, labels)

                # check Nan Loss.
                misc.check_nan_losses(loss)

        # Perform the backward pass.
        optimizer.zero_grad()
        if cfg.SOLVER.USE_MIXED_PRECISION: # Mixed Precision Training
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss_scaler(loss, optimizer, clip_grad=cfg.SOLVER.CLIP_GRAD,
                    parameters=model.parameters(), create_graph=is_second_order)
        else:
            loss.backward()
            # Update the parameters.
            optimizer.step()

        if cfg.DETECTION.ENABLE:
            if cfg.NUM_GPUS > 1:
                loss = du.all_reduce([loss])[0]
            loss = loss.item()

            train_meter.iter_toc()
            # Update and log stats.
            train_meter.update_stats(None, None, None, loss, lr)
        else:
            if isinstance(labels, (dict,)):
                # Compute the verb accuracies.
                verb_top1_acc, verb_top5_acc = metrics.topk_accuracies(preds[0], labels['verb'], (1, 5))

                # Gather all the predictions across all the devices.
                if cfg.NUM_GPUS > 1:
                    loss_verb, verb_top1_acc, verb_top5_acc = du.all_reduce(
                        [loss_verb, verb_top1_acc, verb_top5_acc]
                    )

                # Copy the stats from GPU to CPU (sync point).
                loss_verb, verb_top1_acc, verb_top5_acc = (
                    loss_verb.item(),
                    verb_top1_acc.item(),
                    verb_top5_acc.item(),
                )

                # Compute the noun accuracies.
                noun_top1_acc, noun_top5_acc = metrics.topk_accuracies(preds[1], labels['noun'], (1, 5))

                # Gather all the predictions across all the devices.
                if cfg.NUM_GPUS > 1:
                    loss_noun, noun_top1_acc, noun_top5_acc = du.all_reduce(
                        [loss_noun, noun_top1_acc, noun_top5_acc]
                    )

                # Copy the stats from GPU to CPU (sync point).
                loss_noun, noun_top1_acc, noun_top5_acc = (
                    loss_noun.item(),
                    noun_top1_acc.item(),
                    noun_top5_acc.item(),
                )

                # Compute the action accuracies.
                action_top1_acc, action_top5_acc = metrics.multitask_topk_accuracies((preds[0], preds[1]),
                                                                                     (labels['verb'], labels['noun']),
                                                                                     (1, 5))
                # Gather all the predictions across all the devices.
                if cfg.NUM_GPUS > 1:
                    loss, action_top1_acc, action_top5_acc = du.all_reduce(
                        [loss, action_top1_acc, action_top5_acc]
                    )

                # Copy the stats from GPU to CPU (sync point).
                loss, action_top1_acc, action_top5_acc = (
                    loss.item(),
                    action_top1_acc.item(),
                    action_top5_acc.item(),
                )

                train_meter.iter_toc()
                # Update and log stats.
                train_meter.update_stats(
                    (verb_top1_acc, noun_top1_acc, action_top1_acc),
                    (verb_top5_acc, noun_top5_acc, action_top5_acc),
                    (loss_verb, loss_noun, loss),
                    lr, inputs[0].size(0) * cfg.NUM_GPUS
                )
                if cfg.WANDB.ENABLE and du.is_master_proc(
                    cfg.NUM_GPUS * cfg.NUM_SHARDS
                ):
                    wandb.log(
                            {
                                "Train/loss": loss,
                                "Train/lr": lr[0] if isinstance(lr, (list,)) else lr,
                                "Train/Top1_acc": action_top1_acc,
                                "Train/Top5_acc": action_top5_acc,
                                "Train/verb/loss": loss_verb,
                                "Train/noun/loss": loss_noun,
                                "Train/verb/Top1_acc": verb_top1_acc,
                                "Train/verb/Top5_acc": verb_top5_acc,
                                "Train/noun/Top1_acc": noun_top1_acc,
                                "Train/noun/Top5_acc": noun_top5_acc,
                                "train_step": data_size * cur_epoch + cur_iter,
                            }
                        )     
            else:
                # Compute the errors.
                num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))
                top1_err, top5_err = [
                    (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
                ]

                # Gather all the predictions across all the devices.
                if cfg.NUM_GPUS > 1:
                    loss, top1_err, top5_err = du.all_reduce(
                        [loss, top1_err, top5_err]
                    )

                # Copy the stats from GPU to CPU (sync point).
                loss, top1_err, top5_err = (
                    loss.item(),
                    top1_err.item(),
                    top5_err.item(),
                )

                train_meter.iter_toc()
                # Update and log stats.
                train_meter.update_stats(
                    top1_err, top5_err, loss, lr, inputs[0].size(0) * cfg.NUM_GPUS
                )


                if wandb_log:
                    wandb.log(
                        {
                            "Train/loss": loss,
                            "Train/lr": lr[0] if isinstance(lr, (list,)) else lr,
                            "Train/Top1_err": top1_err,
                            "Train/Top5_err": top5_err,
                            "train_step": data_size * cur_epoch + cur_iter,
                        },
                    )

        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()
    # Log epoch stats.
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()


@torch.no_grad()
def eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, wandb_log=False):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Evaluation mode enabled. The running stats would not be updated.
    model.eval()
    val_meter.iter_tic()

    for cur_iter, (inputs, labels, _, meta) in enumerate(val_loader):
        # Transferthe data to the current GPU device.
        if cfg.NUM_GPUS:
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            if isinstance(labels, (dict,)):
                labels = {k: v.cuda() for k, v in labels.items()}
            else:
                labels = labels.cuda()

        val_meter.data_toc()
        
        with torch.cuda.amp.autocast(enabled=cfg.SOLVER.USE_MIXED_PRECISION):
            if cfg.DETECTION.ENABLE:
                for key, val in meta.items():
                    if isinstance(val, (list,)):
                        for i in range(len(val)):
                            val[i] = val[i].cuda(non_blocking=True)
                    else:
                        meta[key] = val.cuda(non_blocking=True)
                # Compute the predictions.
                preds = model(inputs, meta["boxes"])

                preds = preds.cpu()
                ori_boxes = meta["ori_boxes"].cpu()
                metadata = meta["metadata"].cpu()

                if cfg.NUM_GPUS > 1:
                    preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
                    ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes), dim=0)
                    metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)

                val_meter.iter_toc()
                # Update and log stats.
                val_meter.update_stats(preds.cpu(), ori_boxes.cpu(), metadata.cpu())
            else:
                preds = model(inputs)

                if isinstance(labels, (dict,)):
                    # Compute the verb accuracies.
                    verb_top1_acc, verb_top5_acc = metrics.topk_accuracies(preds[0], labels['verb'], (1, 5))

                    # Combine the errors across the GPUs.
                    if cfg.NUM_GPUS > 1:
                        verb_top1_acc, verb_top5_acc = du.all_reduce([verb_top1_acc, verb_top5_acc])

                    # Copy the errors from GPU to CPU (sync point).
                    verb_top1_acc, verb_top5_acc = verb_top1_acc.item(), verb_top5_acc.item()

                    # Compute the noun accuracies.
                    noun_top1_acc, noun_top5_acc = metrics.topk_accuracies(preds[1], labels['noun'], (1, 5))

                    # Combine the errors across the GPUs.
                    if cfg.NUM_GPUS > 1:
                        noun_top1_acc, noun_top5_acc = du.all_reduce([noun_top1_acc, noun_top5_acc])

                    # Copy the errors from GPU to CPU (sync point).
                    noun_top1_acc, noun_top5_acc = noun_top1_acc.item(), noun_top5_acc.item()

                    # Compute the action accuracies.
                    action_top1_acc, action_top5_acc = metrics.multitask_topk_accuracies((preds[0], preds[1]),
                                                                                        (labels['verb'], labels['noun']),
                                                                                        (1, 5))
                    # Combine the errors across the GPUs.
                    if cfg.NUM_GPUS > 1:
                        action_top1_acc, action_top5_acc = du.all_reduce([action_top1_acc, action_top5_acc])

                    # Copy the errors from GPU to CPU (sync point).
                    action_top1_acc, action_top5_acc = action_top1_acc.item(), action_top5_acc.item()

                    val_meter.iter_toc()
                    # Update and log stats.
                    val_meter.update_stats(
                        (verb_top1_acc, noun_top1_acc, action_top1_acc),
                        (verb_top5_acc, noun_top5_acc, action_top5_acc),
                        inputs[0].size(0) * cfg.NUM_GPUS
                    )
                    val_meter.iter_tic()

                    if wandb_log:
                        wandb.log(
                            {
                                "Val/loss": loss,
                                "Val/Top1_acc": action_top1_acc,
                                "Val/Top5_acc": action_top5_acc,
                                "Val/verb/loss": loss_verb,
                                "Val/verb/Top1_acc": verb_top1_acc,
                                "Val/verb/Top5_acc": verb_top5_acc,
                                "Val/noun/loss": loss_noun,
                                "Val/noun/Top1_acc": noun_top1_acc,
                                "Val/noun/Top5_acc": noun_top5_acc,
                                "val_step": val_step  + cur_iter
                            }
                        )
                else:
                    # Compute the errors.
                    num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))

                    # Combine the errors across the GPUs.
                    top1_err, top5_err = [
                        (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
                    ]
                    if cfg.NUM_GPUS > 1:
                        top1_err, top5_err = du.all_reduce([top1_err, top5_err])

                    # Copy the errors from GPU to CPU (sync point).
                    top1_err, top5_err = top1_err.item(), top5_err.item()

                    val_meter.iter_toc()
                    # Update and log stats.
                    val_meter.update_stats(
                        top1_err, top5_err, inputs[0].size(0) * cfg.NUM_GPUS
                    )
                    val_meter.iter_tic()
        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

    # Log epoch stats.
    is_best_epoch, top1_dict = val_meter.log_epoch_stats(cur_epoch)
    
    if cfg.WANDB.ENABLE and du.is_master_proc(cfg.NUM_GPUS * cfg.NUM_SHARDS):
        if "top1_acc" in top1_dict.keys():
            wandb.log(
                {
                    "Val/epoch/Top1_acc": top1_dict["top1_acc"],
                    "Val/epoch/verb/Top1_acc": top1_dict["verb_top1_acc"],
                    "Val/epoch/noun/Top1_acc": top1_dict["noun_top1_acc"],
                    "Val/epoch/Top5_acc": top1_dict["top5_acc"],
                    "Val/epoch/verb/Top5_acc": top1_dict["verb_top5_acc"],
                    "Val/epoch/noun/Top5_acc": top1_dict["noun_top5_acc"],
                    "epoch": cur_epoch,
                },
            )
        else:
            wandb.log(
                {
                    "Val/epoch/Top1_err": top1_dict["top1_err"], 
                    "Val/epoch/Top5_err": top1_dict["top5_err"],
                    "Val/epoch/mAP": top1_dict["mAP"],
                    "Val/epoch/mAUC": top1_dict["mAUC"],
                    "epoch": cur_epoch
                }
            )

    val_meter.reset()
    return is_best_epoch


def calculate_and_update_precise_bn(loader, model, num_iters=200, use_gpu=True):
    """
    Update the stats in bn layers by calculate the precise stats.
    Args:
        loader (loader): data loader to provide training data.
        model (model): model to update the bn stats.
        num_iters (int): number of iterations to compute and update the bn stats.
    """

    def _gen_loader():
        for inputs, _, _, _ in loader:
            if use_gpu:
                if isinstance(inputs, (list,)):
                    for i in range(len(inputs)):
                        inputs[i] = inputs[i].cuda(non_blocking=True)
                else:
                    inputs = inputs.cuda(non_blocking=True)
            yield inputs

    # Update the bn stats.
    update_bn_stats(model, _gen_loader(), num_iters)


def train(cfg):
    """
    Train a video model for many epochs on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging()

    # Print config.
    logger.info("Train with config:")
    logger.info(pprint.pformat(cfg))

    # Build the video model and print model statistics.
    model = build_model(cfg)
    if cfg.MODEL.MODEL_NAME == 'SlowFast':
        misc.log_model_info(model, cfg, is_train=False)
        # TODO: make it work with Omnivore and TSM


    if cfg.BN.FREEZE:
        model.module.freeze_fn('bn_parameters') if cfg.NUM_GPUS > 1 else model.freeze_fn('bn_parameters')

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)

    # Mixed Precision Training Scaler
    if cfg.SOLVER.USE_MIXED_PRECISION:
        loss_scaler = NativeScaler()
    else:
        loss_scaler = None

    # Load a checkpoint to resume training if applicable.
    if cfg.TRAIN.AUTO_RESUME and cu.has_checkpoint(cfg.OUTPUT_DIR):
        logger.info("Load from last checkpoint.")
        last_checkpoint = cu.get_last_checkpoint(cfg.OUTPUT_DIR)
        checkpoint_epoch = cu.load_checkpoint(
            last_checkpoint, model, cfg.NUM_GPUS > 1, optimizer
        )
        start_epoch = checkpoint_epoch + 1
    elif cfg.TRAIN.CHECKPOINT_FILE_PATH != "" and not cfg.TRAIN.FINETUNE:
        logger.info("Load from given checkpoint file.")
        checkpoint_epoch = cu.load_checkpoint(
            cfg.TRAIN.CHECKPOINT_FILE_PATH,
            model,
            cfg.NUM_GPUS > 1,
            optimizer,
            inflation=cfg.TRAIN.CHECKPOINT_INFLATE,
            convert_from_caffe2=cfg.TRAIN.CHECKPOINT_TYPE == "caffe2",
        )
        start_epoch = checkpoint_epoch + 1
    else:
        start_epoch = 0

    # Create the video train and val loaders.
    if cfg.TRAIN.DATASET != 'epickitchens' or not cfg.EPICKITCHENS.TRAIN_PLUS_VAL:
        train_loader = loader.construct_loader(cfg, "train")
        val_loader = loader.construct_loader(cfg, "val")
    else:
        train_loader = loader.construct_loader(cfg, "train+val")
        val_loader = loader.construct_loader(cfg, "val")

    # Create meters.
    # if cfg.DETECTION.ENABLE:
    #     train_meter = AVAMeter(len(train_loader), cfg, mode="train")
    #     val_meter = AVAMeter(len(val_loader), cfg, mode="val")
    # else:
    #     if cfg.TRAIN.DATASET == 'epickitchens':
    #         train_meter = EPICTrainMeter(len(train_loader), cfg)
    #         val_meter = EPICValMeter(len(val_loader), cfg)
    #     else:
    #         train_meter = TrainMeter(len(train_loader), cfg)
    #         val_meter = ValMeter(len(val_loader), cfg)

    train_meter = TrainMeter(len(train_loader), cfg)
    val_meter = ValMeter(len(val_loader), cfg)

    if cfg.WANDB.ENABLE and du.is_master_proc(cfg.NUM_GPUS * cfg.NUM_SHARDS):
        wandb_log = True
        if cfg.TRAIN.AUTO_RESUME and cfg.WANDB.RUN_ID != "":
            wandb.init(project=cfg.MODEL.MODEL_NAME, config=cfg, sync_tensorboard=True, resume=cfg.WANDB.RUN_ID)
        else:
            wandb.init(project=cfg.MODEL.MODEL_NAME, config=cfg, sync_tensorboard=True)
        wandb.watch(model)

    else:
        wandb_log = False

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))
    
    mixup_fn = None
    
    mixup_active = cfg.MIXUP.MIXUP_ALPHA > 0 or cfg.MIXUP.CUTMIX_ALPHA > 0 or cfg.MIXUP.CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=cfg.MIXUP.MIXUP_ALPHA, 
            cutmix_alpha=cfg.MIXUP.CUTMIX_ALPHA, 
            cutmix_minmax=cfg.MIXUP.CUTMIX_MINMAX,
            prob=cfg.MIXUP.MIXUP_PROB, 
            switch_prob=cfg.MIXUP.MIXUP_SWITCH_PROB, 
            mode=cfg.MIXUP.MIXUP_MODE,
            label_smoothing=cfg.SOLVER.SMOOTHING, 
            num_classes=cfg.MODEL.NUM_CLASSES
        )


    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        # Shuffle the dataset.
        loader.shuffle_dataset(train_loader, cur_epoch)
        # Train for one epoch.
        train_epoch(train_loader, model, optimizer, train_meter, cur_epoch, cfg, wandb_log, loss_scaler=loss_scaler, mixup_fn=mixup_fn)

        # Compute precise BN stats.
        if cfg.BN.USE_PRECISE_STATS and len(get_bn_modules(model)) > 0:
            calculate_and_update_precise_bn(
                train_loader, model, cfg.BN.NUM_BATCHES_PRECISE
            )

        # Save a checkpoint.
        if cu.is_checkpoint_epoch(cur_epoch, cfg.TRAIN.CHECKPOINT_PERIOD):
            print("Checkpoint saving start")
            cu.save_checkpoint(cfg.OUTPUT_DIR, model, optimizer, cur_epoch, cfg, loss_scaler=loss_scaler)
            print("Checkpoint saving end")
        # Evaluate the model on validation set.
        if misc.is_eval_epoch(cfg, cur_epoch):
            print("Evaluation start")
            is_best_epoch = eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, wandb_log)
            if is_best_epoch:
                cu.save_checkpoint(cfg.OUTPUT_DIR, model, optimizer, cur_epoch, cfg, is_best_epoch=is_best_epoch)
            print("Evaluation end")

