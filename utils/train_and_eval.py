import sys

import numpy as np
import torch
import tqdm
from torch.nn import KLDivLoss, MSELoss, L1Loss, CrossEntropyLoss
from torch.nn.functional import cross_entropy

import utils.distributed_utils as utils
from utils.loss import build_target
from utils.tversky_loss import TverskyLoss

mse_loss = MSELoss(size_average=True)
kl_loss = KLDivLoss(size_average=True)
l1_loss = L1Loss(size_average=True)

tversky_loss = TverskyLoss()
CLASS_NAMES = ("background", "plaque", "Stent", "Calcification")


def criterion(inputs, target, num_classes: int = 3):
    losses = {}
    if not isinstance(inputs, dict):
        inputs = {'out': inputs}
    target = build_target(target, num_classes, ignore_index=-1)
    loss_weight = torch.as_tensor([1] + [2] * (num_classes - 1), dtype=inputs['out'].dtype,
                                  device=inputs['out'].device)
    for name, x in inputs.items():
        a = 0.3
        # losses[name] = tversky_loss(x, target)
        # losses[name] = cross_entropy(x, target, weight=loss_weight, label_smoothing=0.1)
        losses[name] = ((1 - a) * cross_entropy(x, target, weight=loss_weight, label_smoothing=0.1)
                        + a * tversky_loss(x, target))
        # Flooding
        # loss = (loss - b).abs() + b
    total_loss = losses['out']
    if len(losses) > 1:
        for k in losses.keys():
            if k != 'out':
                total_loss += 0.3 * losses[k]
    return total_loss


def evaluate(epoch_num, model, data_loader, device, num_classes):
    model.eval()

    confmat = utils.ConfusionMatrix(num_classes)
    val_loss = []
    with torch.no_grad():
        data_loader = tqdm.tqdm(data_loader, file=sys.stdout)
        for image, target in data_loader:
            image, target = image.to(device), target.to(device)

            output = model(image)
            loss = criterion(output, target,
                             num_classes=num_classes)
            if isinstance(output, dict):
                output = output['out']
            confmat.update(target.flatten(), output.argmax(1).flatten())
            val_loss.append(loss.item())
            data_loader.desc = f"[val epoch {epoch_num}] loss: {np.mean(val_loss):.4f}"
        confmat.reduce_from_all_processes()
    dice_per_class = confmat.get_dice_per_class()
    foreground_dice = torch.nanmean(dice_per_class[1:]).item()
    return confmat, foreground_dice, np.mean(val_loss), confmat.get_miou(), dice_per_class.cpu().tolist()


def train_one_epoch(epoch_num, model, optimizer, data_loader, device, num_classes,
                    lr_scheduler, scaler=None):
    model.train()

    confmat = utils.ConfusionMatrix(num_classes)

    train_loss = []
    data_loader = tqdm.tqdm(data_loader, file=sys.stdout)
    for image, target in data_loader:
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target, num_classes=num_classes)

        metric_output = output['out'] if isinstance(output, dict) else output
        confmat.update(target.flatten(), metric_output.argmax(1).flatten())

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        # lr_scheduler.step()

        # if isinstance(output, dict):
        #     output = output['out']
        # confmat.update(target.flatten(), output.argmax(1).flatten())
        # dice.update(output, target)
        train_loss.append(loss.item())

        data_loader.desc = f"[train epoch {epoch_num}] loss: {np.mean(train_loss):.4f} "
    lr_scheduler.step()
    lr = optimizer.param_groups[0]["lr"]
    confmat.reduce_from_all_processes()
    dice_per_class = confmat.get_dice_per_class()
    foreground_dice = torch.nanmean(dice_per_class[1:]).item()
    return np.mean(train_loss), foreground_dice, confmat.get_miou(), lr, dice_per_class.cpu().tolist()
    # return np.mean(train_loss), dice.value.item(), confmat.get_miou(), lr


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.5

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
