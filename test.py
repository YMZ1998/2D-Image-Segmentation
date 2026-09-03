import os

import time
import datetime
import torch

from torch.utils.data import DataLoader

from utils.train_and_eval import CLASS_NAMES, evaluate
from utils.dataset import MyDataset
from parse_args import parse_args, get_model, get_device,get_best_weight_path


def model_test():
    args = parse_args()

    device = get_device(args.device)
    batch_size = args.batch_size
    num_classes = args.num_classes

    val_dataset = MyDataset(args.data_path + "/test", args.image_size)

    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            pin_memory=True)
    model = get_model(args)
    weights_path = get_best_weight_path(args)
    # print(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    start_time = time.time()

    confmat, val_dice, val_loss, val_miou, class_dice = evaluate(
        0, model, val_loader, device=device, num_classes=num_classes)
    class_dice_text = " | ".join(
        f"{name}: {score * 100:.2f}%" for name, score in zip(CLASS_NAMES, class_dice)
    )

    print(f"val_loss: {val_loss:.4f}\n"
          f"val dice: {val_dice * 100:.2f}\n"
          f"val miou: {val_miou * 100:.2f}\n"
          f"Dice per class: {class_dice_text}\n"
          )
    val_info = str(confmat)
    print(val_info)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("test time {}".format(total_time_str))


if __name__ == '__main__':
    model_test()
