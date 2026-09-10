import argparse

import torch

from segmentation_config import CLASS_NAMES, IMAGE_SIZE

from network.UNet import UNet
from network.efficientnet_unet import EfficientUNet


def get_device(requested="cuda"):
    device = torch.device(requested if requested.startswith("cuda") and torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    return device


def get_best_weight_path(args, verbose=True):
    # Keep the existing *_0_* checkpoint names for backward compatibility.
    weights_path = "save_weights/{}_0_best_model.pth".format(args.arch)
    if verbose:
        print("best weight: ", weights_path)
    return weights_path


def get_latest_weight_path(args, verbose=False):
    weights_path = "save_weights/{}_0_latest_model.pth".format(args.arch)
    if verbose:
        print("latest weight: ", weights_path)
    return weights_path


efficientnet_dict = ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2',
                     'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5',
                     'efficientnet_b6', 'efficientnet_b7', 'efficientnet_v2_s']


def get_model(args, pretrain_backbone=True):
    print('★' * 30)
    print(f'model:{args.arch}\n'
          f'epoch:{args.epochs}\n'
          f'batch size:{args.batch_size}\n'
          f'image size:{args.image_size}')
    print('★' * 30)
    device = get_device(args.device)
    if args.arch == 'unet':
        model = UNet(in_channels=args.in_channels, num_classes=args.num_classes, base_c=32).to(device)
    elif args.arch == 'efficientnet' or args.arch in efficientnet_dict:
        model = EfficientUNet(num_classes=args.num_classes, pretrain_backbone=pretrain_backbone,
                              model_name=args.arch).to(device)
    else:
        raise ValueError('arch error')
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="pytorch training")
    parser.add_argument('--arch', '-a', metavar='ARCH', default='efficientnet_b1', help='unet/efficientnet_b1')
    parser.add_argument("--data_path", default="data/oct_augmented_dataset",
                        help="augmented dataset root containing train/ and test/")
    parser.add_argument("--num_classes", default=len(CLASS_NAMES), type=int)
    parser.add_argument("--in_channels", default=1, type=int)
    parser.add_argument("--image_size", default=IMAGE_SIZE, type=int)
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch_size", default=4, type=int)
    parser.add_argument("--epochs", default=300, type=int, metavar="N",
                        help="number of total epochs to train")
    # Optimizer options
    parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--resume', default=0, type=int, help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=1, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--save_best', default=True, type=bool, help='only save best metric weights')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    model = get_model(args)
    print(model)
